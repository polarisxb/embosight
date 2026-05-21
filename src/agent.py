"""EmboSightAgent v1 主循环。

主入口: agent.run(query, env) -> EpisodeResult
内部循环: while not belief.is_confident_to_act(): decide_next(belief)

设计参考: §5
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from src.world_belief import (
    Action,
    Evidence,
    EpisodeResult,
    Hypothesis,
    WorldBelief,
)
from src.memory_manager import MemoryEntry, MemoryManager

logger = logging.getLogger(__name__)


class EmboSightAgent:

    MAX_STEPS = 12
    MAX_RE_OBSERVE = 3
    MAX_ASK_USER = 3

    def __init__(
        self,
        task_decomposer,
        perception,
        safety_classifier,
        grasp_planner,
        action_executor,
        nbv_selector,
        user_channel,
        episode_logger,
        viewpoint_lib,
        llm,
        vlm,
        memory_manager: Optional[MemoryManager] = None,
    ):
        self.task_decomposer = task_decomposer
        self.perception = perception
        self.safety = safety_classifier
        self.grasp_planner = grasp_planner
        self.executor = action_executor
        self.nbv = nbv_selector
        self.user_channel = user_channel
        self.logger = episode_logger
        self.vp_lib = viewpoint_lib
        self.llm = llm
        self.vlm = vlm
        self.memory = memory_manager or MemoryManager()

    # ──────────────────────────────────────
    # 测试用工厂 (with_test_doubles)
    # ──────────────────────────────────────

    @classmethod
    def with_test_doubles(
        cls,
        vp_lib,
        nbv_llm=None,
        **overrides,
    ) -> "EmboSightAgent":
        """构造一个所有依赖都是 None 占位的 agent (decide_next 单测用)。"""
        from src.active_planner import ActiveViewpointSelector
        from tests._mocks import MockLLM, MockVLM
        nbv_llm = nbv_llm or MockLLM(responses=["0"] * 100)
        return cls(
            task_decomposer=overrides.get("task_decomposer"),
            perception=overrides.get("perception"),
            safety_classifier=overrides.get("safety_classifier"),
            grasp_planner=overrides.get("grasp_planner"),
            action_executor=overrides.get("action_executor"),
            nbv_selector=ActiveViewpointSelector(llm=nbv_llm, viewpoint_lib=vp_lib),
            user_channel=overrides.get("user_channel"),
            episode_logger=overrides.get("episode_logger"),
            viewpoint_lib=vp_lib,
            llm=overrides.get("llm", MockLLM([])),
            vlm=overrides.get("vlm", MockVLM([])),
        )

    # ──────────────────────────────────────
    # decide_next (核心决策树, §5.2)
    # ──────────────────────────────────────

    def decide_next(self, belief: WorldBelief) -> Action:
        # ask_user 次数统计
        ask_count = sum(1 for a in belief.action_history if a.kind == "ask_user")

        # 降级: ask_user 超限 → 强制选最佳 hypothesis 直接抓
        if ask_count >= self.MAX_ASK_USER:
            fallback = self._force_best_hypothesis(belief)
            if fallback is not None:
                return fallback

        # 阶段 0: 已 confident → grasp
        if belief.is_confident_to_act():
            return Action(kind="grasp", target_hypothesis=belief.target())

        # 阶段 A: 还没看够 → init view
        if not belief.evidence:
            return Action(kind="observe", viewpoint=self.vp_lib[0])

        target = belief.target()

        # 阶段 B: 没找到 target → NBV / ask_user
        if target is None:
            next_vp = self.nbv.select(
                belief, exclude=belief.used_views(),
                preference="search_target",
            )
            if next_vp is None:
                # LLM semantic fallback: 用 LLM 判断现有 hypothesis 是否语义匹配
                if belief.hypotheses and self._llm_semantic_fallback(belief):
                    target = belief.target()
                    if target is not None:
                        # 成功桥接, 跳过 ask_user, 进入正常决策
                        return self._decide_with_target(belief, target)
                primary = (belief.decomposed.primary_target
                           if belief.decomposed else "目标")
                return Action(
                    kind="ask_user",
                    question=f"我没在场景里看到{primary}, 是不是被挡住了?",
                )
            return Action(kind="observe", viewpoint=next_vp)

        # 阶段 C: re_observe 超限 → ask_user
        if target.times_re_observed >= self.MAX_RE_OBSERVE:
            return Action(kind="ask_user", question=belief.compose_clarification())

        # 阶段 D: 哪轴最不确定就消除哪轴
        axis = belief.most_uncertain_axis()

        # 兜底: label/pos/safety 都 confident 但 grasp 没 plan
        # 阈值与 WorldBelief.DEFAULT_THRESHOLDS 对齐 (真 VLM 在 sim 上的典型输出)
        if (belief.is_label_confident(target)
                and target.position_std_m < 0.10
                and belief.is_safety_confident(target)
                and target.grasp_uncertainty is None):
            return Action(kind="plan_grasp_candidates", target_hypothesis=target)

        if axis == "label":
            if not self._has_zoomed(target):
                return Action(kind="re_observe", target_hypothesis=target,
                              strategy="zoom_in")
            alt2 = (target.label_alternatives[1][0]
                    if len(target.label_alternatives) > 1 else "别的")
            return Action(
                kind="ask_user",
                question=f"我看到一个{target.label}样的东西, 也可能是{alt2}, 您要的是哪个?",
            )

        if axis == "position":
            return Action(kind="re_observe", target_hypothesis=target,
                          strategy="parallax_view")

        if axis == "safety":
            return Action(kind="classify_safety", target_hypothesis=target)

        if axis == "grasp":
            if not target.grasp_candidates:
                return Action(kind="plan_grasp_candidates", target_hypothesis=target)
            if target.pose_uncertainty > 0.5:
                return Action(kind="re_observe", target_hypothesis=target,
                              strategy="parallax_for_pose")
            return Action(
                kind="ask_user",
                question=f"我没法抓到{target.label}, 它现在是横放还是竖放?",
            )

        return Action(kind="give_up",
                      metadata={"reason": "unreachable decision branch"})

    @staticmethod
    def _has_zoomed(h: Hypothesis) -> bool:
        return h.times_re_observed > 0

    def _decide_with_target(self, belief: WorldBelief, target: Hypothesis) -> Action:
        """target 已确定时的决策 (阶段 C-E), 避免代码重复。"""
        if target.times_re_observed >= self.MAX_RE_OBSERVE:
            return Action(kind="ask_user", question=belief.compose_clarification())
        axis = belief.most_uncertain_axis()
        if (belief.is_label_confident(target)
                and target.position_std_m < 0.10
                and belief.is_safety_confident(target)
                and target.grasp_uncertainty is None):
            return Action(kind="plan_grasp_candidates", target_hypothesis=target)
        if axis == "label":
            if not self._has_zoomed(target):
                return Action(kind="re_observe", target_hypothesis=target,
                              strategy="zoom_in")
            alt2 = (target.label_alternatives[1][0]
                    if len(target.label_alternatives) > 1 else "别的")
            return Action(
                kind="ask_user",
                question=f"我看到一个{target.label}样的东西, 也可能是{alt2}, 您要的是哪个?",
            )
        if axis == "position":
            return Action(kind="re_observe", target_hypothesis=target,
                          strategy="parallax_view")
        if axis == "safety":
            return Action(kind="classify_safety", target_hypothesis=target)
        if axis == "grasp":
            if not target.grasp_candidates:
                return Action(kind="plan_grasp_candidates", target_hypothesis=target)
            if target.pose_uncertainty > 0.5:
                return Action(kind="re_observe", target_hypothesis=target,
                              strategy="parallax_for_pose")
            return Action(
                kind="ask_user",
                question=f"我没法抓到{target.label}, 它现在是横放还是竖放?",
            )
        return Action(kind="give_up",
                      metadata={"reason": "unreachable decision branch"})

    def _force_best_hypothesis(self, belief: WorldBelief) -> Optional[Action]:
        """ask_user 超限后降级: 选置信度最高的 hypothesis 强制推进。

        逻辑:
        1. 取 target() (忽略 ambiguity); 若无, 取 label_alternatives[0] 概率最高的 hyp
        2. 强制注入 primary_target 到该 hyp (让 target() 能返回它)
        3. 按缺失阶段返回下一步 action (safety → plan → grasp)
        """
        best = belief.target(ignore_ambiguity=True)
        if best is None and belief.hypotheses:
            # 概率排序选最高置信 hypothesis
            best = max(
                belief.hypotheses,
                key=lambda h: h.label_alternatives[0][1] if h.label_alternatives else 0,
            )
        if best is None:
            return None

        primary = (belief.decomposed.primary_target
                   if belief.decomposed else best.label)
        logger.warning(
            "[agent] ask_user limit reached — forcing best hypothesis: "
            "%s (label='%s') as '%s'",
            best.object_id, best.label, primary,
        )

        # 强制注入 primary_target, 让 target() 能找到它
        from src.perception import _label_key, _shannon
        primary_key = _label_key(primary)
        already = any(
            primary_key in _label_key(lbl)
            for lbl, _ in best.label_alternatives
        )
        if not already:
            best.label_alternatives.append((primary, 0.50))
            total = sum(p for _, p in best.label_alternatives) or 1.0
            best.label_alternatives = sorted(
                ((lbl, p / total) for lbl, p in best.label_alternatives),
                key=lambda x: x[1], reverse=True,
            )
            best.label_entropy = _shannon([p for _, p in best.label_alternatives])
        else:
            # 已有但概率可能不够 — 强制提升到 0.50
            new_alts = []
            for lbl, p in best.label_alternatives:
                if primary_key in _label_key(lbl):
                    new_alts.append((lbl, max(p, 0.50)))
                else:
                    new_alts.append((lbl, p))
            total = sum(p for _, p in new_alts) or 1.0
            best.label_alternatives = sorted(
                ((lbl, p / total) for lbl, p in new_alts),
                key=lambda x: x[1], reverse=True,
            )
            best.label_entropy = _shannon([p for _, p in best.label_alternatives])

        # 按缺失阶段推进
        if not belief.is_safety_confident(best):
            return Action(kind="classify_safety", target_hypothesis=best)
        if not best.grasp_candidates:
            return Action(kind="plan_grasp_candidates", target_hypothesis=best)
        return Action(kind="grasp", target_hypothesis=best)

    def _llm_semantic_fallback(self, belief: WorldBelief) -> bool:
        """LLM 判断现有 hypothesis 中哪个语义等价于 primary_target。

        如果找到匹配, 将 primary_target 注入该 hypothesis 的 alternatives, 返回 True。
        """
        primary = belief.decomposed.primary_target if belief.decomposed else ""
        if not primary:
            return False

        labels = list({h.label for h in belief.hypotheses})
        if not labels:
            return False

        prompt = (
            f"The user wants to pick up '{primary}'. "
            f"A vision model detected these objects in the scene: {labels}. "
            f"Which ONE of these detected objects is most likely to be '{primary}' "
            f"(considering synonyms, visual similarity, or category overlap)? "
            f"Reply with ONLY the object name from the list, or 'none' if no match."
        )
        try:
            answer = self.llm.generate(prompt).strip().lower()
        except Exception as e:
            logger.warning("[semantic_fallback] LLM call failed: %s", e)
            return False

        logger.info("[semantic_fallback] LLM answer='%s' for primary='%s', labels=%s", answer, primary, labels)
        if not answer or answer == "none":
            return False

        # 找到匹配的 hypothesis
        from src.perception import _label_key, _shannon
        answer_key = _label_key(answer)
        matched_h = None
        for h in belief.hypotheses:
            if _label_key(h.label) == answer_key or answer_key in _label_key(h.label):
                matched_h = h
                break
        if matched_h is None:
            return False

        # 注入 primary_target
        primary_key = _label_key(primary)
        already = any(primary_key in _label_key(lbl) for lbl, _ in matched_h.label_alternatives)
        if already:
            # 已存在但可能概率太低, 确保 >= 0.35 (通过 target() 的 0.20 阈值)
            boosted = False
            new_alts = []
            for lbl, p in matched_h.label_alternatives:
                if primary_key in _label_key(lbl) and p < 0.35:
                    new_alts.append((lbl, 0.35))
                    boosted = True
                else:
                    new_alts.append((lbl, p))
            if boosted:
                total = sum(p for _, p in new_alts) or 1.0
                matched_h.label_alternatives = sorted(
                    ((lbl, p / total) for lbl, p in new_alts),
                    key=lambda x: x[1], reverse=True,
                )
                matched_h.label_entropy = _shannon(
                    [p for _, p in matched_h.label_alternatives]
                )
            self._record_label_corrected(primary, answer, method="llm")
            return True

        matched_h.label_alternatives.append((primary, 0.40))
        total = sum(p for _, p in matched_h.label_alternatives) or 1.0
        matched_h.label_alternatives = sorted(
            ((lbl, p / total) for lbl, p in matched_h.label_alternatives),
            key=lambda x: x[1], reverse=True,
        )
        matched_h.label_entropy = _shannon([p for _, p in matched_h.label_alternatives])
        logger.info(
            "[semantic_fallback] LLM matched '%s' → '%s' (injected into %s)",
            primary, answer, matched_h.object_id,
        )
        self._record_label_corrected(primary, answer, method="llm")
        return True

    def _record_label_corrected(
        self, target: str, detected_label: str, method: str = "llm",
    ) -> None:
        """Record a label_corrected event into working memory (dedup per episode)."""
        target = (target or "").strip().lower()
        detected_label = (detected_label or "").strip().lower()
        if not target or not detected_label or detected_label == target:
            return
        for e in self.memory.working_memory:
            if (e.domain == "recognition"
                    and e.event == "label_corrected"
                    and e.context.get("target") == target
                    and e.context.get("detected_label") == detected_label):
                return
        self.memory.record_event(MemoryEntry(
            step=len(self.memory.working_memory),
            domain="recognition",
            event="label_corrected",
            context={
                "target": target,
                "detected_label": detected_label,
                "method": method,
            },
            lesson=f"{target}: {method} matched detected '{detected_label}'",
        ))

    def _apply_recognition_hints(self, belief: WorldBelief) -> None:
        """Merge persisted recognition synonyms into belief.decomposed.primary_target_synonyms.

        Idempotent: dedupes case-insensitively. No-op when decomposed is absent.
        """
        if not belief.decomposed or not belief.decomposed.primary_target:
            return
        hint_syns = self.memory.get_recognition_hints_synonyms(
            belief.decomposed.primary_target,
        )
        if not hint_syns:
            return
        existing = {s.lower() for s in belief.decomposed.primary_target_synonyms}
        added = [s for s in hint_syns if s.lower() not in existing]
        if added:
            belief.decomposed.primary_target_synonyms.extend(added)
            logger.info(
                "[memory] injected %d recognition synonym(s) for '%s': %s",
                len(added), belief.decomposed.primary_target, added,
            )

    def _format_safety_prior_hint(self, label: str) -> Optional[str]:
        """Format historical safety prior into a hint string for SafetyClassifier.

        Returns None if label is empty or no prior is known for this label.
        """
        if not label:
            return None
        prior = self.memory.get_safety_prior(label)
        if not prior:
            return None
        dist = prior.get("dist") or {}
        if not dist:
            return None
        top = prior.get("top_class") or max(dist, key=dist.get)
        n = int(prior.get("observations", 0))
        top_prob = float(dist.get(top, 0.0))
        dist_brief = ", ".join(
            f"{k}={v:.2f}"
            for k, v in sorted(dist.items(), key=lambda x: x[1], reverse=True)[:5]
        )
        return (
            f"Historical safety prior for '{label}' (n={n} obs): "
            f"top class = {top} (avg {top_prob:.2f}). "
            f"Full dist: {dist_brief}. "
            f"Treat this as a Bayesian prior — trust visual evidence if "
            f"it strongly contradicts."
        )

    def _record_safety_classified(self, label: str, ev: Evidence) -> None:
        """Append a safety_classified event to working memory.

        Skips when label is empty or dist is missing/empty (e.g. parse_failed).
        Episode-level dedup is intentionally deferred to consolidation
        (latest dist wins per label).
        """
        label = (label or "").strip().lower()
        if not label:
            return
        payload = ev.raw_payload or {}
        dist = payload.get("dist") or {}
        if not isinstance(dist, dict) or not dist:
            return
        self.memory.record_event(MemoryEntry(
            step=len(self.memory.working_memory),
            domain="safety",
            event="safety_classified",
            context={
                "label": label,
                "dist": dict(dist),
                "entropy": float(payload.get("entropy", 0.0)),
                "reasoning": str(payload.get("reasoning", "")),
            },
            lesson=f"{label}: safety classified",
        ))

    # ──────────────────────────────────────
    # run (主循环, §5.1)
    # ──────────────────────────────────────

    def run(self, query: str, env=None) -> EpisodeResult:
        start = time.time()
        belief = WorldBelief(user_query=query)
        belief.decomposed = self.task_decomposer.decompose_v1(query)
        if self.logger:
            self.logger.start_episode(query)

        # Load long-term memory for this task
        self.memory.working_memory.clear()
        prior = self.memory.load_for_task(
            belief.decomposed.primary_target if belief.decomposed else "",
        )
        if prior:
            logger.info("[agent] loaded prior knowledge:\n%s", prior)
        # Phase 2: inject persisted recognition synonyms into decomposed.synonyms
        self._apply_recognition_hints(belief)

        # 初始 NBV: 至少拍一帧
        self._execute_action(
            Action(kind="observe", viewpoint=self.vp_lib[0]),
            env, belief,
        )

        for step in range(self.MAX_STEPS):
            if self.logger:
                self.logger.log_snapshot(belief.snapshot(step))

            if belief.is_confident_to_act():
                self._execute_action(
                    Action(kind="grasp", target_hypothesis=belief.target()),
                    env, belief,
                )
                if self._latest_grasp_succeeded(belief):
                    self._consolidate_memory(belief, success=True)
                    return self._success_result(belief, start)
                if self._latest_grasp_exhausted_execution_recovery(belief):
                    self._consolidate_memory(belief, success=False)
                    return self._giveup_result(
                        belief,
                        start,
                        reason="execution_recovery_exhausted",
                    )
                # Grasp failed → clear candidates so decide_next re-plans
                # with memory advice (avoids repeating same bad strategy)
                h = belief.target()
                if h is not None:
                    h.grasp_candidates = []
                    h.grasp_strategy = None
                continue

            action = self.decide_next(belief)
            if action.kind == "give_up":
                self._consolidate_memory(belief, success=False)
                return self._giveup_result(
                    belief, start,
                    reason=action.metadata.get("reason"),
                )
            self._execute_action(action, env, belief)
            # decide_next 也可能返回 grasp (如 _force_best_hypothesis),
            # 必须检查成功, 否则 grasp 成功了但 agent 不知道 → MAX_STEPS
            if action.kind == "grasp" and self._latest_grasp_succeeded(belief):
                self._consolidate_memory(belief, success=True)
                return self._success_result(belief, start)
            if (
                action.kind == "grasp"
                and self._latest_grasp_exhausted_execution_recovery(belief)
            ):
                self._consolidate_memory(belief, success=False)
                return self._giveup_result(
                    belief,
                    start,
                    reason="execution_recovery_exhausted",
                )

        self._consolidate_memory(belief, success=False)
        return self._giveup_result(belief, start, reason="MAX_STEPS reached")

    # ──────────────────────────────────────
    # _execute_action (§5.3)
    # ──────────────────────────────────────

    def _execute_action(
        self, action: Action, env, belief: WorldBelief,
    ) -> None:
        belief.action_history.append(action)
        if self.logger:
            self.logger.log_action_start(
                action, belief.snapshot(len(belief.action_history)),
            )
        _ev_pre = len(belief.evidence)

        if action.kind == "observe":
            ev = self.perception.observe(action.viewpoint, env, belief)
            belief.evidence.append(ev)
            self._merge_hypotheses_from_evidence(belief, ev)

        elif action.kind == "re_observe":
            ev = self.perception.re_observe(
                action.target_hypothesis, action.strategy, env, belief,
            )
            action.target_hypothesis.times_re_observed += 1
            belief.evidence.append(ev)
            self._update_hypothesis_from_evidence(action.target_hypothesis, ev)

        elif action.kind == "classify_safety":
            hyp = action.target_hypothesis
            # Phase 2 safety: inject historical prior + record observation
            prior_hint = self._format_safety_prior_hint(hyp.label)
            ev = self.safety.classify(hyp, prior_hint=prior_hint)
            belief.evidence.append(ev)
            hyp.safety_dist = ev.raw_payload.get("dist", {})
            hyp.safety_entropy = ev.raw_payload.get("entropy", 1.0)
            self._record_safety_classified(hyp.label, ev)

        elif action.kind == "plan_grasp_candidates":
            hyp = action.target_hypothesis
            # LLM 策略选择: 根据外观 + 安全属性决定抓取方式
            grasp_advice = self.memory.get_grasp_advice(hyp.label) or ""
            working_advice = self.memory.get_working_summary(domain="grasp")
            memory_advice = "\n".join(filter(None, [grasp_advice, working_advice]))
            strategy = self.grasp_planner.select_strategy(
                hyp, memory_advice=memory_advice, memory=self.memory,
            )
            hyp.grasp_strategy = strategy
            logger.info(
                "[agent] grasp strategy: %s | reason: %s | speech: %s",
                strategy.strategy, strategy.reasoning, strategy.speech,
            )
            # refuse → 拒绝抓取, 向用户警告
            if strategy.strategy == "refuse":
                belief.evidence.append(Evidence(
                    source="grasp_strategy", timestamp=time.time(),
                    raw_payload={"strategy": "refuse", "speech": strategy.speech},
                ))
                return  # 不生成候选, 下轮 decide_next 会 ask_user
            cands = self.grasp_planner.plan(hyp, env)
            # 必须更新 belief.hypotheses 里的 hypothesis (不是 action 上的引用)
            target_id = hyp.object_id
            for h in belief.hypotheses:
                if h.object_id == target_id:
                    h.grasp_candidates = cands
                    h.grasp_strategy = strategy
                    break
            else:
                hyp.grasp_candidates = cands
            belief.evidence.append(Evidence(
                source="grasp_strategy", timestamp=time.time(),
                raw_payload={
                    "strategy": strategy.strategy,
                    "reasoning": strategy.reasoning,
                    "speech": strategy.speech,
                    "n_candidates": len(cands),
                },
            ))

        elif action.kind == "grasp":
            result = self.executor.act(
                action.target_hypothesis, belief.decomposed, env,
            )
            hyp = action.target_hypothesis
            self._annotate_grasp_attempt(hyp, result.attempt)
            action.target_hypothesis.grasp_attempts.append(result.attempt)
            if result.attempt.failure_mode == "success":
                try:
                    verify_ok, conf = self.executor.verify_grasp(
                        action.target_hypothesis, env,
                    )
                except Exception:
                    verify_ok, conf = True, 1.0
                if not verify_ok:
                    result.attempt.failure_mode = "verify_mismatch"
                    result.attempt.diagnostic["verify_confidence"] = conf
                    # 推平 alternatives 让 entropy 持久化 (否则下次 merge 会
                    # 从 alternatives 重算 entropy, 抹掉这里的设置)
                    h = action.target_hypothesis
                    if h.label_alternatives and h.label_alternatives[0][1] > 0.5:
                        new_alts = [(h.label_alternatives[0][0], 0.5)]
                        rest_total = sum(
                            p for _, p in h.label_alternatives[1:]
                        )
                        if rest_total > 0:
                            scale = 0.5 / rest_total
                            new_alts.extend(
                                (lbl, p * scale)
                                for lbl, p in h.label_alternatives[1:]
                            )
                        else:
                            new_alts.append(("not_" + h.label, 0.5))
                        h.label_alternatives = new_alts
                        from src.perception import _shannon
                        h.label_entropy = _shannon([p for _, p in new_alts])
                    h.label_entropy = max(h.label_entropy, 0.6)
                    h.times_re_observed += 1
                    try:
                        self.executor.release_and_retreat(env)
                    except Exception as e:
                        logger.warning(
                            "[agent] release_and_retreat after "
                            "verify_mismatch failed: %s",
                            e,
                        )
            memory_context, memory_lesson = self._grasp_memory_payload(
                hyp, result.attempt,
            )
            if result.attempt.failure_mode == "success":
                self.memory.record_event(MemoryEntry(
                    step=len(belief.action_history),
                    domain="grasp", event="strategy_succeeded",
                    context=memory_context,
                    lesson=memory_lesson,
                ))
            else:
                self.memory.record_event(MemoryEntry(
                    step=len(belief.action_history),
                    domain="grasp", event="strategy_failed",
                    context=memory_context,
                    lesson=memory_lesson,
                ))
            belief.evidence.append(Evidence(
                source="grasp_attempt", timestamp=time.time(),
                raw_payload=result.to_dict(),
            ))

        elif action.kind == "ask_user":
            answer = self.user_channel.ask(action.question)
            belief.consume_user_answer(action.question, answer, self.llm)
            # v1 简化: 把答案转成 Constraint(kind="user_hint") 注入 decomposed.constraints,
            # 这样下一轮 perception/NBV prompt 通过 {constraints} 槽位让 LLM "看见"。
            from src.world_belief import Constraint as _C
            if belief.decomposed is not None:
                belief.decomposed.constraints.append(
                    _C(kind="user_hint", text=answer,
                       reason=f"user answered: {action.question}"),
                )
            belief.evidence.append(Evidence(
                source="user_answer", timestamp=time.time(),
                raw_payload={"q": action.question, "a": answer},
            ))
            if self.logger:
                self.logger.log_user_qa(action.question, answer)

        # prune phantom 每轮 (Edge 9.2)
        belief.prune_phantom_hypotheses()

        # 把本轮新增的 evidence 同步给 logger (Phase 14 replay 依赖)
        if self.logger:
            for _ev in belief.evidence[_ev_pre:]:
                try:
                    self.logger.log_evidence(_ev)
                except Exception as e:
                    logger.warning(f"[agent] log_evidence failed: {e}")

        if self.logger:
            self.logger.log_action_end(
                action, belief.snapshot(len(belief.action_history)),
            )

    # ──────────────────────────────────────
    # 辅助
    # ──────────────────────────────────────

    def _merge_hypotheses_from_evidence(
        self, belief: WorldBelief, ev: Evidence,
    ) -> None:
        if ev.source != "vlm_ground":
            return
        new_hyps_data = ev.raw_payload.get("hypotheses", [])
        for h_dict in new_hyps_data:
            new_h = self._dict_to_hypothesis(h_dict)
            merged = False
            for existing in belief.hypotheses:
                if belief.merge_hypothesis(existing, new_h):
                    merged = True
                    break
            if not merged:
                belief.add_hypothesis(new_h)

        # Recognition memory: CLIP synonym hit
        clip_info = ev.raw_payload.get("clip_injected")
        if clip_info:
            self._record_recognition_synonym(clip_info)

    def _record_recognition_synonym(self, info: dict) -> None:
        """Record a synonym_effective event into working memory (dedup per episode)."""
        target = str(info.get("target", "")).strip().lower()
        synonym = str(info.get("synonym", "")).strip().lower()
        if not target or not synonym or synonym == target:
            return
        # dedupe within episode by (target, synonym)
        for e in self.memory.working_memory:
            if (e.domain == "recognition"
                    and e.event == "synonym_effective"
                    and e.context.get("target") == target
                    and e.context.get("synonym") == synonym):
                return
        sim = float(info.get("sim", 0.0))
        self.memory.record_event(MemoryEntry(
            step=len(self.memory.working_memory),
            domain="recognition",
            event="synonym_effective",
            context={
                "target": target,
                "synonym": synonym,
                "sim": sim,
                "vlm_label": str(info.get("vlm_label", "")).strip().lower(),
            },
            lesson=f"{target}: CLIP hit via '{synonym}' (sim={sim:.2f})",
        ))

    @staticmethod
    def _dict_to_hypothesis(d: dict) -> Hypothesis:
        return Hypothesis(
            object_id=d["object_id"],
            label=d["label"],
            label_alternatives=[(lbl, p) for lbl, p in d["label_alternatives"]],
            label_entropy=d["label_entropy"],
            position_3d=np.array(d["position_3d"], dtype=np.float32),
            position_std_m=d["position_std_m"],
            bbox_per_view={k: tuple(v) for k, v in d.get("bbox_per_view", {}).items()},
            observed_in_views=list(d.get("observed_in_views", [])),
        )

    def _update_hypothesis_from_evidence(
        self, h: Hypothesis, ev: Evidence,
    ) -> None:
        if "hypotheses" in ev.raw_payload and ev.raw_payload["hypotheses"]:
            d = ev.raw_payload["hypotheses"][0]
            new_alts = [(lbl, p) for lbl, p in d.get("label_alternatives", [])]
            if new_alts:
                # Merge with existing alternatives (zoom may drop original labels)
                new_dict = dict(new_alts)
                merged: dict[str, float] = {}
                for lbl, p in h.label_alternatives:
                    if lbl in new_dict:
                        merged[lbl] = (p + new_dict[lbl]) / 2
                    else:
                        merged[lbl] = p / 2  # not confirmed by zoom
                for lbl, p in new_alts:
                    if lbl not in merged:
                        merged[lbl] = p / 2  # new from zoom
                total = sum(merged.values()) or 1.0
                h.label_alternatives = sorted(
                    ((lbl, p / total) for lbl, p in merged.items()),
                    key=lambda x: x[1], reverse=True,
                )
                h.label = h.label_alternatives[0][0]
                from src.perception import _shannon
                h.label_entropy = _shannon([p for _, p in h.label_alternatives])

    def _consolidate_memory(self, belief: WorldBelief, success: bool) -> None:
        """Episode 结束: 将 working memory 精华写入 long-term YAML。"""
        try:
            target_hyp = belief.target()
            self.memory.consolidate(
                success=success,
                object_type=target_hyp.label if target_hyp else "",
            )
        except Exception as e:
            logger.warning("[agent] memory consolidate failed: %s", e)

    @staticmethod
    def _annotate_grasp_attempt(h: Hypothesis, attempt) -> None:
        context, _ = EmboSightAgent._grasp_memory_payload(h, attempt)
        diagnostic = getattr(attempt, "diagnostic", None)
        if diagnostic is None:
            diagnostic = {}
            attempt.diagnostic = diagnostic
        for key in (
            "selected_strategy",
            "executed_strategy",
            "candidate_source",
            "approach_dir",
            "finger_width_m",
            "depth_margin_m",
            "squeeze_extra_steps",
            "grasp_policy_mode",
            "grasp_policy_applied",
            "grasp_policy_profile",
            "legacy_depth_margin_m",
            "legacy_squeeze_extra_steps",
        ):
            if key in context:
                diagnostic.setdefault(key, context[key])

    @staticmethod
    def _grasp_memory_payload(h: Hypothesis, attempt) -> tuple[dict, str]:
        selected = (
            h.grasp_strategy.strategy
            if getattr(h, "grasp_strategy", None) is not None
            else "unknown"
        )
        candidate = getattr(attempt, "candidate", None)
        executed = EmboSightAgent._executed_strategy_name(
            candidate, selected_strategy=selected,
        )
        approach_dir = EmboSightAgent._coerce_position3(
            getattr(candidate, "approach_dir", None),
        )
        context = {
            "strategy": executed,
            "executed_strategy": executed,
            "selected_strategy": selected,
            "candidate_source": str(getattr(candidate, "source", "unknown")),
            "object": h.label,
        }
        diagnostic = getattr(attempt, "diagnostic", {}) or {}
        for key in (
            "grasp_profile",
            "grasp_profile_confidence",
            "grasp_profile_reasons",
            "grasp_policy_mode",
            "grasp_policy_applied",
            "grasp_policy_profile",
            "legacy_depth_margin_m",
            "legacy_squeeze_extra_steps",
        ):
            if key in diagnostic:
                context[key] = diagnostic[key]
        if approach_dir is not None:
            context["approach_dir"] = approach_dir.tolist()
        if candidate is not None:
            context["finger_width_m"] = float(
                getattr(candidate, "finger_width_m", 0.0) or 0.0,
            )
        if "depth_margin_m" in diagnostic:
            context["depth_margin_m"] = float(diagnostic["depth_margin_m"])
        elif getattr(h, "grasp_strategy", None) is not None:
            context["depth_margin_m"] = float(
                getattr(h.grasp_strategy, "depth_margin_m", 0.0) or 0.0,
            )
        if "squeeze_extra_steps" in diagnostic:
            context["squeeze_extra_steps"] = int(diagnostic["squeeze_extra_steps"])
        elif getattr(h, "grasp_strategy", None) is not None:
            context["squeeze_extra_steps"] = int(
                getattr(h.grasp_strategy, "squeeze_extra_steps", 0) or 0,
            )

        candidate_source = context["candidate_source"]
        if getattr(attempt, "failure_mode", None) == "success":
            lesson = (
                f"{h.label}: selected {selected}, executed {executed} "
                f"({candidate_source}) succeeded"
            )
        else:
            failure = getattr(attempt, "failure_mode", "unknown")
            context["failure"] = failure
            lesson = (
                f"{h.label}: selected {selected}, executed {executed} "
                f"({candidate_source}) failed ({failure}), "
                "avoid this executed strategy"
            )
        return context, lesson

    @staticmethod
    def _executed_strategy_name(
        candidate, selected_strategy: str = "unknown",
    ) -> str:
        if candidate is None:
            return selected_strategy
        source = str(getattr(candidate, "source", "") or "")
        if source.startswith("strategy_"):
            return source.removeprefix("strategy_")
        approach_dir = EmboSightAgent._coerce_position3(
            getattr(candidate, "approach_dir", None),
        )
        if source == "vlm_top_grasp" and approach_dir is not None:
            if (
                float(approach_dir[2]) < -0.9
                and abs(float(approach_dir[0])) < 0.2
                and abs(float(approach_dir[1])) < 0.2
            ):
                return "top_down"
        return source or selected_strategy

    def _latest_grasp_succeeded(self, belief: WorldBelief) -> bool:
        h = belief.target()
        if h is None or not h.grasp_attempts:
            return False
        return h.grasp_attempts[-1].failure_mode == "success"

    def _latest_grasp_exhausted_execution_recovery(self, belief: WorldBelief) -> bool:
        h = belief.target()
        if h is None or not h.grasp_attempts:
            return False
        attempt = h.grasp_attempts[-1]
        if attempt.failure_mode == "success":
            return False
        diagnostic = getattr(attempt, "diagnostic", {}) or {}
        return diagnostic.get("execution_recovery_applied") is True

    def _success_result(
        self, belief: WorldBelief, start: float,
    ) -> EpisodeResult:
        h = belief.target()
        result = EpisodeResult(
            success=True,
            target=h,
            speech=self._build_speech(belief, success=True),
            belief_trace=[belief.snapshot(i)
                          for i in range(len(belief.action_history))],
            action_history=list(belief.action_history),
            n_steps=len(belief.action_history),
            elapsed_seconds=time.time() - start,
        )
        if self.logger:
            try:
                self.logger.end_episode(result)
            except Exception as e:
                logger.warning(f"[agent] logger.end_episode failed: {e}")
        return result

    def _giveup_result(
        self, belief: WorldBelief, start: float,
        reason: Optional[str] = None,
    ) -> EpisodeResult:
        result = EpisodeResult(
            success=False,
            target=belief.target(),
            speech=self._build_speech(belief, success=False, reason=reason),
            belief_trace=[],
            action_history=list(belief.action_history),
            n_steps=len(belief.action_history),
            elapsed_seconds=time.time() - start,
            failure_reason=reason,
        )
        if self.logger:
            try:
                self.logger.end_episode(result)
            except Exception as e:
                logger.warning(f"[agent] logger.end_episode failed: {e}")
        return result

    @staticmethod
    def _build_speech(
        belief: WorldBelief, success: bool, reason: Optional[str] = None,
    ) -> str:
        h = belief.target()
        if success and h is not None:
            obj_pos = EmboSightAgent._latest_success_position(
                h, "post_lift_obj_pos",
            )
            if obj_pos is not None:
                return EmboSightAgent._format_success_position(
                    h.label, "当前物体世界坐标", obj_pos,
                )
            eef_pos = EmboSightAgent._latest_success_position(
                h, "post_lift_eef_pos",
            )
            if eef_pos is not None:
                return EmboSightAgent._format_success_position(
                    h.label, "当前夹爪世界坐标", eef_pos,
                )
            estimate = EmboSightAgent._coerce_position3(h.position_3d)
            if estimate is not None:
                return EmboSightAgent._format_success_position(
                    h.label, "目标估计世界坐标", estimate,
                )
            return f"已为您拿到{h.label}。"
        if h is not None:
            return f"我看到一个像{h.label}的东西, 但暂时拿不准。{reason or ''}"
        primary = (belief.decomposed.primary_target
                   if belief.decomposed else "目标")
        return f"我没能找到{primary}。{reason or ''}"

    @staticmethod
    def _latest_success_position(
        h: Hypothesis, diagnostic_key: str,
    ) -> Optional[np.ndarray]:
        for attempt in reversed(getattr(h, "grasp_attempts", [])):
            if getattr(attempt, "failure_mode", None) != "success":
                continue
            diagnostic = getattr(attempt, "diagnostic", {}) or {}
            pos = EmboSightAgent._coerce_position3(diagnostic.get(diagnostic_key))
            if pos is not None:
                return pos
        return None

    @staticmethod
    def _coerce_position3(value) -> Optional[np.ndarray]:
        try:
            pos = np.asarray(value, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if pos.shape[0] < 3 or not np.all(np.isfinite(pos[:3])):
            return None
        return pos[:3]

    @staticmethod
    def _format_success_position(
        label: str, source_text: str, position: np.ndarray,
    ) -> str:
        return (
            f"已为您拿到{label}，{source_text}约 "
            f"x={position[0]:.3f}m，y={position[1]:.3f}m，"
            f"z={position[2]:.3f}m。"
        )
