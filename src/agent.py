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

logger = logging.getLogger(__name__)


class EmboSightAgent:

    MAX_STEPS = 12
    MAX_RE_OBSERVE = 3

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
        if (target.label_entropy < 0.30
                and target.position_std_m < 0.05
                and target.safety_entropy < 0.30
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

    # ──────────────────────────────────────
    # run (主循环, §5.1)
    # ──────────────────────────────────────

    def run(self, query: str, env=None) -> EpisodeResult:
        start = time.time()
        belief = WorldBelief(user_query=query)
        belief.decomposed = self.task_decomposer.decompose_v1(query)
        if self.logger:
            self.logger.start_episode(query)

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
                    return self._success_result(belief, start)
                continue

            action = self.decide_next(belief)
            if action.kind == "give_up":
                return self._giveup_result(
                    belief, start,
                    reason=action.metadata.get("reason"),
                )
            self._execute_action(action, env, belief)

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
            ev = self.safety.classify(action.target_hypothesis)
            belief.evidence.append(ev)
            action.target_hypothesis.safety_dist = ev.raw_payload.get("dist", {})
            action.target_hypothesis.safety_entropy = ev.raw_payload.get("entropy", 1.0)

        elif action.kind == "plan_grasp_candidates":
            cands = self.grasp_planner.plan(action.target_hypothesis, env)
            action.target_hypothesis.grasp_candidates = cands
            belief.evidence.append(Evidence(
                source="depth_projection", timestamp=time.time(),
                raw_payload={"n_candidates": len(cands)},
            ))

        elif action.kind == "grasp":
            result = self.executor.act(
                action.target_hypothesis, belief.decomposed, env,
            )
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
                    self.executor.release_and_retreat(env)
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
                h.label_alternatives = new_alts
                h.label = new_alts[0][0]
                from src.perception import _shannon
                h.label_entropy = _shannon([p for _, p in new_alts])

    def _latest_grasp_succeeded(self, belief: WorldBelief) -> bool:
        h = belief.target()
        if h is None or not h.grasp_attempts:
            return False
        return h.grasp_attempts[-1].failure_mode == "success"

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
            return (f"已为您拿到{h.label}, 在您正前方约 "
                    f"{h.position_3d[0]:.2f}m 处。")
        if h is not None:
            return f"我看到一个像{h.label}的东西, 但暂时拿不准。{reason or ''}"
        primary = (belief.decomposed.primary_target
                   if belief.decomposed else "目标")
        return f"我没能找到{primary}。{reason or ''}"
