# EmboSight 信念状态智能体架构设计 (C+B v1)

> 日期: 2026-05-08
> 状态: 待 review
> 前序: `docs/superpowers/specs/2026-05-06-embodied-enhancement-design.md`
> 关键词: belief-driven agent, query-aware grounding, 4-axis uncertainty, active perception, fail-aware grasp loop

---

## 0. TL;DR

把现行"6 步线性管道"改造成**单一 `WorldBelief` 驱动的智能体循环**，并把 VLM/LLM
当作语义推理引擎使用 (取代手写规则)。共削掉 4 个根因 (开环感知 / 规则编码世界知识 /
标量信心 / 模块各自为政), 系统性消除目前列出的 10 条 P0-P3 问题。

主要改动:

1. 新增显式信念状态 `WorldBelief` (含 4 轴结构化不确定性: label / position / safety / grasp)
2. 主流程从 `decompose → plan → describe → aggregate → decide → execute`
   改为 `while not belief.is_confident_to_act(): decide_next(belief)`
3. 删除 `vlm_grounding.py` 内 Level 0-4 规则匹配 + `safety_gate.py` 关键词表 +
   `object_aliases.yaml` + `safety_rules.yaml`
4. 新增 4 类 fallback action: `re_observe / classify_safety / plan_grasp_candidates / ask_user`
5. ActionExecutor 失败原因结构化回写 belief, 形成"抓取闭环"
6. Post-grasp 验证 (eye_in_hand + VLM) 进入 v1
7. `EpisodeLogger` 全程结构化记录, 支持 replay 测试

---

## 1. Background

### 1.1 现状架构 (基于 2026-05-06 设计稿实施版本)

```
EmboSightPipeline.run(query, env)
    Step 1: TaskDecomposer.decompose(query)             → subtasks[]
    Step 2: ActivePlanner.plan_with_grounding(...)      → observations[] + scene_model
    Step 3: SceneDescriber.describe(...)  × N           → descriptions[]
    Step 4: SceneDescriber.aggregate(...)               → final_desc
    Step 5: ActionDecider.decide(...)                   → action_plan
    Step 6: ActionExecutor.execute_with_scene_model(...)→ action_result
```

各模块在自己的局部上下文里工作, 状态分散在 `observations / descriptions / scene_model /
action_plan` 多个对象中, 模块间用单向数据流串联。

### 1.2 已观察到的 10 个问题

| ID | 等级 | 现象 | 现架构归类 |
|---|---|---|---|
| P0.1 | 致命 | Qwen3-VL 平均每帧只列 1 个物体, 经常漏掉主目标 | 感知层 |
| P0.2 | 致命 | grasp 末端 z 永远卡 0.975 (target 0.944), 无 retry | 执行层 |
| P1.3 | 严重 | bbox 中心采深度可能命中背景, 投影偏差大; 不同物体被错合并 | 感知层 |
| P1.4 | 严重 | Level 0-4 + alias 表 + LLM Level 5, 6 层规则混乱 | 感知层 |
| P1.5 | 严重 | SafetyGate 输出大量 `unknown`, 关键词匹配粗糙 | 安全层 |
| P2.6 | 工程 | VLM 5-15s × 3 相机 + LLM 5-8s + grasp 1-2min, 单 query 2-3 分钟 | 性能 |
| P2.7 | 工程 | 一处出错全链失败, 无熔断, 无 retry, 无降级 | 容错 |
| P2.8 | 工程 | 80 个单测只覆盖 parser/scene_model 算法, sim 集成在用 LLM 作弊 | 测试 |
| P3.9 | 战略 | "管道串行" 不是 "agent 反馈" | 架构 |
| P3.10 | 战略 | RoboCasa 仿真画面与 Qwen3-VL 训练数据域差距大 | 数据 |

### 1.3 根因分析: 10 个症状收敛为 4 条

**根因① 感知是开环, 没有"再看一眼"机制**
VLM 调一次拿结果就完事。Prompt D 故意"反幻觉" → VLM 不知道 query 是什么 → 凭运气列物体 → 漏检主目标。
`active_planner` 已有 NBV 内循环, 但触发条件是"维度覆盖+grounding 阈值",
不是"我对当前 hypothesis 多不确定"。
→ 解释 P0.1, P0.2, P1.3, P2.7

**根因② 把世界知识用规则编码, 不让 LLM 推理**
`vlm_grounding.py` 里 Level 0-4 + alias 表 + semantic_pairs (apple↔fruit) + family penalty +
GT cross-check + generic_labels penalty, 200+ 行 if-else。`SafetyGate` 关键词匹配 (`'ceramic' → fragile`)
是同一回事。这是用代码教 LLM 世界长啥样。
→ 解释 P1.4, P1.5, P0.1 (prompt 没注入 query)

**根因③ 信心是标量, 但不确定性是结构化的**
所有不确定全压进 `query_match_score: float`。但实际有三种独立的不确定:
- 类别 ("是水果但分不清苹果/猕猴桃")
- 位置 ("bbox 中心采到边缘, 深度可能采到背景")
- 风险 ("塑料瓶 or 陶瓷瓶")

加上抓取轴 (姿态/抓点/可达性), 共 4 维。
→ 解释 P1.3 (合并错乱), P1.5 (safety unknown 一锅粥), P1.4 (匹配链路混乱)

**根因④ 模块各自为政, 没有"智能体身份"**
VLM 不知道 user query; SafetyGate 不知道当前 subtask; ActionExecutor 走 legacy / scene_model
两条路径 (`pipeline.py` try/fallback); `env_wrapper.grasp_at` 即使收到 SceneModel
传来的 `target_pos_m`, 内部仍把 `target_body="obj_main"` 当默认 GT 来源 (`_compute_grasp_pose`
+ mini-lift verify 都读 obj_main 真值)——这是 sim-only 的作弊路径, SceneModel 的语义信息没真正进入抓取闭环。
没有共享的 belief, 物理层 retry 有 (`_attempt("try2")`), 但**语义层 retry 缺失**
(换候选物体 / 换抓点 / 问用户都没有), 也没有"问用户"分支。
→ 解释 P3.9, P2.7, P2.8, P0.2 (失败无语义 retry)

P3.10 (sim 域差距) 不是架构问题, 但 `re_observe(zoom_in)` 部分缓解 (放大裁切 → 像素增多)。

### 1.4 为何不能"修 bug 了事"

P0.1 改 prompt 治标; 但下次换个物体还会漏 (因 prompt 不知道 query)。
P0.2 加 retry 治标; 但抓哪、为何失败、试了几次没有显式状态, 永远在写 if-else。
P1.4 调 alias / semantic_pairs 永远填不完。
P1.5 加关键词永远覆盖不全。

→ **修 bug = 在错误抽象上加补丁**, 必须先把抽象换正确。

---

## 2. Goals & Non-Goals

### 2.1 v1 Goals

1. **单一 belief 驱动主循环**: `agent.run` 替换 `pipeline.run`, 状态全部进 `WorldBelief`
2. **4 轴结构化不确定性**: Hypothesis 显式区分 label/position/safety/grasp 不确定度
3. **删光规则编码**: Level 0-4, alias 表, safety_rules YAML 全删, 由 VLM/LLM 直接推理替代
4. **fallback action 闭环**: re_observe (zoom_in / parallax_view / parallax_for_pose) +
   classify_safety + plan_grasp_candidates + ask_user
5. **失败显式回写**: ActionExecutor 失败模式结构化, 写回 belief 触发下一轮决策
6. **Post-grasp 验证**: eye_in_hand + VLM 确认抓到正确物体, 失败回写
7. **EpisodeLogger 可 replay**: 替代 sim 集成测试缺口
8. **FakeUserChannel**: 三种 oracle 来源 (from_query / from_explicit / from_robocasa)

### 2.2 v1 Non-Goals (明确不做)

- ✗ 跨 episode 持久化 belief (留 `session_belief` 接口扩展点)
- ✗ 真实 ASR/TTS 接入 (`VoiceUserChannel` 留接口空实现)
- ✗ 多目标并行任务 (`primary_target` 单目标 + 约束式 `constraints[]`)
- ✗ 学习型阈值 / 在线适应 (阈值硬编码在 `configs/agent.yaml`, 但 per-axis 动态)
- ✗ sim→real domain adaptation (留作 P3.10 后续)
- ✗ 多机器人协作
- ✗ Async/并行感知 (loop 内串行调用)

### 2.3 成功标准

| 维度 | 当前 | v1 目标 |
|---|---|---|
| 主目标识别成功率 (RoboCasa 50 query 测试集) | 漏检率 ~50% | 漏检率 < 15% |
| Grasp 成功率 (仅 graspable 物体) | ~30% (z 卡死) | > 60% |
| 单 query 端到端时延 (含可能 ask_user) | 2-3 min | < 90s 中位数 |
| 失败时给出可操作回应 | 无, 直接 raise | 100% 给出 speech (含建议) |
| 单元测试覆盖 agent 行为 | 0% | decide_next 在 8+ belief 状态下决策被 mock 测覆盖 |

---

## 3. Architecture Overview

### 3.1 顶层数据流

```
              ┌────────────────────────────────────────────────┐
              │                EmboSightAgent.run               │
              │                                                  │
   query ───► │   TaskDecomposer ─► WorldBelief (init)          │
              │                            │                     │
              │                            ▼                     │
              │   ┌─────── while not belief.is_confident_to_act():
              │   │                                              │
              │   │   decide_next(belief) ─► Action              │
              │   │            │                                 │
              │   │            ▼                                 │
              │   │   ┌─ observe        (NBV)                    │
              │   │   ├─ re_observe     (zoom/parallax/pose)     │
              │   │   ├─ classify_safety (LLM only, no sensor)   │
              │   │   ├─ plan_grasp_candidates                   │
              │   │   ├─ grasp           (执行 + verify)         │
              │   │   └─ ask_user        (UserChannel)           │
              │   │            │                                 │
              │   │            ▼                                 │
              │   │   tool 返回 Evidence ─► belief.update()      │
              │   │                                              │
              │   └──────────────────────────────────────────────┤
              │                                                   │
              │   build EpisodeResult (speech + trace)           │
              └────────────────────────────────────────────────┘
                                    │
                                    ▼
                              EpisodeResult
                              ├─ speech (TTS)
                              ├─ belief_trace[]   # 每步 BeliefSnapshot
                              ├─ action_history[]
                              └─ success: bool
```

### 3.2 与现架构对比

| 维度 | 现架构 | v1 |
|---|---|---|
| 状态容器 | observations + scene_model + descriptions 各自分散 | 单一 WorldBelief |
| 决策 | 6 步硬编码顺序 | `decide_next(belief)` 单点决策 |
| 不确定性 | `query_match_score: float` | 4 轴独立 entropy/std |
| 失败处理 | raise + log | 写回 belief → 下一轮决策 |
| Object 知识 | 200+ 行 if-else + alias YAML | VLM/LLM in-context |
| Safety 判定 | 关键词表 + 风险等级硬编码 | LLM 输出 safety_dist |
| 用户交互 | 单向 TTS | 双向 ask/answer (UserChannel) |
| 测试 | 算法单测 | 算法单测 + decide_next 单测 + EpisodeLogger replay |

### 3.3 模块依赖图

```
                     agent.py (EmboSightAgent)
                     │
        ┌────────────┼─────────────┬────────────────┬───────────────┐
        ▼            ▼             ▼                ▼               ▼
   world_belief   task_         active_         user_           episode_
                  decomposer    planner         channel         logger
                                  │
                                  ▼
                              perception (QueryAwareGrounder)
                                  │
                                  ▼
                              vlm_cache
                                  │
                  ┌───────────────┼────────────────┐
                  ▼               ▼                ▼
              safety_gate     grasp_planner    action_executor
              (LLM-based)                          │
                                                   ▼
                                              projection (3D 工具)
                                                   │
                                                   ▼
                                              env_wrapper (现有)
```

---

## 4. Core Data Structures

> 全部位于 `src/world_belief.py`, 纯 Python dataclass + numpy, 无外部依赖。

### 4.1 `Hypothesis` (4 轴不确定性)

```python
from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np

@dataclass
class Pose:
    """物体姿态估计 (6D)。"""
    position: np.ndarray              # (3,) world coord
    rotation_quat: np.ndarray          # (4,) (x, y, z, w)
    upright: bool = True               # 是横还是竖 (粗略, 由 VLM 判)
    
@dataclass
class Hypothesis:
    """场景中一个候选物体, 带 4 轴结构化不确定性。"""
    object_id: str                     # "obj_0", 局部自增
    
    # ──── 1. 类别轴 ────────────────────────
    label: str                         # 当前最佳猜
    label_alternatives: list[tuple[str, float]]  # [("colander", 0.6), ("strainer", 0.3)]
    label_entropy: float               # H(label_alternatives) (温度缩放后, 见 §6.3), 越大越不确定
    
    # ──── 2. 位置轴 ────────────────────────
    position_3d: np.ndarray            # (3,) world coord, best estimate
    position_std_m: float              # 多视角投影 std (m), 单视角时给 prior 0.10
    bbox_per_view: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    
    # ──── 3. 风险轴 ────────────────────────
    safety_dist: dict[str, float] = field(default_factory=dict)
    # 开放 key 字典: 默认 v1 类别 = {safe, fragile, sharp, hot, chemical}
    # 但 prompt 是开放式 JSON, 后续可加 weight (太重) / wet (打滑) 等而不破坏接口
    # e.g. {"safe":0.7, "fragile":0.2, "sharp":0.1}
    safety_entropy: float = 1.0        # 初始为最大熵 (未分类)
    
    # ──── 4. 抓取轴 ────────────────────────
    pose_estimate: Optional[Pose] = None
    pose_uncertainty: float = 1.0
    grasp_candidates: list["GraspCandidate"] = field(default_factory=list)
    grasp_attempts: list["GraspAttempt"] = field(default_factory=list)
    
    @property
    def grasp_feasibility(self) -> float:
        """剩余可用候选中分数最高者。失败过的不重复试。"""
        used = {self._cand_key(a.candidate) for a in self.grasp_attempts}
        unused = [c for c in self.grasp_candidates if self._cand_key(c) not in used]
        return max((c.score for c in unused), default=0.0)
    
    @property
    def grasp_uncertainty(self) -> Optional[float]:
        """grasp 不确定度。
        
        返回 None 表示"尚未规划"——既无 candidates 又无 attempts。这种状态下 grasp 轴
        不参与 `most_uncertain_axis` 排序, 也不阻止 `is_confident_to_act` 的非 grasp 轴 confident
        判定; 这避免 episode 初期所有轴默认满 1.0 时, agent 总是过早跳去 plan_grasp_candidates,
        而忽视了 label/position/safety 还都不确定的事实。
        
        一旦 plan_grasp_candidates 被调用 (即使返回空 candidates) 或有过 attempt, 就开始计算:
        失败 ≥2 次强制为 1.0 (触发 ask_user), 否则 1 - feasibility。
        """
        if not self.grasp_candidates and not self.grasp_attempts:
            return None
        n_fail = sum(1 for a in self.grasp_attempts if a.failure_mode != "success")
        if n_fail >= 2:
            return 1.0
        return 1.0 - self.grasp_feasibility
    
    @staticmethod
    def _cand_key(c: "GraspCandidate") -> tuple:
        return (round(c.point_3d[0], 3), round(c.point_3d[1], 3), round(c.point_3d[2], 3),
                round(c.approach_dir[0], 2), round(c.approach_dir[1], 2), round(c.approach_dir[2], 2))
    
    # ──── 元信息 ────────────────────────
    observed_in_views: list[str] = field(default_factory=list)
    times_re_observed: int = 0          # 在 hypothesis 上做的 re_observe 次数, 防死循环
    last_action_failed: Optional[str] = None
    
    def overall_uncertainty(self) -> float:
        """各轴 max, 决定是否进 is_confident_to_act。各轴归一化到 [0,1]。
        
        grasp_uncertainty=None 时不参与 max——尚未规划的 grasp 轴不应阻塞前置感知。
        """
        norm_pos = min(1.0, self.position_std_m / 0.30)   # 0.3m 当上界
        axes = [self.label_entropy, norm_pos, self.safety_entropy]
        if self.grasp_uncertainty is not None:
            axes.append(self.grasp_uncertainty)
        return max(axes)
```

**为何这 4 轴**:
- label / position / safety 直接对应 P1.3, P1.4, P1.5
- grasp 直接对应 P0.2 (失败状态写不回来)
- 4 轴正交, 决策时可单独瞄准最不确定的轴 (`most_uncertain_axis()`)
- grasp 轴有"未规划 = None"特殊状态: 它是派生量 (依赖 candidates/attempts), 必须显式触发
  `plan_grasp_candidates` 才能开始度量, 不能用默认 1.0 跟其他轴同台 PK

### 4.2 `GraspCandidate` / `GraspAttempt`

```python
@dataclass
class GraspCandidate:
    """单个候选抓点, 由 GraspPlanner 生成。"""
    point_3d: np.ndarray               # 抓点世界坐标
    approach_dir: np.ndarray           # 接近方向 (单位向量, 指向物体)
    finger_width_m: float              # 张开宽度估计
    score: float                       # 0-1: 综合几何 + 姿态 + 可达性
    source: Literal["vlm_top_grasp", "geometric_centroid",
                    "axis_aligned_side", "user_corrected"] = "geometric_centroid"

@dataclass
class GraspAttempt:
    """已经试过的抓取记录。"""
    timestamp: float
    candidate: GraspCandidate          # 用了哪个候选
    failure_mode: Literal[
        "success",
        "hit_z_floor",                 # OSC 卡 z, 没下到目标深度
        "ik_unreachable",              # 工作空间外
        "collision",                   # 撞到其他物体
        "slipped",                     # 关爪后物体掉了
        "verify_mismatch",             # post-grasp VLM 说抓错了
        "timeout"                      # OSC 步数耗尽
    ]
    end_effector_pose_reached: tuple[float, ...]   # (x,y,z,roll,pitch,yaw)
    diagnostic: dict = field(default_factory=dict)
    # diagnostic 例: {"osc_steps": 200, "z_target": 0.944, "z_actual": 0.975,
    #                 "collision_bodies": ["pot"]}
```

### 4.3 `Evidence` / `Action` / `BeliefSnapshot`

```python
@dataclass
class Evidence:
    """一次工具调用的原始结果, 用于审计 + replay。"""
    source: Literal["vlm_ground", "vlm_zoom", "vlm_verify",
                    "llm_safety", "llm_decompose", "user_answer",
                    "grasp_attempt", "depth_projection"]
    timestamp: float
    raw_payload: dict                  # 工具的原始输出 (raw text, JSON, dict, etc.)
    consumed_by: list[str] = field(default_factory=list)   # 哪些 hypothesis 吸收了它

@dataclass
class Action:
    """agent 选出的下一步动作。"""
    kind: Literal[
        "observe",                     # 拍新视角
        "re_observe",                  # 在已有 hypothesis 上重看
        "classify_safety",             # 调 LLM 不动相机
        "plan_grasp_candidates",       # 调 GraspPlanner
        "grasp",                       # 真的抓
        "ask_user",                    # 问用户
        "give_up"                      # 终止
    ]
    target_hypothesis: Optional[Hypothesis] = None
    viewpoint: Optional["Viewpoint"] = None
    strategy: Optional[str] = None     # for re_observe: zoom_in / parallax_view / parallax_for_pose
    question: Optional[str] = None     # for ask_user
    metadata: dict = field(default_factory=dict)

@dataclass
class BeliefSnapshot:
    """某一时刻 belief 的浅拷贝, 用于 EpisodeLogger。"""
    step: int
    timestamp: float
    n_hypotheses: int
    target_summary: Optional[dict]     # target hypothesis 的 to_dict()
    most_uncertain_axis: str
    overall_uncertainty: float
    n_evidence: int
    open_questions_count: int
```

### 4.4 `WorldBelief`

```python
@dataclass
class WorldBelief:
    """主信念状态, 贯穿整个 episode。"""
    user_query: str
    decomposed: Optional["DecomposedTask"] = None  # 见 §4.6
    
    hypotheses: list[Hypothesis] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    action_history: list[Action] = field(default_factory=list)
    
    # 来自 user_answer 的硬约束 (e.g. "用户说在水池左边")
    user_constraints: list[str] = field(default_factory=list)
    
    # ──── 查询接口 ──────────────────────────
    
    def target(self) -> Optional[Hypothesis]:
        """返回当前最匹配 user_query 的 hypothesis。
        
        优先: label 字典里 primary_target 概率高的, 其次 label 文本相似度。
        如果都没有 (没找到目标), 返回 None。
        """
        if not self.hypotheses or not self.decomposed:
            return None
        target_word = self.decomposed.primary_target.lower()
        scored = []
        for h in self.hypotheses:
            # 主要分: label_alternatives 里 target_word 的概率
            prob = next((p for lbl, p in h.label_alternatives if target_word in lbl.lower()), 0.0)
            # 副分: 当前 label 文本匹配
            if target_word in h.label.lower():
                prob = max(prob, 0.5)
            if prob > 0:
                scored.append((prob, h))
        if not scored:
            return None
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[0][1]
    
    def is_confident_to_act(
        self,
        label_thr: Optional[float] = None,
        pos_thr_m: Optional[float] = None,
        safety_thr: Optional[float] = None,
        grasp_thr: Optional[float] = None,
    ) -> bool:
        """所有轴都低于阈值才能动手。阈值若不指定, 走 dynamic_thresholds_for(target)。
        
        grasp_uncertainty=None (尚未规划) 视为不 confident, 触发后续 plan_grasp_candidates。
        """
        h = self.target()
        if h is None:
            return False
        thr = self._dynamic_thresholds(h, label_thr, pos_thr_m, safety_thr, grasp_thr)
        grasp_unc = h.grasp_uncertainty if h.grasp_uncertainty is not None else 1.0
        return (h.label_entropy        < thr["label"]
                and h.position_std_m   < thr["position"]
                and h.safety_entropy   < thr["safety"]
                and grasp_unc          < thr["grasp"])
    
    def most_uncertain_axis(self) -> Literal["label", "position", "safety", "grasp"]:
        """返回最不确定的轴。
        
        grasp 轴的特殊语义: grasp_uncertainty=None 表示"尚未规划", 此时 grasp 不参与排序——
        agent 会先把 label/position/safety 三轴消除到阈值以下, 然后 is_confident_to_act
        发现 grasp 仍 None → False, decide_next 阶段 D 选 plan_grasp_candidates 触发首次规划。
        这样避免初期 4 轴都默认 1.0 时 grasp 占了 max 而过早调 GraspPlanner。
        """
        h = self.target()
        if h is None:
            return "label"   # 没找到目标 → 先消除类别不确定
        norm_pos = min(1.0, h.position_std_m / 0.30)
        scores: dict[str, float] = {
            "label":    h.label_entropy,
            "position": norm_pos,
            "safety":   h.safety_entropy,
        }
        if h.grasp_uncertainty is not None:
            scores["grasp"] = h.grasp_uncertainty
        return max(scores, key=scores.get)   # type: ignore
    
    def used_views(self) -> set[str]:
        return {a.viewpoint.name for a in self.action_history
                if a.kind == "observe" and a.viewpoint is not None}
    
    # ──── 状态修改接口 ──────────────────────
    
    def add_hypothesis(self, h: Hypothesis) -> None: ...
    def merge_hypothesis(self, existing: Hypothesis, new_data: Hypothesis) -> None:
        """新视角的 hypothesis 合并进现有的, 条件:
           position 距离 < 0.15m AND label_alternatives 概率交集 ≥ 0.3。"""
        ...
    def prune_phantom_hypotheses(self) -> int:
        """移除 1 视角孤立 + label_entropy>0.7 + 后续未确认的 hypothesis。返回删除数。"""
        ...
    def consume_user_answer(self, question: str, answer: str, llm) -> None: ...
    def snapshot(self, step: int) -> BeliefSnapshot: ...
    def compose_clarification(self) -> str:
        """根据当前 belief 构造给用户的澄清问题 (供 ask_user 使用)。"""
        ...
    
    # ──── 内部 ───────────────────────────────
    
    def _dynamic_thresholds(
        self, h: Hypothesis,
        label_thr, pos_thr_m, safety_thr, grasp_thr,
    ) -> dict[str, float]:
        """per-axis dynamic 阈值: 高风险物体严格, safe 物体宽松。
        
        定义:
        - high_risk = h.safety_dist.get("sharp",0)+...+h.safety_dist.get("hot",0) > 0.5
        - 默认: label=0.30, position=0.05, safety=0.30, grasp=0.30
        - 高风险: label=0.15, position=0.03, safety=0.15, grasp=0.20
        """
        ...
```

### 4.5 `EpisodeResult`

```python
@dataclass
class EpisodeResult:
    """agent.run 的最终返回。"""
    success: bool
    target: Optional[Hypothesis]
    speech: str                        # TTS 文本
    belief_trace: list[BeliefSnapshot]
    action_history: list[Action]
    n_steps: int
    elapsed_seconds: float
    failure_reason: Optional[str] = None  # 仅 success=False 时
```

### 4.6 `DecomposedTask` / `Constraint`

```python
@dataclass
class Constraint:
    kind: Literal["avoid", "prefer_view", "max_force", "user_hint"]
    target_label: Optional[str] = None  # e.g. "knife" for avoid
    text: Optional[str] = None           # 自然语言提示, e.g. "在水池左边"
    reason: str = ""

@dataclass
class DecomposedTask:
    primary_target: str                # "削皮器"
    constraints: list[Constraint] = field(default_factory=list)
    raw_query: str = ""                # 保留原 query, 给 prompt 用
```

---

## 5. Main Loop & decide_next

### 5.1 `EmboSightAgent.run` 伪代码

```python
class EmboSightAgent:
    
    MAX_STEPS = 12          # 防死循环, 含 observe + re_observe + ask_user + grasp
    MAX_RE_OBSERVE = 3      # 单 hypothesis 上 re_observe 上限
    
    def run(self, query: str, env) -> EpisodeResult:
        belief = WorldBelief(user_query=query)
        belief.decomposed = self.task_decomposer.decompose(query)
        self.logger.start_episode(query)
        
        # 初始 NBV 之前, 至少拍一帧全景
        self._execute_action(Action(kind="observe", viewpoint=self.vp_lib[0]), env, belief)
        
        for step in range(self.MAX_STEPS):
            self.logger.log_snapshot(belief.snapshot(step))
            
            if belief.is_confident_to_act():
                # 真的抓
                self._execute_action(
                    Action(kind="grasp", target_hypothesis=belief.target()),
                    env, belief,
                )
                if self._latest_grasp_succeeded(belief):
                    return self._success_result(belief)
                # 失败已写回 belief, 继续 loop
                continue
            
            action = self.decide_next(belief)
            if action.kind == "give_up":
                return self._giveup_result(belief, reason=action.metadata.get("reason"))
            self._execute_action(action, env, belief)
        
        return self._giveup_result(belief, reason="MAX_STEPS reached")
```

### 5.2 `decide_next` 决策树

> **设计动机** (回应根因②). 这棵树是**显式规则**, 不调 LLM 决策。这看似与"删 200 行规则改
> 用 LLM 推理"的根因②目标矛盾, 但二者并不冲突——
>
> - 根因② 删的是"**世界知识**用 if-else 编码" (`apple↔fruit`, `ceramic→fragile`),
>   这种知识本质上是 LLM/VLM 训练数据里的, 用规则编码必然填不全。
> - decide_next 编码的是"**控制策略**" (按轴优先级路由 action), 这种策略:
>   1. 状态空间小 (4 轴 × 4 阈值状态 = 16 种典型组合, 不是开放语义空间)
>   2. **可单测** (mock belief 状态, 断言 action.kind, 见 §6.2 单测清单)
>   3. **可审计** (epi 日志里看 action 序列就知道为什么), 这对视障辅助场景的可靠性是首要
>   4. LLM 决策不是免费午餐 (5-8s 调一次 + 不确定性), 不该把每步控制都交给它
>
> 所以 v1 的取舍: **世界知识让 LLM/VLM 推, 控制策略用规则树**。如果未来观察到决策树跟不上
> 场景复杂度 (e.g. 6 轴以上的不确定性, 跨 episode reasoning), 再考虑把 decide_next 部分
> 替换成 LLM-policy (留 §13 v2 工作)。

```python
def decide_next(self, belief: WorldBelief) -> Action:
    
    # ── 阶段 A: 还没看够 ──
    if not belief.evidence:
        return Action(kind="observe", viewpoint=self.vp_lib[0])
    
    target = belief.target()
    
    # ── 阶段 B: 没找到 target, 多看 ──
    if target is None:
        next_vp = self.nbv.select(
            belief, exclude=belief.used_views(),
            preference="search_target",
        )
        if next_vp is None:
            return Action(
                kind="ask_user",
                question=f"我没在场景里看到{belief.decomposed.primary_target}, 是不是被挡住了?",
            )
        return Action(kind="observe", viewpoint=next_vp)
    
    # ── 阶段 C: 找到了, 但太多次 re_observe → 问用户 ──
    if target.times_re_observed >= self.MAX_RE_OBSERVE:
        return Action(kind="ask_user", question=belief.compose_clarification())
    
    # ── 阶段 D: 哪轴最不确定就消除哪轴 ──
    axis = belief.most_uncertain_axis()
    
    if axis == "label":
        if self._has_unzoomed_view(target):
            return Action(kind="re_observe", target_hypothesis=target, strategy="zoom_in")
        # 已经 zoom 过了还不确定 → ask_user
        return Action(kind="ask_user",
                      question=f"我看到一个{target.label}样的东西, 也可能是{target.label_alternatives[1][0]}, 您要的是哪个?")
    
    if axis == "position":
        return Action(kind="re_observe", target_hypothesis=target, strategy="parallax_view")
    
    if axis == "safety":
        return Action(kind="classify_safety", target_hypothesis=target)
    
    if axis == "grasp":
        if not target.grasp_candidates:
            return Action(kind="plan_grasp_candidates", target_hypothesis=target)
        if target.pose_uncertainty > 0.5:
            return Action(kind="re_observe", target_hypothesis=target, strategy="parallax_for_pose")
        # 候选有, 姿态也估了, 但 grasp_feasibility 还低 → 候选都不可达
        return Action(kind="ask_user",
                      question=f"我没法抓到{target.label}, 它现在是横放还是竖放? 还是被挡住一部分?")
    
    return Action(kind="give_up", metadata={"reason": "unreachable decision branch"})
```

### 5.3 `_execute_action` 调度

```python
def _execute_action(self, action: Action, env, belief: WorldBelief) -> None:
    belief.action_history.append(action)
    self.logger.log_action_start(action, belief.snapshot(len(belief.action_history)))
    
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
        action.target_hypothesis.safety_dist = ev.raw_payload["dist"]
        action.target_hypothesis.safety_entropy = ev.raw_payload["entropy"]
    
    elif action.kind == "plan_grasp_candidates":
        cands = self.grasp_planner.plan(action.target_hypothesis, env)
        action.target_hypothesis.grasp_candidates = cands
        belief.evidence.append(Evidence(
            source="depth_projection", timestamp=time.time(),
            raw_payload={"n_candidates": len(cands)},
        ))
    
    elif action.kind == "grasp":
        result = self.executor.act(action.target_hypothesis, belief.decomposed, env)
        action.target_hypothesis.grasp_attempts.append(result.attempt)
        # post-grasp verify (即使物理 success 也要 verify)
        if result.attempt.failure_mode == "success":
            verify_ok, conf = self.executor.verify_grasp(action.target_hypothesis, env)
            if not verify_ok:
                result.attempt.failure_mode = "verify_mismatch"
                result.attempt.diagnostic["verify_confidence"] = conf
                action.target_hypothesis.label_entropy = max(
                    action.target_hypothesis.label_entropy, 0.6,
                )
                # ★ 关键: verify 失败必须释放夹爪 + 撤回到 pre_grasp 高度,
                # 否则后续 observe/zoom_in 都被夹爪本身遮挡, 死锁。
                # release_and_retreat 内部: open_gripper → 提升 0.10m → 标记 hypothesis 已被
                # "扰动过" (set last_action_failed="verify_mismatch", times_re_observed += 1)
                self.executor.release_and_retreat(env)
        belief.evidence.append(Evidence(
            source="grasp_attempt", timestamp=time.time(),
            raw_payload=result.to_dict(),
        ))
    
    elif action.kind == "ask_user":
        answer = self.user_channel.ask(action.question)
        belief.consume_user_answer(action.question, answer, self.llm)
        belief.evidence.append(Evidence(
            source="user_answer", timestamp=time.time(),
            raw_payload={"q": action.question, "a": answer},
        ))
    
    self.logger.log_action_end(action, belief.snapshot(len(belief.action_history)))
```

### 5.4 决策优先级总结

```
1. 没观察过        → observe (init view)
2. 没找到 target   → NBV observe
3. re_observe 超限 → ask_user (clarification)
4. label 最不确定  → re_observe(zoom_in)         — 1 次后还不准 → ask_user
5. position 最不确定 → re_observe(parallax_view)
6. safety 最不确定  → classify_safety
7. grasp 最不确定  → plan_grasp_candidates / re_observe(parallax_for_pose) / ask_user
8. 全部够确定     → grasp (含 post-grasp verify)
```

---

## 6. Module Specs

### 6.1 `src/world_belief.py`

**职责**: 纯数据结构 + invariants + 查询/修改 API。无 IO, 无 LLM/VLM 调用。

**接口**: 见 §4。

**不变量** (单测验证):
- `belief.target()` 始终返回 `belief.hypotheses` 中之一或 None
- `add_hypothesis` 后 `len(hypotheses)` += 1
- `merge_hypothesis` 后 `len(hypotheses)` 不变
- `prune_phantom_hypotheses` 仅删除满足 (1 view + entropy>0.7 + age>3 step) 的
- `most_uncertain_axis` 返回值在 `{"label","position","safety","grasp"}` 中
- 当 `target()` 为 None 时, `is_confident_to_act` 必为 False

**单测清单** (TDD, 先写):
1. 空 belief: target=None, is_confident=False, axis="label"
2. 单 hypothesis 全 confident: is_confident=True
3. 4 轴各自最大: most_uncertain_axis 正确返回
4. 高风险物体 (sharp>0.5): dynamic 阈值收紧
5. merge 距离临界 (0.149 vs 0.151)
6. merge label 不交集时不合并
7. prune 1 view + 高熵 + 步数过期 → 删除
8. consume_user_answer 改 boost/demote/constraint

### 6.2 `src/agent.py` (新)

**职责**: 主循环 + decide_next + action 调度。

**接口**:
```python
class EmboSightAgent:
    def __init__(
        self,
        config_path: str,
        env=None,
        session_belief: Optional[WorldBelief] = None,   # 扩展点
        user_channel: Optional[UserChannel] = None,
    ): ...
    
    def run(self, query: str, env=None) -> EpisodeResult: ...
    def decide_next(self, belief: WorldBelief) -> Action: ...
```

**依赖** (DI 注入):
- `TaskDecomposer`, `QueryAwareGrounder`, `SafetyClassifier`,
  `GraspPlanner`, `ActionExecutor`, `ActiveViewpointSelector`,
  `UserChannel`, `EpisodeLogger`, `LLMBackend`, `VLMBackend`, `VLMCache`

**单测清单**:
1. mock 所有工具, mock belief 处于 8 个不同状态, decide_next 选对 action
2. MAX_STEPS 触发 → giveup_result
3. MAX_RE_OBSERVE 触发 → ask_user
4. grasp success 直接 return success_result
5. grasp verify_mismatch → label_entropy 提升, 不返回, 继续 loop
6. ask_user 答案进 belief, 下一轮决策受影响

### 6.3 `src/perception.py` (新, 取代 vlm_grounding.py)

**职责**: VLM grounding + zoom + parallax, 直出 Hypothesis。

```python
class QueryAwareGrounder:
    
    def __init__(
        self, vlm: VLMBackend, llm: LLMBackend, cache: VLMCache,
        ground_prompt_path: str = "prompts/perception/query_aware_ground.txt",
        zoom_prompt_path: str = "prompts/perception/zoom_disambiguate.txt",
        parallax_prompt_path: str = "prompts/perception/parallax_localize.txt",
        pose_prompt_path: str = "prompts/perception/pose_estimation.txt",
        verify_prompt_path: str = "prompts/perception/verify_grasp.txt",
    ): ...
    
    # ── 主入口 ──
    def observe(self, viewpoint: Viewpoint, env, belief: WorldBelief) -> Evidence:
        """拍 viewpoint, query-aware VLM, 返回 Evidence (含 hypotheses[])。"""
    
    def re_observe(
        self, target: Hypothesis, strategy: str, env, belief: WorldBelief,
    ) -> Evidence:
        """根据 strategy 重看 target。
        strategy ∈ {"zoom_in", "parallax_view", "parallax_for_pose"}
        """
    
    def verify_grasp(self, target: Hypothesis, env) -> tuple[bool, float]:
        """eye_in_hand 拍, VLM 确认夹爪里是不是 target。"""
    
    # ── 私有 ──
    def _build_query_aware_prompt(self, query: str, constraints: list[Constraint]) -> str: ...
    def _parse_to_hypotheses(self, raw: str, viewpoint, env) -> list[Hypothesis]:
        """VLM JSON → Hypothesis (含 label_alternatives + label_entropy)。
        
        VLM 直接输出 alternatives + 概率, 我们做两步:
        
        1. 概率温度缩放 (calibration): VLM 自报概率严重过自信 (常给 0.95 的赌博式 top1)。
           应用 temperature τ=1.5 (configurable):
               p_i' = p_i^(1/τ) / Σ p_j^(1/τ)
           τ>1 让分布更平坦, top1 概率从 0.95 → ~0.75, 让 H 真的能反映模糊度,
           否则 label_entropy 几乎永远 < 0.30 阈值, 永远不触发 zoom_in。
        
        2. 计算熵: H = -Σ p_i' log(p_i')
        
        温度参数放 configs/agent.yaml `perception.label_temperature`, 默认 1.5,
        实测后调整 (见 §10.3 demo_queries 跑出来的 entropy 直方图)。
        """
    def _select_parallax_vp(self, target: Hypothesis) -> Viewpoint:
        """选夹角最大的未用视角。"""
    def _select_side_vp(self, target: Hypothesis) -> Viewpoint:
        """选侧面视角 (frontview / robotview), 用于姿态估计。"""
    def _crop_image(self, image_path: str, bbox: tuple) -> str:
        """裁切 bbox + 边距 padding, 写到临时文件返回路径。"""
```

**Prompt 关键结构** (`prompts/perception/query_aware_ground.txt`):

```
You are looking at a kitchen scene. The user is asking for: {primary_target}.

Constraints (from user):
{constraints}

Your task:
1. Find ALL objects on the countertop / table that could possibly be {primary_target}
   or visually similar (e.g. fruits if user asks for an apple).
2. For each object, output:
   - bbox_2d: [x1, y1, x2, y2] in pixels, image is {img_w}x{img_h}
   - label: your best name
   - alternatives: top 3 (label, probability) tuples summing to ≤1.0
       Example: [("colander", 0.6), ("strainer", 0.3), ("basket", 0.1)]
       
       BE CONSERVATIVE with probabilities — if multiple labels look possible, distribute
       probability mass; do NOT give 0.95 unless the object is unmistakable (e.g. a clean
       bright apple). When in doubt 0.5/0.3/0.2 is more useful than 0.95/0.03/0.02
       because downstream reasons over uncertainty. If uncertain, you HELP us by saying so.
       
   - confidence: 0-1 (how sure you are something is THERE)
   - visible_features: 1 sentence

3. If you don't see {primary_target} at all, list the most visually similar objects
   you DO see (fall back gracefully).

JSON: {"objects": [...]}
```

> 即便 prompt 强调 conservative, agent 端仍**双保险**做 temperature 缩放
> (`_parse_to_hypotheses` 步骤 1), 防 VLM 不听话。

**关键差异 vs 老 vlm_grounding**:
- ✅ Prompt 注入 `primary_target` (老的 Prompt D 故意不注入, 漏检根因)
- ✅ 输出 `alternatives` 概率分布, 直出 entropy
- ❌ 删除 alias 反向匹配
- ❌ 删除 semantic_pairs / family penalty
- ❌ 删除 GT cross-check (RoboCasa 真实测试用 GT 是作弊, 留 logger 里做事后审计而非分数)

### 6.4 `src/safety_gate.py` (改造, 缩水)

```python
class SafetyClassifier:
    
    def __init__(
        self, llm: LLMBackend,
        prompt_path: str = "prompts/safety/classify.txt",
    ): ...
    
    def classify(self, hyp: Hypothesis) -> Evidence:
        """LLM 输出 safety_dist + entropy。
        
        Returns Evidence with raw_payload = {
            "dist": {"safe":..., "fragile":..., "sharp":..., "hot":..., "chemical":...},
            "entropy": float,
            "reasoning": str,
        }
        """
```

**Prompt 关键结构** (`prompts/safety/classify.txt`):

```
Object label: {label}
Top alternatives: {alternatives_top3}
Visual features (multi-view): {features}
Pose estimate: {pose_summary}

Context: 视障用户在厨房, 机器人即将抓取该物体。

请输出该物体的安全风险概率分布 (sum to 1.0):
- safe       : 普通无风险物品 (水果/塑料瓶/木勺等)
- fragile    : 易碎 (玻璃/陶瓷/瓷器)
- sharp      : 锋利 (刀/碎片/带刃)
- hot        : 高温 (刚出炉/装热汤的容器)
- chemical   : 化学品 (清洁剂/漂白剂)

JSON:
{
  "dist": {...},
  "reasoning": "<1 句>"
}
```

**LLM 处理**: agent 端调用后, `entropy = -Σ p log p`, 写进 `Hypothesis.safety_entropy`。

**关于风险类别**: v1 固定 5 类 (safe/fragile/sharp/hot/chemical) 用于 prompt 模板 + dynamic
threshold 里的 `high_risk` 触发条件 (sharp+hot+chemical 总和 > 0.5)。但 `safety_dist` 是
开放 key dict, 后续可扩 `weight` (太重抓不动) / `wet` (打滑) 等而无需改数据结构——只需新增
prompt 选项 + 在 `_dynamic_thresholds` 里把新 key 加入"高风险"判定。这是 v2 工作。

**删除**: `_FEATURE_RISK_KEYWORDS` 关键词表, `safety_rules.yaml`, `_detect_feature_risks` 文本扫。

### 6.5 `src/grasp_planner.py` (新)

```python
class GraspPlanner:
    
    def __init__(self, vlm: VLMBackend, env, prompt_path: str): ...
    
    def plan(self, hyp: Hypothesis, env) -> list[GraspCandidate]:
        """生成 N 个候选 (默认 3-5)。
        策略:
        1. geometric_centroid: position_3d + 顶抓 (approach=-z)
        2. axis_aligned_side : 若 pose_estimate.upright=False, 侧抓
        3. vlm_top_grasp     : VLM 看 eye_in_hand 图建议 "best grip point"
        过滤: 工作空间可达性 (env.is_reachable(point, approach))
        排序: score (高到低)
        """
    
    def regenerate_after_failure(
        self, hyp: Hypothesis, last_attempt: GraspAttempt,
    ) -> list[GraspCandidate]:
        """失败后重生候选, 排除已失败的 + 调整 score。"""
```

### 6.6 `src/active_planner.py` (改造)

保留 `ViewpointLibrary` + `Viewpoint` 数据结构。

**删除**:
- `plan(subtasks, env)` 主入口
- `plan_with_grounding(...)`
- `_update_coverage` (维度覆盖)
- `_is_sufficient` (LLM 早停)
- `_build_nbv_prompt` (老的)

**新接口**:
```python
class ActiveViewpointSelector:
    
    def select(
        self,
        belief: WorldBelief,
        exclude: set[str],
        preference: Literal["search_target","disambiguate_label",
                            "parallax_position","grasp_pose"] = "search_target",
    ) -> Optional[Viewpoint]:
        """LLM 决策, 返回 Viewpoint 或 None (无可选)。"""
```

**Prompt** (`prompts/agent/nbv_select.txt`): 让 LLM 看 belief 摘要 + preference, 选索引。

### 6.7 `src/action_executor.py` (改造)

**删除**:
- `execute(plan, env)` (legacy)
- `execute_with_scene_model(plan, scene_model, env)` (老接口)

**新接口**:
```python
@dataclass
class ActionResult:
    success: bool
    attempt: GraspAttempt
    new_observations: list[Evidence] = field(default_factory=list)

class ActionExecutor:
    
    def act(
        self, target: Hypothesis, decomposed: DecomposedTask, env,
    ) -> ActionResult:
        """执行抓取。
        
        步骤:
        1. 选 grasp_candidates 中分数最高的未试过的
        2. env.move_to_pre_grasp(candidate)  ← 失败 → "ik_unreachable"
        3. env.descend(candidate.point_3d)   ← 失败 (z 卡死) → "hit_z_floor"
        4. env.close_gripper()
        5. env.lift()                        ← 检测到 slip → "slipped"
        全程记录 GraspAttempt.diagnostic (osc steps, z 实际值, collision pairs)。
        """
    
    def verify_grasp(self, target: Hypothesis, env) -> tuple[bool, float]:
        """eye_in_hand + VLM 验证。返回 (is_correct, confidence)。"""
    
    def release_and_retreat(self, env, retreat_height_m: float = 0.10) -> None:
        """verify_mismatch 后放下物体并撤回, 解锁后续观察。
        
        步骤:
        1. env.open_gripper()           # 物体落回桌面
        2. 末端上升 retreat_height_m    # 让物体不被夹爪遮挡
        3. (不归原位; agent 后续会按 NBV 选下一视角)
        
        必须调用; 否则下一轮 observe/zoom 会被夹爪遮挡, 死锁。
        """
```

### 6.8 `src/task_decomposer.py` (改造)

**改输出**: `decompose(query: str) -> DecomposedTask`

老版输出 `list[Subtask]`, 新版输出 `DecomposedTask(primary_target, constraints[])`。

**Prompt** (`prompts/agent/decompose.txt`):

```
Query: {query}

请输出:
{
  "primary_target": "<目标物体名, 中文或英文>",
  "constraints": [
    {"kind": "avoid", "target_label": "knife", "reason": "用户提到避开"},
    {"kind": "user_hint", "text": "在水池左边", "reason": "位置提示"}
  ]
}

规则:
1. primary_target 是用户最想要的那一个东西
2. constraints 包含: 避让物体 (avoid) / 位置提示 (user_hint) / 视角偏好 (prefer_view)
3. 不需要 "拆成多个子任务", v1 单目标
```

### 6.9 `src/user_channel.py` (新)

```python
from typing import Protocol

class UserChannel(Protocol):
    def ask(self, question: str, context: dict | None = None) -> str: ...

class FakeUserChannel:
    """LLM 扮演视障用户。"""
    
    SYSTEM = """你是一名视障用户, 正在使用辅助机器人。
    - 你看不见任何视觉细节, 但记得自己想要什么、家里大致布局。
    - 不要假装看到颜色/形状; 但可以说"通常它放在水池左边"这种记忆。
    - 简短自然回答 (1 句话)。
    - 如果机器人列出选项, 选最像你想要的那个。
    """
    
    def __init__(self, llm: LLMBackend, intent: str):
        self.llm = llm
        self.intent = intent
        self.history: list[tuple[str, str]] = []
    
    @classmethod
    def from_query(cls, llm, query: str) -> "FakeUserChannel":
        """LLM 从 query 提取 intent。"""
        prompt = f"用户说: '{query}'。一句话提取他真实想要的物体: "
        intent = llm.generate(prompt).strip()
        return cls(llm, intent)
    
    @classmethod
    def from_explicit(cls, llm, intent: str) -> "FakeUserChannel":
        return cls(llm, intent)
    
    @classmethod
    def from_robocasa(cls, llm, env) -> "FakeUserChannel":
        """从 env._get_obj_type_map() 取 obj_main 类型。"""
        type_map = env._get_obj_type_map()
        intent = type_map.get("obj_main", "unknown")
        return cls(llm, f"我想要那个 {intent}")
    
    def ask(self, question: str, context: dict | None = None) -> str:
        prompt = (
            f"你的真实意图: {self.intent}\n\n"
            f"对话历史:\n{self._format_history()}\n\n"
            f"机器人问: {question}\n你的回答:"
        )
        ans = self.llm.generate(prompt, system=self.SYSTEM).strip()
        self.history.append((question, ans))
        return ans
    
    def _format_history(self) -> str:
        return "\n".join(f"  Q: {q}\n  A: {a}" for q, a in self.history) or "(无)"

class CLIUserChannel:
    def ask(self, question: str, context=None) -> str:
        print(f"\n[Agent] {question}")
        return input("[You] ").strip()

class VoiceUserChannel:
    """v1 留接口, 不实现 (raise NotImplementedError)。"""
    def __init__(self, tts, asr): ...
    def ask(self, question, context=None): raise NotImplementedError
```

### 6.10 `src/episode_logger.py` (新)

```python
@dataclass
class EpisodeRecord:
    query: str
    start_time: float
    snapshots: list[BeliefSnapshot]
    actions: list[Action]                 # action_history 的拷贝
    evidence: list[Evidence]              # 全部 evidence (含 raw_payload)
    user_qa: list[tuple[str, str]]
    final_result: Optional[EpisodeResult] = None

class EpisodeLogger:
    
    def __init__(self, log_dir: str = "logs/episodes"): ...
    
    def start_episode(self, query: str) -> None: ...
    def log_snapshot(self, snap: BeliefSnapshot) -> None: ...
    def log_action_start(self, action: Action, snap: BeliefSnapshot) -> None: ...
    def log_action_end(self, action: Action, snap: BeliefSnapshot) -> None: ...
    def log_user_qa(self, q: str, a: str) -> None: ...
    def end_episode(self, result: EpisodeResult) -> str:
        """save → 返回 JSON path。"""
    
    @classmethod
    def load(cls, json_path: str) -> EpisodeRecord:
        """从 JSON 反序列化为 EpisodeRecord, 不重放, 仅读取。"""
    
    @classmethod
    def replay(
        cls, json_path: str, agent_factory,
    ) -> EpisodeResult:
        """从日志 mock 出 VLM/LLM 工具 (用记录的 raw_payload), 重跑 agent.decide_next。
        
        实现:
        1. record = cls.load(json_path)
        2. 构造 MockFromRecord(record, source) 给每个 source 类型
        3. agent = agent_factory(mocks)
        4. result = agent.run(record.query, mock_env)
        5. return result
        
        用途:
        - 重构后回归测试: 同样的 evidence 序列, decide_next 是否还做出同样决策
        - 追溯 demo 失败原因
        """
```

**JSON Schema** (示例):
```json
{
  "query": "帮我拿削皮器",
  "start_time": 1715159832.4,
  "snapshots": [
    {"step": 0, "n_hypotheses": 0, "most_uncertain_axis": "label", ...}
  ],
  "actions": [
    {"kind": "observe", "viewpoint": "robot0_agentview_center"},
    ...
  ],
  "evidence": [
    {"source": "vlm_ground", "raw_payload": {"objects": [...]}, ...}
  ],
  "user_qa": [],
  "final_result": {"success": true, "speech": "...", ...}
}
```

### 6.11 `src/vlm_cache.py` (新)

```python
class VLMCache:
    """in-memory cache. 按 (image_hash, prompt_hash) 去重。
    生命周期: episode 级 (agent.run 开头 clear)。
    """
    
    def __init__(self, max_size: int = 100): ...
    
    def get(self, image_path: str, prompt: str) -> Optional[str]: ...
    def put(self, image_path: str, prompt: str, response: str) -> None: ...
    def clear(self) -> None: ...
    def stats(self) -> dict:
        """return {"hits": int, "misses": int, "hit_rate": float}"""
```

**hash**: `image_hash = sha256(image_bytes)`, `prompt_hash = sha256(prompt.encode())`

### 6.12 `src/projection.py` (从 scene_model 拆出)

保留 `depth_buffer_to_meters`, `project_bbox_to_world`, `compute_intrinsics` 三个纯函数。

**删除**: 老的 `SceneModel` 类 (合并 / get_best_match / add_view 等), `GroundedObject` 类。

---

## 7. Prompts 新结构

```
prompts/
  agent/
    decompose.txt              # task_decomposer 用 (输出 DecomposedTask)
    nbv_select.txt             # ActiveViewpointSelector 用
    user_answer_parse.txt      # WorldBelief.consume_user_answer 用 (LLM 把答案转指令)
  perception/
    query_aware_ground.txt     # QueryAwareGrounder.observe
    zoom_disambiguate.txt      # re_observe(zoom_in)
    parallax_localize.txt      # re_observe(parallax_view)
    pose_estimation.txt        # re_observe(parallax_for_pose)
    verify_grasp.txt           # post-grasp eye_in_hand verify
  safety/
    classify.txt               # SafetyClassifier
  grasp/
    suggest_top_grasp.txt      # vlm_top_grasp 候选
  user/
    fake_user_system.txt       # FakeUserChannel 系统 prompt
```

**删除老 prompts**:
- `prompts/active_planner.txt`
- `prompts/active_planner_grounding_aware.txt`
- `prompts/scene_describer.txt`        ← scene_describer 整个被 perception 替代
- `prompts/vlm_grounding.txt`          ← 被 query_aware_ground 替代
- `prompts/action_decider.txt`          ← 决策逻辑进 agent.decide_next, 不再用 LLM 决策动作
- `prompts/task_decompose.txt`          ← 重写成 prompts/agent/decompose.txt

---

## 8. Configs

### 8.1 新建 `configs/agent.yaml`

```yaml
# EmboSight Agent v1 主配置

agent:
  max_steps: 12               # 主循环硬上限
  max_re_observe: 3           # 单 hypothesis 上 re_observe 上限

# 各轴 confidence 阈值 (低于此值才 act)
thresholds:
  default:
    label_entropy: 0.30
    position_std_m: 0.05
    safety_entropy: 0.30
    grasp_uncertainty: 0.30
  high_risk:                  # safety_dist 中 sharp+hot+chemical > 0.5 时启用
    label_entropy: 0.15
    position_std_m: 0.03
    safety_entropy: 0.15
    grasp_uncertainty: 0.20

# Hypothesis 合并 / 剪枝
# 注意: 这些阈值是 v1 初始猜测, 需在 §10.3 demo_queries 上实测后调整;
# 加 TODO(v1.1) 记录初值 vs 实测调优值
belief:
  merge_distance_m: 0.15          # TODO(v1.1): RoboCasa 厨房物体密度大, 可能要降到 0.10
  merge_label_intersection_min: 0.30  # TODO(v1.1): zoom 前后 alternatives 可能整组变化, 可能要降到 0.20
  prune_phantom_min_steps: 3      # 1 view 孤立 + 高熵 + age>3 → prune

# Perception 调参
perception:
  label_temperature: 1.5          # VLM 概率温度缩放, p_i' = p_i^(1/τ)/Σ; >1 让分布更平
                                  # TODO(v1.1): 跑 demo 后看 label_entropy 直方图调整

# VLM cache
cache:
  enabled: true
  max_size: 100                   # 100 个 (image,prompt) 条目
  ttl: episode                    # episode 结束 clear

# Verify
verify:
  enabled: true                   # post-grasp verify, v1 开
  min_confidence: 0.6             # VLM 低于此值认为 verify 失败

# 日志
logger:
  log_dir: logs/episodes
  save_json: true
  save_belief_trace: true

# 工具实现选择
implementations:
  user_channel: fake_from_robocasa   # fake_from_query / fake_from_explicit / fake_from_robocasa / cli
  perception: query_aware            # query_aware (新) / legacy_d (老 vlm_grounding, fallback)
  safety: llm_classify               # llm_classify (新) / legacy_yaml (老)
  task_decomposer: constraint_aware  # constraint_aware (新) / legacy_subtasks (老)
```

### 8.2 删除 (在 v1 完成后)

- `configs/object_aliases.yaml`
- `configs/safety_rules.yaml`

### 8.3 保留

- `configs/viewpoints.yaml` (NBV 视角库, v1 不动)
- `configs/default.yaml` (顶层 LLM/VLM 配置, 加 agent 段)

---

## 9. Edge Cases

| Case | 表现 | 处理 |
|---|---|---|
| **9.1 同物体在两视角不同 label** ("fork" vs "spoon") | 老: 按距离合并, label 一处 | 新: 合并条件加 `label_alternatives` 概率交集 ≥ 0.3。否则保留为 2 个 hypothesis + 各自高 entropy → re_observe(zoom_in) 消歧 |
| **9.2 VLM 幻觉** (孤立 1 视角凭空生成) | 老: GT cross-check 扣 0.3 倍分 (但 GT 真实场景不可用) | 新: `prune_phantom_hypotheses` 在每轮结束调一次, 删除 (1 view + entropy>0.7 + step>3) |
| **9.3 Re-observe 反而置信下降** (zoom 后说 "啥也不像") | 老: score 单调上升, 处理不了负证据 | 新: entropy 双向变化; 若 re_observe 后 entropy 增加 → `times_re_observed += 2` (惩罚式), 更快进 ask_user |
| **9.4 用户答非所问** ("不知道") | N/A (无 ask_user) | `consume_user_answer` 里 LLM 解析时输出 `{"unhelpful":true}` → agent 改用 NBV 探索, 不再问类似问题 |
| **9.5 多视角 3D 位置差 > 0.15m** | 老: 当成两个物体 | 新: 保留两个 hypothesis 但 `position_std_m=0.15` 显式标; 触发 parallax_view 第三视角投票 |
| **9.6 Grasp 物理 success 但抓错了** | 老: 直接成功 return | 新: post-grasp verify (eye_in_hand + VLM); verify_mismatch → ① `release_and_retreat` 放下物体 + 撤回 (否则后续观察被夹爪遮挡死锁) ② `label_entropy` 拉到 0.6 ③ `times_re_observed += 1` 标"已扰动" → 下轮 zoom_in 或 ask_user |
| **9.7 MAX_STEPS 用尽** | 老: raise | 新: `_giveup_result` 把 belief.target() 描述给用户 ("我看到一个 yellow round thing 在中间, 可能是您要的, 您要我试一下吗?") |
| **9.8 VLM 调用失败 (timeout/JSON 解析失败)** | 老: 跳过该视角, 继续 | 新: Evidence 标记 source="vlm_failed", agent 选 NBV 换视角再试, 而不是傻等 |
| **9.9 NBV 没有可选视角** | 老: fallback 到第一个未用 | 新: `ActiveViewpointSelector.select` 返回 None → agent 进 ask_user 分支 |
| **9.10 用户回答让 belief 变成"两选一"** | N/A | `consume_user_answer` 设 `boost_hypothesis` + `demote_hypothesis`, 下轮决策应进 grasp |
| **9.11 Action 失败但环境状态变了** (e.g. 撞掉了别的东西) | 老: 不知道 | 新: ActionExecutor 在失败时调一次 `perception.observe(eye_in_hand)`, 把新视角加 evidence; 必要时整个 belief.prune_stale_hypotheses |
| **9.12 同时有多个 hypothesis 都匹配 query** (3 个 apple) | 老: get_best_match 取分最高, 单选 | 新: target() 仅在 top1 与 top2 概率差 > 0.2 时返回 top1; 否则返回 None → agent 进 ask_user ("有 3 个苹果, 您要哪个?") |

---

## 10. Testing Strategy

### 10.1 单元测试 (TDD, 每模块独立)

| 模块 | 测试数 | 覆盖 |
|---|---|---|
| `world_belief.py` | 25+ | 不变量 + target/most_uncertain_axis/merge/prune/consume_user_answer |
| `agent.py` `decide_next` | 15+ | 8 种 belief 状态 + MAX_STEPS + MAX_RE_OBSERVE + axis 路由 |
| `perception.py` | 10+ | parse 解析 + zoom 裁切 + parallax 视角选择 |
| `safety_gate.py` (`SafetyClassifier`) | 5+ | dist 解析 + entropy 计算 |
| `grasp_planner.py` | 8+ | 候选生成 + 可达性过滤 + regenerate |
| `action_executor.py` | 10+ | 各 failure_mode 正确分类 + verify 流程 |
| `user_channel.py` | 8+ | 三种 oracle + 历史维护 + unhelpful answer |
| `episode_logger.py` | 6+ | save/load + replay 决策一致性 |
| `vlm_cache.py` | 5+ | hit/miss/eviction/clear |

总计 100+ 单测 (vs 现 80 个)。

### 10.2 EpisodeLogger Replay 测试

**思路**: 用 sim 跑出一批"已知正确" episode, 序列化为 JSON, 进 `tests/episodes/golden/`。
每次重构后 `pytest tests/replay_test.py`, 用 mock LLM/VLM (返回记录的 raw_payload),
验证 agent.decide_next 决策**等价**——而非严格 sequence 相等。

**为何不要严格 sequence 相等**: golden 是某次具体跑的产物, 但只要我们改了任何 prompt 或
阈值, 决策细节会变 (e.g. 多 zoom 一次 vs 直接 ask_user 都可能是合理的), 严格断言会让
合理改动也"测试失败", 沦为"重构 == 改 golden"。所以只断言**契约级等价**:

```python
# tests/replay_test.py
@pytest.mark.parametrize("episode_path", glob("tests/episodes/golden/*.json"))
def test_replay_decision_consistency(episode_path):
    record = EpisodeLogger.load(episode_path)
    mock_perception = MockFromRecord(record, "vlm_ground")
    mock_safety = MockFromRecord(record, "llm_safety")
    ...
    agent = EmboSightAgent.with_mocks(...)
    result = agent.run(record.query, mock_env)
    
    # ── L1 契约: 终态等价 ───────────────────────────
    # 给定相同的 evidence 流, success 必须一致
    assert result.success == record.final_result.success

    # ── L2 契约: action 集合等价 (顺序可变) ──────────
    # 用了哪些 action kind 必须一致 (如果 golden 触发了 ask_user, 重跑也应该)
    golden_kinds = {a.kind for a in record.actions}
    actual_kinds = {a.kind for a in result.action_history}
    assert actual_kinds == golden_kinds, (
        f"action kind 集合不等: golden={golden_kinds}, actual={actual_kinds}"
    )

    # ── L3 契约: 步数同量级 ──────────────────────────
    # 重跑步数不能比 golden 多 50% (说明决策路径完全跑偏)
    assert len(result.action_history) <= len(record.actions) * 1.5

    # ── L4 契约: 关键 axis 命中 ──────────────────────
    # 如果 golden 在某步消除了 label_entropy (zoom_in), 重跑也应该至少触发过一次
    # zoom_in (按 axis-route 必然性, 不按时间点)
    if any(a.strategy == "zoom_in" for a in record.actions if a.kind == "re_observe"):
        assert any(a.strategy == "zoom_in"
                   for a in result.action_history if a.kind == "re_observe"), \
            "golden zoom 过, replay 没 zoom — 路由策略可能退化"
```

> golden episode 的精确 action sequence 仅作"参考路径", 不作断言。这种 4 层契约把 LLM 输出
> 抖动从测试失败信号里隔离出去, 同时仍能抓"决策路径完全跑偏" / "ask_user 触发条件改了"
> 这种真正的回归。
>
> 严格 sequence 相等留作 **opt-in** 测试 (`@pytest.mark.strict_replay`), 仅在 PR 改
> `decide_next` 树结构时手动启用。

→ 直接补上 P2.8 sim 集成测试断层。

### 10.3 Sim 端到端测试

**保留** 现有 `scripts/run_sim_query.py` 之类的脚本 (改成调 agent.run 而非 pipeline.run)。
增加 5 个 RoboCasa 标准 query 作为 demo 检查清单, 每次 PR 必跑:

```
demo_queries:
  - "帮我拿苹果"          (基础: label 唯一)
  - "我要那个削皮器"        (中等: VLM 容易认错为刀)
  - "拿水池左边的瓶子"      (含位置约束)
  - "帮我取碗, 避开刀"     (含 avoid constraint)
  - "我要那个红色的"        (模糊, 必触发 ask_user)
```

通过标准: 5/5 全部 success 或 ask_user 后 success, 无 raise。

---

## 11. Migration Plan

每步**先写测试再写实现 (TDD)**, 单步可跑通才进下一步。

```
Step 1.  src/world_belief.py                                  [纯数据, 无依赖]
         + tests/test_world_belief.py (25 测试)
         CHECKPOINT: pytest tests/test_world_belief.py 全过

Step 2.  src/vlm_cache.py + tests
         CHECKPOINT: 5 测试全过

Step 3.  src/episode_logger.py + tests
         CHECKPOINT: save/load 通

Step 4.  src/user_channel.py + tests (mock LLM)
         CHECKPOINT: FakeUserChannel 三种构造 + ask 通

Step 5.  src/perception.py (QueryAwareGrounder.observe + parse)
         + prompts/perception/query_aware_ground.txt
         + tests (mock VLM)
         CHECKPOINT: 10 测试 (parse / 解析 alternatives / 计算 entropy)

Step 6.  src/safety_gate.py (新 SafetyClassifier, 删老 SafetyGate)
         + prompts/safety/classify.txt + tests
         CHECKPOINT: dist 解析 + entropy 计算

Step 7.  src/grasp_planner.py + tests
         + prompts/grasp/suggest_top_grasp.txt
         CHECKPOINT: 候选生成 + 可达性 mock

Step 8.  src/action_executor.py 改造
         + 增加 GraspAttempt.failure_mode 分类
         + verify_grasp 实现
         + tests
         CHECKPOINT: 各 failure_mode 单测通

Step 9.  src/active_planner.py 改造 (ActiveViewpointSelector)
         + prompts/agent/nbv_select.txt
         + tests (mock LLM)
         CHECKPOINT: 4 种 preference 各通

Step 10. src/task_decomposer.py 改造
         + prompts/agent/decompose.txt
         + tests
         CHECKPOINT: 输出 DecomposedTask 含 constraints

Step 11. src/agent.py 主循环
         + tests (15+ decide_next 测试)
         CHECKPOINT: 全部 mock 后, run() 跑通 5 种场景

Step 12. src/perception.py re_observe + verify_grasp 实现
         + 4 个新 prompt
         CHECKPOINT: zoom/parallax/pose/verify 各跑通 mock

Step 13. configs/agent.yaml 完整
         旧 configs 标 deprecated (尚不删)
         入口 src/__main__.py 切换走 agent
         CHECKPOINT: 一次完整 sim run 跑通

Step 14. EpisodeLogger replay 测试入 CI
         5 个 golden episode 录入
         CHECKPOINT: replay 测试 5/5 通

Step 15. 删除老代码
         - src/pipeline.py
         - src/vlm_grounding.py
         - src/scene_model.py 中除 projection 外全部 (移到 src/projection.py)
         - src/scene_describer.py 整个 (功能被 perception 吸收)
         - src/action_decider.py (LLM 决策被 agent.decide_next 替代)
         - configs/object_aliases.yaml
         - configs/safety_rules.yaml
         - prompts/ 老文件
         CHECKPOINT: 完整测试套件 + sim 5 query 全过, 老代码彻底清理
```

每个 CHECKPOINT 都是一个 git commit, 单步出问题易回滚。

---

## 12. Demo Story

### 12.1 4 个差异化故事点 (相对于"单纯 grounding accuracy")

**故事 1: "看不清就再看一眼"**
- 用户: "帮我拿削皮器"
- agent 全景视图看到 yellow round thing, label_entropy=0.6
- decide_next 选 re_observe(zoom_in)
- 裁切 bbox 重新 VQA, VLM 看清是 lemon, alternatives=[("lemon",0.85),("orange",0.10)]
- entropy 降到 0.18, 但用户要的是削皮器 → target() 返回 None → 进 NBV 找别处
- → 演示价值: agent 主动消歧, 而非把 lemon 误认为 peeler

**故事 2: "材质拿不准, 调 LLM 问"**
- agent 看到一个杯子, VLM 说 "ceramic-like surface"
- safety_entropy = 0.7 (老的关键词表会直接打 "fragile", 但其实可能是塑料)
- decide_next 选 classify_safety
- LLM 综合 features+context 输出 dist={"safe":0.5,"fragile":0.4,"unknown":0.1}, entropy 仍 0.62
- → high-risk 阈值收紧 → 继续 re_observe(zoom_in)
- 第二轮 LLM 看更清晰图, 输出 {"safe":0.85,"fragile":0.1,...} → entropy=0.25 → 进抓
- → 演示价值: agent 不在风险判断上瞎赌

**故事 3: "抓不到就换抓点"**
- agent 第一次 grasp_attempt: candidate=top_grasp, failure_mode="hit_z_floor"
- pose_uncertainty 从 0.3 拉到 0.7
- decide_next 选 re_observe(parallax_for_pose)
- 侧面视角看到物体是横放的 → pose.upright=False
- regenerate_after_failure 生成 axis_aligned_side 候选
- 第二次 grasp_attempt: candidate=side_grasp, failure_mode="success" → verify 通过
- → 演示价值: 闭环, 失败有反馈

**故事 4: "实在不行问用户"**
- 场景里 3 个红色物体, target() 因为 top1/top2 差距 < 0.2 返回 None
- decide_next → ask_user("我看到 3 个红色的物体: 一个圆形的、一个细长的、一个方形的, 您要哪个?")
- FakeUser (intent="番茄") 答 "圆形的"
- consume_user_answer LLM 解析 → boost 圆形 hypothesis
- 下轮 target() 返回它 → 进 grasp
- → 演示价值: 真正的"双向"agent, 不是单向通报

### 12.2 现 demo_queries.md 升级建议

把现 5 个 query 标记为 v0; 新增 5 个故意触发 fallback 的 query 作 v1 demo:

```
v1_demo_queries:
  - text: "帮我拿削皮器"          # 触发 zoom_in 消歧
    expects: [observe, re_observe, grasp, success]
  - text: "拿那个杯子"             # 触发 classify_safety
    expects: [observe, classify_safety, re_observe, grasp, success]
  - text: "我要个横放的瓶子"        # 触发 parallax_for_pose
    expects: [observe, plan_grasp, grasp(fail:hit_z), re_observe, grasp, success]
  - text: "我要红色的"             # 触发 ask_user
    expects: [observe, ask_user, grasp, success]
  - text: "拿那个削皮器, 避开刀"   # 测 constraints
    expects: [decompose(constraints=[avoid:knife]), observe, grasp, success]
```

---

## 13. Open Questions / Future Work

> v1 不解决, 但接口需要预留。

### 13.1 Session-level 持久化 belief
**问题**: 用户连续说 "拿削皮器, 然后拿水杯", 第二次能否复用第一次的场景观察?
**v1 决定**: fresh_per_episode + 接口预留 `session_belief` 参数。
**后续**: 增加 `WorldBelief.spawn_subbelief(new_query)`, hypothesis 复用但 evidence/open_questions 重置。

### 13.2 真实 ASR/TTS 接入
**问题**: demo 阶段用文本/FakeUser 够, 上真机要换语音。
**v1 决定**: `VoiceUserChannel` 占位类, 接口与 FakeUser/CLI 一致。
**后续**: 接 Whisper / 国内厂商的 STT/TTS。

### 13.3 多目标并行
**问题**: "拿苹果和橙子" 这种。
**v1 决定**: `DecomposedTask.primary_target` 单值, 多目标当 v2 设计。
**后续**: `decompose` 返回 `list[DecomposedTask]`, agent 多 episode 串行 OR 单 episode 多 target slot。

### 13.4 Sim→Real domain adaptation (P3.10)
**问题**: RoboCasa 渲染质量低, Qwen3-VL 域差距大。
**v1 决定**: `re_observe(zoom_in)` 部分缓解 (像素增多)。
**后续**: 选项: (a) 替换 RoboCasa 为更逼真渲染 (Isaac Sim, BlenderProc); (b) 加合成域随机化; (c) 微调 VLM 在仿真图上。

### 13.5 阈值学习 / 在线适应
**问题**: 现在阈值硬编码, 不同物体/场景应该不同。
**v1 决定**: per-axis dynamic (高风险 vs 低风险) 是粗略动态化。
**后续**: bandit / Bayesian update 阈值。

### 13.6 多模态 evidence (触觉, 力反馈)
**问题**: grasp 后能不能让力传感器告诉 agent 抓到的是软物还是硬物?
**v1 决定**: 仅视觉 evidence。
**后续**: `Evidence(source="force_sensor", ...)` 加进 belief。

---

## 14. Appendix A: 接口映射 (老 → 新)

| 现接口 | 新接口 |
|---|---|
| `EmboSightPipeline.run(query, env)` | `EmboSightAgent.run(query, env) -> EpisodeResult` |
| `TaskDecomposer.decompose(query) -> list[Subtask]` | `TaskDecomposer.decompose(query) -> DecomposedTask` |
| `ActivePlanner.plan(subtasks, env) -> list[Observation]` | `ActiveViewpointSelector.select(belief, exclude, preference) -> Optional[Viewpoint]` |
| `ActivePlanner.plan_with_grounding(...)` | (主循环吸收, 删除) |
| `SceneDescriber.describe(image_path, viewpoint, subtasks)` | `QueryAwareGrounder.observe(viewpoint, env, belief) -> Evidence` |
| `SceneDescriber.aggregate(descriptions)` | (取消, 由 belief 直接生成 speech) |
| `VLMGrounder.ground(image_path) -> list[GroundedCandidate]` | `QueryAwareGrounder.observe(...).hypotheses` |
| `VLMGrounder.match_query(candidates, query, gt)` | (取消, prompt 中已注入 query) |
| `SceneModel.add_view(...)` | `belief.merge_hypothesis(...)` |
| `SceneModel.get_best_match(...)` | `belief.target()` |
| `SafetyGate.check(grounded_object) -> SafetyDecision` | `SafetyClassifier.classify(hyp) -> Evidence`; gate 由 `is_confident_to_act` 决策 |
| `SafetyGate.update_object_safety(grounded)` | (取消, 改写 `hyp.safety_dist`) |
| `ActionDecider.decide(query, desc) -> ActionPlan` | (取消, agent.decide_next 替代) |
| `ActionExecutor.execute(plan, env)` | `ActionExecutor.act(target, decomposed, env) -> ActionResult` |
| `ActionExecutor.execute_with_scene_model(plan, model, env)` | (取消, 同上) |

## 15. Appendix B: 完整文件改动表

```
新建:
  src/world_belief.py
  src/agent.py
  src/perception.py
  src/grasp_planner.py
  src/user_channel.py
  src/episode_logger.py
  src/vlm_cache.py
  src/projection.py
  
  configs/agent.yaml
  
  prompts/agent/decompose.txt
  prompts/agent/nbv_select.txt
  prompts/agent/user_answer_parse.txt
  prompts/perception/query_aware_ground.txt
  prompts/perception/zoom_disambiguate.txt
  prompts/perception/parallax_localize.txt
  prompts/perception/pose_estimation.txt
  prompts/perception/verify_grasp.txt
  prompts/safety/classify.txt
  prompts/grasp/suggest_top_grasp.txt
  prompts/user/fake_user_system.txt
  
  tests/test_world_belief.py
  tests/test_agent_decide_next.py
  tests/test_perception.py
  tests/test_safety_classifier.py
  tests/test_grasp_planner.py
  tests/test_action_executor.py
  tests/test_user_channel.py
  tests/test_episode_logger.py
  tests/test_vlm_cache.py
  tests/replay_test.py
  tests/episodes/golden/   (5 个 JSON)

改造:
  src/safety_gate.py        (-150 +40, 重命名内部类为 SafetyClassifier)
  src/active_planner.py     (-200 +80, ActiveViewpointSelector, 删 plan/plan_with_grounding)
  src/action_executor.py    (-100 +120, 接 Hypothesis, 加 verify_grasp + GraspAttempt)
  src/task_decomposer.py    (改 -> DecomposedTask)
  src/__init__.py           (导出新增模块)
  configs/default.yaml      (添加 agent: 段)

删除:
  src/pipeline.py
  src/vlm_grounding.py
  src/scene_model.py        (功能拆到 world_belief + projection)
  src/scene_describer.py
  src/action_decider.py
  
  configs/object_aliases.yaml
  configs/safety_rules.yaml
  
  prompts/active_planner.txt
  prompts/active_planner_grounding_aware.txt
  prompts/scene_describer.txt
  prompts/vlm_grounding.txt
  prompts/action_decider.txt
  prompts/task_decompose.txt
```

---

## 16. 实施前置条件

实施开始前需确认:
1. ✅ 用户已 review 此设计文档并批准 (本步骤)
2. ⏳ 仿真环境 `env_wrapper` 当前能稳定 `observe(viewpoint)` (已知 OK)
3. ⏳ `env_wrapper` 暴露 `is_reachable(point, approach_dir)` API (新增, 需要在 `env_wrapper` 加方法)
4. ⏳ `env_wrapper` 暴露 `descend(z_target)` 失败时回报 z_actual (现有方法需补 diagnostic)
5. ⏳ `LLMBackend.generate(prompt, system, json_mode)` 已稳定 (现有, 已用)
6. ⏳ `VLMBackend.describe(image_path, prompt)` 已稳定 (现有, 已用)

如果 (3)(4) 未就绪, 实施 Step 8 (action_executor 改造) 前需先扩展 `env_wrapper`。

---

## 16.5 Review Findings & Document Revisions (2026-05-08 自审)

文档完稿后做了一轮自审, 包含 (a) 4 条根因 vs `src/` 实际代码核实, (b) 设计细节漏洞修补。
全部修订已 in-place 写入对应章节, 此处列变更摘要供 reviewer 快速对照。

### 16.5.1 4 条根因 vs 代码核实

| 根因 | 验证方式 | 结论 |
|---|---|---|
| ① 感知开环 | `vlm_grounding.py:145-178` `ground()` 调一次 return; `prompts/vlm_grounding.txt` 不注入 query; `active_planner.py:303-328` 早停=`grounding_score AND coverage` | **ACCURATE** |
| ② 规则编码 | `_score_candidate` Level 0-4 + `semantic_pairs` (~25 对) + GT cross-check + family penalty + generic penalty 单函数 ~140 行; `safety_gate.py:56-61` `_FEATURE_RISK_KEYWORDS`; `configs/{object_aliases,safety_rules}.yaml` 30+ 类硬编码 | **ACCURATE, 200 行的数字保守** |
| ③ 标量信心 | 32 处 `query_match_score: float`, 跨 5 个文件; `safety_gate.py:127` `min(query_match, position_conf)` 把所有不确定压成 1 个标量 | **ACCURATE** (现状是 2 标量, 设计稿正确指出应升 4 维) |
| ④ 模块各自为政 | VLM prompt 不知 query (✓); SafetyGate 接口无 subtask (✓); ActionExecutor 双路径 try/fallback (✓); env_wrapper.grasp_at 内部仍读 obj_main 真值 (✓ 但表述精确化为"内部 GT 验证", 不是"完全绕过") | **ACCURATE 主论点; 1 子论点措辞已精确化 (§1.3)** |

**总体**: 4/4 条根因主方向都站得住脚, 不是公关式自批评, 是代码层精确观察。

### 16.5.2 设计漏洞修补 (6 处)

| # | 章节 | 问题 | 修订 |
|---|---|---|---|
| F1 | §1.3 根因④ | "env_wrapper.grasp 完全绕过 SceneModel" 措辞过强 | 改成"`grasp_at` 内部仍把 `target_body=obj_main` 当默认 GT 来源, 这是 sim-only 的作弊路径, SceneModel 的语义信息没真正进入抓取闭环"; 同时指出"物理层 retry 有, 但语义层 retry 缺失" |
| F2 | §4.1 / §4.4 | `grasp_uncertainty` 初始退化: episode 初期所有轴默认 1.0, grasp 这种派生量会被 max 误选, 触发过早 plan_grasp_candidates | `grasp_uncertainty` 改返回 `Optional[float]`: 无 candidates 又无 attempts 时返回 `None`; `most_uncertain_axis` 跳过 `None`; `is_confident_to_act` 把 `None` 视为不 confident; `overall_uncertainty` 不参与 max |
| F3 | §4.1 / §6.3 / §8.1 | VLM 概率校准差, 直出 alternatives 概率会让 `label_entropy` 几乎永远 < 阈值, zoom_in 永不触发 | (a) prompt 加 "BE CONSERVATIVE" 指令; (b) `_parse_to_hypotheses` 增加温度缩放 `p_i' = p_i^(1/τ)/Σ`, τ 可配 (默认 1.5); (c) `configs/agent.yaml` 加 `perception.label_temperature` |
| F4 | §4.1 / §6.4 / §8.1 | `safety_dist` 5 类固定可能不够 (weight, wet 缺); 合并阈值 `0.15m / 0.30 prob` 拍脑袋 | (a) `safety_dist` 标"开放 key dict", v2 可加 weight/wet; (b) `merge_distance_m` / `merge_label_intersection_min` 加 `TODO(v1.1)` 标待实测调 |
| F5 | §5.2 | 决策树是 if-else, 跟根因②"删 200 行规则"看似矛盾 | 加"设计动机"说明: 删的是**世界知识**规则 (LLM 训练数据里的, 必填不全), 编码的是**控制策略** (状态空间小, 可单测可审计); 二者并不冲突 |
| F6 | §5.3 / §6.7 / §9.6 | `verify_mismatch` 后物体仍被夹爪夹起, 后续 observe 被夹爪遮挡 → 死锁 | 加 `ActionExecutor.release_and_retreat`: open_gripper + 提升 0.10m; verify_mismatch 流程强制调用; §9.6 表格更新 |
| F7 | §10.2 | Replay 测试断言"action sequence 严格相等", 任何 prompt/阈值微调都会让 golden 失效 | 改 4 层契约等价 (L1 终态 / L2 action 集合 / L3 步数同量级 / L4 关键 axis 命中); 严格相等改 `@pytest.mark.strict_replay` opt-in |

### 16.5.3 未修订, 但 reviewer 应注意

- **`label_entropy` 计算具体公式** 没在文档里写死 (是用 nats 还是 bits, alternatives 是否包含 "other"), 实施时需要在 §6.3 单测里 pin 住, 否则不同实现互不兼容
- **`MAX_STEPS=12 / MAX_RE_OBSERVE=3`** 是初值, 在跑 §10.3 demo_queries 后可能要调
- **per-axis dynamic 阈值表** (`high_risk` vs `default`) 也是初值, 同上
- **`release_and_retreat` 的撤回方向** 默认 +z 0.10m, 但如果是被 cabinet 顶住会撞 (未来需要 free-space check)

---

## 17. Review Checklist (给用户)

请按以下顺序 review:

- [ ] §1.3 4 条根因是否覆盖你列的全部 10 个问题 (§16.5.1 已附代码核实结论)
- [ ] §2.2 Non-Goals 是否真的能不做 (有没有藏着影响 demo 的事)
- [ ] §4.1 Hypothesis 4 轴定义是否需要加/减/改 (注意 F2 grasp_uncertainty=None 退化, F3 温度缩放, F4 safety_dist 开放 key)
- [ ] §5.2 decide_next 决策树有无遗漏分支 (注意 F5 设计动机已说明它为何用规则)
- [ ] §6 模块接口契约是否清晰 (有没有"我看不出怎么实现"的, 注意 F6 release_and_retreat 已加进 §6.7)
- [ ] §9 Edge cases 12 条够不够 (你想到的别的 case 加进来; 9.6 已含 release/retreat 步骤)
- [ ] §10.2 Replay 测试 4 层契约 (F7) 接受度——是否需要更严或更宽
- [ ] §11 Migration 15 步顺序是否合理 (有没有应该并行的)
- [ ] §12 Demo 4 个故事点你哪个最想要 (实施时优先保它们能演)
- [ ] §13 Open Questions 是否有应该提到 v1 的
- [ ] §16.5 自审表 (F1-F7) 是否还有遗漏

Review 后给我一个 GO 或 修改清单, 我进 writing-plans 阶段开始拆实施任务。
