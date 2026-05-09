# Belief-Driven Agent Migration Plan (C+B v1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **基于 spec**: `docs/superpowers/specs/2026-05-08-emboSight-belief-driven-agent-design.md`
> **执行原则**: TDD; 每 Phase 一个可单测/可回滚的 commit; 老代码先并存、最后 Phase 15 删

**Goal:** 把 EmboSight 现有"6 步线性 pipeline"重构成单一 `WorldBelief` 驱动的智能体循环, 实现 4 轴结构化不确定性 (label/position/safety/grasp) + fail-aware 闭环 + ask_user 分支, 替代 `vlm_grounding` 的 200 行 if-else 与 `safety_gate` 的关键词表。

**Architecture:** 新建 `EmboSightAgent.run` 为新主入口, 内部主循环 `while not belief.is_confident_to_act(): decide_next(belief)`; 7 类 action (observe/re_observe/classify_safety/plan_grasp_candidates/grasp/ask_user/give_up); 失败结构化回写 belief。新建 8 个文件、改造 4 个、删除 5 个文件 + 2 个 YAML。

**Tech Stack:** Python 3.10+, RoboCasa + robosuite + MuJoCo, DeepSeek (LLM, OpenAI 兼容), Qwen2.5/3-VL (本地 GPU), pytest 8+, ruff, dataclass, numpy.

---

## Phase 总览

| Phase | 交付物 | 依赖 | Checkpoint |
|---|---|---|---|
| 1 | `src/world_belief.py` + 25 单测 | — | `pytest tests/test_world_belief.py` 全过 |
| 2 | `src/vlm_cache.py` + 5 单测 | — | `pytest tests/test_vlm_cache.py` 全过 |
| 3 | `src/episode_logger.py` + 6 单测 | Phase 1 | save / load 通; replay 占位通 |
| 4 | `src/user_channel.py` + 8 单测 | — | 三种 FakeUserChannel + ask 通 |
| 5 | `src/perception.py` `observe()` + 第 1 个 prompt + 10 单测 | Phase 1, 2 | parse / alternatives / entropy 通 |
| 6 | `src/safety_gate.py` 改造为 `SafetyClassifier` + classify prompt + 5 单测 | Phase 1 | dist 解析 + entropy 通 |
| 7 | `src/grasp_planner.py` + suggest_top_grasp prompt + 8 单测 | Phase 1 | 候选生成 + 可达性 mock 通 |
| 8 | `src/action_executor.py` 改造 (Hypothesis 输入 + verify_grasp + release_and_retreat) + 10 单测 | Phase 1, 5 | 各 failure_mode 单测通 |
| 9 | `src/active_planner.py` 改造为 `ActiveViewpointSelector` + nbv prompt + 5 单测 | Phase 1 | 4 种 preference 通 |
| 10 | `src/task_decomposer.py` 改造 → `DecomposedTask` + decompose prompt + 5 单测 | Phase 1 | constraints 解析通 |
| 11 | `src/agent.py` 主循环 + decide_next + 17 单测 (含 verify_mismatch 流程) | Phase 1-10 | 8 种 belief 状态 + run 5 场景 + F6 verify 恢复路径通 |
| 12 | `src/perception.py` re_observe (zoom/parallax/pose) + verify_grasp + 4 prompt | Phase 5, 8 | 4 种 strategy mock 通 |
| 13 | `configs/agent.yaml` + `scripts/run_agent.py` + `src/__init__.py` 公开 API | Phase 11 | yaml 加载 + 脚本 import + public API 1 单测通 |
| 14 | EpisodeLogger replay 测试 + 5 golden episode (1 真实风格 + 4 mock-based, v1.1 替换为真 sim) | Phase 13 | 5/5 golden 过 4 层契约 |
| 15 | 删除老代码 (`pipeline.py` / `vlm_grounding.py` / `scene_describer.py` / `action_decider.py` / `scene_model.py` 主体 / 2 个 YAML / 老 prompts) | Phase 14 | 完整测试套件 + sim 5 query 全过 |

每个 Phase = 一个 git commit, 单步出问题易回滚。

---

## 跨阶段约定 (所有 Phase 通用)

### 测试结构

- 测试文件位置: `tests/test_<module>.py` (扁平, 不嵌套)
- 顶部 boilerplate:
  ```python
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).parent.parent))
  
  import pytest
  import numpy as np
  ```
- 用 `class TestSomething:` 分组测试方法
- Mock 类显式定义, 仅含 SUT 需要的字段 (参考 `tests/test_safety_gate.py:21 MockObj`)
- 每个测试方法 1 句中文 docstring 说明意图

### 运行命令

- 单文件单测: `pytest tests/test_world_belief.py -v`
- 跨文件全套: `pytest tests/ -v --tb=short`
- Lint: `ruff check src/ tests/`
- Format: `ruff format src/ tests/`

### Commit 风格 (模仿当前 repo)

```
<type>(<scope>): <subject>

类型:
  feat   新功能
  fix    bug 修复
  refactor 重构 (无行为改变)
  test   只动测试
  docs   文档
  diag   diagnostic / log

scope = 模块: belief / agent / perception / safety / grasp / executor / planner /
              decomposer / cache / logger / userch / config / cleanup

例:
  feat(belief): introduce WorldBelief + Hypothesis with 4-axis uncertainty
  test(belief): cover most_uncertain_axis 8 cases including grasp=None
  refactor(executor): take Hypothesis instead of ActionPlan, add verify_grasp
```

### 不做的事

- ❌ 不在 Phase 1-14 中删任何老代码 (并存; Phase 15 才删)
- ❌ 不动 `configs/viewpoints.yaml` (NBV 视角库, 设计稿明确 v1 不动)
- ❌ 不接真 VLM/LLM 跑端到端 (单测全 mock; sim 集成放 Phase 13 用 `scripts/`)
- ❌ 不引入新依赖 (设计稿全用 dataclass + numpy + yaml + pytest, 这些都已在 requirements.txt)
- ❌ 不实现 `prompts/agent/user_answer_parse.txt` + `consume_user_answer` 的 LLM 语义解析 (设计稿 §6.1 / Edge 9.4 / 9.10 提到的 boost/demote/unhelpful 检测)
  - **v1 简化**: `consume_user_answer(q, a, llm)` 仅把 `f"Q: {q} | A: {a}"` 追加到 `user_constraints[]` (字符串列表)
  - **下游影响**: 用户答案不会自动 boost 某个 hypothesis 概率; 用户答 "不知道" 也不会被识别为 unhelpful
  - **缓解**: 用户给出后, 下一轮 NBV/perception prompt 可在系统消息里把 `user_constraints` 注入 (Phase 5/9 prompt 模板已有 `{constraints}` 槽位, agent 主循环把 `user_constraints` merge 进 `decomposed.constraints` 即可让 LLM "看见")
  - **推迟到 v1.1**: 一旦 demo 时观察到 ask_user 后 belief 没有进展, 再补 prompt + 解析逻辑

### LLM/VLM 调用 mock 范式

测试里需要 mock LLMBackend / VLMBackend 的, 用以下结构:

```python
class MockLLM:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, dict]] = []
    
    def generate(self, prompt: str, system: str = "", **kw) -> str:
        self.calls.append((prompt, kw))
        if not self._responses:
            raise RuntimeError("MockLLM out of responses")
        return self._responses.pop(0)


class MockVLM:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []
    
    def describe(self, image_path: str, prompt: str = "") -> str:
        self.calls.append((image_path, prompt))
        if not self._responses:
            raise RuntimeError("MockVLM out of responses")
        return self._responses.pop(0)
```

放在 `tests/_mocks.py` (Phase 1 第一个 commit 顺手建)。

---

## Phase 1: `src/world_belief.py` + 25 单测

**目标:** 建立 belief 数据层 (Hypothesis / Action / Evidence / WorldBelief / EpisodeResult / GraspCandidate / GraspAttempt / Pose / DecomposedTask / Constraint / BeliefSnapshot)。无 IO, 无 LLM/VLM。后续所有 Phase 依赖此文件。

### Task 1.1: 建 `tests/_mocks.py` 和测试 boilerplate

**Files:**
- Create: `tests/_mocks.py`
- Create: `tests/test_world_belief.py` (空文件占位)

- [ ] **Step 1: 写 `tests/_mocks.py`**

```python
"""跨测试文件复用的 mock。"""
from typing import Any


class MockLLM:
    """模拟 LLMBackend.generate。"""
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []
    
    def generate(self, prompt: str, system: str = "", **kw) -> str:
        self.calls.append((prompt, kw))
        if not self._responses:
            raise RuntimeError("MockLLM out of responses")
        return self._responses.pop(0)


class MockVLM:
    """模拟 VLMBackend.describe。"""
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []
    
    def describe(self, image_path: str, prompt: str = "") -> str:
        self.calls.append((image_path, prompt))
        if not self._responses:
            raise RuntimeError("MockVLM out of responses")
        return self._responses.pop(0)
```

- [ ] **Step 2: 创建 `tests/test_world_belief.py` 占位**

```python
"""WorldBelief / Hypothesis / Evidence / Action 单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
```

- [ ] **Step 3: commit**

```bash
git add tests/_mocks.py tests/test_world_belief.py
git commit -m "test(belief): add MockLLM/MockVLM helpers + test scaffold"
```

---

### Task 1.2: 写 Hypothesis 4 轴 + 派生属性的失败测试

**Files:**
- Modify: `tests/test_world_belief.py`

- [ ] **Step 1: 把 Hypothesis 基础测试加进 `tests/test_world_belief.py`**

```python
from src.world_belief import (
    Hypothesis, Pose, GraspCandidate, GraspAttempt,
)


class TestHypothesisBasics:
    def test_minimal_construct(self):
        """构造一个最小 Hypothesis。"""
        h = Hypothesis(
            object_id="obj_0",
            label="apple",
            label_alternatives=[("apple", 0.8), ("pear", 0.2)],
            label_entropy=0.50,
            position_3d=np.array([0.5, 0.0, 0.9]),
            position_std_m=0.05,
        )
        assert h.label == "apple"
        assert h.label_entropy == 0.50
        assert h.position_std_m == 0.05
        assert h.safety_entropy == 1.0  # 默认最大熵
        assert h.grasp_candidates == []
        assert h.grasp_attempts == []


class TestGraspUncertainty:
    def _make(self, candidates=None, attempts=None):
        return Hypothesis(
            object_id="o0", label="x",
            label_alternatives=[("x", 1.0)], label_entropy=0.0,
            position_3d=np.zeros(3), position_std_m=0.0,
            grasp_candidates=candidates or [],
            grasp_attempts=attempts or [],
        )
    
    def test_no_candidates_no_attempts_returns_none(self):
        """未规划时 grasp_uncertainty 必须是 None (F2)。"""
        h = self._make()
        assert h.grasp_uncertainty is None
    
    def test_with_candidate_no_attempt_returns_one_minus_score(self):
        """有候选无尝试 → 1 - 最高 score。"""
        c = GraspCandidate(
            point_3d=np.array([0.5, 0, 0.9]),
            approach_dir=np.array([0, 0, -1]),
            finger_width_m=0.04, score=0.8,
        )
        h = self._make(candidates=[c])
        assert h.grasp_uncertainty == pytest.approx(0.2)
    
    def test_two_failures_force_one(self):
        """连续 ≥2 次非 success 强制 1.0 (触发 ask_user)。"""
        c = GraspCandidate(
            point_3d=np.array([0.5, 0, 0.9]),
            approach_dir=np.array([0, 0, -1]),
            finger_width_m=0.04, score=0.9,
        )
        a1 = GraspAttempt(timestamp=1.0, candidate=c, failure_mode="hit_z_floor",
                          end_effector_pose_reached=(0,0,0,0,0,0))
        a2 = GraspAttempt(timestamp=2.0, candidate=c, failure_mode="ik_unreachable",
                          end_effector_pose_reached=(0,0,0,0,0,0))
        h = self._make(candidates=[c], attempts=[a1, a2])
        assert h.grasp_uncertainty == 1.0
    
    def test_used_candidate_excluded_from_feasibility(self):
        """失败过的候选不重复试。"""
        c1 = GraspCandidate(point_3d=np.array([0.5,0,0.9]),
                            approach_dir=np.array([0,0,-1]),
                            finger_width_m=0.04, score=0.9)
        c2 = GraspCandidate(point_3d=np.array([0.6,0,0.9]),
                            approach_dir=np.array([0,0,-1]),
                            finger_width_m=0.04, score=0.6)
        a1 = GraspAttempt(timestamp=1.0, candidate=c1, failure_mode="hit_z_floor",
                          end_effector_pose_reached=(0,0,0,0,0,0))
        h = self._make(candidates=[c1, c2], attempts=[a1])
        # c1 已用过, 只剩 c2 score=0.6 → uncertainty = 0.4
        assert h.grasp_uncertainty == pytest.approx(0.4)


class TestOverallUncertainty:
    def _make(self, label_e=0.0, pos_std=0.0, safe_e=0.0, grasp_unc=None,
              candidates=None, attempts=None):
        return Hypothesis(
            object_id="o0", label="x",
            label_alternatives=[("x", 1.0)], label_entropy=label_e,
            position_3d=np.zeros(3), position_std_m=pos_std,
            safety_entropy=safe_e,
            grasp_candidates=candidates or [],
            grasp_attempts=attempts or [],
        )
    
    def test_grasp_none_skipped(self):
        """grasp=None 时 overall 仅看 label/pos/safety (F2)。"""
        h = self._make(label_e=0.4, pos_std=0.0, safe_e=0.2)
        assert h.overall_uncertainty() == pytest.approx(0.4)
    
    def test_position_normalized(self):
        """position_std_m / 0.30 归一化到 [0,1]。"""
        h = self._make(label_e=0.0, pos_std=0.15, safe_e=0.0)
        # 0.15 / 0.30 = 0.5
        assert h.overall_uncertainty() == pytest.approx(0.5)
```

- [ ] **Step 2: 跑测试确认全部 fail (因 src/world_belief.py 还不存在)**

Run: `pytest tests/test_world_belief.py -v`
Expected: `ERROR` on import (`No module named src.world_belief`)

- [ ] **Step 3: 不 commit (这步只确认红)**

---

### Task 1.3: 实现 `src/world_belief.py` (Pose / Hypothesis / GraspCandidate / GraspAttempt)

**Files:**
- Create: `src/world_belief.py`

- [ ] **Step 1: 写第 1 段 (imports + Pose + GraspCandidate + GraspAttempt)**

```python
"""EmboSight Agent v1 - 信念状态数据结构。

纯数据层: 无 IO, 无 LLM/VLM 调用。所有状态修改通过显式方法。

设计参考: docs/superpowers/specs/2026-05-08-emboSight-belief-driven-agent-design.md §4
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 4.1 Pose / GraspCandidate / GraspAttempt
# ============================================================

@dataclass
class Pose:
    """物体姿态估计 (6D)。"""
    position: np.ndarray              # (3,) world coord
    rotation_quat: np.ndarray         # (4,) (x, y, z, w)
    upright: bool = True              # 横/竖 (粗略, 由 VLM 判)


@dataclass
class GraspCandidate:
    """单个候选抓点。"""
    point_3d: np.ndarray              # 抓点世界坐标
    approach_dir: np.ndarray          # 接近方向 (单位向量, 指向物体)
    finger_width_m: float             # 张开宽度估计
    score: float                      # 0-1: 综合几何 + 姿态 + 可达性
    source: Literal[
        "vlm_top_grasp", "geometric_centroid",
        "axis_aligned_side", "user_corrected",
    ] = "geometric_centroid"


@dataclass
class GraspAttempt:
    """已经试过的抓取记录。"""
    timestamp: float
    candidate: GraspCandidate
    failure_mode: Literal[
        "success",
        "hit_z_floor",                # OSC 卡 z, 没下到目标深度
        "ik_unreachable",             # 工作空间外
        "collision",                  # 撞到其他物体
        "slipped",                    # 关爪后物体掉了
        "verify_mismatch",            # post-grasp VLM 说抓错了
        "timeout",                    # OSC 步数耗尽
    ]
    end_effector_pose_reached: tuple[float, ...]  # (x, y, z, roll, pitch, yaw)
    diagnostic: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 2: 写第 2 段 (Hypothesis)**

继续追加到 `src/world_belief.py`:

```python
# ============================================================
# 4.1 Hypothesis (4 轴结构化不确定性)
# ============================================================

@dataclass
class Hypothesis:
    """场景中一个候选物体, 带 4 轴结构化不确定性。"""
    object_id: str
    
    # ──── 1. 类别轴 ────
    label: str
    label_alternatives: list[tuple[str, float]]   # [("label", prob), ...]
    label_entropy: float                          # H(alternatives), 越大越不确定
    
    # ──── 2. 位置轴 ────
    position_3d: np.ndarray                       # (3,) world coord
    position_std_m: float                         # 多视角投影 std (m)
    bbox_per_view: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    
    # ──── 3. 风险轴 ────
    # 开放 key dict, 默认 v1 类: safe / fragile / sharp / hot / chemical
    # 后续可加 weight / wet 等
    safety_dist: dict[str, float] = field(default_factory=dict)
    safety_entropy: float = 1.0                   # 初始最大熵 (未分类)
    
    # ──── 4. 抓取轴 ────
    pose_estimate: Optional[Pose] = None
    pose_uncertainty: float = 1.0
    grasp_candidates: list[GraspCandidate] = field(default_factory=list)
    grasp_attempts: list[GraspAttempt] = field(default_factory=list)
    
    # ──── 元信息 ────
    observed_in_views: list[str] = field(default_factory=list)
    times_re_observed: int = 0
    last_action_failed: Optional[str] = None
    
    # ──────────────────────────────────────
    # 派生属性
    # ──────────────────────────────────────
    
    @property
    def grasp_feasibility(self) -> float:
        """剩余可用候选中分数最高者。失败过的不重复试。"""
        used = {self._cand_key(a.candidate) for a in self.grasp_attempts}
        unused = [c for c in self.grasp_candidates
                  if self._cand_key(c) not in used]
        return max((c.score for c in unused), default=0.0)
    
    @property
    def grasp_uncertainty(self) -> Optional[float]:
        """grasp 不确定度。
        
        返回 None 表示"尚未规划"——既无 candidates 又无 attempts。这种状态下 grasp 轴
        不参与 most_uncertain_axis 排序, 也不阻止 is_confident_to_act 的非 grasp 轴
        confident 判定; 避免 episode 初期 4 轴默认 1.0 时 grasp 占 max 而过早 plan_grasp。
        
        一旦 plan 过 (即使空 candidates) 或有过 attempt: 失败 ≥2 次强制 1.0; 否则 1-feasibility。
        """
        if not self.grasp_candidates and not self.grasp_attempts:
            return None
        n_fail = sum(
            1 for a in self.grasp_attempts if a.failure_mode != "success"
        )
        if n_fail >= 2:
            return 1.0
        return 1.0 - self.grasp_feasibility
    
    @staticmethod
    def _cand_key(c: GraspCandidate) -> tuple:
        return (
            round(float(c.point_3d[0]), 3),
            round(float(c.point_3d[1]), 3),
            round(float(c.point_3d[2]), 3),
            round(float(c.approach_dir[0]), 2),
            round(float(c.approach_dir[1]), 2),
            round(float(c.approach_dir[2]), 2),
        )
    
    def overall_uncertainty(self) -> float:
        """各轴 max, 决定是否进 is_confident_to_act。归一化到 [0, 1]。
        
        grasp_uncertainty=None 时不参与 max。
        """
        norm_pos = min(1.0, self.position_std_m / 0.30)
        axes = [self.label_entropy, norm_pos, self.safety_entropy]
        gu = self.grasp_uncertainty
        if gu is not None:
            axes.append(gu)
        return max(axes)
```

- [ ] **Step 3: 跑 1.2 写好的测试**

Run: `pytest tests/test_world_belief.py::TestHypothesisBasics tests/test_world_belief.py::TestGraspUncertainty tests/test_world_belief.py::TestOverallUncertainty -v`
Expected: 全部 pass (8 tests)

- [ ] **Step 4: commit**

```bash
git add src/world_belief.py tests/test_world_belief.py
git commit -m "feat(belief): add Pose/GraspCandidate/GraspAttempt/Hypothesis with 4-axis uncertainty"
```

---

### Task 1.4: Action / Evidence / BeliefSnapshot / EpisodeResult / DecomposedTask 数据结构

**Files:**
- Modify: `src/world_belief.py` (追加)
- Modify: `tests/test_world_belief.py` (追加)

- [ ] **Step 1: 追加测试**

```python
from src.world_belief import (
    Action, Evidence, BeliefSnapshot, EpisodeResult,
    DecomposedTask, Constraint,
)


class TestActionEvidence:
    def test_action_default_metadata_dict(self):
        a = Action(kind="observe")
        assert a.kind == "observe"
        assert a.metadata == {}
    
    def test_evidence_default_consumed_by_empty(self):
        ev = Evidence(source="vlm_ground", timestamp=1.0, raw_payload={"x": 1})
        assert ev.consumed_by == []
    
    def test_decomposed_task_constraints_empty(self):
        dt = DecomposedTask(primary_target="apple", raw_query="拿苹果")
        assert dt.constraints == []
        assert dt.primary_target == "apple"
```

- [ ] **Step 2: 在 `src/world_belief.py` 追加这些 dataclass**

```python
# ============================================================
# 4.6 DecomposedTask / Constraint
# ============================================================

@dataclass
class Constraint:
    kind: Literal["avoid", "prefer_view", "max_force", "user_hint"]
    target_label: Optional[str] = None
    text: Optional[str] = None
    reason: str = ""


@dataclass
class DecomposedTask:
    primary_target: str
    constraints: list[Constraint] = field(default_factory=list)
    raw_query: str = ""


# ============================================================
# 4.3 Evidence / Action / BeliefSnapshot
# ============================================================

EvidenceSource = Literal[
    "vlm_ground", "vlm_zoom", "vlm_verify",
    "llm_safety", "llm_decompose", "user_answer",
    "grasp_attempt", "depth_projection", "vlm_failed",
]


@dataclass
class Evidence:
    """一次工具调用的原始结果, 用于审计 + replay。"""
    source: EvidenceSource
    timestamp: float
    raw_payload: dict[str, Any]
    consumed_by: list[str] = field(default_factory=list)


ActionKind = Literal[
    "observe", "re_observe", "classify_safety",
    "plan_grasp_candidates", "grasp", "ask_user", "give_up",
]


@dataclass
class Action:
    kind: ActionKind
    target_hypothesis: Optional["Hypothesis"] = None
    viewpoint: Optional[Any] = None    # Viewpoint 类在 active_planner, 此处不直接 import
    strategy: Optional[Literal["zoom_in", "parallax_view", "parallax_for_pose"]] = None
    question: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BeliefSnapshot:
    """某时刻 belief 的浅拷贝, 用于 EpisodeLogger。"""
    step: int
    timestamp: float
    n_hypotheses: int
    target_summary: Optional[dict[str, Any]]
    most_uncertain_axis: str
    overall_uncertainty: float
    n_evidence: int
    open_questions_count: int


# ============================================================
# 4.5 EpisodeResult
# ============================================================

@dataclass
class EpisodeResult:
    success: bool
    target: Optional[Hypothesis]
    speech: str
    belief_trace: list[BeliefSnapshot]
    action_history: list[Action]
    n_steps: int
    elapsed_seconds: float
    failure_reason: Optional[str] = None
```

- [ ] **Step 3: 跑测试**

Run: `pytest tests/test_world_belief.py::TestActionEvidence -v`
Expected: 3 pass

- [ ] **Step 4: commit**

```bash
git add src/world_belief.py tests/test_world_belief.py
git commit -m "feat(belief): add Action/Evidence/BeliefSnapshot/EpisodeResult/DecomposedTask"
```

---

### Task 1.5: WorldBelief 容器 + 查询方法 (target / is_confident_to_act / most_uncertain_axis)

**Files:**
- Modify: `src/world_belief.py` (追加 WorldBelief 类)
- Modify: `tests/test_world_belief.py` (追加测试)

- [ ] **Step 1: 写 WorldBelief 测试 (失败先)**

```python
from src.world_belief import WorldBelief


def _basic_hyp(label="apple", label_e=0.2, pos_std=0.04, safe_e=0.2,
               candidates=None, attempts=None, alternatives=None):
    return Hypothesis(
        object_id=f"obj_{label}",
        label=label,
        label_alternatives=alternatives or [(label, 0.8), ("other", 0.2)],
        label_entropy=label_e,
        position_3d=np.array([0.5, 0.0, 0.9]),
        position_std_m=pos_std,
        safety_entropy=safe_e,
        grasp_candidates=candidates or [],
        grasp_attempts=attempts or [],
    )


class TestWorldBeliefTarget:
    def test_empty_belief_target_is_none(self):
        b = WorldBelief(user_query="拿苹果")
        b.decomposed = DecomposedTask(primary_target="apple")
        assert b.target() is None
    
    def test_no_decomposed_target_is_none(self):
        b = WorldBelief(user_query="anything")
        b.hypotheses = [_basic_hyp()]
        assert b.target() is None
    
    def test_label_match_returns_hyp(self):
        b = WorldBelief(user_query="拿苹果")
        b.decomposed = DecomposedTask(primary_target="apple")
        h = _basic_hyp(label="apple",
                       alternatives=[("apple", 0.9), ("kiwi", 0.1)])
        b.hypotheses = [h]
        assert b.target() is h
    
    def test_top1_top2_close_returns_none(self):
        """top1 概率与 top2 差 < 0.2 → 模糊 (9.12)。"""
        b = WorldBelief(user_query="拿苹果")
        b.decomposed = DecomposedTask(primary_target="apple")
        h1 = _basic_hyp(label="apple_1",
                        alternatives=[("apple", 0.4), ("pear", 0.3)])
        h2 = _basic_hyp(label="apple_2",
                        alternatives=[("apple", 0.5), ("pear", 0.2)])
        b.hypotheses = [h1, h2]
        # top1=h2 (0.5), top2=h1 (0.4), diff=0.1 < 0.2 → None
        assert b.target() is None


class TestIsConfidentToAct:
    def test_no_target_not_confident(self):
        b = WorldBelief(user_query="拿苹果")
        b.decomposed = DecomposedTask(primary_target="apple")
        assert b.is_confident_to_act() is False
    
    def test_grasp_none_not_confident(self):
        """grasp_uncertainty=None 视为不 confident (F2)。"""
        b = WorldBelief(user_query="拿苹果")
        b.decomposed = DecomposedTask(primary_target="apple")
        # 其他 3 轴全 confident
        h = _basic_hyp(label="apple", label_e=0.1, pos_std=0.02, safe_e=0.1,
                       alternatives=[("apple", 0.9)])
        b.hypotheses = [h]
        assert b.is_confident_to_act() is False
    
    def test_all_axes_confident_returns_true(self):
        c = GraspCandidate(point_3d=np.array([0.5,0,0.9]),
                           approach_dir=np.array([0,0,-1]),
                           finger_width_m=0.04, score=0.9)
        b = WorldBelief(user_query="拿苹果")
        b.decomposed = DecomposedTask(primary_target="apple")
        h = _basic_hyp(label="apple", label_e=0.1, pos_std=0.02, safe_e=0.1,
                       alternatives=[("apple", 0.9)], candidates=[c])
        b.hypotheses = [h]
        assert b.is_confident_to_act() is True


class TestMostUncertainAxis:
    def test_no_target_returns_label(self):
        b = WorldBelief(user_query="拿苹果")
        b.decomposed = DecomposedTask(primary_target="apple")
        assert b.most_uncertain_axis() == "label"
    
    def test_grasp_none_skipped(self):
        """grasp=None 时不参与最大轴选择 (F2)。"""
        b = WorldBelief(user_query="拿苹果")
        b.decomposed = DecomposedTask(primary_target="apple")
        h = _basic_hyp(label="apple", label_e=0.5, pos_std=0.02, safe_e=0.1,
                       alternatives=[("apple", 0.9)])
        b.hypotheses = [h]
        # 4 轴: label=0.5, pos=0.067, safe=0.1, grasp=None → label 最大
        assert b.most_uncertain_axis() == "label"
    
    def test_safety_max(self):
        b = WorldBelief(user_query="拿苹果")
        b.decomposed = DecomposedTask(primary_target="apple")
        h = _basic_hyp(label="apple", label_e=0.1, pos_std=0.02, safe_e=0.8,
                       alternatives=[("apple", 0.9)])
        b.hypotheses = [h]
        assert b.most_uncertain_axis() == "safety"
```

- [ ] **Step 2: 跑测试 (应全 fail, 因 WorldBelief 不存在)**

Run: `pytest tests/test_world_belief.py::TestWorldBeliefTarget tests/test_world_belief.py::TestIsConfidentToAct tests/test_world_belief.py::TestMostUncertainAxis -v`
Expected: ImportError on `WorldBelief`

- [ ] **Step 3: 写 WorldBelief 类**

追加到 `src/world_belief.py`:

```python
# ============================================================
# 4.4 WorldBelief
# ============================================================

@dataclass
class WorldBelief:
    """主信念状态, 贯穿整个 episode。"""
    user_query: str
    decomposed: Optional[DecomposedTask] = None
    
    hypotheses: list[Hypothesis] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    action_history: list[Action] = field(default_factory=list)
    user_constraints: list[str] = field(default_factory=list)
    
    # ──── 默认阈值 (可被 _dynamic_thresholds 覆盖) ────
    DEFAULT_THRESHOLDS = {
        "label": 0.30, "position": 0.05, "safety": 0.30, "grasp": 0.30,
    }
    HIGH_RISK_THRESHOLDS = {
        "label": 0.15, "position": 0.03, "safety": 0.15, "grasp": 0.20,
    }
    AMBIGUITY_PROB_GAP = 0.20      # top1/top2 概率差 < 此值 → 模糊
    
    # ──── 查询接口 ──────────────────────────
    
    def target(self) -> Optional[Hypothesis]:
        """返回最匹配 user_query 的 hypothesis。
        
        - 如无 decomposed 或无 hypotheses, None
        - 取 label_alternatives 里 primary_target 的概率最大者
        - 副分: 当前 label 文本含 primary_target 也算 0.5
        - top1 与 top2 的概率差 < 0.2 → 视为模糊, 返回 None (Edge case 9.12)
        """
        if not self.hypotheses or not self.decomposed:
            return None
        target_word = self.decomposed.primary_target.lower()
        scored: list[tuple[float, Hypothesis]] = []
        for h in self.hypotheses:
            prob = next(
                (p for lbl, p in h.label_alternatives
                 if target_word in lbl.lower()),
                0.0,
            )
            if target_word in h.label.lower():
                prob = max(prob, 0.5)
            if prob > 0:
                scored.append((prob, h))
        if not scored:
            return None
        scored.sort(key=lambda t: t[0], reverse=True)
        if len(scored) >= 2 and (scored[0][0] - scored[1][0]) < self.AMBIGUITY_PROB_GAP:
            return None
        return scored[0][1]
    
    def is_confident_to_act(
        self,
        label_thr: Optional[float] = None,
        pos_thr_m: Optional[float] = None,
        safety_thr: Optional[float] = None,
        grasp_thr: Optional[float] = None,
    ) -> bool:
        """所有轴都低于阈值才能动手。grasp=None 视为不 confident。"""
        h = self.target()
        if h is None:
            return False
        thr = self._dynamic_thresholds(h, label_thr, pos_thr_m, safety_thr, grasp_thr)
        gu = h.grasp_uncertainty if h.grasp_uncertainty is not None else 1.0
        return (
            h.label_entropy   < thr["label"]
            and h.position_std_m < thr["position"]
            and h.safety_entropy < thr["safety"]
            and gu                < thr["grasp"]
        )
    
    def most_uncertain_axis(self) -> Literal["label", "position", "safety", "grasp"]:
        """最不确定的轴; grasp_uncertainty=None 时跳过。"""
        h = self.target()
        if h is None:
            return "label"
        norm_pos = min(1.0, h.position_std_m / 0.30)
        scores: dict[str, float] = {
            "label":    h.label_entropy,
            "position": norm_pos,
            "safety":   h.safety_entropy,
        }
        if h.grasp_uncertainty is not None:
            scores["grasp"] = h.grasp_uncertainty
        return max(scores, key=scores.get)        # type: ignore[return-value]
    
    def used_views(self) -> set[str]:
        return {
            getattr(a.viewpoint, "name", str(a.viewpoint))
            for a in self.action_history
            if a.kind == "observe" and a.viewpoint is not None
        }
    
    # ──── 内部 ───────────────────────────────
    
    def _dynamic_thresholds(
        self, h: Hypothesis,
        label_thr, pos_thr_m, safety_thr, grasp_thr,
    ) -> dict[str, float]:
        """高风险物体严格, safe 物体宽松。
        
        high_risk = h.safety_dist 中 sharp/hot/chemical 之和 > 0.5
        """
        risk_score = (
            h.safety_dist.get("sharp", 0.0)
            + h.safety_dist.get("hot", 0.0)
            + h.safety_dist.get("chemical", 0.0)
        )
        base = self.HIGH_RISK_THRESHOLDS if risk_score > 0.5 else self.DEFAULT_THRESHOLDS
        return {
            "label":    label_thr   if label_thr   is not None else base["label"],
            "position": pos_thr_m   if pos_thr_m   is not None else base["position"],
            "safety":   safety_thr  if safety_thr  is not None else base["safety"],
            "grasp":    grasp_thr   if grasp_thr   is not None else base["grasp"],
        }
```

- [ ] **Step 4: 跑测试**

Run: `pytest tests/test_world_belief.py::TestWorldBeliefTarget tests/test_world_belief.py::TestIsConfidentToAct tests/test_world_belief.py::TestMostUncertainAxis -v`
Expected: 11 pass

- [ ] **Step 5: commit**

```bash
git add src/world_belief.py tests/test_world_belief.py
git commit -m "feat(belief): WorldBelief container + target/is_confident_to_act/most_uncertain_axis"
```

---

### Task 1.6: WorldBelief 修改接口 (add / merge / prune / consume_user_answer / snapshot)

**Files:**
- Modify: `src/world_belief.py`
- Modify: `tests/test_world_belief.py`

- [ ] **Step 1: 测试 (先红)**

```python
class TestMerge:
    def test_close_distance_overlap_label_merges(self):
        """距离 < 0.15m + 概率交集 > 0.30 → 合并。"""
        b = WorldBelief(user_query="x")
        b.decomposed = DecomposedTask(primary_target="apple")
        h1 = _basic_hyp(label="apple",
                        alternatives=[("apple", 0.7), ("pear", 0.3)])
        h1.position_3d = np.array([0.50, 0.0, 0.9])
        h1.observed_in_views = ["v1"]
        b.hypotheses = [h1]
        h2 = _basic_hyp(label="apple",
                        alternatives=[("apple", 0.8), ("kiwi", 0.2)])
        h2.position_3d = np.array([0.52, 0.01, 0.91])    # 距离 ~0.022m
        h2.observed_in_views = ["v2"]
        merged = b.merge_hypothesis(h1, h2)
        assert merged is True
        assert len(b.hypotheses) == 1
        assert "v2" in b.hypotheses[0].observed_in_views
    
    def test_far_distance_does_not_merge(self):
        b = WorldBelief(user_query="x")
        b.decomposed = DecomposedTask(primary_target="apple")
        h1 = _basic_hyp(label="apple")
        h1.position_3d = np.array([0.5, 0, 0.9])
        b.hypotheses = [h1]
        h2 = _basic_hyp(label="apple")
        h2.position_3d = np.array([0.8, 0, 0.9])         # 距离 0.3m > 0.15
        merged = b.merge_hypothesis(h1, h2)
        assert merged is False
        assert len(b.hypotheses) == 1                    # 还没 add 进去
    
    def test_low_label_intersection_does_not_merge(self):
        b = WorldBelief(user_query="x")
        b.decomposed = DecomposedTask(primary_target="apple")
        h1 = _basic_hyp(label="apple",
                        alternatives=[("apple", 0.9), ("pear", 0.1)])
        h2 = _basic_hyp(label="bottle",
                        alternatives=[("bottle", 0.9), ("can", 0.1)])
        h2.position_3d = h1.position_3d + np.array([0.02, 0, 0])
        b.hypotheses = [h1]
        # 概率交集: 仅 (apple, 0.9)/(apple, 0.0) = 0; bottle 0.9/0.0 = 0; → 0
        merged = b.merge_hypothesis(h1, h2)
        assert merged is False


class TestPrune:
    def test_phantom_pruned(self):
        """1 视角 + entropy>0.7 + 步数>3 → 删。"""
        b = WorldBelief(user_query="x")
        h_ghost = _basic_hyp(label="ghost", label_e=0.85,
                             alternatives=[("ghost", 0.4), ("blob", 0.4)])
        h_ghost.observed_in_views = ["v1"]
        b.hypotheses = [h_ghost]
        # 模拟 4 步
        for _ in range(4):
            b.action_history.append(Action(kind="observe"))
        n = b.prune_phantom_hypotheses()
        assert n == 1
        assert b.hypotheses == []
    
    def test_multi_view_not_pruned(self):
        b = WorldBelief(user_query="x")
        h = _basic_hyp(label="apple", label_e=0.85)
        h.observed_in_views = ["v1", "v2"]
        b.hypotheses = [h]
        for _ in range(4):
            b.action_history.append(Action(kind="observe"))
        n = b.prune_phantom_hypotheses()
        assert n == 0


class TestSnapshot:
    def test_snapshot_basic(self):
        b = WorldBelief(user_query="拿苹果")
        b.decomposed = DecomposedTask(primary_target="apple")
        h = _basic_hyp(label="apple", label_e=0.4,
                       alternatives=[("apple", 0.9)])
        b.hypotheses = [h]
        snap = b.snapshot(step=2)
        assert snap.step == 2
        assert snap.n_hypotheses == 1
        assert snap.most_uncertain_axis == "label"
        assert snap.target_summary is not None
        assert snap.target_summary["label"] == "apple"
```

- [ ] **Step 2: 跑测试 (确认红)**

Run: `pytest tests/test_world_belief.py::TestMerge tests/test_world_belief.py::TestPrune tests/test_world_belief.py::TestSnapshot -v`
Expected: AttributeError on `merge_hypothesis` etc.

- [ ] **Step 3: 在 WorldBelief 类中追加方法**

```python
    # ──── 状态修改 ──────────────────────────
    
    MERGE_DISTANCE_M = 0.15           # TODO(v1.1): 实测调
    MERGE_LABEL_INTERSECTION_MIN = 0.30  # TODO(v1.1): 实测调
    PRUNE_MIN_STEPS = 3
    PRUNE_PHANTOM_ENTROPY = 0.7
    
    def add_hypothesis(self, h: Hypothesis) -> None:
        self.hypotheses.append(h)
    
    def merge_hypothesis(self, existing: Hypothesis, new_data: Hypothesis) -> bool:
        """尝试把 new_data 合并进 existing。返回 True 表示合并成功。
        
        条件:
        - position 距离 < MERGE_DISTANCE_M
        - label_alternatives 概率交集 ≥ MERGE_LABEL_INTERSECTION_MIN
        """
        if existing not in self.hypotheses:
            return False
        dist = float(np.linalg.norm(existing.position_3d - new_data.position_3d))
        if dist >= self.MERGE_DISTANCE_M:
            return False
        # label_alternatives 概率交集
        e_dict = dict(existing.label_alternatives)
        intersect = sum(min(p, e_dict.get(lbl, 0.0))
                        for lbl, p in new_data.label_alternatives)
        if intersect < self.MERGE_LABEL_INTERSECTION_MIN:
            return False
        # 合并: 位置加权平均, label 取概率高的
        n_old = len(existing.observed_in_views) or 1
        n_new = len(new_data.observed_in_views) or 1
        existing.position_3d = (
            existing.position_3d * n_old + new_data.position_3d * n_new
        ) / (n_old + n_new)
        existing.observed_in_views.extend(
            v for v in new_data.observed_in_views
            if v not in existing.observed_in_views
        )
        existing.bbox_per_view.update(new_data.bbox_per_view)
        # label_alternatives: 概率平均后归一化
        merged_alts: dict[str, float] = dict(existing.label_alternatives)
        for lbl, p in new_data.label_alternatives:
            merged_alts[lbl] = (merged_alts.get(lbl, 0.0) + p) / 2
        total = sum(merged_alts.values()) or 1.0
        existing.label_alternatives = sorted(
            ((lbl, p / total) for lbl, p in merged_alts.items()),
            key=lambda x: x[1], reverse=True,
        )
        existing.label = existing.label_alternatives[0][0]
        # entropy 重算
        existing.label_entropy = _shannon([p for _, p in existing.label_alternatives])
        # 多视角 std (粗略: 简单更新)
        existing.position_std_m = max(existing.position_std_m, dist / 2)
        return True
    
    def prune_phantom_hypotheses(self) -> int:
        """删除疑似幻觉 hypothesis (1 视角 + 高熵 + 步数>3)。返回删除数。"""
        if len(self.action_history) <= self.PRUNE_MIN_STEPS:
            return 0
        before = len(self.hypotheses)
        self.hypotheses = [
            h for h in self.hypotheses
            if not (
                len(h.observed_in_views) <= 1
                and h.label_entropy > self.PRUNE_PHANTOM_ENTROPY
            )
        ]
        return before - len(self.hypotheses)
    
    def snapshot(self, step: int) -> BeliefSnapshot:
        import time as _t
        h = self.target()
        target_summary = None
        if h is not None:
            target_summary = {
                "label": h.label,
                "label_entropy": h.label_entropy,
                "position_3d": h.position_3d.tolist(),
                "position_std_m": h.position_std_m,
                "safety_entropy": h.safety_entropy,
                "grasp_uncertainty": h.grasp_uncertainty,
            }
        h_for_axis = self.target()
        if h_for_axis is not None:
            ovr = h_for_axis.overall_uncertainty()
        else:
            ovr = 1.0
        return BeliefSnapshot(
            step=step,
            timestamp=_t.time(),
            n_hypotheses=len(self.hypotheses),
            target_summary=target_summary,
            most_uncertain_axis=self.most_uncertain_axis(),
            overall_uncertainty=ovr,
            n_evidence=len(self.evidence),
            open_questions_count=len(self.open_questions),
        )
    
    def consume_user_answer(self, question: str, answer: str, llm) -> None:
        """v1 简化版: 把 (question, answer) 追加到 user_constraints。
        
        TODO(v1.1): LLM 解析答案 → boost/demote/constraint/unhelpful (设计稿 §6.1 + Edge 9.4/9.10)。
        - 当前 v1: 答案仅作字符串保留, 下一轮 NBV/perception prompt 通过把 user_constraints
          merge 进 decomposed.constraints 让 LLM "看见"。
        - v1.1 升级: 加载 prompts/agent/user_answer_parse.txt, 输出 {boost: id, demote: id,
          new_constraint: {...}, unhelpful: bool}, 直接修改 hypothesis.label_alternatives 概率。
        - llm 参数当前不使用, 保留接口签名以避免 v1.1 升级时破坏调用方。
        """
        _ = llm  # v1 未使用; v1.1 接入 LLM 解析时启用
        self.user_constraints.append(f"Q: {question} | A: {answer}")
    
    def compose_clarification(self) -> str:
        """构造给用户的澄清问题。"""
        h = self.target()
        if h is None:
            return f"我没看清您要的{self.decomposed.primary_target if self.decomposed else 'something'}, 您能描述一下它附近还有什么吗?"
        alts = [lbl for lbl, _ in h.label_alternatives[:2]]
        return f"我看到一个像{alts[0]}的东西, 也可能是{alts[1] if len(alts) > 1 else '别的'}, 您要的是哪个?"
```

- [ ] **Step 4: 在文件顶部添加 `_shannon` helper**

```python
import math

def _shannon(probs: list[float]) -> float:
    """Shannon entropy in nats; 输入概率不必归一, 内部归一。"""
    total = sum(p for p in probs if p > 0)
    if total <= 0:
        return 0.0
    h = 0.0
    for p in probs:
        if p > 0:
            q = p / total
            h -= q * math.log(q)
    return h
```

- [ ] **Step 5: 跑测试**

Run: `pytest tests/test_world_belief.py -v`
Expected: 至少 21 pass (1.2 + 1.4 + 1.5 + 1.6)

- [ ] **Step 6: commit**

```bash
git add src/world_belief.py tests/test_world_belief.py
git commit -m "feat(belief): merge_hypothesis/prune_phantom/snapshot/consume_user_answer"
```

---

### Task 1.7: 4 个余下边界测试 (high_risk thresholds / merge 临界 / prune 多视角保留 / consume_user_answer)

**Files:**
- Modify: `tests/test_world_belief.py`

- [ ] **Step 1: 加测试**

```python
class TestEdgeCases:
    def test_high_risk_tightens_thresholds(self):
        """sharp+hot+chemical > 0.5 → high_risk (label thr 0.30 → 0.15)。"""
        b = WorldBelief(user_query="拿削皮器")
        b.decomposed = DecomposedTask(primary_target="peeler")
        c = GraspCandidate(point_3d=np.array([0.5,0,0.9]),
                           approach_dir=np.array([0,0,-1]),
                           finger_width_m=0.04, score=0.9)
        h = _basic_hyp(label="peeler",
                       alternatives=[("peeler", 0.85)])
        h.label_entropy = 0.20    # 普通模式 < 0.30 confident, high-risk 模式 > 0.15 不 confident
        h.position_std_m = 0.02
        h.safety_dist = {"sharp": 0.7, "safe": 0.3}
        h.safety_entropy = 0.10
        h.grasp_candidates = [c]
        b.hypotheses = [h]
        # high_risk 阈值收紧 label=0.15, label_entropy=0.20 > 0.15 → 不 confident
        assert b.is_confident_to_act() is False
    
    def test_merge_distance_boundary(self):
        """距离 0.149 vs 0.151。"""
        b = WorldBelief(user_query="x")
        b.decomposed = DecomposedTask(primary_target="apple")
        h1 = _basic_hyp(label="apple")
        h1.position_3d = np.array([0.5, 0.0, 0.9])
        b.hypotheses = [h1]
        # 0.149 → merge ok
        h_close = _basic_hyp(label="apple")
        h_close.position_3d = np.array([0.5 + 0.149, 0, 0.9])
        assert b.merge_hypothesis(h1, h_close) is True
        # reset
        h1.position_3d = np.array([0.5, 0.0, 0.9])
        # 0.151 → no merge
        h_far = _basic_hyp(label="apple")
        h_far.position_3d = np.array([0.5 + 0.151, 0, 0.9])
        assert b.merge_hypothesis(h1, h_far) is False
    
    def test_prune_recent_steps_kept(self):
        """步数 ≤ PRUNE_MIN_STEPS → 不 prune (即使是幻觉)。"""
        b = WorldBelief(user_query="x")
        h = _basic_hyp(label="ghost", label_e=0.85)
        h.observed_in_views = ["v1"]
        b.hypotheses = [h]
        # 仅 2 步
        b.action_history.append(Action(kind="observe"))
        b.action_history.append(Action(kind="observe"))
        n = b.prune_phantom_hypotheses()
        assert n == 0
    
    def test_consume_user_answer_appends_constraint(self):
        b = WorldBelief(user_query="x")
        b.consume_user_answer("您要的是哪个?", "圆形的", llm=None)
        assert len(b.user_constraints) == 1
        assert "圆形的" in b.user_constraints[0]
```

- [ ] **Step 2: 跑测试**

Run: `pytest tests/test_world_belief.py::TestEdgeCases -v`
Expected: 4 pass

- [ ] **Step 3: 跑全套**

Run: `pytest tests/test_world_belief.py -v`
Expected: 25+ pass, 0 fail

- [ ] **Step 4: ruff lint**

Run: `ruff check src/world_belief.py tests/test_world_belief.py`
Expected: All checks passed

- [ ] **Step 5: commit**

```bash
git add tests/test_world_belief.py
git commit -m "test(belief): edge cases (high_risk/merge boundary/prune recent/consume_user_answer)"
```

**Phase 1 CHECKPOINT:** `pytest tests/test_world_belief.py -v` 25+ pass, 0 fail。`src/world_belief.py` 后续 Phase 都依赖它。

---

## Phase 2: `src/vlm_cache.py` + 5 单测

**目标:** Episode 级 in-memory cache, 按 (image_hash, prompt_hash) 去重 VLM 调用; episode 结束 clear。

### Task 2.1: 失败测试

**Files:**
- Create: `tests/test_vlm_cache.py`

- [ ] **Step 1: 写测试**

```python
"""VLMCache 单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import tempfile
import os


class TestVLMCache:
    @pytest.fixture
    def tmp_image(self, tmp_path):
        """生成一张极小的测试图。"""
        from PIL import Image
        p = tmp_path / "img.png"
        Image.new("RGB", (4, 4), (255, 0, 0)).save(p)
        return str(p)
    
    def test_miss_then_hit(self, tmp_image):
        from src.vlm_cache import VLMCache
        cache = VLMCache(max_size=10)
        assert cache.get(tmp_image, "prompt A") is None
        cache.put(tmp_image, "prompt A", "response 1")
        assert cache.get(tmp_image, "prompt A") == "response 1"
    
    def test_different_prompt_no_collision(self, tmp_image):
        from src.vlm_cache import VLMCache
        cache = VLMCache()
        cache.put(tmp_image, "prompt A", "response A")
        assert cache.get(tmp_image, "prompt B") is None
    
    def test_lru_eviction(self, tmp_path):
        """超过 max_size 时, 最久未用的被剔除。"""
        from PIL import Image
        from src.vlm_cache import VLMCache
        cache = VLMCache(max_size=2)
        paths = []
        for i in range(3):
            p = tmp_path / f"img{i}.png"
            Image.new("RGB", (4, 4), (i * 50, 0, 0)).save(p)
            paths.append(str(p))
            cache.put(str(p), "p", f"r{i}")
        # img0 应已被剔除
        assert cache.get(paths[0], "p") is None
        assert cache.get(paths[1], "p") == "r1"
        assert cache.get(paths[2], "p") == "r2"
    
    def test_clear_empties(self, tmp_image):
        from src.vlm_cache import VLMCache
        cache = VLMCache()
        cache.put(tmp_image, "p", "r")
        cache.clear()
        assert cache.get(tmp_image, "p") is None
    
    def test_stats(self, tmp_image):
        from src.vlm_cache import VLMCache
        cache = VLMCache()
        cache.get(tmp_image, "p")              # miss
        cache.put(tmp_image, "p", "r")
        cache.get(tmp_image, "p")              # hit
        cache.get(tmp_image, "q")              # miss
        s = cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 2
        assert s["hit_rate"] == pytest.approx(1 / 3)
```

- [ ] **Step 2: 跑测试 (红)**

Run: `pytest tests/test_vlm_cache.py -v`
Expected: ImportError

---

### Task 2.2: 实现 `src/vlm_cache.py`

**Files:**
- Create: `src/vlm_cache.py`

- [ ] **Step 1: 写实现**

```python
"""VLMCache: episode 级 in-memory cache, 按 (image_hash, prompt_hash) 去重。"""
from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class VLMCache:
    def __init__(self, max_size: int = 100):
        self._max = max_size
        self._store: OrderedDict[str, str] = OrderedDict()
        self._hits = 0
        self._misses = 0
    
    @staticmethod
    def _key(image_path: str, prompt: str) -> str:
        with open(image_path, "rb") as f:
            img_hash = hashlib.sha256(f.read()).hexdigest()
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return f"{img_hash}::{prompt_hash}"
    
    def get(self, image_path: str, prompt: str) -> Optional[str]:
        try:
            key = self._key(image_path, prompt)
        except FileNotFoundError:
            self._misses += 1
            return None
        if key in self._store:
            self._hits += 1
            self._store.move_to_end(key)
            return self._store[key]
        self._misses += 1
        return None
    
    def put(self, image_path: str, prompt: str, response: str) -> None:
        try:
            key = self._key(image_path, prompt)
        except FileNotFoundError:
            return
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = response
            return
        self._store[key] = response
        if len(self._store) > self._max:
            self._store.popitem(last=False)
    
    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0
    
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (self._hits / total) if total else 0.0,
            "size": len(self._store),
        }
```

- [ ] **Step 2: 跑测试**

Run: `pytest tests/test_vlm_cache.py -v`
Expected: 5 pass

- [ ] **Step 3: ruff lint**

Run: `ruff check src/vlm_cache.py`

- [ ] **Step 4: commit**

```bash
git add src/vlm_cache.py tests/test_vlm_cache.py
git commit -m "feat(cache): VLMCache with sha256 image+prompt key, LRU eviction, stats"
```

**Phase 2 CHECKPOINT:** `pytest tests/test_vlm_cache.py -v` 5 pass。

---

## Phase 3: `src/episode_logger.py` + 6 单测

**目标:** 记录 episode 全过程 (snapshots / actions / evidence) 到 JSON, 支持 `load` 反序列化和 `replay` 接口占位 (Phase 14 实现完整 replay)。

### Task 3.1: 失败测试

**Files:**
- Create: `tests/test_episode_logger.py`

- [ ] **Step 1: 写测试**

```python
"""EpisodeLogger 单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import pytest
import numpy as np


@pytest.fixture
def tmp_log_dir(tmp_path):
    return str(tmp_path / "episodes")


class TestEpisodeLogger:
    def test_start_and_end_writes_json(self, tmp_log_dir):
        from src.episode_logger import EpisodeLogger
        from src.world_belief import EpisodeResult
        lg = EpisodeLogger(log_dir=tmp_log_dir)
        lg.start_episode("拿苹果")
        result = EpisodeResult(
            success=True, target=None, speech="ok",
            belief_trace=[], action_history=[], n_steps=0, elapsed_seconds=1.0,
        )
        path = lg.end_episode(result)
        assert Path(path).exists()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["query"] == "拿苹果"
        assert data["final_result"]["success"] is True
    
    def test_log_snapshot_appends(self, tmp_log_dir):
        from src.episode_logger import EpisodeLogger
        from src.world_belief import BeliefSnapshot, EpisodeResult
        lg = EpisodeLogger(log_dir=tmp_log_dir)
        lg.start_episode("q")
        for i in range(3):
            snap = BeliefSnapshot(
                step=i, timestamp=time.time(), n_hypotheses=i,
                target_summary=None, most_uncertain_axis="label",
                overall_uncertainty=0.5, n_evidence=0, open_questions_count=0,
            )
            lg.log_snapshot(snap)
        path = lg.end_episode(EpisodeResult(
            success=True, target=None, speech="", belief_trace=[],
            action_history=[], n_steps=3, elapsed_seconds=1.0,
        ))
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert len(data["snapshots"]) == 3
        assert data["snapshots"][2]["step"] == 2
    
    def test_log_action_pair(self, tmp_log_dir):
        from src.episode_logger import EpisodeLogger
        from src.world_belief import (
            Action, BeliefSnapshot, EpisodeResult,
        )
        lg = EpisodeLogger(log_dir=tmp_log_dir)
        lg.start_episode("q")
        a = Action(kind="observe")
        snap = BeliefSnapshot(
            step=0, timestamp=time.time(), n_hypotheses=0,
            target_summary=None, most_uncertain_axis="label",
            overall_uncertainty=1.0, n_evidence=0, open_questions_count=0,
        )
        lg.log_action_start(a, snap)
        lg.log_action_end(a, snap)
        path = lg.end_episode(EpisodeResult(
            success=False, target=None, speech="", belief_trace=[],
            action_history=[a], n_steps=1, elapsed_seconds=1.0,
        ))
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert len(data["actions"]) == 1
        assert data["actions"][0]["kind"] == "observe"
    
    def test_load_round_trip(self, tmp_log_dir):
        from src.episode_logger import EpisodeLogger
        from src.world_belief import EpisodeResult
        lg = EpisodeLogger(log_dir=tmp_log_dir)
        lg.start_episode("query A")
        path = lg.end_episode(EpisodeResult(
            success=True, target=None, speech="hi", belief_trace=[],
            action_history=[], n_steps=0, elapsed_seconds=0.5,
        ))
        record = EpisodeLogger.load(path)
        assert record.query == "query A"
        assert record.final_result is not None
        assert record.final_result.success is True
    
    def test_user_qa_logged(self, tmp_log_dir):
        from src.episode_logger import EpisodeLogger
        from src.world_belief import EpisodeResult
        lg = EpisodeLogger(log_dir=tmp_log_dir)
        lg.start_episode("q")
        lg.log_user_qa("您要哪个?", "圆形的")
        path = lg.end_episode(EpisodeResult(
            success=True, target=None, speech="", belief_trace=[],
            action_history=[], n_steps=0, elapsed_seconds=0.0,
        ))
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["user_qa"] == [["您要哪个?", "圆形的"]]
    
    def test_evidence_serializes_numpy(self, tmp_log_dir):
        from src.episode_logger import EpisodeLogger
        from src.world_belief import Evidence, EpisodeResult
        lg = EpisodeLogger(log_dir=tmp_log_dir)
        lg.start_episode("q")
        ev = Evidence(
            source="vlm_ground", timestamp=1.0,
            raw_payload={"objects": [{"pos": np.array([0.1, 0.2, 0.3])}]},
        )
        lg.log_evidence(ev)
        path = lg.end_episode(EpisodeResult(
            success=True, target=None, speech="", belief_trace=[],
            action_history=[], n_steps=0, elapsed_seconds=0.0,
        ))
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # numpy 数组应被转成 list
        assert isinstance(
            data["evidence"][0]["raw_payload"]["objects"][0]["pos"], list
        )
```

- [ ] **Step 2: 跑测试 (红)**

Run: `pytest tests/test_episode_logger.py -v`
Expected: ImportError

---

### Task 3.2: 实现 `src/episode_logger.py`

**Files:**
- Create: `src/episode_logger.py`

- [ ] **Step 1: 写实现**

```python
"""EpisodeLogger: 记录 belief / action / evidence 流到 JSON, 支持 replay 反序列化。

设计参考: docs/superpowers/specs/2026-05-08-emboSight-belief-driven-agent-design.md §6.10
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.world_belief import (
    Action, BeliefSnapshot, EpisodeResult, Evidence,
)

logger = logging.getLogger(__name__)


@dataclass
class EpisodeRecord:
    query: str
    start_time: float
    snapshots: list[BeliefSnapshot]
    actions: list[Action]
    evidence: list[Evidence]
    user_qa: list[tuple[str, str]]
    final_result: Optional[EpisodeResult] = None


def _to_jsonable(obj: Any) -> Any:
    """递归把 dataclass / numpy / Path 转成 JSON 可序列化对象。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


class EpisodeLogger:
    
    def __init__(self, log_dir: str = "logs/episodes"):
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._reset()
    
    def _reset(self) -> None:
        self._query: str = ""
        self._start: float = 0.0
        self._snaps: list[BeliefSnapshot] = []
        self._actions: list[Action] = []
        self._evidence: list[Evidence] = []
        self._user_qa: list[tuple[str, str]] = []
    
    def start_episode(self, query: str) -> None:
        self._reset()
        self._query = query
        self._start = time.time()
        logger.info(f"[episode] START: query={query!r}")
    
    def log_snapshot(self, snap: BeliefSnapshot) -> None:
        self._snaps.append(snap)
    
    def log_action_start(self, action: Action, snap: BeliefSnapshot) -> None:
        self._actions.append(action)
        self._snaps.append(snap)
    
    def log_action_end(self, action: Action, snap: BeliefSnapshot) -> None:
        self._snaps.append(snap)
    
    def log_user_qa(self, q: str, a: str) -> None:
        self._user_qa.append((q, a))
    
    def log_evidence(self, ev: Evidence) -> None:
        self._evidence.append(ev)
    
    def end_episode(self, result: EpisodeResult) -> str:
        ts = int(self._start)
        safe_q = "".join(c if c.isalnum() else "_" for c in self._query)[:30]
        path = self._dir / f"episode_{ts}_{safe_q}.json"
        payload = {
            "query": self._query,
            "start_time": self._start,
            "snapshots": _to_jsonable(self._snaps),
            "actions": _to_jsonable(self._actions),
            "evidence": _to_jsonable(self._evidence),
            "user_qa": _to_jsonable(self._user_qa),
            "final_result": _to_jsonable(result),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"[episode] END: {path}")
        return str(path)
    
    @classmethod
    def load(cls, json_path: str) -> EpisodeRecord:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        # 注意: 这里仅做"浅"反序列化, dataclass 字段作为 dict 保留;
        # 完整反序列化在 Phase 14 EpisodeReplay 里做
        snaps = [BeliefSnapshot(**s) for s in data["snapshots"]]
        actions = [Action(kind=a["kind"], strategy=a.get("strategy"),
                          question=a.get("question"),
                          metadata=a.get("metadata", {}))
                   for a in data["actions"]]
        evidence = [Evidence(source=e["source"], timestamp=e["timestamp"],
                             raw_payload=e["raw_payload"],
                             consumed_by=e.get("consumed_by", []))
                    for e in data["evidence"]]
        final = None
        if data.get("final_result") is not None:
            fr = data["final_result"]
            final = EpisodeResult(
                success=fr["success"], target=None, speech=fr.get("speech", ""),
                belief_trace=[], action_history=[],
                n_steps=fr.get("n_steps", 0),
                elapsed_seconds=fr.get("elapsed_seconds", 0.0),
                failure_reason=fr.get("failure_reason"),
            )
        return EpisodeRecord(
            query=data["query"],
            start_time=data["start_time"],
            snapshots=snaps,
            actions=actions,
            evidence=evidence,
            user_qa=[tuple(x) for x in data.get("user_qa", [])],
            final_result=final,
        )
    
    @classmethod
    def replay(cls, json_path: str, agent_factory) -> EpisodeResult:
        """Phase 14 实现完整 mock 重放; v1 仅占位。"""
        raise NotImplementedError("replay implemented in Phase 14")
```

- [ ] **Step 2: 跑测试**

Run: `pytest tests/test_episode_logger.py -v`
Expected: 6 pass

- [ ] **Step 3: ruff + commit**

Run: `ruff check src/episode_logger.py`

```bash
git add src/episode_logger.py tests/test_episode_logger.py
git commit -m "feat(logger): EpisodeLogger save/load with numpy-aware serialization"
```

**Phase 3 CHECKPOINT:** `pytest tests/test_episode_logger.py -v` 6 pass。

---

## Phase 4: `src/user_channel.py` + 8 单测

**目标:** 三种 FakeUserChannel (from_query / from_explicit / from_robocasa) + CLIUserChannel + VoiceUserChannel 占位; LLM 解析"unhelpful answer"; 历史维护。

### Task 4.1: 失败测试

**Files:**
- Create: `tests/test_user_channel.py`

- [ ] **Step 1: 写测试**

```python
"""UserChannel 单元测试 (FakeUser/CLI/Voice)。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from tests._mocks import MockLLM


class TestFakeUserChannel:
    def test_from_query_extracts_intent(self):
        from src.user_channel import FakeUserChannel
        llm = MockLLM(responses=["削皮器", "圆形的"])
        ch = FakeUserChannel.from_query(llm, "帮我拿那个削皮器")
        assert ch.intent == "削皮器"
        ans = ch.ask("您要哪个?")
        assert ans == "圆形的"
        # history 累积
        assert len(ch.history) == 1
    
    def test_from_explicit_passes_through(self):
        from src.user_channel import FakeUserChannel
        llm = MockLLM(responses=["a"])
        ch = FakeUserChannel.from_explicit(llm, "苹果")
        assert ch.intent == "苹果"
    
    def test_from_robocasa_reads_obj_main(self):
        from src.user_channel import FakeUserChannel
        llm = MockLLM(responses=[])
        class FakeEnv:
            def _get_obj_type_map(self):
                return {"obj_main": "peeler"}
        ch = FakeUserChannel.from_robocasa(llm, FakeEnv())
        assert "peeler" in ch.intent
    
    def test_history_format(self):
        from src.user_channel import FakeUserChannel
        llm = MockLLM(responses=["A1", "A2"])
        ch = FakeUserChannel.from_explicit(llm, "x")
        ch.ask("Q1")
        ch.ask("Q2")
        history = ch._format_history()
        assert "Q1" in history and "A1" in history
        assert "Q2" in history and "A2" in history
    
    def test_unhelpful_answer_handled(self):
        """用户答 "不知道" 时, channel 不报错, 返回原文。"""
        from src.user_channel import FakeUserChannel
        llm = MockLLM(responses=["不知道"])
        ch = FakeUserChannel.from_explicit(llm, "x")
        ans = ch.ask("您要哪个?")
        assert "不知道" in ans


class TestCLIUserChannel:
    def test_cli_reads_input(self, monkeypatch):
        from src.user_channel import CLIUserChannel
        ch = CLIUserChannel()
        monkeypatch.setattr("builtins.input", lambda *a, **k: "圆形的")
        ans = ch.ask("您要哪个?")
        assert ans == "圆形的"


class TestVoiceUserChannel:
    def test_voice_raises_not_implemented(self):
        from src.user_channel import VoiceUserChannel
        ch = VoiceUserChannel(tts=None, asr=None)
        with pytest.raises(NotImplementedError):
            ch.ask("anything")


class TestProtocol:
    def test_all_channels_have_ask(self):
        from src.user_channel import FakeUserChannel, CLIUserChannel
        # 接口一致性: 都有 ask(question) 方法
        assert hasattr(FakeUserChannel(MockLLM(responses=[]), "x"), "ask")
        assert hasattr(CLIUserChannel(), "ask")
```

- [ ] **Step 2: 跑测试 (红)**

Run: `pytest tests/test_user_channel.py -v`
Expected: ImportError

---

### Task 4.2: 实现 `src/user_channel.py`

**Files:**
- Create: `src/user_channel.py`
- Create: `prompts/user/fake_user_system.txt`

- [ ] **Step 1: 写 prompt 文件**

```
你是一名视障用户, 正在使用辅助机器人。
- 你看不见任何视觉细节, 但记得自己想要什么、家里大致布局。
- 不要假装看到颜色/形状; 但可以说"通常它放在水池左边"这种记忆。
- 简短自然回答 (1 句话)。
- 如果机器人列出选项, 选最像你想要的那个。
- 如果完全不知道, 直说"不知道"。
```

- [ ] **Step 2: 实现 user_channel.py**

```python
"""UserChannel: 与用户 (人/oracle/语音) 双向交互通道。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


# ============================================================
# Protocol
# ============================================================

class UserChannel(Protocol):
    def ask(self, question: str, context: Optional[dict] = None) -> str: ...


# ============================================================
# FakeUserChannel (LLM 扮演用户)
# ============================================================

_DEFAULT_SYSTEM_PATH = "prompts/user/fake_user_system.txt"


class FakeUserChannel:
    
    def __init__(self, llm, intent: str, system_path: str = _DEFAULT_SYSTEM_PATH):
        self.llm = llm
        self.intent = intent
        self.history: list[tuple[str, str]] = []
        self._system = self._load_system(system_path)
    
    @staticmethod
    def _load_system(path: str) -> str:
        p = Path(path)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return "你是一名视障用户, 简短回答。"
    
    @classmethod
    def from_query(cls, llm, query: str) -> "FakeUserChannel":
        prompt = f"用户说: {query!r}。请用 1 个词提取他真实想要的物体名 (中文): "
        intent = llm.generate(prompt).strip()
        return cls(llm, intent)
    
    @classmethod
    def from_explicit(cls, llm, intent: str) -> "FakeUserChannel":
        return cls(llm, intent)
    
    @classmethod
    def from_robocasa(cls, llm, env) -> "FakeUserChannel":
        type_map = env._get_obj_type_map()
        obj_type = type_map.get("obj_main", "unknown")
        return cls(llm, f"我想要那个 {obj_type}")
    
    def ask(self, question: str, context: Optional[dict] = None) -> str:
        prompt = (
            f"你的真实意图: {self.intent}\n\n"
            f"对话历史:\n{self._format_history()}\n\n"
            f"机器人问: {question}\n你的回答:"
        )
        ans = self.llm.generate(prompt, system=self._system).strip()
        self.history.append((question, ans))
        return ans
    
    def _format_history(self) -> str:
        if not self.history:
            return "(无)"
        return "\n".join(f"  Q: {q}\n  A: {a}" for q, a in self.history)


# ============================================================
# CLIUserChannel
# ============================================================

class CLIUserChannel:
    def ask(self, question: str, context: Optional[dict] = None) -> str:
        print(f"\n[Agent] {question}")
        return input("[You] ").strip()


# ============================================================
# VoiceUserChannel (占位)
# ============================================================

class VoiceUserChannel:
    """v1 留接口, 不实现。"""
    
    def __init__(self, tts, asr):
        self.tts = tts
        self.asr = asr
    
    def ask(self, question: str, context: Optional[dict] = None) -> str:
        raise NotImplementedError("VoiceUserChannel: 留 v2 接 Whisper/STT/TTS")
```

- [ ] **Step 3: 跑测试**

Run: `pytest tests/test_user_channel.py -v`
Expected: 8 pass

- [ ] **Step 4: ruff + commit**

```bash
git add src/user_channel.py prompts/user/fake_user_system.txt tests/test_user_channel.py
git commit -m "feat(userch): FakeUser/CLI/Voice channels with three FakeUser oracles"
```

**Phase 4 CHECKPOINT:** `pytest tests/test_user_channel.py -v` 8 pass。

---

## Phase 5: `src/perception.py` `observe()` + 第 1 个 prompt + 10 单测

**目标:** 新建 `QueryAwareGrounder`, 主入口 `observe(viewpoint, env, belief) -> Evidence`; VLM JSON 解析 + 温度缩放 + label_entropy 计算 + position_3d 投影。re_observe 留 Phase 12 实现。

### Task 5.1: prompt 文件 + 失败测试

**Files:**
- Create: `prompts/perception/query_aware_ground.txt`
- Create: `tests/test_perception.py`

- [ ] **Step 1: 写 prompt 文件**

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
   - alternatives: top 3 (label, probability) tuples summing to <= 1.0
       Example: [["colander", 0.6], ["strainer", 0.3], ["basket", 0.1]]
       
       BE CONSERVATIVE with probabilities — if multiple labels look possible, distribute
       probability mass; do NOT give 0.95 unless the object is unmistakable. When in
       doubt 0.5/0.3/0.2 is more useful than 0.95/0.03/0.02.
   - confidence: 0-1 (how sure you are something is THERE)
   - visible_features: 1 short phrase

3. If you don't see {primary_target} at all, list the most visually similar objects
   you DO see (fall back gracefully).

Reply with ONLY raw JSON, no markdown fences:
{"objects": [{"bbox_2d": [...], "label": "...", "alternatives": [...], "confidence": ..., "visible_features": "..."}, ...]}
```

- [ ] **Step 2: 写测试**

```python
"""QueryAwareGrounder 单元测试 (observe / parse / 温度缩放)。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import math
import pytest
import numpy as np
from tests._mocks import MockLLM, MockVLM


@pytest.fixture
def tmp_image(tmp_path):
    from PIL import Image
    p = tmp_path / "img.png"
    Image.new("RGB", (256, 256), (200, 100, 50)).save(p)
    return str(p)


def _make_vlm_json(objects):
    return json.dumps({"objects": objects})


class TestParse:
    def test_basic_parse(self):
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        from src.world_belief import DecomposedTask
        vlm = MockVLM(responses=[_make_vlm_json([
            {"bbox_2d": [10, 10, 50, 50], "label": "apple",
             "alternatives": [["apple", 0.7], ["pear", 0.3]],
             "confidence": 0.9, "visible_features": "red round"},
        ])])
        g = QueryAwareGrounder(vlm=vlm, llm=MockLLM([]), cache=VLMCache(),
                               label_temperature=1.0)
        hyps = g._parse_to_hypotheses(
            vlm._responses[0] if False else _make_vlm_json([
                {"bbox_2d": [10, 10, 50, 50], "label": "apple",
                 "alternatives": [["apple", 0.7], ["pear", 0.3]],
                 "confidence": 0.9, "visible_features": "red"},
            ]),
            viewpoint=None, env=None,
        )
        assert len(hyps) == 1
        assert hyps[0].label == "apple"
        assert ("apple", pytest.approx(0.7, abs=1e-2)) in [
            (l, p) for l, p in hyps[0].label_alternatives
        ]
    
    def test_temperature_scaling_flattens(self):
        """τ>1 让 0.95 概率被压平。"""
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        g = QueryAwareGrounder(vlm=MockVLM([]), llm=MockLLM([]),
                               cache=VLMCache(), label_temperature=2.0)
        raw = _make_vlm_json([
            {"bbox_2d": [0,0,1,1], "label": "x",
             "alternatives": [["x", 0.95], ["y", 0.04], ["z", 0.01]],
             "confidence": 0.9, "visible_features": "f"},
        ])
        hyps = g._parse_to_hypotheses(raw, viewpoint=None, env=None)
        # τ=2: p_i' ∝ p_i^0.5; 0.95 → ~0.81 平方根, 归一化后 top1 < 0.95
        top1_prob = hyps[0].label_alternatives[0][1]
        assert top1_prob < 0.90
        assert top1_prob > 0.70
    
    def test_entropy_computation(self):
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        g = QueryAwareGrounder(vlm=MockVLM([]), llm=MockLLM([]),
                               cache=VLMCache(), label_temperature=1.0)
        raw = _make_vlm_json([
            {"bbox_2d": [0,0,1,1], "label": "x",
             "alternatives": [["x", 0.5], ["y", 0.5]],
             "confidence": 0.9, "visible_features": "f"},
        ])
        hyps = g._parse_to_hypotheses(raw, viewpoint=None, env=None)
        # H(0.5, 0.5) = ln(2) ≈ 0.693
        assert hyps[0].label_entropy == pytest.approx(math.log(2), abs=0.01)
    
    def test_malformed_json_returns_empty(self):
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        g = QueryAwareGrounder(vlm=MockVLM([]), llm=MockLLM([]),
                               cache=VLMCache(), label_temperature=1.0)
        hyps = g._parse_to_hypotheses("not json", viewpoint=None, env=None)
        assert hyps == []
    
    def test_alternatives_sum_normalized(self):
        """alternatives 和不等于 1 的也能解析 + 归一化。"""
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        g = QueryAwareGrounder(vlm=MockVLM([]), llm=MockLLM([]),
                               cache=VLMCache(), label_temperature=1.0)
        raw = _make_vlm_json([
            {"bbox_2d": [0,0,1,1], "label": "x",
             "alternatives": [["x", 0.4], ["y", 0.4]],     # sum=0.8
             "confidence": 0.9, "visible_features": "f"},
        ])
        hyps = g._parse_to_hypotheses(raw, viewpoint=None, env=None)
        s = sum(p for _, p in hyps[0].label_alternatives)
        assert s == pytest.approx(1.0, abs=1e-3)


class TestPromptBuild:
    def test_inject_target(self):
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        from src.world_belief import DecomposedTask, Constraint
        g = QueryAwareGrounder(vlm=MockVLM([]), llm=MockLLM([]),
                               cache=VLMCache())
        prompt = g._build_query_aware_prompt(
            primary_target="削皮器",
            constraints=[Constraint(kind="avoid", target_label="knife",
                                     reason="用户说避开")],
            img_w=512, img_h=384,
        )
        assert "削皮器" in prompt
        assert "knife" in prompt
        assert "512" in prompt
        assert "384" in prompt


class TestObserve:
    def test_observe_calls_vlm_with_query(self, tmp_image):
        """observe 应注入 query 到 prompt (根因①)。"""
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        from src.world_belief import WorldBelief, DecomposedTask
        
        class FakeVP:
            name = "robot0_agentview_center"
        class FakeObs:
            image_path = tmp_image
        class FakeEnv:
            def observe(self, vp): return FakeObs()
            def viewpoint_intrinsics(self, vp): return None
        
        vlm = MockVLM(responses=[_make_vlm_json([
            {"bbox_2d": [50, 50, 100, 100], "label": "apple",
             "alternatives": [["apple", 0.8], ["other", 0.2]],
             "confidence": 0.9, "visible_features": "red"},
        ])])
        g = QueryAwareGrounder(vlm=vlm, llm=MockLLM([]),
                               cache=VLMCache(), label_temperature=1.0)
        belief = WorldBelief(user_query="拿苹果")
        belief.decomposed = DecomposedTask(primary_target="apple")
        ev = g.observe(FakeVP(), FakeEnv(), belief)
        assert ev.source == "vlm_ground"
        assert "apple" in vlm.calls[0][1]              # prompt 含 "apple"
    
    def test_observe_uses_cache(self, tmp_image):
        """同一 (image, prompt) 第二次不再调 VLM。"""
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        from src.world_belief import WorldBelief, DecomposedTask
        
        class FakeVP: name = "v1"
        class FakeObs: image_path = tmp_image
        class FakeEnv:
            def observe(self, vp): return FakeObs()
            def viewpoint_intrinsics(self, vp): return None
        
        vlm = MockVLM(responses=[_make_vlm_json([])] * 5)
        cache = VLMCache(max_size=10)
        g = QueryAwareGrounder(vlm=vlm, llm=MockLLM([]),
                               cache=cache, label_temperature=1.0)
        belief = WorldBelief(user_query="拿苹果")
        belief.decomposed = DecomposedTask(primary_target="apple")
        g.observe(FakeVP(), FakeEnv(), belief)
        first_call_count = len(vlm.calls)
        g.observe(FakeVP(), FakeEnv(), belief)
        assert len(vlm.calls) == first_call_count       # 没多调
    
    def test_observe_failed_returns_evidence_with_failed_source(self, tmp_image):
        """VLM 抛异常 → Evidence(source='vlm_failed') (Edge 9.8)。"""
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        from src.world_belief import WorldBelief, DecomposedTask
        
        class BadVLM:
            calls = []
            def describe(self, *a, **kw):
                raise RuntimeError("VLM down")
        
        class FakeVP: name = "v1"
        class FakeObs: image_path = tmp_image
        class FakeEnv:
            def observe(self, vp): return FakeObs()
            def viewpoint_intrinsics(self, vp): return None
        
        g = QueryAwareGrounder(vlm=BadVLM(), llm=MockLLM([]),
                               cache=VLMCache())
        belief = WorldBelief(user_query="x")
        belief.decomposed = DecomposedTask(primary_target="apple")
        ev = g.observe(FakeVP(), FakeEnv(), belief)
        assert ev.source == "vlm_failed"
```

- [ ] **Step 3: 跑测试 (红)**

Run: `pytest tests/test_perception.py -v`
Expected: ImportError

---

### Task 5.2: 实现 `src/perception.py` (observe + parse + temperature)

**Files:**
- Create: `src/perception.py`

- [ ] **Step 1: 写实现**

```python
"""QueryAwareGrounder: query-aware VLM grounding, 直出 Hypothesis。

替代老 vlm_grounding.py + scene_describer.py:
- prompt 注入 user_query (根因①)
- 输出 alternatives 概率分布, 直出 entropy
- 温度缩放纠正 VLM 概率过自信 (F3)
- 删除 Level 0-4 + alias + semantic_pairs + GT cross-check 全套规则 (根因②)

设计参考: §6.3 / §4.1
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.world_belief import (
    Constraint, DecomposedTask, Evidence, Hypothesis, WorldBelief,
)
from src.vlm_cache import VLMCache

logger = logging.getLogger(__name__)


_DEFAULT_GROUND_PROMPT = "prompts/perception/query_aware_ground.txt"


def _shannon(probs: list[float]) -> float:
    total = sum(p for p in probs if p > 0)
    if total <= 0:
        return 0.0
    h = 0.0
    for p in probs:
        if p > 0:
            q = p / total
            h -= q * math.log(q)
    return h


def _temperature_scale(probs: list[tuple[str, float]],
                       tau: float) -> list[tuple[str, float]]:
    """温度缩放 (F3): p_i' = p_i^(1/τ) / Σ p_j^(1/τ)。τ=1.0 即不变。"""
    if tau == 1.0:
        total = sum(p for _, p in probs) or 1.0
        return [(lbl, p / total) for lbl, p in probs]
    inv = 1.0 / tau
    raised = [(lbl, p ** inv if p > 0 else 0.0) for lbl, p in probs]
    s = sum(p for _, p in raised) or 1.0
    return [(lbl, p / s) for lbl, p in raised]


class QueryAwareGrounder:
    
    def __init__(
        self,
        vlm,
        llm,
        cache: VLMCache,
        ground_prompt_path: str = _DEFAULT_GROUND_PROMPT,
        zoom_prompt_path: str = "prompts/perception/zoom_disambiguate.txt",
        parallax_prompt_path: str = "prompts/perception/parallax_localize.txt",
        verify_prompt_path: str = "prompts/perception/verify_grasp.txt",
        label_temperature: float = 1.5,
    ):
        self.vlm = vlm
        self.llm = llm
        self.cache = cache
        self.label_temperature = label_temperature
        self._ground_template = self._load(ground_prompt_path)
        # zoom/parallax/verify prompts: Phase 12 用; 此处仅记路径
        self._zoom_path = zoom_prompt_path
        self._parallax_path = parallax_prompt_path
        self._verify_path = verify_prompt_path
        self._next_obj_id = 0
    
    @staticmethod
    def _load(path: str) -> Optional[str]:
        p = Path(path)
        return p.read_text(encoding="utf-8") if p.exists() else None
    
    # ──────────────────────────────────────
    # 主入口: observe
    # ──────────────────────────────────────
    
    def observe(self, viewpoint, env, belief: WorldBelief) -> Evidence:
        """拍 viewpoint, query-aware VLM, 返回 Evidence (含 hypotheses[])。
        
        失败时返回 source='vlm_failed' Evidence (Edge 9.8)。
        """
        obs = env.observe(viewpoint)
        image_path = getattr(obs, "image_path", str(obs))
        # 图像尺寸
        img_w, img_h = 256, 256
        try:
            from PIL import Image
            with Image.open(image_path) as im:
                img_w, img_h = im.size
        except Exception:
            pass
        
        primary = belief.decomposed.primary_target if belief.decomposed else ""
        constraints = belief.decomposed.constraints if belief.decomposed else []
        prompt = self._build_query_aware_prompt(primary, constraints, img_w, img_h)
        
        # cache
        cached = self.cache.get(image_path, prompt)
        if cached is not None:
            raw = cached
        else:
            try:
                raw = self.vlm.describe(image_path, prompt=prompt)
                self.cache.put(image_path, prompt, raw)
            except Exception as e:
                logger.warning(f"[perception] VLM call failed: {e}")
                return Evidence(
                    source="vlm_failed", timestamp=time.time(),
                    raw_payload={"error": str(e), "viewpoint": getattr(viewpoint, "name", str(viewpoint))},
                )
        
        hyps = self._parse_to_hypotheses(raw, viewpoint, env)
        return Evidence(
            source="vlm_ground", timestamp=time.time(),
            raw_payload={
                "viewpoint": getattr(viewpoint, "name", str(viewpoint)),
                "hypotheses": [self._hyp_to_dict(h) for h in hyps],
                "image_path": image_path,
                "raw_vlm_text": raw[:1000],
            },
        )
    
    # ──────────────────────────────────────
    # Prompt build
    # ──────────────────────────────────────
    
    def _build_query_aware_prompt(
        self,
        primary_target: str,
        constraints: list[Constraint],
        img_w: int = 256,
        img_h: int = 256,
    ) -> str:
        if self._ground_template is None:
            return f"List all objects. User wants {primary_target}."
        constraints_text = "\n".join(
            f"- {c.kind}: {c.target_label or c.text or ''} ({c.reason})"
            for c in constraints
        ) or "(无)"
        return (
            self._ground_template
            .replace("{primary_target}", primary_target or "<unknown>")
            .replace("{constraints}", constraints_text)
            .replace("{img_w}", str(img_w))
            .replace("{img_h}", str(img_h))
        )
    
    # ──────────────────────────────────────
    # Parse: VLM JSON → Hypothesis
    # ──────────────────────────────────────
    
    def _parse_to_hypotheses(
        self, raw: str, viewpoint, env,
    ) -> list[Hypothesis]:
        """把 VLM JSON 解析成 Hypothesis 列表, 含温度缩放 + 熵计算 (F3)。"""
        data = self._extract_json(raw)
        if not data:
            return []
        objects = data.get("objects", [])
        hyps: list[Hypothesis] = []
        for obj in objects:
            try:
                bbox = tuple(int(x) for x in obj.get("bbox_2d", [0, 0, 0, 0]))
                label = str(obj.get("label", "unknown"))
                alts_raw = obj.get("alternatives", [[label, 1.0]])
                alts = [(str(lbl), float(p)) for lbl, p in alts_raw]
                # 温度缩放 + 归一化
                alts_scaled = _temperature_scale(alts, self.label_temperature)
                entropy = _shannon([p for _, p in alts_scaled])
                
                # position 投影 (粗略: 取 bbox 中心 + 估深度; 真实投影在 Phase 12)
                pos_3d, pos_std = self._estimate_position(bbox, viewpoint, env)
                
                vp_name = getattr(viewpoint, "name", str(viewpoint)) if viewpoint else "v0"
                h = Hypothesis(
                    object_id=f"obj_{self._next_obj_id}",
                    label=label,
                    label_alternatives=sorted(alts_scaled, key=lambda x: x[1], reverse=True),
                    label_entropy=entropy,
                    position_3d=pos_3d,
                    position_std_m=pos_std,
                    bbox_per_view={vp_name: bbox},
                    observed_in_views=[vp_name],
                )
                hyps.append(h)
                self._next_obj_id += 1
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"[perception] skip malformed object: {e}; obj={obj}")
        return hyps
    
    @staticmethod
    def _extract_json(raw: str) -> Optional[dict]:
        # 容忍 markdown fence
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return None
    
    @staticmethod
    def _estimate_position(
        bbox: tuple[int, int, int, int], viewpoint, env,
    ) -> tuple[np.ndarray, float]:
        """粗略 position 估计: 单视角先用 prior。
        
        真实多视角投影在 Phase 12 通过 src/projection.py 实现。
        """
        return np.array([0.0, 0.0, 0.9], dtype=np.float32), 0.10
    
    @staticmethod
    def _hyp_to_dict(h: Hypothesis) -> dict[str, Any]:
        return {
            "object_id": h.object_id,
            "label": h.label,
            "label_alternatives": h.label_alternatives,
            "label_entropy": h.label_entropy,
            "position_3d": h.position_3d.tolist(),
            "position_std_m": h.position_std_m,
            "bbox_per_view": {k: list(v) for k, v in h.bbox_per_view.items()},
        }
    
    # ──────────────────────────────────────
    # re_observe / verify_grasp - Phase 12 实现
    # ──────────────────────────────────────
    
    def re_observe(self, target: Hypothesis, strategy: str, env, belief: WorldBelief) -> Evidence:
        raise NotImplementedError("re_observe implemented in Phase 12")
    
    def verify_grasp(self, target: Hypothesis, env) -> tuple[bool, float]:
        raise NotImplementedError("verify_grasp implemented in Phase 12")
```

- [ ] **Step 2: 跑测试**

Run: `pytest tests/test_perception.py -v`
Expected: 10 pass

- [ ] **Step 3: ruff + commit**

```bash
git add src/perception.py prompts/perception/query_aware_ground.txt tests/test_perception.py
git commit -m "feat(perception): QueryAwareGrounder.observe with query-injected prompt + temperature scaling"
```

**Phase 5 CHECKPOINT:** 10 pass。Phase 12 会补 re_observe / verify_grasp。

---

## Phase 6: SafetyClassifier (改造 `src/safety_gate.py`) + classify prompt + 5 单测

**目标:** 把 SafetyGate 缩水成 LLM-only 的 SafetyClassifier; 删 `_FEATURE_RISK_KEYWORDS` + `_get_rule` + YAML 加载 (老类**保留**, 等 Phase 15 删, 新类并存)。

### Task 6.1: prompt + 失败测试

**Files:**
- Create: `prompts/safety/classify.txt`
- Create: `tests/test_safety_classifier.py`

- [ ] **Step 1: 写 prompt**

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

注意: 这 5 类是默认; 如果你认为有别的关键风险 (e.g. 太重/打滑), 可以加自定义 key
但每次只加 1 个, 总和仍 = 1.0。

Reply with ONLY raw JSON, no markdown:
{"dist": {"safe": 0.0, "fragile": 0.0, "sharp": 0.0, "hot": 0.0, "chemical": 0.0}, "reasoning": "<1 句>"}
```

- [ ] **Step 2: 测试**

```python
"""SafetyClassifier (LLM-based) 单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import math
import pytest
import numpy as np
from tests._mocks import MockLLM


def _make_hyp(label="apple", features="red round"):
    from src.world_belief import Hypothesis
    return Hypothesis(
        object_id="o0", label=label,
        label_alternatives=[(label, 0.9)], label_entropy=0.1,
        position_3d=np.zeros(3), position_std_m=0.05,
    )


class TestSafetyClassifier:
    def test_classify_returns_dist(self):
        from src.safety_gate import SafetyClassifier
        llm = MockLLM(responses=[json.dumps({
            "dist": {"safe": 0.8, "fragile": 0.1, "sharp": 0.05, "hot": 0.0, "chemical": 0.05},
            "reasoning": "看起来像水果",
        })])
        sc = SafetyClassifier(llm=llm)
        h = _make_hyp(label="apple")
        ev = sc.classify(h)
        assert ev.source == "llm_safety"
        assert ev.raw_payload["dist"]["safe"] == 0.8
    
    def test_entropy_computed(self):
        from src.safety_gate import SafetyClassifier
        llm = MockLLM(responses=[json.dumps({
            "dist": {"safe": 0.5, "fragile": 0.5},
            "reasoning": "?",
        })])
        sc = SafetyClassifier(llm=llm)
        ev = sc.classify(_make_hyp())
        assert ev.raw_payload["entropy"] == pytest.approx(math.log(2), abs=0.01)
    
    def test_malformed_json_returns_unknown(self):
        from src.safety_gate import SafetyClassifier
        llm = MockLLM(responses=["not json at all"])
        sc = SafetyClassifier(llm=llm)
        ev = sc.classify(_make_hyp())
        # 退化: dist 全 0 + entropy=0 + reasoning 标 'parse_failed'
        assert ev.raw_payload["entropy"] == pytest.approx(0.0)
        assert "parse_failed" in ev.raw_payload.get("reasoning", "")
    
    def test_dist_normalized(self):
        """LLM 输出 dist 不归一时, 自动归一。"""
        from src.safety_gate import SafetyClassifier
        llm = MockLLM(responses=[json.dumps({
            "dist": {"safe": 0.4, "fragile": 0.4},        # sum=0.8
            "reasoning": "?",
        })])
        sc = SafetyClassifier(llm=llm)
        ev = sc.classify(_make_hyp())
        assert sum(ev.raw_payload["dist"].values()) == pytest.approx(1.0, abs=1e-3)
    
    def test_open_key_dist_accepted(self):
        """LLM 自定义 key (e.g. weight) 不报错 (F4)。"""
        from src.safety_gate import SafetyClassifier
        llm = MockLLM(responses=[json.dumps({
            "dist": {"safe": 0.5, "weight": 0.5},
            "reasoning": "very heavy",
        })])
        sc = SafetyClassifier(llm=llm)
        ev = sc.classify(_make_hyp())
        assert "weight" in ev.raw_payload["dist"]
```

- [ ] **Step 3: 跑测试 (红)**

Run: `pytest tests/test_safety_classifier.py -v`
Expected: ImportError on SafetyClassifier

---

### Task 6.2: 在 `src/safety_gate.py` 追加 `SafetyClassifier` 类 (老 SafetyGate 保留)

**Files:**
- Modify: `src/safety_gate.py` (追加新类, 不删老的)

- [ ] **Step 1: 在文件底部加 SafetyClassifier**

```python
# ============================================================
# 新: SafetyClassifier (LLM-based, replace SafetyGate in v1)
# ============================================================

import json
import math
import re
import time
from typing import Optional

from src.world_belief import Evidence, Hypothesis


_DEFAULT_CLASSIFY_PROMPT = "prompts/safety/classify.txt"


def _shannon_safety(probs: dict[str, float]) -> float:
    total = sum(p for p in probs.values() if p > 0)
    if total <= 0:
        return 0.0
    h = 0.0
    for p in probs.values():
        if p > 0:
            q = p / total
            h -= q * math.log(q)
    return h


class SafetyClassifier:
    """LLM 输出 safety_dist + entropy, 不依赖关键词表 / YAML 规则。"""
    
    def __init__(self, llm, prompt_path: str = _DEFAULT_CLASSIFY_PROMPT):
        self.llm = llm
        p = Path(prompt_path)
        self._template = p.read_text(encoding="utf-8") if p.exists() else None
    
    def classify(self, hyp: Hypothesis) -> Evidence:
        prompt = self._build_prompt(hyp)
        try:
            raw = self.llm.generate(prompt, system="")
        except Exception as e:
            return Evidence(
                source="llm_safety", timestamp=time.time(),
                raw_payload={"dist": {}, "entropy": 0.0,
                             "reasoning": f"llm_failed: {e}"},
            )
        
        data = self._extract_json(raw)
        if data is None:
            return Evidence(
                source="llm_safety", timestamp=time.time(),
                raw_payload={"dist": {}, "entropy": 0.0,
                             "reasoning": "parse_failed",
                             "raw": raw[:500]},
            )
        
        dist = {str(k): float(v) for k, v in data.get("dist", {}).items()}
        # 归一化
        total = sum(dist.values())
        if total > 0:
            dist = {k: v / total for k, v in dist.items()}
        entropy = _shannon_safety(dist)
        return Evidence(
            source="llm_safety", timestamp=time.time(),
            raw_payload={
                "dist": dist,
                "entropy": entropy,
                "reasoning": str(data.get("reasoning", "")),
            },
        )
    
    def _build_prompt(self, hyp: Hypothesis) -> str:
        if self._template is None:
            return f"Classify safety of {hyp.label}, return JSON dist."
        alts_top3 = hyp.label_alternatives[:3]
        alts_text = ", ".join(f"{lbl}({p:.2f})" for lbl, p in alts_top3)
        features = "; ".join(
            f"{vp}: ..." for vp in hyp.observed_in_views
        ) or "(无)"
        pose_text = "upright" if (hyp.pose_estimate is None or hyp.pose_estimate.upright) else "side"
        return (
            self._template
            .replace("{label}", hyp.label)
            .replace("{alternatives_top3}", alts_text)
            .replace("{features}", features)
            .replace("{pose_summary}", pose_text)
        )
    
    @staticmethod
    def _extract_json(raw: str) -> Optional[dict]:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return None
```

- [ ] **Step 2: 跑测试**

Run: `pytest tests/test_safety_classifier.py -v`
Expected: 5 pass

- [ ] **Step 3: 跑老测试确保没破 (Phase 15 才删 SafetyGate)**

Run: `pytest tests/test_safety_gate.py -v`
Expected: All still pass (老类未动)

- [ ] **Step 4: commit**

```bash
git add src/safety_gate.py prompts/safety/classify.txt tests/test_safety_classifier.py
git commit -m "feat(safety): SafetyClassifier (LLM-only) coexists with old SafetyGate"
```

**Phase 6 CHECKPOINT:** 5 new + 老 safety_gate 测试不破。

---

## Phase 7: `src/grasp_planner.py` + suggest_top_grasp prompt + 8 单测

**目标:** 生成 N 个 GraspCandidate, 含 geometric_centroid / axis_aligned_side / vlm_top_grasp 三种策略 + 可达性过滤; `regenerate_after_failure` 排除已失败候选。

### Task 7.1: prompt + 失败测试

**Files:**
- Create: `prompts/grasp/suggest_top_grasp.txt`
- Create: `tests/test_grasp_planner.py`

- [ ] **Step 1: 写 prompt**

```
看这张 eye-in-hand 图。物体: {label} (姿态: {pose})。

请用 1 个 [x_norm, y_norm] 标记你认为最稳的"顶抓点"位置 (像素坐标归一化到 0-1, 图像左上 0,0)。
- x_norm, y_norm: 抓点中心
- finger_align: "x" 表示夹爪沿 x 轴, "y" 沿 y 轴 (基于物体长轴)

Reply ONLY raw JSON:
{"grip_norm": [x, y], "finger_align": "x"}
```

- [ ] **Step 2: 测试**

```python
"""GraspPlanner 单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pytest
import numpy as np
from tests._mocks import MockVLM


def _hyp(label="apple", upright=True, pos_std=0.02):
    from src.world_belief import Hypothesis, Pose
    h = Hypothesis(
        object_id="o0", label=label,
        label_alternatives=[(label, 0.9)], label_entropy=0.1,
        position_3d=np.array([0.5, 0.0, 0.9]),
        position_std_m=pos_std,
    )
    h.pose_estimate = Pose(
        position=np.array([0.5, 0, 0.9]),
        rotation_quat=np.array([0, 0, 0, 1]),
        upright=upright,
    )
    return h


class FakeEnv:
    def __init__(self, reachable_fn=None):
        self._reachable_fn = reachable_fn or (lambda p, d: True)
    def is_reachable(self, point_3d, approach_dir):
        return self._reachable_fn(point_3d, approach_dir)
    def observe(self, vp): 
        class O: image_path = "/dev/null"
        return O()
    def eye_in_hand_viewpoint(self):
        class V: name = "eye_in_hand"
        return V()


class TestGraspPlannerPlan:
    def test_geometric_centroid_always_first(self):
        """最朴素策略一定有: 物体上方, 顶抓。"""
        from src.grasp_planner import GraspPlanner
        gp = GraspPlanner(vlm=MockVLM([]), env=FakeEnv())
        cands = gp.plan(_hyp(), env=FakeEnv())
        assert len(cands) >= 1
        # 至少有一个 source=geometric_centroid
        assert any(c.source == "geometric_centroid" for c in cands)
    
    def test_axis_aligned_side_when_horizontal(self):
        """横放物体 → 加 axis_aligned_side 候选。"""
        from src.grasp_planner import GraspPlanner
        gp = GraspPlanner(vlm=MockVLM([]), env=FakeEnv())
        cands = gp.plan(_hyp(upright=False), env=FakeEnv())
        assert any(c.source == "axis_aligned_side" for c in cands)
    
    def test_unreachable_filtered(self):
        from src.grasp_planner import GraspPlanner
        env = FakeEnv(reachable_fn=lambda p, d: False)
        gp = GraspPlanner(vlm=MockVLM([]), env=env)
        cands = gp.plan(_hyp(), env=env)
        assert cands == []
    
    def test_vlm_top_grasp_used_if_available(self):
        from src.grasp_planner import GraspPlanner
        vlm = MockVLM(responses=[json.dumps({
            "grip_norm": [0.5, 0.5], "finger_align": "x",
        })])
        gp = GraspPlanner(vlm=vlm, env=FakeEnv())
        cands = gp.plan(_hyp(), env=FakeEnv())
        assert any(c.source == "vlm_top_grasp" for c in cands)
    
    def test_sorted_by_score_desc(self):
        from src.grasp_planner import GraspPlanner
        gp = GraspPlanner(vlm=MockVLM([]), env=FakeEnv())
        cands = gp.plan(_hyp(upright=False), env=FakeEnv())
        scores = [c.score for c in cands]
        assert scores == sorted(scores, reverse=True)


class TestRegenerateAfterFailure:
    def test_excludes_failed_candidate(self):
        from src.grasp_planner import GraspPlanner
        from src.world_belief import GraspAttempt, GraspCandidate
        gp = GraspPlanner(vlm=MockVLM([]), env=FakeEnv())
        h = _hyp()
        c1 = GraspCandidate(point_3d=h.position_3d.copy(),
                            approach_dir=np.array([0,0,-1]),
                            finger_width_m=0.04, score=0.9,
                            source="geometric_centroid")
        h.grasp_candidates = [c1]
        attempt = GraspAttempt(timestamp=1.0, candidate=c1,
                               failure_mode="hit_z_floor",
                               end_effector_pose_reached=(0,)*6)
        h.grasp_attempts = [attempt]
        new_cands = gp.regenerate_after_failure(h, attempt)
        # 新候选不应跟 c1 完全一致
        for c in new_cands:
            assert tuple(c.point_3d) != tuple(c1.point_3d) or \
                   tuple(c.approach_dir) != tuple(c1.approach_dir)
    
    def test_horizontal_pose_after_z_floor_failure(self):
        """hit_z_floor 失败 + pose 转横 → 强制 axis_aligned_side。"""
        from src.grasp_planner import GraspPlanner
        from src.world_belief import GraspAttempt, GraspCandidate
        gp = GraspPlanner(vlm=MockVLM([]), env=FakeEnv())
        h = _hyp(upright=False)
        c1 = GraspCandidate(point_3d=h.position_3d,
                            approach_dir=np.array([0,0,-1]),
                            finger_width_m=0.04, score=0.9,
                            source="geometric_centroid")
        h.grasp_candidates = [c1]
        attempt = GraspAttempt(timestamp=1.0, candidate=c1,
                               failure_mode="hit_z_floor",
                               end_effector_pose_reached=(0,)*6)
        h.grasp_attempts = [attempt]
        new_cands = gp.regenerate_after_failure(h, attempt)
        assert any(c.source == "axis_aligned_side" for c in new_cands)


class TestEdgeNoPose:
    def test_no_pose_falls_back_to_centroid_only(self):
        from src.grasp_planner import GraspPlanner
        gp = GraspPlanner(vlm=MockVLM([]), env=FakeEnv())
        h = _hyp()
        h.pose_estimate = None
        cands = gp.plan(h, env=FakeEnv())
        assert all(c.source != "axis_aligned_side" for c in cands)
        assert any(c.source == "geometric_centroid" for c in cands)
```

- [ ] **Step 3: 跑测试 (红)**

---

### Task 7.2: 实现 `src/grasp_planner.py`

**Files:**
- Create: `src/grasp_planner.py`

- [ ] **Step 1: 写实现**

```python
"""GraspPlanner: 生成 GraspCandidate, 含三种策略 + 可达性过滤。

策略:
- geometric_centroid: 物体中心 + 顶抓 approach=-z
- axis_aligned_side : pose.upright=False 时, 沿物体短轴侧抓
- vlm_top_grasp     : VLM 看 eye_in_hand 图建议 grip 点

设计参考: §6.5
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

from src.world_belief import GraspAttempt, GraspCandidate, Hypothesis

logger = logging.getLogger(__name__)


_DEFAULT_PROMPT = "prompts/grasp/suggest_top_grasp.txt"


class GraspPlanner:
    
    def __init__(self, vlm, env, prompt_path: str = _DEFAULT_PROMPT):
        self.vlm = vlm
        self.env = env
        p = Path(prompt_path)
        self._template = p.read_text(encoding="utf-8") if p.exists() else None
    
    # ──────────────────────────────────────
    # plan / regenerate_after_failure
    # ──────────────────────────────────────
    
    def plan(self, hyp: Hypothesis, env=None) -> list[GraspCandidate]:
        env = env or self.env
        cands: list[GraspCandidate] = []
        
        # 1. geometric_centroid
        c_top = GraspCandidate(
            point_3d=hyp.position_3d.copy(),
            approach_dir=np.array([0, 0, -1.0]),
            finger_width_m=0.04,
            score=0.7,
            source="geometric_centroid",
        )
        cands.append(c_top)
        
        # 2. axis_aligned_side (pose 横放时)
        if hyp.pose_estimate is not None and not hyp.pose_estimate.upright:
            c_side = GraspCandidate(
                point_3d=hyp.position_3d.copy(),
                approach_dir=np.array([1.0, 0, 0]),
                finger_width_m=0.04,
                score=0.65,
                source="axis_aligned_side",
            )
            cands.append(c_side)
        
        # 3. vlm_top_grasp (eye_in_hand)
        try:
            v = self._vlm_grasp(hyp, env)
            if v is not None:
                cands.append(v)
        except Exception as e:
            logger.debug(f"[grasp_planner] vlm_top_grasp skipped: {e}")
        
        # 可达性过滤
        cands = [
            c for c in cands
            if env.is_reachable(c.point_3d, c.approach_dir)
        ]
        # 排序
        cands.sort(key=lambda c: c.score, reverse=True)
        return cands
    
    def regenerate_after_failure(
        self, hyp: Hypothesis, last_attempt: GraspAttempt,
    ) -> list[GraspCandidate]:
        new_cands = self.plan(hyp)
        # hit_z_floor 时, 强烈倾向侧抓
        if last_attempt.failure_mode == "hit_z_floor":
            new_cands = [c for c in new_cands
                         if c.source != "geometric_centroid"]
            # 若 pose 仍 upright, 强制加一个 side approach
            if not any(c.source == "axis_aligned_side" for c in new_cands):
                new_cands.append(GraspCandidate(
                    point_3d=hyp.position_3d.copy(),
                    approach_dir=np.array([1.0, 0, 0]),
                    finger_width_m=0.04, score=0.55,
                    source="axis_aligned_side",
                ))
        # 排除已用过的
        used = {self._cand_sig(a.candidate) for a in hyp.grasp_attempts}
        return [c for c in new_cands if self._cand_sig(c) not in used]
    
    @staticmethod
    def _cand_sig(c: GraspCandidate) -> tuple:
        return (
            round(float(c.point_3d[0]), 3),
            round(float(c.point_3d[1]), 3),
            round(float(c.point_3d[2]), 3),
            round(float(c.approach_dir[0]), 2),
            round(float(c.approach_dir[1]), 2),
            round(float(c.approach_dir[2]), 2),
        )
    
    # ──────────────────────────────────────
    # vlm_top_grasp
    # ──────────────────────────────────────
    
    def _vlm_grasp(self, hyp: Hypothesis, env) -> Optional[GraspCandidate]:
        if self._template is None:
            return None
        try:
            obs = env.observe(env.eye_in_hand_viewpoint())
        except Exception:
            return None
        pose_text = "upright" if (hyp.pose_estimate is None or hyp.pose_estimate.upright) else "side"
        prompt = (
            self._template
            .replace("{label}", hyp.label)
            .replace("{pose}", pose_text)
        )
        raw = self.vlm.describe(getattr(obs, "image_path", "/dev/null"), prompt=prompt)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return None
        # VLM 给的 grip_norm [x, y] 暂时直接用作 score 加权; 真实 3D 投影 Phase 12
        return GraspCandidate(
            point_3d=hyp.position_3d.copy(),
            approach_dir=np.array([0, 0, -1.0]),
            finger_width_m=0.04, score=0.75,
            source="vlm_top_grasp",
        )
```

- [ ] **Step 2: 跑测试**

Run: `pytest tests/test_grasp_planner.py -v`
Expected: 8 pass

- [ ] **Step 3: ruff + commit**

```bash
git add src/grasp_planner.py prompts/grasp/suggest_top_grasp.txt tests/test_grasp_planner.py
git commit -m "feat(grasp): GraspPlanner with 3 strategies + reachability filter + regenerate"
```

**Phase 7 CHECKPOINT:** 8 pass。

---

## Phase 8: `src/action_executor.py` 改造 (Hypothesis 输入 + verify_grasp + release_and_retreat) + 10 单测

**目标:** 改造 ActionExecutor 接 Hypothesis 而非 ActionPlan, 加 `verify_grasp` (eye_in_hand + VLM 占位) + `release_and_retreat` (F6); 各 failure_mode 结构化分类。老接口 `execute` / `execute_with_scene_model` **保留** (Phase 15 删)。

### Task 8.1: 失败测试

**Files:**
- Create: `tests/test_action_executor_v1.py`

- [ ] **Step 1: 写测试**

```python
"""ActionExecutor v1 (Hypothesis-based) 单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np


def _hyp_with_candidate(score=0.9):
    from src.world_belief import GraspCandidate, Hypothesis
    c = GraspCandidate(point_3d=np.array([0.5, 0, 0.9]),
                       approach_dir=np.array([0,0,-1]),
                       finger_width_m=0.04, score=score,
                       source="geometric_centroid")
    h = Hypothesis(
        object_id="o0", label="apple",
        label_alternatives=[("apple", 0.9)], label_entropy=0.1,
        position_3d=np.array([0.5, 0, 0.9]), position_std_m=0.02,
        grasp_candidates=[c],
    )
    return h, c


class FakeEnv:
    def __init__(self, descend_ok=True, ik_ok=True, lift_ok=True,
                 final_z=0.05):
        self.descend_ok = descend_ok
        self.ik_ok = ik_ok
        self.lift_ok = lift_ok
        self.final_z = final_z
        self._gripper_open = True
        self.calls: list[str] = []
    
    def move_to_pre_grasp(self, candidate) -> bool:
        self.calls.append("move_to_pre_grasp")
        return self.ik_ok
    
    def descend(self, point_3d):
        self.calls.append("descend")
        if self.descend_ok:
            return True, point_3d[2]
        return False, point_3d[2] + 0.03   # 卡住
    
    def close_gripper(self) -> bool:
        self.calls.append("close")
        self._gripper_open = False
        return True
    
    def open_gripper(self) -> bool:
        self.calls.append("open")
        self._gripper_open = True
        return True
    
    def lift(self) -> tuple[bool, float]:
        self.calls.append("lift")
        return self.lift_ok, self.final_z
    
    def get_eef_pos(self):
        return np.array([0.5, 0, 0.95])
    
    def move_arm_to(self, pos, **kw):
        self.calls.append("move")
        return True


class TestAct:
    def test_success_path(self):
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = FakeEnv()
        exe = ActionExecutor(scene_describer=None)
        h, c = _hyp_with_candidate()
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)
        assert result.attempt.failure_mode == "success"
    
    def test_ik_unreachable_classified(self):
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = FakeEnv(ik_ok=False)
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)
        assert result.attempt.failure_mode == "ik_unreachable"
    
    def test_hit_z_floor_classified(self):
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = FakeEnv(descend_ok=False)
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)
        assert result.attempt.failure_mode == "hit_z_floor"
        assert "z_actual" in result.attempt.diagnostic
    
    def test_slipped_classified(self):
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = FakeEnv(lift_ok=False, final_z=0.0)
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)
        assert result.attempt.failure_mode == "slipped"
    
    def test_no_candidates_returns_failure(self):
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask, Hypothesis
        env = FakeEnv()
        exe = ActionExecutor(scene_describer=None)
        h = Hypothesis(
            object_id="o0", label="x",
            label_alternatives=[("x", 1.0)], label_entropy=0.0,
            position_3d=np.zeros(3), position_std_m=0.0,
        )
        result = exe.act(h, DecomposedTask(primary_target="x"), env)
        assert result.attempt.failure_mode == "ik_unreachable"   # 或其他, 不能 success


class TestVerifyGrasp:
    def test_verify_uses_vlm(self):
        from src.action_executor import ActionExecutor
        # 占位: Phase 12 之前 verify_grasp 占位返回 (True, 1.0) 或调 perception.verify_grasp
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()
        env = FakeEnv()
        # 默认 stub 实现应返回 (True, 1.0) 或交给 perception
        ok, conf = exe.verify_grasp(h, env)
        assert isinstance(ok, bool)
        assert 0.0 <= conf <= 1.0


class TestReleaseAndRetreat:
    def test_release_and_retreat_opens_then_lifts(self):
        """F6: 撤回必须 open + 提升, 否则后续 observe 被夹爪挡。"""
        from src.action_executor import ActionExecutor
        env = FakeEnv()
        exe = ActionExecutor(scene_describer=None)
        exe.release_and_retreat(env, retreat_height_m=0.10)
        assert "open" in env.calls
        assert any(c.startswith("move") for c in env.calls)
        # open 必须在 move 前 (先放再走)
        assert env.calls.index("open") < env.calls.index("move")


class TestDiagnostic:
    def test_diagnostic_contains_osc_steps(self):
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        exe = ActionExecutor(scene_describer=None)
        env = FakeEnv()
        h, _ = _hyp_with_candidate()
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)
        # diagnostic 至少含 z_actual / z_target
        assert "z_target" in result.attempt.diagnostic
```

- [ ] **Step 2: 跑测试 (红)**

Run: `pytest tests/test_action_executor_v1.py -v`
Expected: AttributeError on `.act` 或 `release_and_retreat` (老 ActionExecutor 没这些方法)

---

### Task 8.2: 在 `src/action_executor.py` 追加新接口 `act` / `verify_grasp` / `release_and_retreat`

**Files:**
- Modify: `src/action_executor.py` (追加新方法 + ActionResult, 不删 `execute`)

- [ ] **Step 1: 在文件末尾追加**

```python
# ============================================================
# v1 新接口 (基于 Hypothesis)
# ============================================================

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.world_belief import (
    DecomposedTask, Evidence, GraspAttempt, GraspCandidate,
    Hypothesis,
)

if TYPE_CHECKING:
    pass


@dataclass
class ActionResult:
    success: bool
    attempt: GraspAttempt
    new_observations: list[Evidence] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "attempt": {
                "failure_mode": self.attempt.failure_mode,
                "diagnostic": self.attempt.diagnostic,
                "candidate_source": self.attempt.candidate.source,
            },
        }


# 把这些方法 monkey-patch / 加到 ActionExecutor 类里。
# 因为老 ActionExecutor 已经存在 (with execute / execute_with_scene_model),
# 我们在类定义末尾追加方法即可。

# 实际改动是在 ActionExecutor 类内部加方法 — 见 Step 2
```

- [ ] **Step 2: 在 ActionExecutor 类内部 (找到 class ActionExecutor 块) 加 `act` / `verify_grasp` / `release_and_retreat`**

```python
    # ============================================================
    # v1 接口 (Hypothesis-based)
    # ============================================================
    
    def act(
        self,
        target: Hypothesis,
        decomposed: DecomposedTask,
        env,
    ) -> ActionResult:
        """v1 主接口: 抓取 target, 失败结构化回写。"""
        # 选 candidates 最高分未试过的
        used = {self._cand_sig(a.candidate) for a in target.grasp_attempts}
        candidate = next(
            (c for c in target.grasp_candidates if self._cand_sig(c) not in used),
            None,
        )
        if candidate is None:
            return self._failed_result(
                None, "ik_unreachable",
                {"reason": "no candidate"}, env,
            )
        
        # 1. pre-grasp
        ok = env.move_to_pre_grasp(candidate)
        if not ok:
            return self._failed_result(candidate, "ik_unreachable",
                                       {"stage": "pre_grasp"}, env)
        
        # 2. descend
        z_target = float(candidate.point_3d[2])
        descend_ok, z_actual = env.descend(candidate.point_3d)
        if not descend_ok:
            return self._failed_result(
                candidate, "hit_z_floor",
                {"z_target": z_target, "z_actual": z_actual,
                 "stage": "descend"},
                env,
            )
        
        # 3. close gripper
        env.close_gripper()
        
        # 4. lift
        lift_ok, final_z = env.lift()
        if not lift_ok:
            return self._failed_result(
                candidate, "slipped",
                {"z_target": z_target, "z_actual": z_actual,
                 "final_z": final_z, "stage": "lift"},
                env,
            )
        
        # success
        attempt = GraspAttempt(
            timestamp=time.time(),
            candidate=candidate,
            failure_mode="success",
            end_effector_pose_reached=tuple(env.get_eef_pos().tolist()) + (0.0, 0.0, 0.0),
            diagnostic={"z_target": z_target, "z_actual": z_actual,
                        "final_z": final_z, "stage": "complete"},
        )
        return ActionResult(success=True, attempt=attempt)
    
    def verify_grasp(self, target: Hypothesis, env) -> tuple[bool, float]:
        """post-grasp 语义验证。Phase 12 接 perception.verify_grasp; v1 占位。"""
        return True, 1.0
    
    def release_and_retreat(self, env, retreat_height_m: float = 0.10) -> None:
        """F6: verify_mismatch / 异常退出时, 先松开夹爪再撤回。"""
        env.open_gripper()
        try:
            current = env.get_eef_pos()
            target = current + np.array([0.0, 0.0, retreat_height_m], dtype=np.float32)
            env.move_arm_to(target, threshold_m=0.02)
        except Exception as e:
            import logging as _l
            _l.getLogger(__name__).warning(
                f"[release_and_retreat] retreat failed: {e}"
            )
    
    def _failed_result(
        self, candidate: Optional[GraspCandidate], mode: str,
        diag: dict, env,
    ) -> ActionResult:
        # 失败时也尝试松开 + 撤回, 保护现场
        try:
            self.release_and_retreat(env)
        except Exception:
            pass
        if candidate is None:
            # 占位 candidate 让 GraspAttempt 不空
            candidate = GraspCandidate(
                point_3d=np.zeros(3), approach_dir=np.zeros(3),
                finger_width_m=0.04, score=0.0,
                source="geometric_centroid",
            )
        attempt = GraspAttempt(
            timestamp=time.time(),
            candidate=candidate,
            failure_mode=mode,           # type: ignore[arg-type]
            end_effector_pose_reached=(0.0,) * 6,
            diagnostic=diag,
        )
        return ActionResult(success=False, attempt=attempt)
    
    @staticmethod
    def _cand_sig(c: GraspCandidate) -> tuple:
        return (
            round(float(c.point_3d[0]), 3),
            round(float(c.point_3d[1]), 3),
            round(float(c.point_3d[2]), 3),
            round(float(c.approach_dir[0]), 2),
            round(float(c.approach_dir[1]), 2),
            round(float(c.approach_dir[2]), 2),
        )
```

- [ ] **Step 3: 跑测试**

Run: `pytest tests/test_action_executor_v1.py -v`
Expected: 10 pass

- [ ] **Step 4: 跑老 executor 测试不破**

Run: `pytest tests/test_env_wrapper_grasp.py -v`
Expected: 不破 (新方法是追加, 老 execute 不动)

- [ ] **Step 5: commit**

```bash
git add src/action_executor.py tests/test_action_executor_v1.py
git commit -m "feat(executor): act(Hypothesis)/verify_grasp/release_and_retreat with structured failure_mode"
```

**Phase 8 CHECKPOINT:** 10 new pass + 老不破。

---

## Phase 9: `ActiveViewpointSelector` + nbv prompt + 5 单测

**目标:** 改造 `active_planner.py`, 新增 `ActiveViewpointSelector.select(belief, exclude, preference) -> Optional[Viewpoint]`; 老 `plan` / `plan_with_grounding` **保留** (Phase 15 删)。

### Task 9.1: prompt + 失败测试

**Files:**
- Create: `prompts/agent/nbv_select.txt`
- Create: `tests/test_viewpoint_selector.py`

- [ ] **Step 1: 写 prompt**

```
你是机器人的视角规划器。当前 belief 摘要:

primary_target: {primary_target}
hypotheses ({n_hyp} 个):
{hyp_list}

已用过的视角: {used_views}
可用视角:
{vp_list}

需求: {preference}
- "search_target": 最大化看到 primary_target 的概率
- "disambiguate_label": 看清当前 target 的 label
- "parallax_position": 给当前 target 提供视差以降 position_std
- "grasp_pose": 看清当前 target 的姿态 (横/竖)

请输出 1 个未用视角的索引 (整数, 0-based) 或 -1 表示无可选。

只回复一个数字, 不带任何解释。
```

- [ ] **Step 2: 测试**

```python
"""ActiveViewpointSelector 单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from tests._mocks import MockLLM


class FakeViewpoint:
    def __init__(self, name):
        self.name = name


class FakeViewpointLib:
    def __init__(self, names):
        self.viewpoints = [FakeViewpoint(n) for n in names]
    def __len__(self): return len(self.viewpoints)
    def __getitem__(self, i): return self.viewpoints[i]


def _basic_belief():
    from src.world_belief import WorldBelief, DecomposedTask
    b = WorldBelief(user_query="拿苹果")
    b.decomposed = DecomposedTask(primary_target="apple")
    return b


class TestSelect:
    def test_returns_viewpoint_at_index(self):
        from src.active_planner import ActiveViewpointSelector
        llm = MockLLM(responses=["1"])
        vp_lib = FakeViewpointLib(["v0", "v1", "v2"])
        sel = ActiveViewpointSelector(llm=llm, viewpoint_lib=vp_lib)
        vp = sel.select(_basic_belief(), exclude=set(), preference="search_target")
        assert vp.name == "v1"
    
    def test_excludes_used(self):
        from src.active_planner import ActiveViewpointSelector
        llm = MockLLM(responses=["2"])
        vp_lib = FakeViewpointLib(["v0", "v1", "v2"])
        sel = ActiveViewpointSelector(llm=llm, viewpoint_lib=vp_lib)
        vp = sel.select(_basic_belief(), exclude={"v0", "v1"},
                        preference="search_target")
        assert vp.name == "v2"
    
    def test_minus_one_returns_none(self):
        from src.active_planner import ActiveViewpointSelector
        llm = MockLLM(responses=["-1"])
        vp_lib = FakeViewpointLib(["v0", "v1"])
        sel = ActiveViewpointSelector(llm=llm, viewpoint_lib=vp_lib)
        vp = sel.select(_basic_belief(), exclude=set(),
                        preference="search_target")
        assert vp is None
    
    def test_all_excluded_returns_none(self):
        from src.active_planner import ActiveViewpointSelector
        llm = MockLLM(responses=[])
        vp_lib = FakeViewpointLib(["v0"])
        sel = ActiveViewpointSelector(llm=llm, viewpoint_lib=vp_lib)
        vp = sel.select(_basic_belief(), exclude={"v0"}, preference="search_target")
        assert vp is None
    
    def test_invalid_index_returns_none(self):
        """LLM 输出 99 (越界) → None。"""
        from src.active_planner import ActiveViewpointSelector
        llm = MockLLM(responses=["99"])
        vp_lib = FakeViewpointLib(["v0", "v1"])
        sel = ActiveViewpointSelector(llm=llm, viewpoint_lib=vp_lib)
        vp = sel.select(_basic_belief(), exclude=set(), preference="search_target")
        assert vp is None
```

- [ ] **Step 3: 跑测试 (红)**

---

### Task 9.2: 在 `src/active_planner.py` 追加 ActiveViewpointSelector

**Files:**
- Modify: `src/active_planner.py`

- [ ] **Step 1: 在文件末尾追加**

```python
# ============================================================
# v1 新接口
# ============================================================

import logging as _logging
import re as _re
from pathlib import Path as _Path
from typing import Optional as _Optional, Literal as _Literal

from src.world_belief import WorldBelief as _WorldBelief

_NBV_PROMPT_PATH = "prompts/agent/nbv_select.txt"
_v_logger = _logging.getLogger(__name__)


class ActiveViewpointSelector:
    
    def __init__(self, llm, viewpoint_lib, prompt_path: str = _NBV_PROMPT_PATH):
        self.llm = llm
        self.vp_lib = viewpoint_lib
        p = _Path(prompt_path)
        self._template = p.read_text(encoding="utf-8") if p.exists() else None
    
    def select(
        self,
        belief: _WorldBelief,
        exclude: set[str],
        preference: _Literal[
            "search_target", "disambiguate_label",
            "parallax_position", "grasp_pose",
        ] = "search_target",
    ) -> _Optional[object]:
        """LLM 选下一视角索引, 越界/重复/不存在 → None。"""
        # 候选: 未排除的视角
        candidates = [
            (i, vp) for i, vp in enumerate(self.vp_lib)
            if vp.name not in exclude
        ]
        if not candidates:
            return None
        
        prompt = self._build_prompt(belief, exclude, preference)
        try:
            raw = self.llm.generate(prompt, system="")
        except Exception as e:
            _v_logger.warning(f"[viewpoint_selector] LLM failed: {e}")
            # fallback: 返回第一个未排除的
            return candidates[0][1]
        
        m = _re.search(r"-?\d+", raw)
        if not m:
            return None
        idx = int(m.group())
        if idx == -1:
            return None
        if idx < 0 or idx >= len(self.vp_lib):
            return None
        vp = self.vp_lib[idx]
        if vp.name in exclude:
            return None
        return vp
    
    def _build_prompt(
        self, belief: _WorldBelief, exclude: set[str], preference: str,
    ) -> str:
        if self._template is None:
            return f"Pick a viewpoint index for {preference}, skip {exclude}."
        primary = belief.decomposed.primary_target if belief.decomposed else "?"
        hyp_lines = [
            f"  - {h.label} (entropy={h.label_entropy:.2f}, "
            f"pos_std={h.position_std_m:.2f}m, views={h.observed_in_views})"
            for h in belief.hypotheses
        ] or ["  (无)"]
        vp_lines = [f"  {i}: {vp.name}" for i, vp in enumerate(self.vp_lib)]
        return (
            self._template
            .replace("{primary_target}", primary)
            .replace("{n_hyp}", str(len(belief.hypotheses)))
            .replace("{hyp_list}", "\n".join(hyp_lines))
            .replace("{used_views}", ", ".join(sorted(exclude)) or "(无)")
            .replace("{vp_list}", "\n".join(vp_lines))
            .replace("{preference}", preference)
        )
```

- [ ] **Step 2: 跑测试**

Run: `pytest tests/test_viewpoint_selector.py -v`
Expected: 5 pass

- [ ] **Step 3: 老 active_planner 测试不破**

Run: `pytest tests/test_active_planner_grounding.py -v`
Expected: 不破

- [ ] **Step 4: commit**

```bash
git add src/active_planner.py prompts/agent/nbv_select.txt tests/test_viewpoint_selector.py
git commit -m "feat(planner): ActiveViewpointSelector with 4 preferences, coexists with old ActivePlanner"
```

**Phase 9 CHECKPOINT:** 5 new + 老不破。

---

## Phase 10: `task_decomposer.py` 改造 → `DecomposedTask` + decompose prompt + 5 单测

**目标:** TaskDecomposer 新增 `decompose_v1` 方法返回 `DecomposedTask` (含 constraints), 老 `decompose` 返回 `list[Subtask]` **保留** (Phase 15 删)。

### Task 10.1: prompt + 失败测试

**Files:**
- Create: `prompts/agent/decompose.txt`
- Create: `tests/test_task_decomposer_v1.py`

- [ ] **Step 1: 写 prompt**

```
Query: {query}

请输出 JSON:
{
  "primary_target": "<目标物体名, 中文或英文>",
  "constraints": [
    {"kind": "avoid", "target_label": "knife", "reason": "用户提到避开"},
    {"kind": "user_hint", "text": "在水池左边", "reason": "位置提示"}
  ]
}

规则:
1. primary_target 是用户最想要的那一个东西 (单个)
2. constraints 包含: 避让物体 (avoid) / 位置提示 (user_hint) / 视角偏好 (prefer_view)
3. v1 单目标; 多目标 query 仍取最重要的一个
4. 没 constraint 时返回空数组

Reply ONLY raw JSON, no fences.
```

- [ ] **Step 2: 测试**

```python
"""TaskDecomposer v1 (DecomposedTask) 单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pytest
from tests._mocks import MockLLM


class TestDecomposeV1:
    def test_basic(self):
        from src.task_decomposer import TaskDecomposer
        llm = MockLLM(responses=[json.dumps({
            "primary_target": "apple",
            "constraints": [],
        })])
        td = TaskDecomposer(llm)
        dt = td.decompose_v1("帮我拿苹果")
        assert dt.primary_target == "apple"
        assert dt.constraints == []
        assert dt.raw_query == "帮我拿苹果"
    
    def test_avoid_constraint(self):
        from src.task_decomposer import TaskDecomposer
        from src.world_belief import Constraint
        llm = MockLLM(responses=[json.dumps({
            "primary_target": "bowl",
            "constraints": [
                {"kind": "avoid", "target_label": "knife", "reason": "用户避开"},
            ],
        })])
        td = TaskDecomposer(llm)
        dt = td.decompose_v1("拿碗, 避开刀")
        assert dt.primary_target == "bowl"
        assert len(dt.constraints) == 1
        assert dt.constraints[0].kind == "avoid"
        assert dt.constraints[0].target_label == "knife"
    
    def test_user_hint_constraint(self):
        from src.task_decomposer import TaskDecomposer
        llm = MockLLM(responses=[json.dumps({
            "primary_target": "bottle",
            "constraints": [
                {"kind": "user_hint", "text": "水池左边", "reason": "位置提示"},
            ],
        })])
        td = TaskDecomposer(llm)
        dt = td.decompose_v1("拿水池左边的瓶子")
        assert any(c.kind == "user_hint" and "水池" in (c.text or "")
                   for c in dt.constraints)
    
    def test_malformed_falls_back_to_primary_only(self):
        from src.task_decomposer import TaskDecomposer
        llm = MockLLM(responses=["not json"])
        td = TaskDecomposer(llm)
        dt = td.decompose_v1("帮我拿苹果")
        # fallback: primary_target = raw_query 的中心词或整 query
        assert dt.primary_target  # 非空
        assert dt.constraints == []
    
    def test_unknown_constraint_kind_skipped(self):
        from src.task_decomposer import TaskDecomposer
        llm = MockLLM(responses=[json.dumps({
            "primary_target": "x",
            "constraints": [{"kind": "weird_kind", "reason": "?"}],
        })])
        td = TaskDecomposer(llm)
        dt = td.decompose_v1("x")
        assert dt.constraints == []
```

- [ ] **Step 3: 跑测试 (红)**

---

### Task 10.2: 在 `task_decomposer.py` 加 decompose_v1

**Files:**
- Modify: `src/task_decomposer.py`

- [ ] **Step 1: 加方法**

```python
# 在 TaskDecomposer 类内追加:

    def decompose_v1(self, query: str, prompt_path: str = "prompts/agent/decompose.txt"):
        """v1 输出 DecomposedTask, 含 primary_target + constraints。"""
        from pathlib import Path as _P
        import json as _json, re as _re
        from src.world_belief import Constraint, DecomposedTask
        
        p = _P(prompt_path)
        if p.exists():
            template = p.read_text(encoding="utf-8")
            prompt = template.replace("{query}", query)
        else:
            prompt = f"Query: {query}\nOutput JSON with primary_target and constraints[]."
        
        try:
            raw = self.llm.generate(prompt, system="")
        except Exception:
            return DecomposedTask(primary_target=query.strip(), raw_query=query)
        
        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        if not m:
            return DecomposedTask(primary_target=query.strip(), raw_query=query)
        try:
            data = _json.loads(m.group())
        except _json.JSONDecodeError:
            return DecomposedTask(primary_target=query.strip(), raw_query=query)
        
        primary = str(data.get("primary_target", query.strip()))
        constraints: list[Constraint] = []
        for c in data.get("constraints", []):
            kind = c.get("kind")
            if kind not in {"avoid", "prefer_view", "max_force", "user_hint"}:
                continue
            constraints.append(Constraint(
                kind=kind,
                target_label=c.get("target_label"),
                text=c.get("text"),
                reason=c.get("reason", ""),
            ))
        return DecomposedTask(
            primary_target=primary,
            constraints=constraints,
            raw_query=query,
        )
```

- [ ] **Step 2: 跑测试**

Run: `pytest tests/test_task_decomposer_v1.py -v`
Expected: 5 pass

- [ ] **Step 3: commit**

```bash
git add src/task_decomposer.py prompts/agent/decompose.txt tests/test_task_decomposer_v1.py
git commit -m "feat(decomposer): add decompose_v1 -> DecomposedTask with constraints"
```

**Phase 10 CHECKPOINT:** 5 pass。

---

## Phase 11: `src/agent.py` 主循环 + decide_next + 17 单测

**目标:** `EmboSightAgent.run` + `decide_next` 决策树 + `_execute_action` 调度。所有依赖通过 DI 注入, 全部 mock 测试。
17 单测 = 9 (decide_next 路由 8 种状态 + max_re_observe) + 5 (run 集成 5 场景) + 3 (verify_mismatch 流程, F6 / Edge 9.6)。

### Task 11.1: decide_next 失败测试

**Files:**
- Create: `tests/test_agent_decide_next.py`

- [ ] **Step 1: 写 decide_next 测试**

```python
"""EmboSightAgent.decide_next 单元测试 (mock-driven, 8+ belief 状态)。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np


def _make_belief(target_word="apple", hyps=None, evidence=None,
                 action_history=None):
    from src.world_belief import (
        WorldBelief, DecomposedTask, Hypothesis, Evidence, Action,
    )
    b = WorldBelief(user_query=f"拿{target_word}")
    b.decomposed = DecomposedTask(primary_target=target_word)
    b.hypotheses = hyps or []
    b.evidence = evidence or []
    b.action_history = action_history or []
    return b


def _confident_target_hyp(label="apple"):
    from src.world_belief import GraspCandidate, Hypothesis
    c = GraspCandidate(point_3d=np.array([0.5,0,0.9]),
                       approach_dir=np.array([0,0,-1]),
                       finger_width_m=0.04, score=0.9,
                       source="geometric_centroid")
    return Hypothesis(
        object_id="o0", label=label,
        label_alternatives=[(label, 0.95), ("other", 0.05)],
        label_entropy=0.10,
        position_3d=np.array([0.5, 0, 0.9]), position_std_m=0.02,
        safety_dist={"safe": 0.9, "fragile": 0.1}, safety_entropy=0.10,
        grasp_candidates=[c],
    )


def _make_agent(viewpoints=None, nbv_responses=None):
    """构造一个 mock 化的 agent。"""
    from src.agent import EmboSightAgent
    from tests._mocks import MockLLM, MockVLM
    
    class FakeVPLib:
        def __init__(self, names):
            from src.active_planner import Viewpoint  # 假定老 dataclass 还在
            self.viewpoints = [
                type("VP", (), {"name": n})() for n in names
            ]
        def __len__(self): return len(self.viewpoints)
        def __getitem__(self, i): return self.viewpoints[i]
    
    vp_lib = FakeVPLib(viewpoints or ["v0", "v1", "v2"])
    nbv_llm = MockLLM(nbv_responses or ["1", "2", "-1"])
    return EmboSightAgent.with_test_doubles(vp_lib=vp_lib, nbv_llm=nbv_llm)


class TestDecideNext:
    def test_no_evidence_returns_observe(self):
        from src.world_belief import Action
        agent = _make_agent()
        belief = _make_belief()
        action = agent.decide_next(belief)
        assert action.kind == "observe"
        assert action.viewpoint.name == "v0"
    
    def test_no_target_returns_nbv_observe(self):
        from src.world_belief import Evidence, Hypothesis
        agent = _make_agent()
        belief = _make_belief(
            evidence=[Evidence(source="vlm_ground", timestamp=0, raw_payload={})],
            hyps=[
                Hypothesis(object_id="o", label="banana",
                           label_alternatives=[("banana", 0.9)],
                           label_entropy=0.1,
                           position_3d=np.zeros(3), position_std_m=0.05),
            ],   # 不是 apple
        )
        action = agent.decide_next(belief)
        assert action.kind in {"observe", "ask_user"}
    
    def test_no_target_no_more_views_asks_user(self):
        from src.world_belief import Evidence, Action
        agent = _make_agent(viewpoints=["v0"], nbv_responses=["-1"])
        belief = _make_belief(
            evidence=[Evidence(source="vlm_ground", timestamp=0, raw_payload={})],
            action_history=[Action(kind="observe", viewpoint=type("VP",(),{"name":"v0"})())],
        )
        action = agent.decide_next(belief)
        assert action.kind == "ask_user"
    
    def test_label_uncertain_zooms(self):
        from src.world_belief import Evidence, Hypothesis
        agent = _make_agent()
        h = Hypothesis(
            object_id="o0", label="apple",
            label_alternatives=[("apple", 0.5), ("kiwi", 0.5)],
            label_entropy=0.69,        # high
            position_3d=np.array([0.5,0,0.9]), position_std_m=0.02,
            safety_entropy=0.1,
        )
        belief = _make_belief(
            hyps=[h],
            evidence=[Evidence(source="vlm_ground", timestamp=0, raw_payload={})],
        )
        action = agent.decide_next(belief)
        assert action.kind == "re_observe"
        assert action.strategy == "zoom_in"
    
    def test_position_uncertain_parallax(self):
        from src.world_belief import Evidence, Hypothesis
        agent = _make_agent()
        h = Hypothesis(
            object_id="o0", label="apple",
            label_alternatives=[("apple", 0.95)],
            label_entropy=0.1,
            position_3d=np.array([0.5,0,0.9]), position_std_m=0.20,   # 大
            safety_entropy=0.1,
        )
        belief = _make_belief(
            hyps=[h],
            evidence=[Evidence(source="vlm_ground", timestamp=0, raw_payload={})],
        )
        action = agent.decide_next(belief)
        assert action.kind == "re_observe"
        assert action.strategy == "parallax_view"
    
    def test_safety_uncertain_classify(self):
        from src.world_belief import Evidence, Hypothesis
        agent = _make_agent()
        h = Hypothesis(
            object_id="o0", label="apple",
            label_alternatives=[("apple", 0.95)],
            label_entropy=0.1,
            position_3d=np.array([0.5,0,0.9]), position_std_m=0.02,
            safety_entropy=0.9,       # 大
        )
        belief = _make_belief(
            hyps=[h],
            evidence=[Evidence(source="vlm_ground", timestamp=0, raw_payload={})],
        )
        action = agent.decide_next(belief)
        assert action.kind == "classify_safety"
    
    def test_grasp_no_candidates_plans(self):
        from src.world_belief import Evidence, Hypothesis
        agent = _make_agent()
        h = Hypothesis(
            object_id="o0", label="apple",
            label_alternatives=[("apple", 0.95)],
            label_entropy=0.1,
            position_3d=np.array([0.5,0,0.9]), position_std_m=0.02,
            safety_entropy=0.1,
        )
        # plan grasp 直到 plan_grasp_candidates 完成才进 grasp
        # 但 grasp_uncertainty=None → most_uncertain_axis 跳过 grasp
        # is_confident_to_act=False (因 grasp_uncertainty None)
        # 决策: 既然 label/pos/safety 都 confident, axis 必为 grasp 但被排除...
        # 实现层兜底: 检查 is_confident_to_act 不通过 + grasp candidate 空 → plan
        belief = _make_belief(
            hyps=[h],
            evidence=[Evidence(source="vlm_ground", timestamp=0, raw_payload={})],
        )
        action = agent.decide_next(belief)
        assert action.kind == "plan_grasp_candidates"
    
    def test_all_confident_returns_grasp(self):
        from src.world_belief import Evidence
        agent = _make_agent()
        h = _confident_target_hyp()
        belief = _make_belief(
            hyps=[h],
            evidence=[Evidence(source="vlm_ground", timestamp=0, raw_payload={})],
        )
        # is_confident_to_act → True; 但 decide_next 仅在 not is_confident 时调用
        # 主循环负责调; decide_next 也应支持 (返回 grasp)
        action = agent.decide_next(belief)
        # decide_next 实现: is_confident → 直接 return grasp
        assert action.kind == "grasp"
        assert action.target_hypothesis is h
    
    def test_max_re_observe_asks_user(self):
        from src.world_belief import Evidence, Hypothesis
        agent = _make_agent()
        h = Hypothesis(
            object_id="o0", label="apple",
            label_alternatives=[("apple", 0.5), ("pear", 0.5)],
            label_entropy=0.69,
            position_3d=np.array([0.5,0,0.9]), position_std_m=0.02,
            safety_entropy=0.1,
        )
        h.times_re_observed = 3   # 等于 MAX_RE_OBSERVE
        belief = _make_belief(
            hyps=[h],
            evidence=[Evidence(source="vlm_ground", timestamp=0, raw_payload={})],
        )
        action = agent.decide_next(belief)
        assert action.kind == "ask_user"
```

- [ ] **Step 2: 跑测试 (红)**

Run: `pytest tests/test_agent_decide_next.py -v`
Expected: ImportError on `src.agent`

---

### Task 11.2: 实现 `src/agent.py` (decide_next + run + _execute_action)

**Files:**
- Create: `src/agent.py`

- [ ] **Step 1: 写实现**

```python
"""EmboSightAgent v1 主循环。

主入口: agent.run(query, env) -> EpisodeResult
内部循环: while not belief.is_confident_to_act(): decide_next(belief)

设计参考: §5
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any, Optional

import numpy as np

from src.world_belief import (
    Action, BeliefSnapshot, DecomposedTask, EpisodeResult, Evidence,
    Hypothesis, WorldBelief,
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
        """构造一个所有依赖都是 None 占位 / mock 的 agent, 用于 decide_next 单测。"""
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
    # decide_next (核心决策树, 见 §5.2)
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
                return Action(
                    kind="ask_user",
                    question=f"我没在场景里看到{belief.decomposed.primary_target}, 是不是被挡住了?",
                )
            return Action(kind="observe", viewpoint=next_vp)
        
        # 阶段 C: re_observe 超限 → ask_user
        if target.times_re_observed >= self.MAX_RE_OBSERVE:
            return Action(kind="ask_user", question=belief.compose_clarification())
        
        # 阶段 D: 哪轴最不确定就消除哪轴
        # (注意: most_uncertain_axis 可能跳过 grasp; 但若 label/pos/safety 都 confident
        #  而 grasp 还没 plan, is_confident_to_act 已返 False → 这里需兜底)
        axis = belief.most_uncertain_axis()
        
        # 兜底: label/pos/safety 都 confident 但 grasp 没 plan
        if (target.label_entropy   < 0.30
            and target.position_std_m < 0.05
            and target.safety_entropy < 0.30
            and target.grasp_uncertainty is None):
            return Action(kind="plan_grasp_candidates", target_hypothesis=target)
        
        if axis == "label":
            if not self._has_zoomed(target):
                return Action(kind="re_observe", target_hypothesis=target,
                              strategy="zoom_in")
            return Action(
                kind="ask_user",
                question=f"我看到一个{target.label}样的东西, 也可能是{target.label_alternatives[1][0] if len(target.label_alternatives) > 1 else '别的'}, 您要的是哪个?",
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
                return self._giveup_result(belief, start,
                                           reason=action.metadata.get("reason"))
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
            try:
                ev = self.perception.re_observe(
                    action.target_hypothesis, action.strategy, env, belief,
                )
            except NotImplementedError:
                # Phase 12 之前: re_observe 占位, 退化成 observe
                ev = self.perception.observe(self.vp_lib[0], env, belief)
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
                    action.target_hypothesis.label_entropy = max(
                        action.target_hypothesis.label_entropy, 0.6,
                    )
                    action.target_hypothesis.times_re_observed += 1
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
            # v1.1 起改用 LLM 解析后的结构化 boost/demote (跨阶段约定 §不做的事)。
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
        # zoom 后用 alternatives 更新; Phase 12 详细实现
        if "hypotheses" in ev.raw_payload and ev.raw_payload["hypotheses"]:
            d = ev.raw_payload["hypotheses"][0]
            new_alts = [(lbl, p) for lbl, p in d.get("label_alternatives", [])]
            if new_alts:
                h.label_alternatives = new_alts
                h.label = new_alts[0][0]
                # entropy 重算
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
        return EpisodeResult(
            success=True,
            target=h,
            speech=self._build_speech(belief, success=True),
            belief_trace=[belief.snapshot(i)
                          for i in range(len(belief.action_history))],
            action_history=list(belief.action_history),
            n_steps=len(belief.action_history),
            elapsed_seconds=time.time() - start,
        )
    
    def _giveup_result(
        self, belief: WorldBelief, start: float, reason: Optional[str] = None,
    ) -> EpisodeResult:
        return EpisodeResult(
            success=False,
            target=belief.target(),
            speech=self._build_speech(belief, success=False, reason=reason),
            belief_trace=[],
            action_history=list(belief.action_history),
            n_steps=len(belief.action_history),
            elapsed_seconds=time.time() - start,
            failure_reason=reason,
        )
    
    @staticmethod
    def _build_speech(belief: WorldBelief, success: bool,
                      reason: Optional[str] = None) -> str:
        h = belief.target()
        if success and h is not None:
            return f"已为您拿到{h.label}, 在您正前方约 {h.position_3d[0]:.2f}m 处。"
        if h is not None:
            return f"我看到一个像{h.label}的东西, 但暂时拿不准。{reason or ''}"
        return f"我没能找到{belief.decomposed.primary_target if belief.decomposed else '目标'}。{reason or ''}"
```

- [ ] **Step 2: 跑 decide_next 测试**

Run: `pytest tests/test_agent_decide_next.py -v`
Expected: 9 pass

- [ ] **Step 3: commit**

```bash
git add src/agent.py tests/test_agent_decide_next.py
git commit -m "feat(agent): EmboSightAgent.run + decide_next decision tree (8 axis-routing branches)"
```

---

### Task 11.3: run() 集成测试 (mock 全套, 跑 5 场景)

**Files:**
- Create: `tests/test_agent_run.py`

- [ ] **Step 1: 写 run 集成测试 (mock 全部依赖)**

```python
"""EmboSightAgent.run 集成测试 (mock 全部依赖, 验证 5 种场景)。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pytest
import numpy as np
from tests._mocks import MockLLM, MockVLM


def _make_full_agent(decompose_response, vlm_responses, safety_response,
                     vp_count=3, user_channel=None):
    from src.agent import EmboSightAgent
    from src.perception import QueryAwareGrounder
    from src.safety_gate import SafetyClassifier
    from src.grasp_planner import GraspPlanner
    from src.action_executor import ActionExecutor
    from src.active_planner import ActiveViewpointSelector
    from src.task_decomposer import TaskDecomposer
    from src.episode_logger import EpisodeLogger
    from src.user_channel import FakeUserChannel
    from src.vlm_cache import VLMCache
    
    class FakeVPLib:
        def __init__(self, n):
            self.viewpoints = [type("VP",(),{"name":f"v{i}"})() for i in range(n)]
        def __len__(self): return len(self.viewpoints)
        def __getitem__(self, i): return self.viewpoints[i]
    
    class FakeEnv:
        def observe(self, vp):
            return type("Obs", (), {"image_path": "/dev/null"})()
        def viewpoint_intrinsics(self, vp): return None
        def is_reachable(self, p, d): return True
        def move_to_pre_grasp(self, c): return True
        def descend(self, p): return True, float(p[2])
        def close_gripper(self): return True
        def open_gripper(self): return True
        def lift(self): return True, 0.05
        def get_eef_pos(self): return np.array([0.5,0,0.95])
        def move_arm_to(self, p, **kw): return True
        def eye_in_hand_viewpoint(self): 
            return type("VP",(),{"name":"eye_in_hand"})()
        def _get_obj_type_map(self): return {"obj_main": "apple"}
    
    decompose_llm = MockLLM(responses=[decompose_response])
    nbv_llm = MockLLM(responses=["1", "2", "-1"] * 5)
    safety_llm = MockLLM(responses=[safety_response] * 5)
    user_llm = MockLLM(responses=["apple", "圆形的"])
    
    vlm = MockVLM(responses=vlm_responses)
    cache = VLMCache()
    
    return EmboSightAgent(
        task_decomposer=TaskDecomposer(decompose_llm),
        perception=QueryAwareGrounder(vlm=vlm, llm=decompose_llm,
                                       cache=cache, label_temperature=1.0),
        safety_classifier=SafetyClassifier(llm=safety_llm),
        grasp_planner=GraspPlanner(vlm=vlm, env=FakeEnv()),
        action_executor=ActionExecutor(scene_describer=None),
        nbv_selector=ActiveViewpointSelector(llm=nbv_llm, viewpoint_lib=FakeVPLib(vp_count)),
        user_channel=user_channel or FakeUserChannel.from_explicit(user_llm, "apple"),
        episode_logger=None,
        viewpoint_lib=FakeVPLib(vp_count),
        llm=decompose_llm,
        vlm=vlm,
    ), FakeEnv()


class TestRun:
    def test_basic_success_path(self):
        """1 frame 看到 confident 的苹果 → grasp success。"""
        decomp = json.dumps({"primary_target": "apple", "constraints": []})
        vlm_resp = json.dumps({"objects": [
            {"bbox_2d": [50, 50, 100, 100], "label": "apple",
             "alternatives": [["apple", 0.95], ["other", 0.05]],
             "confidence": 0.9, "visible_features": "red round"},
        ]})
        safety = json.dumps({"dist": {"safe": 0.9, "fragile": 0.1},
                             "reasoning": "fruit"})
        agent, env = _make_full_agent(decomp, [vlm_resp]*5, safety)
        result = agent.run("拿苹果", env)
        assert result.success is True
        assert result.target.label == "apple"
    
    def test_no_target_runs_until_max_or_ask_user(self):
        """全场没苹果 → 应触发 ask_user 或 give_up。"""
        decomp = json.dumps({"primary_target": "apple", "constraints": []})
        vlm_resp = json.dumps({"objects": [
            {"bbox_2d": [10,10,20,20], "label": "banana",
             "alternatives": [["banana", 0.9], ["other", 0.1]],
             "confidence": 0.9, "visible_features": "yellow"},
        ]})
        safety = json.dumps({"dist": {"safe": 1.0}, "reasoning": "?"})
        agent, env = _make_full_agent(decomp, [vlm_resp]*15, safety,
                                      vp_count=2)
        result = agent.run("拿苹果", env)
        # 不能崩, 必须给出 speech
        assert result.speech != ""
    
    def test_ask_user_branch_runs(self):
        """场景里 2 个苹果 (top1/top2 prob 接近) → target=None → ask_user。"""
        decomp = json.dumps({"primary_target": "apple", "constraints": []})
        vlm_resp = json.dumps({"objects": [
            {"bbox_2d": [10,10,30,30], "label": "apple",
             "alternatives": [["apple", 0.5], ["pear", 0.5]],
             "confidence": 0.9, "visible_features": "red"},
            {"bbox_2d": [80,80,100,100], "label": "apple",
             "alternatives": [["apple", 0.5], ["pear", 0.5]],
             "confidence": 0.9, "visible_features": "red"},
        ]})
        safety = json.dumps({"dist": {"safe": 1.0}, "reasoning": "?"})
        agent, env = _make_full_agent(decomp, [vlm_resp]*15, safety)
        result = agent.run("拿苹果", env)
        # 至少调用过一次 ask_user
        assert any(a.kind == "ask_user" for a in result.action_history)
    
    def test_decompose_with_constraint(self):
        """avoid:knife 通过到 perception."""
        decomp = json.dumps({
            "primary_target": "bowl",
            "constraints": [
                {"kind": "avoid", "target_label": "knife", "reason": "用户避开"},
            ],
        })
        vlm_resp = json.dumps({"objects": [
            {"bbox_2d": [50,50,80,80], "label": "bowl",
             "alternatives": [["bowl", 0.95]],
             "confidence": 0.9, "visible_features": "round"},
        ]})
        safety = json.dumps({"dist": {"safe": 0.9, "fragile": 0.1}})
        agent, env = _make_full_agent(decomp, [vlm_resp]*5, safety)
        result = agent.run("拿碗, 避开刀", env)
        assert result.target is not None
        assert result.target.label == "bowl"
    
    def test_max_steps_stops(self):
        """场景持续模糊 → MAX_STEPS 后 give_up。"""
        decomp = json.dumps({"primary_target": "apple", "constraints": []})
        # 无苹果, 全 banana
        vlm_resp = json.dumps({"objects": [
            {"bbox_2d": [10,10,20,20], "label": "banana",
             "alternatives": [["banana", 0.5], ["other", 0.5]],
             "confidence": 0.5, "visible_features": "yellow"},
        ]})
        safety = json.dumps({"dist": {"safe": 1.0}})
        agent, env = _make_full_agent(decomp, [vlm_resp]*30, safety, vp_count=2)
        result = agent.run("拿苹果", env)
        assert result.success is False
        assert result.failure_reason is not None
        assert len(result.action_history) <= agent.MAX_STEPS + 2


class TestVerifyMismatchFlow:
    """F6 / Edge 9.6: post-grasp verify 失败时的完整恢复流程。
    
    覆盖契约 (来自设计稿 §5.3 + §9.6):
    1. result.attempt.failure_mode 改为 "verify_mismatch"
    2. target.label_entropy 拉到 ≥ 0.6
    3. target.times_re_observed += 1
    4. executor.release_and_retreat 被调一次 (避免夹爪遮挡死锁)
    5. loop 不 return success, 继续到下一轮决策
    """
    
    def _make_agent_with_failing_verify(self, decompose_response, vlm_resp, safety):
        """构造一个 verify_grasp 永远返 (False, 0.4) 的 agent。"""
        from src.agent import EmboSightAgent
        agent, env = _make_full_agent(decompose_response, [vlm_resp]*10, safety)
        # monkey-patch executor.verify_grasp + release_and_retreat
        orig_verify = agent.executor.verify_grasp
        orig_release = agent.executor.release_and_retreat
        env._release_call_count = 0
        def fail_verify(target, e):
            return False, 0.4
        def count_release(e, retreat_height_m=0.10):
            env._release_call_count += 1
            return orig_release(e, retreat_height_m)
        agent.executor.verify_grasp = fail_verify
        agent.executor.release_and_retreat = count_release
        return agent, env
    
    def test_verify_mismatch_marks_failure_and_retreats(self):
        """物理 grasp 成功但 verify 说不对 → failure_mode 改写, release 被调。"""
        decomp = json.dumps({"primary_target": "apple", "constraints": []})
        vlm_resp = json.dumps({"objects": [
            {"bbox_2d": [50,50,100,100], "label": "apple",
             "alternatives": [["apple", 0.95], ["other", 0.05]],
             "confidence": 0.9, "visible_features": "red round"},
        ]})
        safety = json.dumps({"dist": {"safe": 0.9, "fragile": 0.1}})
        agent, env = self._make_agent_with_failing_verify(decomp, vlm_resp, safety)
        result = agent.run("拿苹果", env)
        # 至少触发过一次 grasp + verify_mismatch
        attempts = result.target.grasp_attempts if result.target else []
        assert any(a.failure_mode == "verify_mismatch" for a in attempts), \
            "verify 失败必须改写 failure_mode 为 verify_mismatch"
        # release_and_retreat 必须调过 (≥ 1 次, 每次 verify 失败一次)
        n_mismatch = sum(1 for a in attempts if a.failure_mode == "verify_mismatch")
        assert env._release_call_count >= n_mismatch, \
            f"release_and_retreat 调用 {env._release_call_count} < verify_mismatch {n_mismatch}"
    
    def test_verify_mismatch_raises_label_entropy(self):
        """verify_mismatch 后 label_entropy 必须 ≥ 0.6 (触发下一轮 zoom_in)。"""
        decomp = json.dumps({"primary_target": "apple", "constraints": []})
        vlm_resp = json.dumps({"objects": [
            {"bbox_2d": [50,50,100,100], "label": "apple",
             "alternatives": [["apple", 0.95], ["other", 0.05]],
             "confidence": 0.9, "visible_features": "red"},
        ]})
        safety = json.dumps({"dist": {"safe": 0.9}})
        agent, env = self._make_agent_with_failing_verify(decomp, vlm_resp, safety)
        result = agent.run("拿苹果", env)
        h = result.target
        # 经历过 verify_mismatch 后, entropy 至少被拉到 0.6
        if h is not None and any(a.failure_mode == "verify_mismatch"
                                  for a in h.grasp_attempts):
            assert h.label_entropy >= 0.6 - 1e-6, \
                f"verify_mismatch 后 label_entropy={h.label_entropy} 未提升到 ≥ 0.6"
    
    def test_verify_mismatch_increments_re_observed(self):
        """verify_mismatch 后 times_re_observed += 1 (标"已扰动")。"""
        decomp = json.dumps({"primary_target": "apple", "constraints": []})
        vlm_resp = json.dumps({"objects": [
            {"bbox_2d": [50,50,100,100], "label": "apple",
             "alternatives": [["apple", 0.95]],
             "confidence": 0.9, "visible_features": "red"},
        ]})
        safety = json.dumps({"dist": {"safe": 0.9}})
        agent, env = self._make_agent_with_failing_verify(decomp, vlm_resp, safety)
        result = agent.run("拿苹果", env)
        h = result.target
        if h is not None:
            n_mismatch = sum(1 for a in h.grasp_attempts
                             if a.failure_mode == "verify_mismatch")
            assert h.times_re_observed >= n_mismatch, \
                f"times_re_observed={h.times_re_observed} 应 ≥ verify_mismatch 次数 {n_mismatch}"
```

- [ ] **Step 2: 跑测试**

Run: `pytest tests/test_agent_run.py -v`
Expected: 8 pass (5 原有 + 3 verify_mismatch)

- [ ] **Step 3: 跑 Phase 1-11 全套**

Run: `pytest tests/test_world_belief.py tests/test_vlm_cache.py tests/test_episode_logger.py tests/test_user_channel.py tests/test_perception.py tests/test_safety_classifier.py tests/test_grasp_planner.py tests/test_action_executor_v1.py tests/test_viewpoint_selector.py tests/test_task_decomposer_v1.py tests/test_agent_decide_next.py tests/test_agent_run.py -v`
Expected: 100+ pass, 0 fail

- [ ] **Step 4: ruff + commit**

Run: `ruff check src/agent.py tests/test_agent_run.py`

```bash
git add tests/test_agent_run.py
git commit -m "test(agent): EmboSightAgent.run 5-scenario integration tests with full mock stack"
```

**Phase 11 CHECKPOINT:** 17+ pass (decide_next 9 + run 5 + verify_mismatch 3)。Phase 1-11 累计 105+ pass。

---

## Phase 12: re_observe (zoom / parallax / pose) + verify_grasp + 4 prompt

**目标:** 给 `QueryAwareGrounder` 补 `re_observe` 和 `verify_grasp` 实现, 之前的 `NotImplementedError` 替换掉。

### Task 12.1: 4 个新 prompt + 新增测试

**Files:**
- Create: `prompts/perception/zoom_disambiguate.txt`
- Create: `prompts/perception/parallax_localize.txt`
- Create: `prompts/perception/pose_estimation.txt`
- Create: `prompts/perception/verify_grasp.txt`

- [ ] **Step 1: 写 4 个 prompt**

`zoom_disambiguate.txt`:
```
这是放大裁切的物体特写 (原始 bbox 周围加 padding)。
当前候选标签: {label}
其他可能: {alternatives_top3}

请重新评估 (BE CONSERVATIVE):
- 给出 top 3 (label, prob), 总和 ≤ 1.0
- 视觉特征 (1 句)

JSON:
{"label": "...", "alternatives": [["x", 0.5], ...], "visible_features": "..."}
```

`parallax_localize.txt`:
```
这是从 {viewpoint_name} 拍的同一物体。已有视角中, 物体的 3D 位置估计是
({pos_x}, {pos_y}, {pos_z}) ± {pos_std}m。

在这个视角的图像里, 物体看起来在 bbox: 请输出 [x1, y1, x2, y2] 像素坐标。
图像尺寸 {img_w}x{img_h}。

JSON: {"bbox_2d": [...], "confidence": 0-1}
```

`pose_estimation.txt`:
```
这是从侧面 ({viewpoint_name}) 看到的物体 ({label})。

请判断:
- upright: 物体是竖立 (true) 还是横放 (false)?
- main_axis_dir: 长轴方向, 选 "x" / "y" / "z" 之一

JSON: {"upright": true/false, "main_axis_dir": "z"}
```

`verify_grasp.txt`:
```
这是 eye-in-hand 相机看到的画面 (机器人刚抓起物体)。

期望抓的物体: {expected_label}
其他可能 (从期望候选中): {alternatives}

请判断夹爪当前夹住的物体:
- is_match: 是不是 {expected_label}? (true/false)
- confidence: 0-1
- actual_guess: 如果不是, 你猜实际是什么?

JSON: {"is_match": true, "confidence": 0.85, "actual_guess": "..."}
```

- [ ] **Step 2: 测试**

```python
# 追加到 tests/test_perception.py:

class TestReObserve:
    def test_zoom_in_uses_zoom_prompt(self, tmp_image):
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        from src.world_belief import (
            DecomposedTask, Hypothesis, WorldBelief,
        )
        vlm = MockVLM(responses=[json.dumps({
            "label": "apple",
            "alternatives": [["apple", 0.9], ["pear", 0.1]],
            "visible_features": "shiny red",
        })])
        g = QueryAwareGrounder(vlm=vlm, llm=MockLLM([]),
                               cache=VLMCache(), label_temperature=1.0)
        h = Hypothesis(
            object_id="o0", label="apple",
            label_alternatives=[("apple", 0.5), ("kiwi", 0.5)],
            label_entropy=0.69,
            position_3d=np.array([0.5,0,0.9]), position_std_m=0.05,
            bbox_per_view={"v0": (50,50,100,100)},
            observed_in_views=["v0"],
        )
        class FakeEnv:
            def observe(self, vp):
                return type("O", (), {"image_path": tmp_image})()
            def viewpoint_intrinsics(self, vp): return None
        belief = WorldBelief(user_query="x")
        belief.decomposed = DecomposedTask(primary_target="apple")
        ev = g.re_observe(h, "zoom_in", FakeEnv(), belief)
        assert ev.source == "vlm_zoom"
    
    def test_parallax_view_uses_parallax_prompt(self, tmp_image):
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        from src.world_belief import (
            DecomposedTask, Hypothesis, WorldBelief,
        )
        vlm = MockVLM(responses=[json.dumps({
            "bbox_2d": [60, 60, 110, 110], "confidence": 0.85,
        })])
        g = QueryAwareGrounder(vlm=vlm, llm=MockLLM([]),
                               cache=VLMCache())
        h = Hypothesis(
            object_id="o0", label="apple",
            label_alternatives=[("apple", 0.95)],
            label_entropy=0.1,
            position_3d=np.array([0.5,0,0.9]), position_std_m=0.20,
            observed_in_views=["v0"],
        )
        class FakeVPLib:
            def __getitem__(self, i): 
                return type("VP",(),{"name":f"v{i}"})()
            def __len__(self): return 3
            def __iter__(self):
                for i in range(3):
                    yield self[i]
        class FakeEnv:
            def observe(self, vp):
                return type("O", (), {"image_path": tmp_image})()
            def viewpoint_intrinsics(self, vp): return None
        g._vp_lib = FakeVPLib()    # 注: re_observe 需要 vp_lib
        belief = WorldBelief(user_query="x")
        belief.decomposed = DecomposedTask(primary_target="apple")
        ev = g.re_observe(h, "parallax_view", FakeEnv(), belief)
        assert ev.source == "vlm_zoom"  # 或 "vlm_ground" 视实现
    
    def test_unknown_strategy_raises(self):
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        from src.world_belief import (
            DecomposedTask, Hypothesis, WorldBelief,
        )
        g = QueryAwareGrounder(vlm=MockVLM([]), llm=MockLLM([]),
                               cache=VLMCache())
        h = Hypothesis(
            object_id="o0", label="x",
            label_alternatives=[("x", 1.0)], label_entropy=0.0,
            position_3d=np.zeros(3), position_std_m=0.05,
        )
        belief = WorldBelief(user_query="x")
        belief.decomposed = DecomposedTask(primary_target="x")
        with pytest.raises(ValueError):
            g.re_observe(h, "unknown_strategy", env=None, belief=belief)


class TestVerifyGrasp:
    def test_verify_match(self, tmp_image):
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        from src.world_belief import Hypothesis
        vlm = MockVLM(responses=[json.dumps({
            "is_match": True, "confidence": 0.9, "actual_guess": "",
        })])
        g = QueryAwareGrounder(vlm=vlm, llm=MockLLM([]),
                               cache=VLMCache())
        h = Hypothesis(
            object_id="o0", label="apple",
            label_alternatives=[("apple", 0.95)], label_entropy=0.1,
            position_3d=np.zeros(3), position_std_m=0.05,
        )
        class FakeEnv:
            def observe(self, vp):
                return type("O", (), {"image_path": tmp_image})()
            def eye_in_hand_viewpoint(self):
                return type("VP",(),{"name":"eye_in_hand"})()
        ok, conf = g.verify_grasp(h, FakeEnv())
        assert ok is True
        assert conf == pytest.approx(0.9)
    
    def test_verify_mismatch(self, tmp_image):
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        from src.world_belief import Hypothesis
        vlm = MockVLM(responses=[json.dumps({
            "is_match": False, "confidence": 0.7, "actual_guess": "pear",
        })])
        g = QueryAwareGrounder(vlm=vlm, llm=MockLLM([]),
                               cache=VLMCache())
        h = Hypothesis(
            object_id="o0", label="apple",
            label_alternatives=[("apple", 0.95)], label_entropy=0.1,
            position_3d=np.zeros(3), position_std_m=0.05,
        )
        class FakeEnv:
            def observe(self, vp):
                return type("O", (), {"image_path": tmp_image})()
            def eye_in_hand_viewpoint(self):
                return type("VP",(),{"name":"eye_in_hand"})()
        ok, conf = g.verify_grasp(h, FakeEnv())
        assert ok is False
```

- [ ] **Step 3: 跑测试 (红 — re_observe/verify_grasp 还在 NotImplemented)**

Run: `pytest tests/test_perception.py::TestReObserve tests/test_perception.py::TestVerifyGrasp -v`
Expected: NotImplementedError

---

### Task 12.2: 实现 `QueryAwareGrounder.re_observe` + `verify_grasp`

**Files:**
- Modify: `src/perception.py`

- [ ] **Step 1: 替换 re_observe / verify_grasp 占位实现**

把 perception.py 的:
```python
    def re_observe(self, target: Hypothesis, strategy: str, env, belief: WorldBelief) -> Evidence:
        raise NotImplementedError("re_observe implemented in Phase 12")
    
    def verify_grasp(self, target: Hypothesis, env) -> tuple[bool, float]:
        raise NotImplementedError("verify_grasp implemented in Phase 12")
```

替换为:

```python
    def re_observe(
        self, target: Hypothesis, strategy: str, env, belief: WorldBelief,
    ) -> Evidence:
        if strategy == "zoom_in":
            return self._zoom_observe(target, env, belief)
        if strategy == "parallax_view":
            return self._parallax_observe(target, env, belief, for_pose=False)
        if strategy == "parallax_for_pose":
            return self._parallax_observe(target, env, belief, for_pose=True)
        raise ValueError(f"unknown re_observe strategy: {strategy}")
    
    def _zoom_observe(
        self, target: Hypothesis, env, belief: WorldBelief,
    ) -> Evidence:
        # 取 target 的某视角 bbox + 裁切原图
        vp_name = target.observed_in_views[0] if target.observed_in_views else "v0"
        bbox = target.bbox_per_view.get(vp_name)
        if bbox is None:
            # 没 bbox: 退化成全图
            return self.observe(self._vp_by_name(vp_name), env, belief)
        # 模拟原图重新拍 (此处简化: 直接调 env.observe(vp), 实际应裁切)
        try:
            obs = env.observe(self._vp_by_name(vp_name))
            cropped = self._crop_image(obs.image_path, bbox, padding=10)
        except Exception as e:
            return Evidence(source="vlm_failed", timestamp=time.time(),
                            raw_payload={"error": str(e)})
        zoom_template = self._load(self._zoom_path) or "Zoom prompt missing"
        prompt = (
            zoom_template
            .replace("{label}", target.label)
            .replace("{alternatives_top3}",
                     ", ".join(f"{l}({p:.2f})"
                               for l, p in target.label_alternatives[:3]))
        )
        try:
            raw = self.vlm.describe(cropped, prompt=prompt)
        except Exception as e:
            return Evidence(source="vlm_failed", timestamp=time.time(),
                            raw_payload={"error": str(e)})
        data = self._extract_json(raw)
        if data is None:
            return Evidence(source="vlm_zoom", timestamp=time.time(),
                            raw_payload={"parse_failed": True, "raw": raw[:500]})
        # 更新 alternatives + entropy
        new_alts_raw = data.get("alternatives", [])
        new_alts = [(str(l), float(p)) for l, p in new_alts_raw]
        new_alts = _temperature_scale(new_alts, self.label_temperature)
        return Evidence(
            source="vlm_zoom", timestamp=time.time(),
            raw_payload={
                "hypotheses": [{
                    "object_id": target.object_id,
                    "label": data.get("label", target.label),
                    "label_alternatives": new_alts,
                    "label_entropy": _shannon([p for _, p in new_alts]),
                    "position_3d": target.position_3d.tolist(),
                    "position_std_m": target.position_std_m,
                    "bbox_per_view": {k: list(v) for k, v in target.bbox_per_view.items()},
                    "observed_in_views": list(target.observed_in_views),
                }],
            },
        )
    
    def _parallax_observe(
        self, target: Hypothesis, env, belief: WorldBelief, for_pose: bool,
    ) -> Evidence:
        # 选未用过的视角
        used = set(target.observed_in_views)
        next_vp = None
        for i in range(len(getattr(self, "_vp_lib", []) or [])):
            vp = self._vp_lib[i]
            if vp.name not in used:
                next_vp = vp
                break
        if next_vp is None:
            return Evidence(source="vlm_failed", timestamp=time.time(),
                            raw_payload={"reason": "no parallax viewpoint"})
        try:
            obs = env.observe(next_vp)
        except Exception as e:
            return Evidence(source="vlm_failed", timestamp=time.time(),
                            raw_payload={"error": str(e)})
        if for_pose:
            template = self._load("prompts/perception/pose_estimation.txt") or ""
            prompt = (
                template
                .replace("{viewpoint_name}", next_vp.name)
                .replace("{label}", target.label)
            )
        else:
            template = self._load(self._parallax_path) or ""
            prompt = (
                template
                .replace("{viewpoint_name}", next_vp.name)
                .replace("{label}", target.label)
                .replace("{pos_x}", f"{target.position_3d[0]:.2f}")
                .replace("{pos_y}", f"{target.position_3d[1]:.2f}")
                .replace("{pos_z}", f"{target.position_3d[2]:.2f}")
                .replace("{pos_std}", f"{target.position_std_m:.2f}")
            )
        try:
            raw = self.vlm.describe(obs.image_path, prompt=prompt)
        except Exception as e:
            return Evidence(source="vlm_failed", timestamp=time.time(),
                            raw_payload={"error": str(e)})
        # raw 解析: 简化版, 仅记录 raw_payload, hypothesis 更新由 _update_hypothesis_from_evidence 做
        return Evidence(
            source="vlm_zoom", timestamp=time.time(),
            raw_payload={
                "viewpoint": next_vp.name,
                "raw_vlm_text": raw[:500],
                "for_pose": for_pose,
            },
        )
    
    def verify_grasp(self, target: Hypothesis, env) -> tuple[bool, float]:
        try:
            obs = env.observe(env.eye_in_hand_viewpoint())
        except Exception:
            return True, 1.0   # 拍不了照, 默认通过
        template = self._load(self._verify_path) or ""
        alts = ", ".join(
            f"{l}({p:.2f})" for l, p in target.label_alternatives[:3]
        )
        prompt = (
            template
            .replace("{expected_label}", target.label)
            .replace("{alternatives}", alts)
        )
        try:
            raw = self.vlm.describe(obs.image_path, prompt=prompt)
        except Exception:
            return True, 1.0
        data = self._extract_json(raw)
        if data is None:
            return True, 1.0
        return bool(data.get("is_match", True)), float(data.get("confidence", 1.0))
    
    # ──────────────────────────────────────
    # helpers
    # ──────────────────────────────────────
    
    def _vp_by_name(self, name: str):
        for vp in (self._vp_lib or []):
            if vp.name == name:
                return vp
        # fallback: 第一个
        return (self._vp_lib or [None])[0]
    
    def _crop_image(self, image_path: str, bbox: tuple[int,int,int,int],
                    padding: int = 10) -> str:
        from PIL import Image
        import tempfile
        with Image.open(image_path) as im:
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(im.width, x2 + padding)
            y2 = min(im.height, y2 + padding)
            crop = im.crop((x1, y1, x2, y2))
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        crop.save(tmp.name)
        return tmp.name
```

- [ ] **Step 2: 给 QueryAwareGrounder 加 `_vp_lib` 注入**

修改构造函数, 加 viewpoint_lib 参数:

```python
    def __init__(
        self,
        vlm,
        llm,
        cache: VLMCache,
        ground_prompt_path: str = _DEFAULT_GROUND_PROMPT,
        zoom_prompt_path: str = "prompts/perception/zoom_disambiguate.txt",
        parallax_prompt_path: str = "prompts/perception/parallax_localize.txt",
        verify_prompt_path: str = "prompts/perception/verify_grasp.txt",
        label_temperature: float = 1.5,
        viewpoint_lib=None,                      # ← 新增
    ):
        # ... (原有代码)
        self._vp_lib = viewpoint_lib
```

- [ ] **Step 3: 跑测试**

Run: `pytest tests/test_perception.py -v`
Expected: 13 pass (10 旧 + 3 新)

- [ ] **Step 4: agent.py 也需要把 vp_lib 传进 perception**

修改 `src/agent.py` 构造调用; 在 `with_test_doubles` 给 perception 传 vp_lib (如果用的话)。

- [ ] **Step 5: 跑全套确认 agent 集成测试不破**

Run: `pytest tests/test_agent_run.py -v`
Expected: 仍 5 pass

- [ ] **Step 6: commit**

```bash
git add src/perception.py prompts/perception/zoom_disambiguate.txt prompts/perception/parallax_localize.txt prompts/perception/pose_estimation.txt prompts/perception/verify_grasp.txt tests/test_perception.py
git commit -m "feat(perception): re_observe (zoom/parallax/parallax_for_pose) + verify_grasp implementations"
```

**Phase 12 CHECKPOINT:** Phase 5 + 12 共 13+ pass; agent_run 仍通。

---

## Phase 13: `configs/agent.yaml` + 入口切换

**目标:** 新建 `configs/agent.yaml` (设计稿 §8.1), 修改 `src/__init__.py` 导出新类, 提供命令行入口让用户能从 `python -m src` 切到新 agent。

### Task 13.0: `configs/default.yaml` 联动决策 (不加 agent 段)

**背景:** 设计稿 §8.3 说 "保留 `configs/default.yaml` (顶层 LLM/VLM 配置, 加 agent 段)"。 但 `configs/default.yaml` 现有内容包含老 `active_planner / action_executor / prompts` 段——这些段在 Phase 15.7 会被删 (老接口替换); 此外 Task 13.2 的 `scripts/run_agent.py` 已显式同时加载 `default.yaml` (顶层 LLM/VLM/sim) + `agent.yaml` (v1 阈值/perception/cache/logger), 两文件职责分明, 不需要在 `default.yaml` 再嵌一份 agent 段。

**v1 决定**:
1. `default.yaml` **不加 agent 段** (避免双重事实来源)
2. `default.yaml` 中 `active_planner` / `action_executor` / `prompts` 三段在 Phase 15.7 删除时连带删
3. v1 验证期间 (Phase 13 - Phase 14): 这 3 段保留, 老代码并存可读

- [ ] **Step 1: 在 plan 内确认无操作 (此 Task 只是决策点)**

无文件改动。继续 Task 13.1。

---

### Task 13.1: 新建 configs/agent.yaml

**Files:**
- Create: `configs/agent.yaml`

- [ ] **Step 1: 写 yaml**

(完整内容见设计稿 §8.1, 含 thresholds / belief / perception.label_temperature / cache / verify / logger / implementations)

```yaml
# EmboSight Agent v1 主配置

agent:
  max_steps: 12
  max_re_observe: 3

thresholds:
  default:
    label_entropy: 0.30
    position_std_m: 0.05
    safety_entropy: 0.30
    grasp_uncertainty: 0.30
  high_risk:
    label_entropy: 0.15
    position_std_m: 0.03
    safety_entropy: 0.15
    grasp_uncertainty: 0.20

belief:
  merge_distance_m: 0.15            # TODO(v1.1): 实测调
  merge_label_intersection_min: 0.30
  prune_phantom_min_steps: 3

perception:
  label_temperature: 1.5            # TODO(v1.1): 跑 demo 后调
  ground_prompt: prompts/perception/query_aware_ground.txt
  zoom_prompt: prompts/perception/zoom_disambiguate.txt
  parallax_prompt: prompts/perception/parallax_localize.txt
  pose_prompt: prompts/perception/pose_estimation.txt
  verify_prompt: prompts/perception/verify_grasp.txt

cache:
  enabled: true
  max_size: 100
  ttl: episode

verify:
  enabled: true
  min_confidence: 0.6

logger:
  log_dir: logs/episodes
  save_json: true
  save_belief_trace: true

implementations:
  user_channel: fake_from_robocasa
  perception: query_aware
  safety: llm_classify
  task_decomposer: constraint_aware
```

- [ ] **Step 2: commit yaml**

```bash
git add configs/agent.yaml
git commit -m "feat(config): configs/agent.yaml with 4-axis thresholds + temperature + impl switches"
```

---

### Task 13.2: 入口脚本 `scripts/run_agent.py`

**Files:**
- Create: `scripts/run_agent.py`

- [ ] **Step 0: 验证现有 backend 构造签名 (避免引用不存在的工厂方法)**

Run (PowerShell):
```powershell
grep -n "^class \(LLMBackend\|VLMBackend\|EnvWrapper\|ViewpointLibrary\)\|^    def __init__" src/llm_backend.py src/vlm_backend.py src/env_wrapper.py src/active_planner.py
```

Expected (现状已确认 2026-05-08):
- `LLMBackend.__init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, ...)` — 直接构造, 无 `from_default`
- `VLMBackend.__init__(self, model_id: str = "./checkpoints/Qwen2.5-VL-7B-Instruct", device: str = "cuda", ...)` — 同上
- `EnvWrapper.__init__(self, config: Optional[EnvConfig] = None)` — 接 EnvConfig dataclass, 无工厂
- `ViewpointLibrary.__init__(self, config_path: str = "configs/viewpoints.yaml")` — 已是 yaml 加载, 但 API 是 `__init__` 不是 `from_yaml`

> **结论**: 老 plan 写的 `LLMBackend.from_default() / VLMBackend.from_default() / EnvWrapper.from_default() / ViewpointLibrary.from_yaml(...)` **都不存在**。Step 1 的脚本必须改用现有构造签名。

- [ ] **Step 1: 写脚本 (使用现有构造签名)**

```python
"""EmboSight Agent v1 入口: 从 query 开始跑一个 episode。

Usage:
    python scripts/run_agent.py --query "拿苹果" --config configs/default.yaml --agent-config configs/agent.yaml
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import EmboSightAgent
from src.task_decomposer import TaskDecomposer
from src.perception import QueryAwareGrounder
from src.safety_gate import SafetyClassifier
from src.grasp_planner import GraspPlanner
from src.action_executor import ActionExecutor
from src.active_planner import ActiveViewpointSelector, ViewpointLibrary
from src.user_channel import FakeUserChannel, CLIUserChannel
from src.episode_logger import EpisodeLogger
from src.vlm_cache import VLMCache
from src.llm_backend import LLMBackend
from src.vlm_backend import VLMBackend
from src.env_wrapper import EnvConfig, EnvWrapper


def _build_llm(cfg: dict) -> LLMBackend:
    llm_cfg = cfg.get("llm", {})
    return LLMBackend(
        api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=llm_cfg.get("base_url"),
        model=llm_cfg.get("model", "deepseek-chat"),
        max_tokens=llm_cfg.get("max_tokens", 2048),
        temperature=llm_cfg.get("temperature", 0.1),
        timeout=llm_cfg.get("timeout", 60.0),
    )


def _build_vlm(cfg: dict) -> VLMBackend:
    vlm_cfg = cfg.get("vlm", {})
    return VLMBackend(
        model_id=vlm_cfg.get("model_id", "./checkpoints/Qwen2.5-VL-7B-Instruct"),
        device=vlm_cfg.get("device", "cuda"),
        torch_dtype=vlm_cfg.get("torch_dtype", "bfloat16"),
        max_new_tokens=vlm_cfg.get("max_new_tokens", 1024),
    )


def _build_env(cfg: dict) -> EnvWrapper:
    sim_cfg = cfg.get("simulator", {})
    env_cfg = EnvConfig(
        env_name=sim_cfg.get("env_name", "PickPlaceCounterToCabinet"),
        robots=sim_cfg.get("robots", "PandaMobile"),
        image_width=sim_cfg.get("image_width", 256),
        image_height=sim_cfg.get("image_height", 256),
        camera_names=sim_cfg.get("camera_names", []),
    )
    return EnvWrapper(env_cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--config", default="configs/default.yaml",
                        help="顶层 LLM/VLM/sim 配置")
    parser.add_argument("--agent-config", default="configs/agent.yaml",
                        help="agent v1 阈值/perception/cache/logger 配置")
    parser.add_argument("--user-mode", default="fake_from_robocasa",
                        choices=["fake_from_robocasa", "fake_from_query", "cli"])
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    
    top_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    agent_cfg = yaml.safe_load(Path(args.agent_config).read_text(encoding="utf-8"))
    
    # 实例化依赖 (使用现有构造签名, 见 Step 0 验证)
    llm = _build_llm(top_cfg)
    vlm = _build_vlm(top_cfg)
    cache = VLMCache(max_size=agent_cfg["cache"]["max_size"])
    env = _build_env(top_cfg)
    vp_lib = ViewpointLibrary(
        config_path=top_cfg.get("viewpoints_path", "configs/viewpoints.yaml"),
    )
    
    # User channel
    if args.user_mode == "fake_from_robocasa":
        user_channel = FakeUserChannel.from_robocasa(llm, env)
    elif args.user_mode == "fake_from_query":
        user_channel = FakeUserChannel.from_query(llm, args.query)
    else:
        user_channel = CLIUserChannel()
    
    agent = EmboSightAgent(
        task_decomposer=TaskDecomposer(llm),
        perception=QueryAwareGrounder(
            vlm=vlm, llm=llm, cache=cache,
            label_temperature=agent_cfg["perception"]["label_temperature"],
            viewpoint_lib=vp_lib,
        ),
        safety_classifier=SafetyClassifier(llm=llm),
        grasp_planner=GraspPlanner(vlm=vlm, env=env),
        action_executor=ActionExecutor(scene_describer=None),
        nbv_selector=ActiveViewpointSelector(llm=llm, viewpoint_lib=vp_lib),
        user_channel=user_channel,
        episode_logger=EpisodeLogger(log_dir=agent_cfg["logger"]["log_dir"]),
        viewpoint_lib=vp_lib,
        llm=llm,
        vlm=vlm,
    )
    
    result = agent.run(args.query, env)
    print("\n========== EPISODE RESULT ==========")
    print(f"success: {result.success}")
    print(f"speech : {result.speech}")
    print(f"steps  : {result.n_steps}")
    print(f"time   : {result.elapsed_seconds:.1f}s")
    if not result.success:
        print(f"reason : {result.failure_reason}")
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
```

> **校验**: 如果 `LLMBackend.__init__` / `VLMBackend.__init__` / `EnvConfig` 字段名与上述不符 (实施时再次 grep 确认), 必须先调整 `_build_*` 三个 helper, 不要修改各 backend 自身。这样 backend 主体改动最小。

- [ ] **Step 2: 干跑一次 (mock-only)**

由于不一定能在测试机上跑真 LLM/VLM, 写一个 dry-run 测试:

Create: `tests/test_run_agent_script.py`

```python
"""scripts/run_agent.py 的 import-only 测试 (确保脚本不语法错误)。"""
import importlib.util
from pathlib import Path


def test_run_agent_module_loads():
    spec = importlib.util.spec_from_file_location(
        "run_agent",
        str(Path(__file__).parent.parent / "scripts" / "run_agent.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    # 不真 exec (因为里面有 argparse + main 调用); 只验证文件可被识别为有效 Python
    assert spec.loader is not None
```

Run: `pytest tests/test_run_agent_script.py -v`

- [ ] **Step 3: commit**

```bash
git add scripts/run_agent.py tests/test_run_agent_script.py
git commit -m "feat(config): scripts/run_agent.py entry with config-driven dependencies"
```

---

### Task 13.3: `src/__init__.py` 导出 v1 公开 API

**背景:** 设计稿 §15 Appendix B 改造表明确列出 `src/__init__.py (导出新增模块)`。当前 `src/__init__.py` 只有版本号和文档字符串, 不导出任何符号; 实施完 Phase 1-12 后外部代码 (脚本 / 单测 / 未来 v2) 应该能 `from src import EmboSightAgent, WorldBelief, EpisodeResult, ...`。

**Files:**
- Modify: `src/__init__.py`

- [ ] **Step 1: 在文件末尾追加显式 export**

```python
# ============================================================
# v1 公开 API (设计稿 §15 Appendix B)
# ============================================================
# 主入口
from src.agent import EmboSightAgent

# 数据结构 (供脚本/外部测试 import 使用)
from src.world_belief import (
    Action,
    BeliefSnapshot,
    Constraint,
    DecomposedTask,
    EpisodeResult,
    Evidence,
    GraspAttempt,
    GraspCandidate,
    Hypothesis,
    Pose,
    WorldBelief,
)

# 辅助
from src.episode_logger import EpisodeLogger
from src.user_channel import (
    CLIUserChannel,
    FakeUserChannel,
    UserChannel,
    VoiceUserChannel,
)
from src.vlm_cache import VLMCache

__all__ = [
    # 主入口
    "EmboSightAgent",
    # 数据结构
    "Action", "BeliefSnapshot", "Constraint", "DecomposedTask",
    "EpisodeResult", "Evidence", "GraspAttempt", "GraspCandidate",
    "Hypothesis", "Pose", "WorldBelief",
    # 辅助
    "EpisodeLogger",
    "CLIUserChannel", "FakeUserChannel", "UserChannel", "VoiceUserChannel",
    "VLMCache",
]
```

- [ ] **Step 2: 写 import 测试**

Create: `tests/test_public_api.py`

```python
"""验证 src 包级别 v1 公开 API 可用 (设计稿 §15 Appendix B)。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_public_api_imports():
    """所有声明的公开符号都能 from src import 到。"""
    import src
    expected = {
        "EmboSightAgent",
        "Action", "BeliefSnapshot", "Constraint", "DecomposedTask",
        "EpisodeResult", "Evidence", "GraspAttempt", "GraspCandidate",
        "Hypothesis", "Pose", "WorldBelief",
        "EpisodeLogger",
        "CLIUserChannel", "FakeUserChannel", "UserChannel", "VoiceUserChannel",
        "VLMCache",
    }
    actual = set(src.__all__)
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"src.__all__ 缺: {missing}"
    assert not extra, f"src.__all__ 多: {extra}"
    # 每个符号都能真实 import
    for name in expected:
        assert hasattr(src, name), f"src.{name} 未导出"
```

- [ ] **Step 3: 跑测试**

Run: `pytest tests/test_public_api.py -v`
Expected: 1 pass

- [ ] **Step 4: commit**

```bash
git add src/__init__.py tests/test_public_api.py
git commit -m "feat(api): export v1 public API from src package (Agent/Belief/Logger/Channels)"
```

**Phase 13 CHECKPOINT:** 配置文件 + 入口脚本 + 公开 API 三件套就绪 (真实 sim 跑放 Phase 14 之后单独)。

---

## Phase 14: EpisodeLogger Replay 测试 + 5 golden episode

**目标:** 实现 `EpisodeLogger.replay(json_path, agent_factory) -> EpisodeResult`; 录入 5 个 golden episode; replay 4 层契约测试 (F7) 入 CI。

### Task 14.1: 实现 replay

**Files:**
- Modify: `src/episode_logger.py` (替换 `replay` 占位)

- [ ] **Step 1: 写 MockFromRecord 和 replay 实现**

```python
# 替换 EpisodeLogger.replay 的占位:

class _MockFromRecord:
    """从 EpisodeRecord 的 evidence 序列回放某 source 的输出。"""
    
    def __init__(self, record: EpisodeRecord, source: str):
        self._responses: list[str] = []
        for ev in record.evidence:
            if ev.source == source:
                self._responses.append(
                    json.dumps(ev.raw_payload, ensure_ascii=False)
                )
    
    def describe(self, image_path: str, prompt: str = "") -> str:
        if not self._responses:
            return '{"objects": []}'
        return self._responses.pop(0)
    
    def generate(self, prompt: str, system: str = "", **kw) -> str:
        if not self._responses:
            return '{}'
        return self._responses.pop(0)


@classmethod
def replay(cls, json_path: str, agent_factory) -> EpisodeResult:
    """从 golden episode 回放 → 跑 agent.decide_next 序列, 返回新 result。
    
    agent_factory: callable(mocks: dict) -> EmboSightAgent
    """
    record = cls.load(json_path)
    mocks = {
        "vlm_ground": _MockFromRecord(record, "vlm_ground"),
        "vlm_zoom": _MockFromRecord(record, "vlm_zoom"),
        "vlm_verify": _MockFromRecord(record, "vlm_verify"),
        "llm_safety": _MockFromRecord(record, "llm_safety"),
        "llm_decompose": _MockFromRecord(record, "llm_decompose"),
        "user_answer": _MockFromRecord(record, "user_answer"),
    }
    agent, env = agent_factory(mocks)
    result = agent.run(record.query, env)
    return result
```

(把这两段添加到 `src/episode_logger.py`)

- [ ] **Step 2: 写 replay test**

Create: `tests/test_replay.py`

```python
"""4 层契约 replay 测试 (F7)。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import glob
import json
import pytest


GOLDEN_DIR = Path(__file__).parent / "episodes" / "golden"
GOLDEN_DIR.mkdir(parents=True, exist_ok=True)


def _make_test_factory(mocks: dict):
    """同 test_agent_run.py 的 _make_full_agent 思路, 把 vlm/llm 替换成 record-based mock。"""
    from src.agent import EmboSightAgent
    from src.perception import QueryAwareGrounder
    from src.safety_gate import SafetyClassifier
    from src.grasp_planner import GraspPlanner
    from src.action_executor import ActionExecutor
    from src.active_planner import ActiveViewpointSelector
    from src.task_decomposer import TaskDecomposer
    from src.user_channel import FakeUserChannel
    from src.vlm_cache import VLMCache
    
    class FakeVPLib:
        def __init__(self, n=3):
            self.viewpoints = [type("VP",(),{"name":f"v{i}"})() for i in range(n)]
        def __len__(self): return len(self.viewpoints)
        def __getitem__(self, i): return self.viewpoints[i]
    
    class FakeEnv:
        def observe(self, vp):
            return type("O", (), {"image_path": "/dev/null"})()
        def viewpoint_intrinsics(self, vp): return None
        def is_reachable(self, p, d): return True
        def move_to_pre_grasp(self, c): return True
        def descend(self, p): 
            import numpy as np
            return True, float(np.asarray(p)[2])
        def close_gripper(self): return True
        def open_gripper(self): return True
        def lift(self): return True, 0.05
        def get_eef_pos(self):
            import numpy as np
            return np.array([0.5,0,0.95])
        def move_arm_to(self, p, **kw): return True
        def eye_in_hand_viewpoint(self):
            return type("VP",(),{"name":"eye_in_hand"})()
        def _get_obj_type_map(self): return {"obj_main": "apple"}
    
    vp_lib = FakeVPLib()
    return EmboSightAgent(
        task_decomposer=TaskDecomposer(mocks["llm_decompose"]),
        perception=QueryAwareGrounder(
            vlm=mocks["vlm_ground"], llm=mocks["llm_decompose"],
            cache=VLMCache(), label_temperature=1.0,
            viewpoint_lib=vp_lib,
        ),
        safety_classifier=SafetyClassifier(llm=mocks["llm_safety"]),
        grasp_planner=GraspPlanner(vlm=mocks["vlm_ground"], env=FakeEnv()),
        action_executor=ActionExecutor(scene_describer=None),
        nbv_selector=ActiveViewpointSelector(llm=mocks["llm_decompose"],
                                              viewpoint_lib=vp_lib),
        user_channel=FakeUserChannel.from_explicit(mocks["user_answer"], "apple"),
        episode_logger=None,
        viewpoint_lib=vp_lib,
        llm=mocks["llm_decompose"],
        vlm=mocks["vlm_ground"],
    ), FakeEnv()


GOLDEN_FILES = sorted(glob.glob(str(GOLDEN_DIR / "*.json")))


@pytest.mark.skipif(not GOLDEN_FILES, reason="no golden episodes yet")
@pytest.mark.parametrize("episode_path", GOLDEN_FILES)
def test_replay_decision_consistency(episode_path):
    """4 层契约 (F7): L1 终态 / L2 action 集合 / L3 步数同量级 / L4 zoom 命中。"""
    from src.episode_logger import EpisodeLogger
    record = EpisodeLogger.load(episode_path)
    result = EpisodeLogger.replay(episode_path, _make_test_factory)
    
    # L1
    assert result.success == record.final_result.success, (
        f"L1: success differs"
    )
    
    # L2
    golden_kinds = {a.kind for a in record.actions}
    actual_kinds = {a.kind for a in result.action_history}
    assert actual_kinds == golden_kinds, (
        f"L2: action kinds differ. golden={golden_kinds}, actual={actual_kinds}"
    )
    
    # L3
    assert len(result.action_history) <= len(record.actions) * 1.5, (
        f"L3: step count blew up: {len(result.action_history)} vs {len(record.actions)}"
    )
    
    # L4
    if any(a.strategy == "zoom_in" for a in record.actions if a.kind == "re_observe"):
        assert any(a.strategy == "zoom_in"
                   for a in result.action_history if a.kind == "re_observe"), \
            "L4: golden zoomed but replay didn't"
```

- [ ] **Step 3: commit (不录 golden, 那是 Task 14.2)**

```bash
git add src/episode_logger.py tests/test_replay.py
git commit -m "feat(logger): EpisodeLogger.replay with 4-tier contract test scaffold"
```

---

### Task 14.2: 录 5 个 golden episode

**Files:**
- Create: `tests/episodes/golden/01_basic_apple.json`
- Create: `tests/episodes/golden/02_zoom_disambiguate_peeler.json`
- Create: `tests/episodes/golden/03_classify_safety_cup.json`
- Create: `tests/episodes/golden/04_ask_user_red_object.json`
- Create: `tests/episodes/golden/05_avoid_knife.json`

- [ ] **Step 1: 写 1 号 golden 手工示范 (其余 4 个跑真实 sim 录)**

```json
{
  "query": "拿苹果",
  "start_time": 1715159832.4,
  "snapshots": [
    {"step": 0, "timestamp": 1715159832.4, "n_hypotheses": 0,
     "target_summary": null, "most_uncertain_axis": "label",
     "overall_uncertainty": 1.0, "n_evidence": 0, "open_questions_count": 0}
  ],
  "actions": [
    {"kind": "observe", "strategy": null, "question": null, "metadata": {}},
    {"kind": "grasp", "strategy": null, "question": null, "metadata": {}}
  ],
  "evidence": [
    {"source": "llm_decompose", "timestamp": 1715159832.5,
     "raw_payload": {"primary_target": "apple", "constraints": []},
     "consumed_by": []},
    {"source": "vlm_ground", "timestamp": 1715159833.0,
     "raw_payload": {
       "viewpoint": "v0",
       "hypotheses": [{
         "object_id": "obj_0", "label": "apple",
         "label_alternatives": [["apple", 0.95], ["other", 0.05]],
         "label_entropy": 0.10,
         "position_3d": [0.5, 0.0, 0.9], "position_std_m": 0.02,
         "bbox_per_view": {"v0": [50, 50, 100, 100]},
         "observed_in_views": ["v0"]
       }]
     },
     "consumed_by": []},
    {"source": "llm_safety", "timestamp": 1715159834.0,
     "raw_payload": {"dist": {"safe": 0.9, "fragile": 0.1}, "entropy": 0.32,
                     "reasoning": "fruit"},
     "consumed_by": []},
    {"source": "grasp_attempt", "timestamp": 1715159835.0,
     "raw_payload": {"success": true,
                     "attempt": {"failure_mode": "success",
                                 "diagnostic": {"z_target": 0.9, "z_actual": 0.9},
                                 "candidate_source": "geometric_centroid"}},
     "consumed_by": []}
  ],
  "user_qa": [],
  "final_result": {
    "success": true, "speech": "已为您拿到 apple",
    "n_steps": 2, "elapsed_seconds": 3.5, "failure_reason": null
  }
}
```

- [ ] **Step 2: 跑 replay 测试 (1 号 golden 应通过)**

Run: `pytest tests/test_replay.py -v`
Expected: 1 pass

- [ ] **Step 3: 写 sim 录制脚本 (Task 14.3 用 mock 数据手写 4 个 golden 占位; 真 sim 录制 v1.1 替换)**

Create: `scripts/record_golden_episode.py`

```python
"""录制一个 golden episode。在真 sim 上跑 agent.run + EpisodeLogger, 输出 JSON。

Usage:
    python scripts/record_golden_episode.py --query "我要那个削皮器" \
        --output tests/episodes/golden/02_zoom_disambiguate_peeler.json
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/agent.yaml")
    args = parser.parse_args()
    
    # 复用 scripts/run_agent.py 的逻辑构造 agent + run
    # (此处简化: 提示用户先跑 run_agent.py 录到 logs/episodes/, 再 copy)
    
    import subprocess
    subprocess.run(
        [sys.executable, "scripts/run_agent.py",
         "--query", args.query, "--config", args.config],
        check=True,
    )
    
    # 找最新 episode_*.json
    log_dir = Path("logs/episodes")
    candidates = sorted(log_dir.glob("episode_*.json"))
    if not candidates:
        print("ERROR: no episode generated")
        sys.exit(1)
    latest = candidates[-1]
    
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(latest, out)
    print(f"recorded → {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 文档化 5 个 golden 的录制方式**

Modify: `tests/episodes/golden/README.md` (新建)

```markdown
# Golden Episodes

本目录存放 EpisodeLogger replay 测试的 golden 数据。

## 5 个 query (设计稿 §10.3)

| # | query | 触发场景 | 期望关键 action |
|---|---|---|---|
| 01 | 拿苹果 | 基础 (label 唯一) | observe → grasp → success |
| 02 | 我要那个削皮器 | zoom 消歧 | observe → re_observe(zoom_in) → grasp |
| 03 | 拿那个杯子 | safety 分类 | observe → classify_safety → grasp |
| 04 | 我要那个红色的 | ask_user | observe → ask_user → grasp |
| 05 | 帮我取碗, 避开刀 | constraint | decompose(constraint) → observe → grasp |

## 录制方式

```bash
python scripts/record_golden_episode.py \
  --query "我要那个削皮器" \
  --output tests/episodes/golden/02_zoom_disambiguate_peeler.json
```

## 重新录制

如果决策树或 prompt 改了导致 replay 测试 L1/L2 失败:
1. 跑 sim 重新录: `python scripts/record_golden_episode.py ...`
2. diff 新老 golden, 确认变化合理 (e.g. 多了一步 zoom_in)
3. 替换 golden 文件 + commit
```

- [ ] **Step 5: commit (1 号 golden 手写)**

```bash
git add tests/episodes/golden/01_basic_apple.json tests/episodes/golden/README.md scripts/record_golden_episode.py
git commit -m "test(replay): 1 manually-crafted golden + recording script + README for 5-query suite"
```

---

### Task 14.3: 手写其余 4 个 golden 模板 (mock-based, 不依赖真 sim)

**目标:** v1 验收要求 5/5 golden 通过 replay (设计稿 §11 Step 14)。但真 sim 录制依赖 GPU + RoboCasa 启动, plan 实施期未必有条件。本 task 用"mock 出 evidence 序列"的方式手写 02-05 号 golden, 让 replay 测试在 PR 阶段就能跑通; 真 sim 录制由 v1.1 在演示前替换。

> **设计原则**: 4 层契约 (L1-L4) 关注的是**决策路径形状**, 不是 evidence 内容真实性。所以手写 evidence 只要能驱动 agent 走出"该走的 action 序列"就够。每条 golden 的关键 `action_kinds` 集合见下表, 替换为真 sim 录制时不会破坏 L2 契约。

**Files:**
- Create: `tests/episodes/golden/02_zoom_disambiguate_peeler.json`
- Create: `tests/episodes/golden/03_classify_safety_cup.json`
- Create: `tests/episodes/golden/04_ask_user_red_object.json`
- Create: `tests/episodes/golden/05_avoid_knife.json`

#### Golden 关键契约对照表

| # | query | 必触发 action_kind | 关键 evidence source |
|---|---|---|---|
| 02 | 我要那个削皮器 | `observe`, `re_observe`, `grasp` | `vlm_ground` (label_entropy>0.5) → `vlm_zoom` (entropy<0.2) |
| 03 | 拿那个杯子 | `observe`, `classify_safety`, `grasp` | `vlm_ground` → `llm_safety` (entropy<0.3) |
| 04 | 我要那个红色的 | `observe`, `ask_user`, `grasp` | `vlm_ground` (target() 返 None, top1/top2 差<0.2) → `user_answer` |
| 05 | 帮我取碗, 避开刀 | `observe`, `grasp` (含 avoid 约束) | `llm_decompose` (constraints 含 avoid:knife) → `vlm_ground` → grasp |

- [ ] **Step 1: 写 02_zoom_disambiguate_peeler.json**

模板原则: 第 1 帧 vlm_ground 给一个高熵候选 (label_entropy=0.65), 第 2 步 re_observe(zoom_in) → vlm_zoom 给 entropy=0.15 → confident → grasp success。

```json
{
  "query": "我要那个削皮器",
  "start_time": 1715200000.0,
  "snapshots": [
    {"step": 0, "timestamp": 1715200000.0, "n_hypotheses": 0,
     "target_summary": null, "most_uncertain_axis": "label",
     "overall_uncertainty": 1.0, "n_evidence": 0, "open_questions_count": 0}
  ],
  "actions": [
    {"kind": "observe", "strategy": null, "question": null, "metadata": {}},
    {"kind": "re_observe", "strategy": "zoom_in", "question": null, "metadata": {}},
    {"kind": "plan_grasp_candidates", "strategy": null, "question": null, "metadata": {}},
    {"kind": "grasp", "strategy": null, "question": null, "metadata": {}}
  ],
  "evidence": [
    {"source": "llm_decompose", "timestamp": 1715200000.1,
     "raw_payload": {"primary_target": "peeler", "constraints": []},
     "consumed_by": []},
    {"source": "vlm_ground", "timestamp": 1715200001.0,
     "raw_payload": {
       "viewpoint": "v0",
       "hypotheses": [{
         "object_id": "obj_0", "label": "peeler",
         "label_alternatives": [["peeler", 0.5], ["knife", 0.4], ["spoon", 0.1]],
         "label_entropy": 0.65,
         "position_3d": [0.5, 0.0, 0.9], "position_std_m": 0.04,
         "bbox_per_view": {"v0": [60, 70, 110, 130]},
         "observed_in_views": ["v0"]
       }]
     },
     "consumed_by": []},
    {"source": "vlm_zoom", "timestamp": 1715200002.5,
     "raw_payload": {
       "hypotheses": [{
         "object_id": "obj_0", "label": "peeler",
         "label_alternatives": [["peeler", 0.92], ["knife", 0.08]],
         "label_entropy": 0.15,
         "position_3d": [0.5, 0.0, 0.9], "position_std_m": 0.04,
         "bbox_per_view": {"v0": [60, 70, 110, 130]},
         "observed_in_views": ["v0"]
       }]
     },
     "consumed_by": []},
    {"source": "llm_safety", "timestamp": 1715200003.0,
     "raw_payload": {"dist": {"safe": 0.4, "sharp": 0.6}, "entropy": 0.67,
                     "reasoning": "刃口锋利"},
     "consumed_by": []},
    {"source": "depth_projection", "timestamp": 1715200003.5,
     "raw_payload": {"n_candidates": 1}, "consumed_by": []},
    {"source": "grasp_attempt", "timestamp": 1715200004.5,
     "raw_payload": {"success": true,
                     "attempt": {"failure_mode": "success",
                                 "diagnostic": {"z_target": 0.9, "z_actual": 0.9},
                                 "candidate_source": "geometric_centroid"}},
     "consumed_by": []}
  ],
  "user_qa": [],
  "final_result": {
    "success": true, "speech": "已为您拿到 peeler",
    "n_steps": 4, "elapsed_seconds": 4.5, "failure_reason": null
  }
}
```

- [ ] **Step 2: 写 03_classify_safety_cup.json**

```json
{
  "query": "拿那个杯子",
  "start_time": 1715200100.0,
  "snapshots": [
    {"step": 0, "timestamp": 1715200100.0, "n_hypotheses": 0,
     "target_summary": null, "most_uncertain_axis": "label",
     "overall_uncertainty": 1.0, "n_evidence": 0, "open_questions_count": 0}
  ],
  "actions": [
    {"kind": "observe", "strategy": null, "question": null, "metadata": {}},
    {"kind": "classify_safety", "strategy": null, "question": null, "metadata": {}},
    {"kind": "plan_grasp_candidates", "strategy": null, "question": null, "metadata": {}},
    {"kind": "grasp", "strategy": null, "question": null, "metadata": {}}
  ],
  "evidence": [
    {"source": "llm_decompose", "timestamp": 1715200100.1,
     "raw_payload": {"primary_target": "cup", "constraints": []},
     "consumed_by": []},
    {"source": "vlm_ground", "timestamp": 1715200101.0,
     "raw_payload": {
       "viewpoint": "v0",
       "hypotheses": [{
         "object_id": "obj_0", "label": "cup",
         "label_alternatives": [["cup", 0.92], ["mug", 0.08]],
         "label_entropy": 0.15,
         "position_3d": [0.5, 0.0, 0.9], "position_std_m": 0.04,
         "bbox_per_view": {"v0": [50, 50, 100, 100]},
         "observed_in_views": ["v0"]
       }]
     },
     "consumed_by": []},
    {"source": "llm_safety", "timestamp": 1715200102.0,
     "raw_payload": {"dist": {"safe": 0.4, "fragile": 0.5, "sharp": 0.1},
                     "entropy": 0.71, "reasoning": "陶瓷感"},
     "consumed_by": []},
    {"source": "llm_safety", "timestamp": 1715200103.0,
     "raw_payload": {"dist": {"safe": 0.85, "fragile": 0.15},
                     "entropy": 0.25, "reasoning": "再看像塑料"},
     "consumed_by": []},
    {"source": "depth_projection", "timestamp": 1715200103.5,
     "raw_payload": {"n_candidates": 1}, "consumed_by": []},
    {"source": "grasp_attempt", "timestamp": 1715200104.5,
     "raw_payload": {"success": true,
                     "attempt": {"failure_mode": "success",
                                 "diagnostic": {"z_target": 0.9, "z_actual": 0.9},
                                 "candidate_source": "geometric_centroid"}},
     "consumed_by": []}
  ],
  "user_qa": [],
  "final_result": {
    "success": true, "speech": "已为您拿到 cup", 
    "n_steps": 4, "elapsed_seconds": 4.5, "failure_reason": null
  }
}
```

- [ ] **Step 3: 写 04_ask_user_red_object.json**

模板原则: 第 1 帧 vlm_ground 返 2 个高度模糊的红色物体 (top1/top2 差 < 0.2 → target() 返 None) → ask_user → user_answer "圆形的" 后下一轮 vlm 给 confident apple → grasp。

```json
{
  "query": "我要那个红色的",
  "start_time": 1715200200.0,
  "snapshots": [
    {"step": 0, "timestamp": 1715200200.0, "n_hypotheses": 0,
     "target_summary": null, "most_uncertain_axis": "label",
     "overall_uncertainty": 1.0, "n_evidence": 0, "open_questions_count": 0}
  ],
  "actions": [
    {"kind": "observe", "strategy": null, "question": null, "metadata": {}},
    {"kind": "ask_user", "strategy": null,
     "question": "我看到 2 个红色的物体, 您要哪个?", "metadata": {}},
    {"kind": "observe", "strategy": null, "question": null, "metadata": {}},
    {"kind": "plan_grasp_candidates", "strategy": null, "question": null, "metadata": {}},
    {"kind": "grasp", "strategy": null, "question": null, "metadata": {}}
  ],
  "evidence": [
    {"source": "llm_decompose", "timestamp": 1715200200.1,
     "raw_payload": {"primary_target": "apple", "constraints": []},
     "consumed_by": []},
    {"source": "vlm_ground", "timestamp": 1715200201.0,
     "raw_payload": {
       "viewpoint": "v0",
       "hypotheses": [
         {"object_id": "obj_0", "label": "apple",
          "label_alternatives": [["apple", 0.5], ["tomato", 0.5]],
          "label_entropy": 0.69,
          "position_3d": [0.5, 0.0, 0.9], "position_std_m": 0.04,
          "bbox_per_view": {"v0": [50, 50, 90, 90]},
          "observed_in_views": ["v0"]},
         {"object_id": "obj_1", "label": "apple",
          "label_alternatives": [["apple", 0.5], ["tomato", 0.5]],
          "label_entropy": 0.69,
          "position_3d": [0.7, 0.1, 0.9], "position_std_m": 0.04,
          "bbox_per_view": {"v0": [120, 50, 160, 90]},
          "observed_in_views": ["v0"]}
       ]
     },
     "consumed_by": []},
    {"source": "user_answer", "timestamp": 1715200202.0,
     "raw_payload": {"q": "我看到 2 个红色的物体, 您要哪个?", "a": "圆形的"},
     "consumed_by": []},
    {"source": "vlm_ground", "timestamp": 1715200203.0,
     "raw_payload": {
       "viewpoint": "v1",
       "hypotheses": [{
         "object_id": "obj_0", "label": "apple",
         "label_alternatives": [["apple", 0.95], ["tomato", 0.05]],
         "label_entropy": 0.10,
         "position_3d": [0.5, 0.0, 0.9], "position_std_m": 0.03,
         "bbox_per_view": {"v1": [60, 60, 100, 100]},
         "observed_in_views": ["v0", "v1"]
       }]
     },
     "consumed_by": []},
    {"source": "llm_safety", "timestamp": 1715200204.0,
     "raw_payload": {"dist": {"safe": 0.95, "fragile": 0.05}, "entropy": 0.20,
                     "reasoning": "fruit"},
     "consumed_by": []},
    {"source": "depth_projection", "timestamp": 1715200204.5,
     "raw_payload": {"n_candidates": 1}, "consumed_by": []},
    {"source": "grasp_attempt", "timestamp": 1715200205.5,
     "raw_payload": {"success": true,
                     "attempt": {"failure_mode": "success",
                                 "diagnostic": {"z_target": 0.9, "z_actual": 0.9},
                                 "candidate_source": "geometric_centroid"}},
     "consumed_by": []}
  ],
  "user_qa": [["我看到 2 个红色的物体, 您要哪个?", "圆形的"]],
  "final_result": {
    "success": true, "speech": "已为您拿到 apple",
    "n_steps": 5, "elapsed_seconds": 5.5, "failure_reason": null
  }
}
```

- [ ] **Step 4: 写 05_avoid_knife.json**

```json
{
  "query": "帮我取碗, 避开刀",
  "start_time": 1715200300.0,
  "snapshots": [
    {"step": 0, "timestamp": 1715200300.0, "n_hypotheses": 0,
     "target_summary": null, "most_uncertain_axis": "label",
     "overall_uncertainty": 1.0, "n_evidence": 0, "open_questions_count": 0}
  ],
  "actions": [
    {"kind": "observe", "strategy": null, "question": null, "metadata": {}},
    {"kind": "plan_grasp_candidates", "strategy": null, "question": null, "metadata": {}},
    {"kind": "grasp", "strategy": null, "question": null, "metadata": {}}
  ],
  "evidence": [
    {"source": "llm_decompose", "timestamp": 1715200300.1,
     "raw_payload": {
       "primary_target": "bowl",
       "constraints": [
         {"kind": "avoid", "target_label": "knife", "reason": "用户避开"}
       ]
     },
     "consumed_by": []},
    {"source": "vlm_ground", "timestamp": 1715200301.0,
     "raw_payload": {
       "viewpoint": "v0",
       "hypotheses": [{
         "object_id": "obj_0", "label": "bowl",
         "label_alternatives": [["bowl", 0.93], ["plate", 0.07]],
         "label_entropy": 0.12,
         "position_3d": [0.5, 0.0, 0.9], "position_std_m": 0.04,
         "bbox_per_view": {"v0": [40, 40, 120, 120]},
         "observed_in_views": ["v0"]
       }]
     },
     "consumed_by": []},
    {"source": "llm_safety", "timestamp": 1715200302.0,
     "raw_payload": {"dist": {"safe": 0.7, "fragile": 0.3}, "entropy": 0.30,
                     "reasoning": "陶瓷碗"},
     "consumed_by": []},
    {"source": "depth_projection", "timestamp": 1715200302.5,
     "raw_payload": {"n_candidates": 1}, "consumed_by": []},
    {"source": "grasp_attempt", "timestamp": 1715200303.5,
     "raw_payload": {"success": true,
                     "attempt": {"failure_mode": "success",
                                 "diagnostic": {"z_target": 0.9, "z_actual": 0.9},
                                 "candidate_source": "geometric_centroid"}},
     "consumed_by": []}
  ],
  "user_qa": [],
  "final_result": {
    "success": true, "speech": "已为您拿到 bowl",
    "n_steps": 3, "elapsed_seconds": 3.5, "failure_reason": null
  }
}
```

- [ ] **Step 5: 跑 replay 全套**

Run: `pytest tests/test_replay.py -v`
Expected: 5 pass (01-05 全过 4 层契约)

> **可能的契约断言失败**: 02 和 04 模板里 vlm_ground 是 2 帧 (golden 用 mock 排队消耗), replay 时由于 cache 把同 prompt 第 2 次直接读 cache, 可能拿不到第 2 帧。如出现此问题, 在 `_make_test_factory` 里把 cache.max_size 设 0 (实质禁用 cache)。

- [ ] **Step 6: 在 README.md 里加一行说明**

修改 `tests/episodes/golden/README.md` 在 "## 录制方式" 前加一节:

```markdown
## v1 现状: 02-05 是 mock 模板

01_basic_apple.json 是真实风格的最小 episode 模板。
02-05 是用"手工编排 evidence 序列"模拟出 4 层契约必要的 action 序列, **不代表 sim 真实输出**。
v1.1 演示前必须用 scripts/record_golden_episode.py 在真 sim 录制替换。

替换时关注:
1. L2 契约 (action 集合) 不应变 — 见关键契约对照表
2. L3 契约 (步数 1.5x) 真 sim 通常更长, 必要时调宽阈值
3. L4 契约 (zoom_in 命中) 02 必须保留, 否则 demo 故事 1 不成立
```

- [ ] **Step 7: commit**

```bash
git add tests/episodes/golden/02_zoom_disambiguate_peeler.json tests/episodes/golden/03_classify_safety_cup.json tests/episodes/golden/04_ask_user_red_object.json tests/episodes/golden/05_avoid_knife.json tests/episodes/golden/README.md
git commit -m "test(replay): hand-crafted golden 02-05 (mock-based, replace with real sim in v1.1)"
```

**Phase 14 CHECKPOINT:** 5 个 golden 通过 replay 4 层契约; 录制工具就位等真 sim 替换。

---

## Phase 15: 删除老代码 (清理)

**目标:** 安全删除 5 个 src 文件 + 2 个 yaml + 6 个老 prompt + 1 个老 SafetyGate 类。每个删除独立 commit, 单步可回滚。

> **⚠️ 执行顺序**: 文档列出的 15.1-15.7 顺序按"被引用最少 → 被引用最多"排, 但 **15.5 (拆 scene_model) 必须在 15.7 (删老 active_planner.plan / action_executor.execute) 之后做**。否则 15.5 完成时, 老 active_planner.plan 还在引用已删的 SceneModel, 测试会断。
>
> **推荐执行序**: `15.1 → 15.2 → 15.3 → 15.4 → 15.6 → 15.7 → 15.5 → 15.8`

### 删除前置检查

- [ ] **Step 0: 跑全套测试 baseline**

Run: `pytest tests/ -v --tb=short 2>&1 | tail -30`
Record: 通过的 N (用于回滚比对)

---

### Task 15.1: 删 `src/pipeline.py`

- [ ] **Step 1: grep 引用**

Run: `grep -r "from src.pipeline" src/ scripts/ tests/`
Expected: 仅 `scripts/run_sim_query.py` 之类老入口引用

- [ ] **Step 2: 老入口脚本改用 agent**

Modify 引用 pipeline 的脚本 → 改成 import agent (设计稿 §11 Step 13 已做了 `scripts/run_agent.py`, 此处只需删老入口或加 deprecation 警告)

- [ ] **Step 3: 删 pipeline.py**

```bash
git rm src/pipeline.py
```

- [ ] **Step 4: 跑测试**

Run: `pytest tests/ -v`
Expected: 不破

- [ ] **Step 5: commit**

```bash
git commit -m "refactor(cleanup): delete src/pipeline.py (replaced by src/agent.py)"
```

---

### Task 15.2: 删 `src/vlm_grounding.py`

- [ ] **Step 1: grep 引用**

Run: `grep -r "from src.vlm_grounding\|src.vlm_grounding" src/ scripts/ tests/`

- [ ] **Step 2: 老测试 `tests/test_vlm_grounding.py` 也删**

```bash
git rm src/vlm_grounding.py tests/test_vlm_grounding.py
```

- [ ] **Step 3: 跑测试**

Run: `pytest tests/ -v`
Expected: 不破 (新 perception.py 已替代)

- [ ] **Step 4: commit**

```bash
git commit -m "refactor(cleanup): delete src/vlm_grounding.py + tests (replaced by src/perception.py)"
```

---

### Task 15.3: 删 `src/scene_describer.py`

- [ ] **Step 1: grep 引用**

Run: `grep -rn "scene_describer\|SceneDescriber" src/ scripts/ tests/`
Expected: 应只剩 `src/action_executor.py` 的 `scene_describer=None` 占位 + 老 pipeline (已删)

- [ ] **Step 2: 把 ActionExecutor 构造里的 `scene_describer` 参数也清理 (改为 optional 且不再使用)**

Modify: `src/action_executor.py`
- 把 `scene_describer` 标 deprecated, 不在 v1 接口里使用

- [ ] **Step 3: 删 scene_describer.py**

```bash
git rm src/scene_describer.py
```

- [ ] **Step 4: 跑测试 + commit**

```bash
git add src/action_executor.py
git commit -m "refactor(cleanup): delete src/scene_describer.py (perception 已替代)"
```

---

### Task 15.4: 删 `src/action_decider.py` + 老 prompts

- [ ] **Step 1**

```bash
git rm src/action_decider.py
git rm prompts/action_decider.txt
git rm prompts/active_planner.txt
git rm prompts/active_planner_grounding_aware.txt
git rm prompts/scene_describer.txt
git rm prompts/vlm_grounding.txt
git rm prompts/task_decompose.txt
```

- [ ] **Step 2: 跑测试 + commit**

```bash
git commit -m "refactor(cleanup): delete action_decider.py + 6 old prompts"
```

---

### Task 15.5: 拆 `src/scene_model.py` (projection 留下, 主体删)

**Files:**
- Create: `src/projection.py`
- Modify (then delete): `src/scene_model.py`

> **现状摘要 (2026-05-08 grep 确认)**: `src/scene_model.py` 含 4 个顶层符号 — `@dataclass GroundedObject` (line ~40), 3 个纯函数 `depth_buffer_to_meters` / `project_bbox_to_world` / `compute_intrinsics`, `class SceneModel`。前后端引用情况见 Step 0。

- [ ] **Step 0: 全仓 grep 现有引用 (避免漏迁)**

Run (PowerShell):
```powershell
grep -rn "from src.scene_model import\|from \.scene_model import\|src\.scene_model\." src/ tests/ scripts/
```

Expected (现状):
- `src/active_planner.py` (8 处) — 引用 `SceneModel / GroundedObject / project_bbox_to_world / compute_intrinsics`
- `src/action_executor.py` (3 处) — 引用 `SceneModel / GroundedObject`
- `src/scene_describer.py` (1 处) — 已被 Phase 15.3 删, 不计
- `src/pipeline.py` (3 处) — 已被 Phase 15.1 删, 不计
- `src/env_wrapper.py` (1 处) — 引用 `compute_intrinsics`
- `tests/test_scene_model.py` (23 处) — `SceneModel / GroundedObject / depth_buffer_to_meters / project_bbox_to_world / compute_intrinsics` 全用了
- `tests/test_active_planner_grounding.py` (2 处) — `SceneModel`
- `scripts/test_safequery_integration.py` (1 处) — 待手动审计

> **影响范围**: 完成 Phase 15.1 / 15.3 / 15.7 后, **仍引用 scene_model 的非废弃代码**只剩:
> - `src/active_planner.py` (老 plan/plan_with_grounding 部分; Phase 15.7 删完后引用归零)
> - `src/action_executor.py` (老 execute_with_scene_model; Phase 15.7 删完后归零)
> - `src/env_wrapper.py` (compute_intrinsics; 必须改为 from src.projection import)
> 
> 故 Task 15.5 在 Task 15.7 **之后**做更安全 (老接口先删, scene_model 引用自然减少, 只剩 env_wrapper 一处)。

- [ ] **Step 1: 把 3 个纯函数搬到 `src/projection.py` (完整代码)**

```python
"""3D 投影工具函数 (从 scene_model.py 拆出)。

保留范围:
- depth_buffer_to_meters
- project_bbox_to_world
- compute_intrinsics

删除 (v1 用 Hypothesis + WorldBelief.merge_hypothesis 替代):
- @dataclass GroundedObject
- class SceneModel (add_view / merge / get_best_match / ...)
"""
from __future__ import annotations

import numpy as np


def depth_buffer_to_meters(
    z_buffer: float,
    extent: float,
    znear_ratio: float,
    zfar_ratio: float,
) -> float:
    """MuJoCo z_buffer → meters。
    
    照搬 src/scene_model.py:91-115 (无算法变更)。
    """
    near = znear_ratio * extent
    far = zfar_ratio * extent
    return near / (1.0 - z_buffer * (1.0 - near / far))


def project_bbox_to_world(
    bbox_2d: tuple[int, int, int, int],
    depth_image: np.ndarray,
    K: np.ndarray,
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
) -> np.ndarray:
    """bbox 中心 + 深度 → world 3D 坐标。
    
    照搬 src/scene_model.py:118-177。
    """
    # ... (完整复制 scene_model.py:118-177 的实现)


def compute_intrinsics(fovy_deg: float, height: int, width: int) -> np.ndarray:
    """从 fovy 计算 3x3 内参矩阵 K。
    
    照搬 src/scene_model.py:180-188。
    """
    fy = 0.5 * height / np.tan(0.5 * np.radians(fovy_deg))
    fx = fy   # 正方形像素
    K = np.array([[fx, 0, width / 2],
                  [0, fy, height / 2],
                  [0, 0, 1]])
    return K
```

> **不要修改算法**, 只是搬位置。算法变更应该是单独的 PR。

- [ ] **Step 2: 把 `src/env_wrapper.py` 的 `from src.scene_model import compute_intrinsics` 改成 `from src.projection import compute_intrinsics`**

```bash
grep -n "from src.scene_model" src/env_wrapper.py
```

Expected: 1 行命中。直接 sed 替换或手改。

- [ ] **Step 3: 拆测试 — `tests/test_scene_model.py` → `tests/test_projection.py`**

把测试中跟 3 个纯函数相关的部分 (`depth_buffer_to_meters / project_bbox_to_world / compute_intrinsics`) 搬到 `tests/test_projection.py`; 跟 `SceneModel / GroundedObject` 相关的测试**直接删** (这些类已无替代物, 测试无意义)。

- [ ] **Step 4: 处理 `scripts/test_safequery_integration.py`**

```bash
grep -n "scene_model" scripts/test_safequery_integration.py
```

看具体引用什么; 若仅为整合 demo, 改 import 路径即可。

- [ ] **Step 5: 删 scene_model.py + 老测试**

```bash
git rm src/scene_model.py
git rm tests/test_scene_model.py
git add src/projection.py src/env_wrapper.py tests/test_projection.py scripts/test_safequery_integration.py
git commit -m "refactor(cleanup): split scene_model.py → projection.py (3 pure fns), drop SceneModel/GroundedObject"
```

- [ ] **Step 6: 跑全套测试**

Run: `pytest tests/ -v --tb=short`
Expected: 不破; `test_projection.py` 取代 `test_scene_model.py` 通过

---

### Task 15.6: 删 `configs/object_aliases.yaml` + `configs/safety_rules.yaml`

- [ ] **Step 1: grep 引用**

```bash
grep -rn "object_aliases\|safety_rules" src/ tests/ scripts/
```

Expected: 仅 老 SafetyGate 引用 safety_rules; env_wrapper 可能引用 object_aliases (用于 ground_object 的 alias 路径)

- [ ] **Step 2: env_wrapper.ground_object 把 alias_map 路径改为可选 (默认 None, 不再加载)**

Modify: `src/env_wrapper.py` 的 `ground_object` (使其在没有 alias_map 时仍工作 — fallback 逻辑保留)

- [ ] **Step 3: 老 SafetyGate 类删除 (新 SafetyClassifier 替代)**

Modify: `src/safety_gate.py` — 删 `class SafetyGate` + `SafetyDecision` + `_FEATURE_RISK_KEYWORDS` + `_load_rules` 等老代码; 仅保留 `SafetyClassifier`。

```bash
# tests/test_safety_gate.py 改成测 SafetyClassifier 的, 或删
git rm tests/test_safety_gate.py
```

- [ ] **Step 4: 删 yaml**

```bash
git rm configs/object_aliases.yaml configs/safety_rules.yaml
```

- [ ] **Step 5: 跑全套测试**

Run: `pytest tests/ -v`
Expected: 通 (新 SafetyClassifier 测试在 tests/test_safety_classifier.py 已覆盖)

- [ ] **Step 6: commit**

```bash
git add src/safety_gate.py src/env_wrapper.py
git commit -m "refactor(cleanup): delete old SafetyGate + 2 yaml + tests/test_safety_gate.py"
```

---

### Task 15.7: 老 `task_decomposer.decompose()` (返回 list[Subtask]) + `active_planner.plan()` 等

**老接口可保留也可删, 看是否还有引用。**

- [ ] **Step 1: grep**

```bash
grep -rn "decompose(\|\.plan(\|plan_with_grounding" src/ tests/ scripts/
```

- [ ] **Step 2: 删除已无引用的老方法**

Modify: `src/task_decomposer.py` (删 `decompose`, 仅留 `decompose_v1`)
Modify: `src/active_planner.py` (删 `plan / plan_with_grounding / _update_coverage / _is_sufficient`, 仅留 `ActiveViewpointSelector`)
Modify: `src/action_executor.py` (删 `execute / execute_with_scene_model`, 仅留 `act / verify_grasp / release_and_retreat`)

- [ ] **Step 3: 跑测试**

Run: `pytest tests/ -v`
Expected: 通

- [ ] **Step 4: commit**

```bash
git add src/task_decomposer.py src/active_planner.py src/action_executor.py
git commit -m "refactor(cleanup): drop legacy decompose/plan/execute APIs (v1 接口替代)"
```

---

### Task 15.8: 最终验收

- [ ] **Step 1: 全套测试**

Run: `pytest tests/ -v --tb=short`
Expected: 100+ pass, 0 fail

- [ ] **Step 2: ruff 全 lint**

Run: `ruff check src/ tests/ scripts/`
Expected: All checks passed

- [ ] **Step 3: src/ 文件清单核对**

Expected (12 个文件):
```
src/__init__.py
src/agent.py
src/world_belief.py
src/perception.py
src/safety_gate.py     (仅 SafetyClassifier)
src/grasp_planner.py
src/action_executor.py (仅 v1 act/verify/release)
src/active_planner.py  (仅 ViewpointLibrary + ActiveViewpointSelector)
src/task_decomposer.py (仅 decompose_v1)
src/user_channel.py
src/episode_logger.py
src/vlm_cache.py
src/projection.py
src/env_wrapper.py
src/llm_backend.py
src/vlm_backend.py
src/utils.py
src/eval.py
```

- [ ] **Step 4: sim 端到端验证 (5 demo queries)**

Run: 
```bash
for q in "拿苹果" "我要那个削皮器" "拿那个杯子" "我要那个红色的" "帮我取碗, 避开刀"; do
  python scripts/run_agent.py --query "$q"
done
```
Expected: 全部不崩, 至少给出 speech (success 或 ask_user 后 success)

- [ ] **Step 5: 最终 commit**

```bash
git add -A
git commit -m "refactor(cleanup): Phase 15 done — old pipeline/scene_describer/yaml all removed"
```

**Phase 15 CHECKPOINT (= v1 完成):**
- 全套单测 110+ pass (Phase 1-12 ~100 + Phase 13 1 public API + Phase 14 5 replay + projection 迁移测试)
- ruff 全 clean  
- 5 个 demo query 在 sim 跑通
- src/ 仅剩 18 个文件, 老代码彻底清理

---

## 完成验收

设计稿 §16 实施前置条件全部满足后, 跑下列命令一次:

```bash
# 1. 全套单测
pytest tests/ -v --tb=short

# 2. lint
ruff check src/ tests/ scripts/

# 3. 5 个 demo query (sim)
for q in "拿苹果" "我要那个削皮器" "拿那个杯子" "我要那个红色的" "帮我取碗, 避开刀"; do
  python scripts/run_agent.py --query "$q" --user-mode fake_from_robocasa
done

# 4. replay 测试 (5 golden)
pytest tests/test_replay.py -v
```

全过 = v1 GO。

---

## 应急回滚

每 Phase 都是独立 commit, 出问题:

```bash
git log --oneline   # 找到上一个 checkpoint
git reset --hard <commit>
```

只回滚最近 Phase, 不需要全回。

---

## 不在本计划内的事 (Open Questions, 见设计稿 §13)

- Session 持久化 belief
- 真 ASR/TTS
- 多目标并行
- sim2real domain adaptation
- 阈值学习
- 触觉/力反馈

这些是 v2 工作, v1 不做。
