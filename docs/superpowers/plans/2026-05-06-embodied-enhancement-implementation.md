# EmboSight 具身增强实施计划

> **基于 spec**: `docs/superpowers/specs/2026-05-06-embodied-enhancement-design.md`
> **目标导向**: 比赛成品优先，实验/消融留到工程跑通后

**Goal**: 把 EmboSight 升级为完整具身闭环系统 — 用户输入查询 → 机器人主动观察 → 决策 → 风险感知运动 → 抓取 → 语义验证 → 语音反馈，全程在 RoboCasa 仿真中可视化展示。

**Architecture**: 6 步 Pipeline (TaskDecomposer → ActivePlanner → SceneDescriber → Aggregate → ActionDecider → ActionExecutor)。新增模块 `action_decider.py` + `action_executor.py`，扩展 `env_wrapper.py` 实现真实手臂控制和抓取。

**Tech Stack**: Python 3.10+, RoboCasa, robosuite, MuJoCo, DeepSeek API, Qwen2.5-VL, OSC_POSE 控制器

---

## Phase 总览

| Phase | 内容 | 关键交付 |
|---|---|---|
| 1 | 真实手臂移动 + observe 实时刷新 | `move_arm_to`, `get_eef_pos`, observe 刷新 |
| 2 | Object grounding | `ground_object`, `object_aliases.yaml` |
| 3 | 抓取动作 | `grasp_at` |
| 4 | ActionDecider 模块 | `action_decider.py`, prompt |
| 5 | ActionExecutor 模块 | `action_executor.py` (安全路径 + 验证) |
| 6 | Pipeline Step 5-6 接入 | 扩展 `pipeline.py` |
| 7 | 实时可视化 Demo | `--visualize` 模式 |
| 8 | 端到端集成测试 + 演示 | `test_embodied.py`, demo query 集 |

---

## Phase 1: 真实手臂移动 + observe 实时刷新

**目标**: 让 `env.move_arm_to(target_pos_m)` 真的移动机械臂，`env.observe()` 取到的是当前帧而非 reset 静态帧。

### Task 1.1: 加 `get_eef_pos` 接口

**Files:**
- Modify: `src/env_wrapper.py`

- [ ] **Step 1: 添加 `get_eef_pos()` 方法**

读取 `_latest_obs['robot0_eef_pos']` 返回 numpy 数组 (单位 m)。如果 `_latest_obs` 为空则先 reset。

```python
def get_eef_pos(self) -> np.ndarray:
    """获取末端执行器当前世界坐标 (单位: m)"""
    if not self._latest_obs:
        self.reset()
    pos = self._latest_obs.get("robot0_eef_pos")
    if pos is None:
        raise RuntimeError("robot0_eef_pos not in observation")
    return np.asarray(pos, dtype=np.float32)
```

- [ ] **Step 2: 文件顶部加 numpy import**（若尚未导入）

- [ ] **Step 3: 自测**

在 env_wrapper 的 `__main__` 块加：
```python
print("eef_pos:", env.get_eef_pos())
```
运行 `python -m src.env_wrapper`，应输出形如 `[0.5 0.0 1.0]` 的 3D 坐标。

- [ ] **Step 4: commit**

```bash
git add src/env_wrapper.py
git commit -m "feat(env): add get_eef_pos() returning meter coordinates"
```

---

### Task 1.2: 实现真实 `move_arm_to`

**Files:**
- Modify: `src/env_wrapper.py:90-106` (替换 no-op 实现)

- [ ] **Step 1: 改签名**

旧签名 `move_arm_to(pose: 6-tuple, cm/度)` → 新签名 `move_arm_to(target_pos_m, max_steps=200, threshold_m=0.02)`

- [ ] **Step 2: 实现 OSC 控制循环**

```python
def move_arm_to(
    self,
    target_pos_m,
    max_steps: int = 200,
    threshold_m: float = 0.02,
) -> bool:
    """OSC 增量控制移动末端到目标位置 (单位: m)
    
    Returns:
        True if converged within threshold, False otherwise
    """
    target = np.asarray(target_pos_m, dtype=np.float32)
    action_dim = self._env.action_dim
    
    for step in range(max_steps):
        current = self.get_eef_pos()
        delta = target - current
        dist = float(np.linalg.norm(delta))
        
        if dist < threshold_m:
            logger.debug(f"[move_arm_to] converged at step {step}, dist={dist:.4f}m")
            return True
        
        # 限幅: 单步最大 0.05m
        step_size = min(0.05, dist)
        direction = delta / max(dist, 1e-6)
        
        action = np.zeros(action_dim, dtype=np.float32)
        action[0:3] = direction * step_size
        # gripper 维持 0 (不动作)
        
        try:
            obs, _, done, _ = self._env.step(action)
            self._latest_obs = obs
        except Exception as e:
            logger.warning(f"[move_arm_to] env.step failed at step {step}: {e}")
            return False
    
    logger.warning(f"[move_arm_to] max_steps reached, dist={dist:.4f}m")
    return False
```

- [ ] **Step 3: 自测**

在 `__main__` 加：
```python
start = env.get_eef_pos()
target = start + np.array([0.0, 0.0, 0.10])  # 向上 10cm
ok = env.move_arm_to(target)
end = env.get_eef_pos()
print(f"start={start}, end={end}, ok={ok}")
```

预期: `end` 与 `target` 距离 < 0.02m，`ok=True`。

- [ ] **Step 4: commit**

```bash
git commit -am "feat(env): implement real OSC arm control in move_arm_to"
```

---

### Task 1.3: 让 `observe` 刷新到当前帧

**Files:**
- Modify: `src/env_wrapper.py:108-148`

- [ ] **Step 1: observe 加 zero-action step 刷新**

```python
def observe(self, viewpoint) -> "Observation":
    from .active_planner import Observation
    
    if not self._latest_obs:
        self.reset()
    
    # 关键: 做一次 zero-action step 让所有摄像头渲染当前帧
    try:
        zero_action = np.zeros(self._env.action_dim, dtype=np.float32)
        obs, _, _, _ = self._env.step(zero_action)
        self._latest_obs = obs
    except Exception as e:
        logger.warning(f"[observe] zero-step failed, using stale obs: {e}")
    
    camera_name = viewpoint.name
    img_key = f"{camera_name}_image"
    img = self._latest_obs.get(img_key)
    # ... (其余保留)
```

- [ ] **Step 2: 验证 eye-in-hand 跟随手臂**

在 `__main__` 加：
```python
from src.active_planner import ViewpointLibrary
vp_lib = ViewpointLibrary("configs/viewpoints.yaml")
eye = next(v for v in vp_lib.viewpoints if v.name == "robot0_eye_in_hand")

obs1 = env.observe(eye)
env.move_arm_to(env.get_eef_pos() + np.array([0.0, 0.10, 0.0]))
obs2 = env.observe(eye)
print(f"img1={obs1.image_path}, img2={obs2.image_path}")
```

打开两张图肉眼对比应明显不同。

- [ ] **Step 3: commit**

```bash
git commit -am "feat(env): observe now refreshes current frame via zero-action step"
```

---

## Phase 2: Object Grounding

**目标**: 用户说"药瓶"能映射到 RoboCasa 内部对象并返回 meter 坐标。

### Task 2.1: 创建别名映射配置

**Files:**
- Create: `configs/object_aliases.yaml`

- [ ] **Step 1: 列出 RoboCasa 厨房常见对象**

先扫一遍当前任务里实际出现的对象。可在 env_wrapper `__main__` 临时加：
```python
print("body names:", list(env._env.sim.model.body_names))
```
跑一次记录下来。

- [ ] **Step 2: 写映射文件**

```yaml
# configs/object_aliases.yaml
# 中文/英文用户描述 → RoboCasa body name 候选列表
aliases:
  药瓶: [bottle, medicine_bottle, pill_bottle]
  瓶子: [bottle]
  水瓶: [water_bottle, bottle]
  调味瓶: [seasoning_bottle, spice_bottle, bottle]
  杯子: [cup, mug, glass]
  马克杯: [mug]
  锅: [pot, pan, saucepan]
  平底锅: [pan, frying_pan]
  碗: [bowl]
  盘子: [plate, dish]
  刀: [knife]
  勺: [spoon, ladle]
  叉子: [fork]
```

> 实际跑通后再补全。先列 8-10 个常见的。

- [ ] **Step 3: commit**

```bash
git add configs/object_aliases.yaml
git commit -m "feat: add object alias map for grounding Chinese targets"
```

---

### Task 2.2: 实现 `ObjectGrounding` + `ground_object`

**Files:**
- Modify: `src/env_wrapper.py`

- [ ] **Step 1: 加 `ObjectGrounding` dataclass**

放到 EnvConfig 旁边：

```python
@dataclass
class ObjectGrounding:
    user_target: str
    canonical_name: str
    sim_body_name: str
    position_m: tuple[float, float, float]
    confidence: float
    source: str  # "alias_map" | "fuzzy_match" | "llm_match"
```

- [ ] **Step 2: 加私有 `_load_aliases` + `_get_body_pos`**

```python
def _load_aliases(self) -> dict[str, list[str]]:
    path = Path("configs/object_aliases.yaml")
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("aliases", {})

def _get_body_pos(self, body_name: str) -> np.ndarray | None:
    sim = self._env.sim
    try:
        body_id = sim.model.body_name2id(body_name)
        return np.asarray(sim.data.body_xpos[body_id], dtype=np.float32)
    except (KeyError, ValueError):
        return None
```

- [ ] **Step 3: 实现 `ground_object`**

```python
def ground_object(self, user_target: str) -> ObjectGrounding | None:
    if not self._latest_obs:
        self.reset()
    
    aliases = getattr(self, "_aliases", None)
    if aliases is None:
        self._aliases = self._load_aliases()
        aliases = self._aliases
    
    sim_body_names = list(self._env.sim.model.body_names)
    
    # 1) 别名精确匹配
    candidates = aliases.get(user_target, [])
    for canonical in candidates:
        for body in sim_body_names:
            if canonical.lower() in body.lower():
                pos = self._get_body_pos(body)
                if pos is not None:
                    return ObjectGrounding(
                        user_target=user_target,
                        canonical_name=canonical,
                        sim_body_name=body,
                        position_m=tuple(pos.tolist()),
                        confidence=0.9,
                        source="alias_map",
                    )
    
    # 2) body name 模糊匹配 (英文用户输入)
    for body in sim_body_names:
        if user_target.lower() in body.lower():
            pos = self._get_body_pos(body)
            if pos is not None:
                return ObjectGrounding(
                    user_target=user_target,
                    canonical_name=body,
                    sim_body_name=body,
                    position_m=tuple(pos.tolist()),
                    confidence=0.6,
                    source="fuzzy_match",
                )
    
    logger.warning(f"[ground_object] failed to ground '{user_target}'")
    return None
```

- [ ] **Step 4: 自测**

```python
g = env.ground_object("药瓶")
print(g)
g2 = env.ground_object("不存在的物体")
print(g2)  # None
```

- [ ] **Step 5: commit**

```bash
git commit -am "feat(env): implement ground_object with alias map + fuzzy match"
```

---

## Phase 3: 抓取动作

**目标**: `env.grasp_at(target_pos_m)` 完成开夹爪 → 预抓取 → 下降 → 闭夹爪 → 提升。

### Task 3.1: 实现 `grasp_at`

**Files:**
- Modify: `src/env_wrapper.py`

- [ ] **Step 1: 加 `_gripper_action`**

```python
def _gripper_action(self, gripper_value: float, n_steps: int = 10) -> None:
    """单独控制夹爪 (gripper_value: -1 开, +1 关)"""
    action = np.zeros(self._env.action_dim, dtype=np.float32)
    # 夹爪通常是 action 的最后一维 (具体位置看 controller 配置)
    # PandaMobile OSC: 0:3 pos, 3:6 rot, 6 gripper, 7+ base
    gripper_idx = 6  # 校验后调整
    action[gripper_idx] = gripper_value
    for _ in range(n_steps):
        try:
            obs, _, _, _ = self._env.step(action)
            self._latest_obs = obs
        except Exception as e:
            logger.warning(f"[gripper] step failed: {e}")
            break
```

> ⚠️ `gripper_idx` 需要先用 `print(env._env.action_spec)` 确认 PandaMobile 的实际 action layout，可能不是 6。第一次跑前先调试。

- [ ] **Step 2: 实现 `grasp_at`**

```python
def grasp_at(
    self,
    target_pos_m,
    pre_grasp_height_m: float = 0.10,
) -> bool:
    """完整抓取流程: 开爪 → 预抓取 → 下降 → 关爪 → 提升"""
    target = np.asarray(target_pos_m, dtype=np.float32)
    pre_grasp = target + np.array([0.0, 0.0, pre_grasp_height_m], dtype=np.float32)
    
    logger.info(f"[grasp] open gripper")
    self._gripper_action(-1.0, n_steps=8)
    
    logger.info(f"[grasp] move to pre-grasp {pre_grasp}")
    if not self.move_arm_to(pre_grasp):
        return False
    
    logger.info(f"[grasp] descend to target {target}")
    if not self.move_arm_to(target):
        return False
    
    logger.info(f"[grasp] close gripper")
    self._gripper_action(+1.0, n_steps=15)
    
    logger.info(f"[grasp] lift to {pre_grasp}")
    if not self.move_arm_to(pre_grasp):
        return False
    
    return True
```

- [ ] **Step 3: 加 `eye_in_hand_viewpoint()` 辅助方法**

```python
def eye_in_hand_viewpoint(self):
    """快速获取 eye_in_hand viewpoint 对象"""
    from .active_planner import Viewpoint
    return Viewpoint(
        name="robot0_eye_in_hand",
        position=(0, 0, 30),
        orientation=(0, -90, 0),
        purpose="抓取后视觉验证",
    )
```

- [ ] **Step 4: 自测**

```python
g = env.ground_object("药瓶")
if g:
    ok = env.grasp_at(g.position_m)
    print(f"grasp ok={ok}")
    obs = env.observe(env.eye_in_hand_viewpoint())
    print(f"verify image: {obs.image_path}")
```

- [ ] **Step 5: commit**

```bash
git commit -am "feat(env): implement grasp_at with pre-grasp/descend/close/lift"
```

---

## Phase 4: ActionDecider 模块

**目标**: LLM 判断查询是否需要物理动作。

### Task 4.1: 写 prompt

**Files:**
- Create: `prompts/action_decider.txt`

- [ ] **Step 1: 创建 prompt 文件**

```text
你是 EmboSight 视障辅助机器人的行动决策模块。

输入:
- 用户查询 (中文)
- 场景描述 (五维度结构化 JSON)

任务: 判断用户是否需要机器人执行物理动作。

判断规则:
- 含 "拿/取/递/给我/帮我..取" → action_type = "grasp"
- 含 "在哪/有什么/告诉我/描述" → action_type = "none"
- 含 "指向" → action_type = "point"
- 模糊时优先 "none"，避免误抓

输出格式 (严格 JSON, 不加任何额外文字):
{
  "action_type": "grasp" | "point" | "none",
  "target_object": "目标物体的中文名 (none 时填空字符串)",
  "reason": "为什么这样判断",
  "safety_constraints": ["从场景描述里提取的安全相关约束"],
  "require_confirmation": true
}

示例 1:
查询: "帮我拿药瓶"
描述: {"objects":["药瓶","锅"], "safety_alerts":["右侧热锅 [高风险]"]}
输出: {"action_type":"grasp","target_object":"药瓶","reason":"用户明确请求拿取","safety_constraints":["避开右侧热锅"],"require_confirmation":true}

示例 2:
查询: "桌上有什么"
描述: {"objects":["药瓶","碗"]}
输出: {"action_type":"none","target_object":"","reason":"用户只想了解信息","safety_constraints":[],"require_confirmation":false}
```

- [ ] **Step 2: commit**

```bash
git add prompts/action_decider.txt
git commit -m "feat(prompts): add action_decider prompt for intent → action mapping"
```

---

### Task 4.2: 实现 ActionDecider

**Files:**
- Create: `src/action_decider.py`

- [ ] **Step 1: 写完整模块**

```python
"""LLM 行动决策模块 — 判断查询是否需要物理动作。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ActionPlan:
    action_type: str  # "grasp" | "point" | "none"
    target_object: str = ""
    reason: str = ""
    safety_constraints: list[str] = field(default_factory=list)
    require_confirmation: bool = True

    @property
    def needs_execution(self) -> bool:
        return self.action_type in ("grasp", "point")


class ActionDecider:
    def __init__(
        self,
        llm_client,
        prompt_path: str = "prompts/action_decider.txt",
    ) -> None:
        self.llm = llm_client
        self.prompt_path = Path(prompt_path)
        self._system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt not found: {self.prompt_path}")
        return self.prompt_path.read_text(encoding="utf-8")

    def decide(self, query: str, description: Any) -> ActionPlan:
        """根据查询和场景描述决定行动"""
        desc_dict = description.to_dict() if hasattr(description, "to_dict") else description
        user_msg = f"查询: {query}\n场景描述: {json.dumps(desc_dict, ensure_ascii=False)}"
        
        try:
            response = self.llm.generate(
                user_msg,
                system=self._system_prompt,
                json_mode=True,
            )
            data = json.loads(response)
            return ActionPlan(
                action_type=data.get("action_type", "none"),
                target_object=data.get("target_object", ""),
                reason=data.get("reason", ""),
                safety_constraints=data.get("safety_constraints", []),
                require_confirmation=data.get("require_confirmation", True),
            )
        except Exception as e:
            logger.warning(f"[ActionDecider] failed, fallback to none: {e}")
            return ActionPlan(
                action_type="none",
                reason=f"决策失败: {e}",
            )


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    from src.llm_backend import LLMBackend
    
    llm = LLMBackend()
    decider = ActionDecider(llm)
    
    plan1 = decider.decide("帮我拿药瓶", {"objects": ["药瓶"], "safety_alerts": []})
    print("plan1:", plan1)
    
    plan2 = decider.decide("桌上有什么", {"objects": ["药瓶", "碗"]})
    print("plan2:", plan2)
```

- [ ] **Step 2: 自测**

```bash
python -m src.action_decider
```

预期: plan1.action_type == "grasp", plan2.action_type == "none"

- [ ] **Step 3: commit**

```bash
git add src/action_decider.py
git commit -m "feat: ActionDecider module for query → action_type via LLM"
```

---

## Phase 5: ActionExecutor 模块

**目标**: 完整执行 grounding → 风险路径 → 抓取 → 验证 闭环。

### Task 5.1: 数据结构 + 骨架

**Files:**
- Create: `src/action_executor.py`

- [ ] **Step 1: 写数据结构**

```python
"""ActionExecutor — 风险感知运动 + 抓取 + 语义验证闭环 (创新⑤⑥)"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .action_decider import ActionPlan
from .env_wrapper import ObjectGrounding

logger = logging.getLogger(__name__)


@dataclass
class NoGoZone:
    name: str
    center_m: tuple[float, float, float]
    radius_m: float
    risk_level: str  # "high" | "medium" | "low"
    reason: str


@dataclass
class ActionResult:
    success: bool
    executed: bool
    grounding: Optional[ObjectGrounding] = None
    verification_match: bool = False
    message: str = ""
    no_go_zones: list[NoGoZone] = field(default_factory=list)
    waypoints: list[tuple[float, float, float]] = field(default_factory=list)


class ActionExecutor:
    def __init__(
        self,
        scene_describer,
        no_go_radius_m: float = 0.15,
        match_threshold: float = 0.5,
    ) -> None:
        self.describer = scene_describer
        self.no_go_radius_m = no_go_radius_m
        self.match_threshold = match_threshold
```

- [ ] **Step 2: commit**

```bash
git add src/action_executor.py
git commit -m "feat: ActionExecutor skeleton with NoGoZone/ActionResult"
```

---

### Task 5.2: 实现风险区域提取 + 路径规划

**Files:**
- Modify: `src/action_executor.py`

- [ ] **Step 1: 加 `_extract_no_go_zones`**

从 safety_constraints 中匹配关键词提取危险物体，调用 `env.ground_object` 获得位置。

```python
HAZARD_KEYWORDS = {
    "high": ["热", "烫", "火", "刀", "锐"],
    "medium": ["玻璃", "易碎", "尖", "重"],
    "low": ["不稳", "湿"],
}

HAZARD_OBJECTS = ["锅", "刀", "杯", "玻璃", "瓶"]

def _extract_no_go_zones(
    self,
    safety_constraints: list[str],
    env,
) -> list[NoGoZone]:
    zones: list[NoGoZone] = []
    for constraint in safety_constraints:
        risk_level = "low"
        for level, kws in HAZARD_KEYWORDS.items():
            if any(kw in constraint for kw in kws):
                risk_level = level
                break
        
        for obj in HAZARD_OBJECTS:
            if obj in constraint:
                g = env.ground_object(obj)
                if g is not None:
                    zones.append(NoGoZone(
                        name=obj,
                        center_m=g.position_m,
                        radius_m=self.no_go_radius_m,
                        risk_level=risk_level,
                        reason=constraint,
                    ))
                    logger.info(f"[no_go] {obj} at {g.position_m} ({risk_level})")
                    break
    return zones
```

- [ ] **Step 2: 加 `_plan_safe_path`**

```python
def _plan_safe_path(
    self,
    start_m: np.ndarray,
    goal_m: np.ndarray,
    no_go_zones: list[NoGoZone],
) -> list[np.ndarray]:
    """生成 waypoint 列表 (含起点除外的所有中间点和终点)"""
    if not no_go_zones:
        return [goal_m]
    
    # 检查直线是否穿过任一 no-go zone
    blocking_zone = None
    for zone in no_go_zones:
        if self._line_intersects_sphere(start_m, goal_m, np.asarray(zone.center_m), zone.radius_m):
            blocking_zone = zone
            break
    
    if blocking_zone is None:
        return [goal_m]
    
    # 生成绕行 waypoint: 在 zone 侧面拉一个垂直偏移点
    center = np.asarray(blocking_zone.center_m)
    direction = goal_m - start_m
    direction_normed = direction / max(np.linalg.norm(direction), 1e-6)
    perp = np.array([-direction_normed[1], direction_normed[0], 0.0])  # XY 平面垂直
    
    offset = blocking_zone.radius_m + 0.10  # 余量 10cm
    detour = center + perp * offset
    detour[2] = max(detour[2], start_m[2])  # 不要降到地面
    
    logger.info(f"[plan] detour via {detour} to avoid {blocking_zone.name}")
    return [detour, goal_m]

@staticmethod
def _line_intersects_sphere(p1, p2, c, r):
    p1, p2, c = map(np.asarray, (p1, p2, c))
    d = p2 - p1
    f = p1 - c
    a = float(np.dot(d, d))
    b = 2 * float(np.dot(f, d))
    c_ = float(np.dot(f, f)) - r * r
    disc = b * b - 4 * a * c_
    if disc < 0:
        return False
    disc = np.sqrt(disc)
    t1 = (-b - disc) / (2 * a)
    t2 = (-b + disc) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)
```

- [ ] **Step 3: commit**

```bash
git commit -am "feat(executor): no_go_zones extraction + simple safe path planner"
```

---

### Task 5.3: 实现 execute 主流程 + 语义验证

**Files:**
- Modify: `src/action_executor.py`

- [ ] **Step 1: 加 `_verify_consistency`**

```python
@staticmethod
def _tokenize_zh(text: str) -> set[str]:
    """简单分词: 提取所有 2-4 字中文片段"""
    chars = re.sub(r"[^\u4e00-\u9fa5a-zA-Z]", " ", text)
    tokens = set()
    for word in chars.split():
        for n in (2, 3, 4):
            for i in range(len(word) - n + 1):
                tokens.add(word[i:i+n])
        if word:
            tokens.add(word)
    return tokens

def _verify_consistency(
    self,
    target_object: str,
    verify_desc,
) -> tuple[bool, float]:
    """语义匹配: 验证描述中是否包含目标物体特征"""
    target_tokens = self._tokenize_zh(target_object)
    
    desc_text = ""
    if hasattr(verify_desc, "objects"):
        desc_text += " ".join(verify_desc.objects) + " "
    if hasattr(verify_desc, "tactile"):
        desc_text += " ".join(str(t) for t in verify_desc.tactile)
    
    desc_tokens = self._tokenize_zh(desc_text)
    
    if not target_tokens or not desc_tokens:
        return False, 0.0
    
    overlap = target_tokens & desc_tokens
    score = len(overlap) / len(target_tokens)
    return score >= self.match_threshold, score
```

- [ ] **Step 2: 加 `execute`**

```python
def execute(self, plan: ActionPlan, env) -> ActionResult:
    if plan.action_type != "grasp":
        return ActionResult(success=True, executed=False, message="无需物理动作")
    
    # 1) Grounding
    grounding = env.ground_object(plan.target_object)
    if grounding is None:
        return ActionResult(
            success=False,
            executed=False,
            message=f"无法定位目标物体: {plan.target_object}",
        )
    target_pos_m = np.asarray(grounding.position_m, dtype=np.float32)
    
    # 2) 风险感知
    no_go_zones = self._extract_no_go_zones(plan.safety_constraints, env)
    
    # 3) 路径规划
    start = env.get_eef_pos()
    pre_grasp = target_pos_m + np.array([0.0, 0.0, 0.10])
    waypoints = self._plan_safe_path(start, pre_grasp, no_go_zones)
    
    # 4) 沿 waypoint 移动
    for wp in waypoints[:-1]:  # 最后一个由 grasp_at 处理
        env.move_arm_to(wp)
    
    # 5) 抓取
    grasp_ok = env.grasp_at(target_pos_m)
    
    # 6) 语义验证
    verify_obs = env.observe(env.eye_in_hand_viewpoint())
    verify_desc = self.describer.describe(verify_obs.image_path)
    match, score = self._verify_consistency(plan.target_object, verify_desc)
    
    msg = f"{'已拿到' if match else '可能没拿到'}目标 (匹配度 {score:.2f})"
    if hasattr(verify_desc, "to_speech"):
        msg += f" — {verify_desc.to_speech()}"
    
    return ActionResult(
        success=grasp_ok and match,
        executed=True,
        grounding=grounding,
        verification_match=match,
        message=msg,
        no_go_zones=no_go_zones,
        waypoints=[tuple(w.tolist()) for w in waypoints],
    )
```

- [ ] **Step 3: commit**

```bash
git commit -am "feat(executor): execute() with safe path + grasp + semantic verify"
```

---

## Phase 6: Pipeline Step 5-6 接入

### Task 6.1: Pipeline 加 ActionDecider + ActionExecutor

**Files:**
- Modify: `src/pipeline.py`

- [ ] **Step 1: import + 初始化**

```python
from .action_decider import ActionDecider, ActionPlan
from .action_executor import ActionExecutor, ActionResult

# __init__ 末尾
self.action_decider = ActionDecider(
    self.llm,
    prompt_path=self.config.get("prompts", {}).get("action_decider", "prompts/action_decider.txt"),
)
self.action_executor = ActionExecutor(self.scene_describer)
```

- [ ] **Step 2: run() 加 Step 5-6**

在现有 aggregate 之后：

```python
# Step 5: 行动决策
logger.info("[Step 5] 行动决策")
action_plan = self.action_decider.decide(query, final_desc)
logger.info(f"  → {action_plan.action_type} target={action_plan.target_object}")

# Step 6: 行动执行
action_result = None
if action_plan.needs_execution:
    logger.info("[Step 6] 行动执行")
    action_result = self.action_executor.execute(action_plan, env)
    logger.info(f"  → success={action_result.success}, msg={action_result.message}")

# 拼接最终语音
speech = final_desc.to_speech()
if action_result is not None:
    speech += f"\n[行动结果] {action_result.message}"

return {
    "query": query,
    "subtasks": [s.to_dict() if hasattr(s, "to_dict") else s for s in subtasks],
    "observations": [...],
    "description": final_desc.to_dict() if hasattr(final_desc, "to_dict") else final_desc,
    "action_plan": action_plan.__dict__,
    "action_result": action_result.__dict__ if action_result else None,
    "speech": speech,
}
```

- [ ] **Step 3: commit**

```bash
git commit -am "feat(pipeline): integrate Step 5 ActionDecider + Step 6 ActionExecutor"
```

---

## Phase 7: 实时可视化 Demo

**目标**: `python scripts/run_demo.py --query "..." --visualize` 打开 viewer，机器人动作可见。

### Task 7.1: EnvWrapper 支持 viewer 模式

**Files:**
- Modify: `src/env_wrapper.py`

- [ ] **Step 1: EnvConfig 加 `has_renderer` 字段**

```python
@dataclass
class EnvConfig:
    # ... 现有字段
    has_renderer: bool = False  # True = 实时窗口
    has_offscreen_renderer: bool = True  # 默认离屏
```

- [ ] **Step 2: 构造 robosuite env 时传入**

```python
self._env = robosuite.make(
    env_name=self.config.env_name,
    robots=self.config.robots,
    has_renderer=self.config.has_renderer,
    has_offscreen_renderer=self.config.has_offscreen_renderer,
    use_camera_obs=True,
    camera_names=self.config.camera_names,
    # ...
)
```

- [ ] **Step 3: 加 `render()` 方法**

```python
def render(self) -> None:
    if self.config.has_renderer:
        try:
            self._env.render()
        except Exception as e:
            logger.warning(f"[render] failed: {e}")
```

- [ ] **Step 4: 在所有 step 后调用 render**

最简单做法: 在 `move_arm_to` 和 `_gripper_action` 的 `env.step` 后加 `self.render()`。

- [ ] **Step 5: commit**

```bash
git commit -am "feat(env): support has_renderer for real-time MuJoCo viewer"
```

---

### Task 7.2: run_demo 加 `--visualize`

**Files:**
- Modify: `scripts/run_demo.py`

- [ ] **Step 1: argparse 加参数**

```python
parser.add_argument("--visualize", action="store_true", help="开实时 MuJoCo viewer")
```

- [ ] **Step 2: 把 visualize 传给 EnvConfig**

```python
env_cfg = EnvConfig(
    # ...
    has_renderer=args.visualize,
    has_offscreen_renderer=True,  # 同时离屏渲染留 RGB
)
```

- [ ] **Step 3: 加每步骤进度打印**

在 pipeline 里 `logger.info` 已经有了，可在 run_demo 顶部 `logging.basicConfig(level=logging.INFO)` 保证显示。

- [ ] **Step 4: 自测 (服务器有 X 转发或本地)**

```bash
python scripts/run_demo.py --query "帮我拿药瓶" --visualize
```

- [ ] **Step 5: commit**

```bash
git commit -am "feat(demo): add --visualize flag for real-time viewer"
```

---

## Phase 8: 端到端集成测试 + 演示 query 集

### Task 8.1: 集成测试脚本

**Files:**
- Create: `scripts/test_embodied.py`

- [ ] **Step 1: 写测试脚本**

```python
"""端到端具身能力测试 (mock + 真实环境)"""
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
log = logging.getLogger("test_embodied")


def test_arm_control(env):
    log.info("=== Test 1: arm control ===")
    start = env.get_eef_pos()
    target = start + [0.0, 0.0, 0.05]
    ok = env.move_arm_to(target)
    end = env.get_eef_pos()
    log.info(f"  start={start}, end={end}, ok={ok}")
    assert ok, "move_arm_to failed"


def test_grounding(env, target="药瓶"):
    log.info(f"=== Test 2: ground_object('{target}') ===")
    g = env.ground_object(target)
    log.info(f"  {g}")
    return g


def test_grasp(env, grounding):
    log.info("=== Test 3: grasp_at ===")
    if grounding is None:
        log.warning("  skip: no grounding")
        return False
    ok = env.grasp_at(grounding.position_m)
    log.info(f"  ok={ok}")
    return ok


def test_pipeline(query="帮我拿药瓶", visualize=False):
    log.info(f"=== Test 4: full pipeline | query='{query}' ===")
    from src.env_wrapper import EnvConfig, EnvWrapper
    from src.pipeline import EmboSightPipeline
    
    env = EnvWrapper(EnvConfig(has_renderer=visualize))
    env.reset()
    
    test_arm_control(env)
    g = test_grounding(env, "药瓶")
    
    pipeline = EmboSightPipeline("configs/default.yaml")
    result = pipeline.run(query, env)
    
    out = Path("results/test_embodied_result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log.info(f"saved {out}")
    log.info(f"speech: {result['speech']}")
    
    env.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--query", default="帮我拿药瓶")
    p.add_argument("--visualize", action="store_true")
    args = p.parse_args()
    test_pipeline(args.query, args.visualize)
```

- [ ] **Step 2: 运行**

```bash
python scripts/test_embodied.py --query "帮我拿药瓶"
python scripts/test_embodied.py --query "桌上有什么" 
python scripts/test_embodied.py --query "帮我拿杯子"
```

- [ ] **Step 3: commit**

```bash
git add scripts/test_embodied.py
git commit -m "feat: end-to-end embodied capability test script"
```

---

### Task 8.2: Demo query 集

**Files:**
- Create: `docs/demo_queries.md`

- [ ] **Step 1: 列演示用 query**

```markdown
# 比赛演示 Query 集

## A: 纯感知 (action_type=none)
1. 桌子上有什么？
2. 我的药瓶在哪里？
3. 周围有什么危险吗？

## B: 抓取 (action_type=grasp)
4. 帮我拿药瓶
5. 给我一个杯子
6. 帮我把刀拿开

## C: 复合任务
7. 我想喝水，帮帮我
8. 我要吃药，需要药瓶和水

## 演示流程
1. 启动 viewer
2. 依次跑 A 类 query：展示主动观察 + 五维度描述
3. 跑 B 类：展示风险感知运动 + 抓取 + 验证
4. 现场让裁判提一个新 query：证明不是预录制
```

- [ ] **Step 2: commit**

```bash
git add docs/demo_queries.md
git commit -m "docs: add demo query set for competition"
```

---

## 完成标准

- [ ] `python -m src.env_wrapper` 通过自测 (move/grasp/ground 都正常)
- [ ] `python -m src.action_decider` 输出正确的 action_type
- [ ] `python scripts/test_embodied.py --query "帮我拿药瓶"` 端到端通过
- [ ] `python scripts/run_demo.py --query "..." --visualize` 能看到机器人实时动作
- [ ] `results/test_embodied_result.json` 包含完整 6 步输出
- [ ] git log 显示每个 Phase 都有清晰提交

---

## 后续 (比赛后做)

- 实验框架: 6 个消融实验 (见 spec)
- 大规模 query 集 (50 条)
- 量化指标自动统计
- baseline 对比脚本
- 论文写作

## 风险预案

| 风险 | 兜底方案 |
|---|---|
| OSC 控制不收敛 | max_steps=200 超时 + 用关节空间控制兜底 |
| viewer 在服务器开不出来 | 自动 fallback 到 offscreen + 保存 mp4 |
| ground_object 找不到 RoboCasa 内部物体名 | 跑一次 `print(sim.model.body_names)` 手动补 alias map |
| gripper action 维度索引错 | 第一次跑前 `print(env.action_spec)` 确认 |
| RoboCasa 现场没有"药瓶" | 用任务实际包含的物体 (cup/bowl/pan) 演示 |
