# SafeQuery-VLM Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用零样本 VLM 视觉定位 + 多视角融合 + 安全门控, 替换当前基于字符串匹配的 grounding 管道, 打通 `scene_describer → action_executor` 的信息断层, 使系统能可靠、安全地执行视障辅助抓取任务。

**Architecture:** 新建 `vlm_grounding.py` 让 VLM 一次性输出 `{bbox, category, confidence, description, safety}` 结构化结果; 新建 `scene_model.py` 承载 `GroundedObject` 统一对象表示; 新建 `safety_gate.py` 做风险/置信度门控; 改造 `scene_describer` 作为 VLM grounding 的消费者; 改造 `action_executor` 直接消费 `GroundedObject`, 不再独立 grounding。

**Tech Stack:** Qwen2.5-VL-7B (本地) · DeepSeek API · RoboCasa + Robosuite · PyYAML · pytest · Python 3.10

---

## Spec Summary (from user requirements)

国赛级标准, 以"**安全第一 + 准确识别 + 精确抓取**"为核心需求, 提出 SafeQuery-VLM 架构:

| 层 | 责任 | 实现模块 |
|---|---|---|
| L1: Query-Driven Active Perception | 按用户 query 决定需采集哪些视角 | `active_planner.py` (升级) |
| L2: Multi-View VLM Grounding | 多视角 VLM 输出候选物体 + bbox + 置信度 | `vlm_grounding.py` (新) |
| L3: Consistency + Safety Verification | 空间/语义一致性 + 安全评分 + 置信度门控 | `scene_model.py` + `safety_gate.py` (新) |
| L4: Execution + Feedback | 通过所有验证才执行抓取 | `action_executor.py` (改) |

**Not in scope:** Human-in-the-loop via ASR (真语音交互). 本期用 safety gate 自动拒绝替代. 论文里写 future work.

---

## Architectural Decisions

### AD-1: 用 VLM 一次性产出 grounding + description + safety
**不是**: 分三次 VLM 调用 (grounding / description / safety) 再融合.  
**是**: 设计一个 schema, VLM 单次推理输出所有结构化信息.

理由: VLM 推理成本高 (~15s), 一次调用比三次快 3x; 且描述/安全/grounding 本质是同一张图的不同方面, 合并输出信息一致性更好。

### AD-2: Phase 1 决定 bbox 模式还是几何文本模式
**两条实现路径**取决于 Phase 1 探测结果:
- **Path A (bbox mode)**: Qwen2.5-VL bbox 精度 ≥ 60% mIoU → 用 bbox + depth 投影到 3D
- **Path B (geometry mode)**: bbox 精度不够 → 用 VLM 文字描述 (正前方 30cm) + scene_describer 现有几何后处理

Path A 是论文卖点更强的方案; Path B 是不依赖 bbox 精度的保底方案. Phase 1 跑完才决定.

### AD-3: 保留旧 `env.ground_object` 作为 fallback
不删旧代码. Action executor 优先用新 grounding, 新 grounding 失败时降级到旧 alias_map 链. 提交前分阶段 deprecate.

### AD-4: Safety gate 用 YAML 规则表而非 LLM 判定
`configs/safety_rules.yaml` 维护 `{category: risk_level}` 映射.  
**理由**: LLM 判定每次调用成本高、结果不稳定; YAML 表可审计、可扩展、评委答辩好讲"领域知识注入"。

### AD-5: 不做真 human-in-the-loop, 用 safety gate 自动拒绝
低置信度 (< 0.75) 或高风险 (high-risk + conf < 0.9) → 直接 TTS 告知 "暂不执行, 建议您手动确认" + 返回 `success=False`. 不假装等待用户语音输入.

---

## File Structure

### Create

```
src/
├── vlm_grounding.py          # 核心: VLM 视觉定位 + schema 解析
├── scene_model.py            # GroundedObject + SceneModel 聚合
└── safety_gate.py            # 安全评分 + 置信度门控

configs/
└── safety_rules.yaml         # category → {risk_level, reason, zh_name}

tests/
├── test_vlm_grounding.py
├── test_scene_model.py
└── test_safety_gate.py

scripts/
├── probe_vlm_bbox.py         # Phase 1 探测: Qwen bbox 精度
└── probe_robocasa_depth.py   # Phase 1 探测: depth/intrinsics

prompts/
└── vlm_grounding.txt         # VLM 结构化 grounding prompt
```

### Modify

```
src/
├── scene_describer.py        # 输出 SceneModel (含 GroundedObject)
├── action_executor.py        # 消费 SceneModel, 去除独立 ground_object
├── active_planner.py         # grounding-aware viewpoint 选择
├── env_wrapper.py            # 暴露 depth + camera intrinsics
└── pipeline.py               # 串接新信息流

configs/
└── default.yaml              # 新增 vlm_grounding / safety 配置
```

### Deprecate (not delete)

```
src/env_wrapper.py
└── ground_object()           # 保留作 fallback, action_executor 不再默认调用
```

---

## Phase Overview

| Phase | 目标 | 估时 | 可回滚? | 交付物 |
|-------|------|------|--------|--------|
| **1. Probe** | 探测 Qwen VLM bbox 精度 + RoboCasa depth/intrinsics API | 2h | 是 | 两个 log + 决策记录 (决定 Path A/B) |
| **2. VLM Grounding Module** | `vlm_grounding.py` + `prompts/vlm_grounding.txt` + 单元测试 | 5h | 是 | VLM 单视角产出 GroundedCandidate list |
| **3. Scene Model + 3D Fusion** | `scene_model.py` + 多视角聚合 + 3D 投影 | 6h | 是 | GroundedObject 带 3D 坐标 + 跨视角 ID |
| **4. Safety Gate** | `safety_gate.py` + YAML 规则 + 单元测试 | 3h | 是 | 风险门控函数 + 单测全绿 |
| **5. Action Integration** | scene_describer → action_executor 信息流打通 | 4h | 是 (切回旧路径) | 端到端抓取 peeler/condiment_bottle 跑通 |
| **6. Active Planner 升级** | grounding-aware 视角选择 + 消融对比 | 5h | 是 | 3 视角消融数据 + 论文实验章节草稿 |

**总估时: 25 小时** (不含人类在环).

---

## Phase 1: Probe (2h)

### Task 1.1: 探测 Qwen2.5-VL 的 bbox 输出能力

**Files:**
- Create: `scripts/probe_vlm_bbox.py`
- Uses: `src/vlm_backend.py` (existing)

**Success Criteria:** 能拿到 VLM 对目标物体的 bbox 坐标, 且坐标跟人眼判断大致一致 (mIoU > 0.5 判 Path A 可行).

- [ ] **Step 1: Write probe script**

Create `scripts/probe_vlm_bbox.py`:

```python
"""探测 Qwen2.5-VL 的 bbox / grounding 输出能力.

目的: SafeQuery-VLM 架构决定走 bbox 路径还是几何文本 fallback.
Qwen2.5-VL 官方支持 <box> token grounding, 但 7B 模型精度未知.

运行: MUJOCO_GL=egl python scripts/probe_vlm_bbox.py
输出: 对默认场景 agentview 图跑 3 种 prompt, 打印 VLM 返回内容 + 解析后 bbox.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env_wrapper import EnvWrapper, EnvConfig
from src.vlm_backend import VLMBackend


BBOX_PROMPTS = {
    "prompt_A_qwen_native": (
        "Please locate the peeler in this image. "
        "Output the bounding box in the format: "
        "<|box_start|>(x1,y1),(x2,y2)<|box_end|>"
    ),
    "prompt_B_natural": (
        "Look at this image. Find the peeler and give me its position "
        "as bounding box coordinates [x1, y1, x2, y2] where values are "
        "integers in pixels (image is 256x256). "
        "Reply ONLY with the JSON: {\"bbox\": [x1,y1,x2,y2]} or {\"bbox\": null}"
    ),
    "prompt_C_multi_object": (
        "List all task-relevant objects in this image. For each, give:\n"
        "- name (e.g. peeler, condiment_bottle, reamer)\n"
        "- bbox: [x1, y1, x2, y2] in pixels (0-255 range)\n"
        "- confidence: 0.0 to 1.0\n"
        "Reply in JSON: {\"objects\": [{...}, ...]}"
    ),
}


def main():
    env = EnvWrapper(EnvConfig())
    env.reset()

    # 采集默认场景的 agentview_center 图
    from src.active_planner import Viewpoint
    vp = Viewpoint(
        name="robot0_agentview_center",
        position=(0, 0, 60),
        orientation=(0, -45, 0),
        purpose="probe",
    )
    obs = env.observe(vp)
    img_path = obs.image_path
    print(f"\n=== Probe image: {img_path} ===")

    # 打印 episode 真实物体类型 (ground truth)
    type_map = env._get_obj_type_map()
    print(f"Ground truth object categories: {type_map}\n")

    vlm = VLMBackend()

    for name, prompt in BBOX_PROMPTS.items():
        print(f"\n--- {name} ---")
        print(f"Prompt: {prompt[:120]}...")
        try:
            raw = vlm.describe(img_path, prompt=prompt)
            print(f"Raw output ({len(raw)} chars):")
            print(raw[:800])
            bboxes = _parse_bboxes(raw)
            print(f"Parsed bboxes: {bboxes}")
        except Exception as e:
            print(f"ERROR: {e}")

    env.close()


def _parse_bboxes(raw: str) -> list:
    """尝试从 VLM 输出解析 bbox. 多种格式都试一遍."""
    results = []

    # Format 1: Qwen native <|box_start|>(x1,y1),(x2,y2)<|box_end|>
    m = re.findall(
        r"<\|box_start\|>\(?(\d+),\s*(\d+)\)?,\s*\(?(\d+),\s*(\d+)\)?<\|box_end\|>",
        raw,
    )
    for match in m:
        results.append({"format": "qwen_native", "bbox": [int(x) for x in match]})

    # Format 2: JSON {"bbox": [x1,y1,x2,y2]}
    try:
        import json
        text = raw
        if "```" in text:
            mm = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL)
            if mm:
                text = mm.group(1)
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            if "bbox" in data and data["bbox"]:
                results.append({"format": "json_single", "bbox": data["bbox"]})
            if "objects" in data:
                for o in data["objects"]:
                    if o.get("bbox"):
                        results.append({"format": "json_multi", "name": o.get("name"), "bbox": o["bbox"]})
    except Exception as e:
        pass

    return results


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run probe on server**

```bash
git pull
MUJOCO_GL=egl PYTHONUNBUFFERED=1 python scripts/probe_vlm_bbox.py 2>&1 | tee probe_vlm.log
```

Expected output contains:
- 3 个 prompt 的 VLM 原始输出
- 至少一个 prompt 能解析出 bbox (format qwen_native / json_single / json_multi)
- bbox 坐标范围大致合理 (0-256)

- [ ] **Step 3: Decision criteria**

Record in `probe_vlm.log` comments at bottom:

```
[DECISION] VLM bbox mode:
- If any prompt produced parseable bbox AND bbox visually contains target:
  → PATH A (bbox-based grounding)
- If all prompts fail OR bbox grossly wrong (e.g. (0,0,10,10) for a large object):
  → PATH B (geometry-text fallback, use scene_describer's existing pipeline)
```

Post the log to me, I confirm PATH A or B before proceeding.

- [ ] **Step 4: Commit**

```bash
git add scripts/probe_vlm_bbox.py
git commit -m "probe: test Qwen2.5-VL bbox grounding capability"
git push
```

---

### Task 1.2: 探测 RoboCasa 的 depth + camera intrinsics API

**Files:**
- Create: `scripts/probe_robocasa_depth.py`

**Success Criteria:** 能拿到 RGB 图对应的 depth map + camera intrinsic matrix, 且能把 2D 像素 (cx, cy) + depth 反投影成 3D 世界坐标.

- [ ] **Step 1: Write probe script**

Create `scripts/probe_robocasa_depth.py`:

```python
"""探测 RoboCasa 是否暴露 depth image 和 camera intrinsics.

2D bbox → 3D 世界坐标需要:
    1. depth image (每个像素的深度)
    2. camera intrinsic matrix K (fx, fy, cx, cy)
    3. camera extrinsic: 相机在世界坐标系中的位置 + 朝向

RoboCasa 基于 robosuite, robosuite 应该通过 camera_depths=True 暴露 depth.

运行: MUJOCO_GL=egl python scripts/probe_robocasa_depth.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# 临时改造 env_wrapper 调用以请求 depth
import robocasa  # noqa
import robosuite as suite


def main():
    env = suite.make(
        env_name="PickPlaceCounterToCabinet",
        robots="PandaOmron",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["robot0_agentview_center"],
        camera_heights=256,
        camera_widths=256,
        camera_depths=True,           # ← 关键: 请求 depth
        camera_segmentations=None,
        control_freq=20,
    )
    obs = env.reset()

    print("=" * 60)
    print("ROBOCASA DEPTH + INTRINSICS PROBE")
    print("=" * 60)

    print(f"\n--- obs keys ---")
    for k in obs:
        v = obs[k]
        shape = getattr(v, "shape", "N/A")
        dtype = getattr(v, "dtype", type(v).__name__)
        print(f"  {k}: shape={shape}, dtype={dtype}")

    # 1) depth image
    depth_key = "robot0_agentview_center_depth"
    if depth_key in obs:
        depth = obs[depth_key]
        print(f"\n--- depth {depth_key} ---")
        print(f"  shape: {depth.shape}")
        print(f"  range: [{depth.min():.3f}, {depth.max():.3f}]")
        print(f"  dtype: {depth.dtype}")
    else:
        print(f"\n  !!! depth key '{depth_key}' NOT FOUND !!!")

    # 2) camera intrinsics from mujoco model
    sim = env.sim
    print(f"\n--- camera intrinsics from sim.model ---")
    try:
        cam_name = "robot0_agentview_center"
        cam_id = sim.model.camera_name2id(cam_name)
        fovy_deg = sim.model.cam_fovy[cam_id]
        height = 256
        width = 256
        fy = 0.5 * height / np.tan(0.5 * np.radians(fovy_deg))
        fx = fy  # 假设正方形像素
        cx = width / 2
        cy = height / 2
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        print(f"  fovy(deg) = {fovy_deg}")
        print(f"  K matrix:\n{K}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 3) camera extrinsic (position + orientation in world)
    print(f"\n--- camera extrinsic ---")
    try:
        cam_pos = sim.data.cam_xpos[cam_id]
        cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3)
        print(f"  position (world): {cam_pos}")
        print(f"  rotation matrix (world):\n{cam_mat}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 4) 试做一次 2D→3D 反投影验证
    print(f"\n--- 2D→3D backprojection test ---")
    if depth_key in obs:
        depth = obs[depth_key]
        # 取图像中心一点
        u, v = 128, 128
        z = float(depth[v, u])
        # normalized depth buffer [0,1] → real distance (需要 near/far planes)
        extent = sim.model.stat.extent
        near = sim.model.vis.map.znear * extent
        far = sim.model.vis.map.zfar * extent
        real_z = near / (1.0 - z * (1.0 - near / far))
        print(f"  pixel (128,128) depth buffer = {z:.4f}")
        print(f"  near={near:.3f}, far={far:.3f}, real_z={real_z:.3f}m")

        # 像素 → 相机系
        x_cam = (u - cx) * real_z / fx
        y_cam = (v - cy) * real_z / fy
        z_cam = real_z
        # 相机系 → 世界系 (mujoco 相机朝 -z, y 朝上)
        pt_cam = np.array([x_cam, -y_cam, -z_cam])  # convert from image coord to cam
        pt_world = cam_mat @ pt_cam + cam_pos
        print(f"  world coord: {pt_world}")

    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run probe on server**

```bash
git pull
MUJOCO_GL=egl PYTHONUNBUFFERED=1 python scripts/probe_robocasa_depth.py 2>&1 | tee probe_depth.log
```

Expected output contains:
- depth image shape (256, 256) or (256, 256, 1)
- K matrix with sensible fx/fy (should be ~222 for 60 deg fovy on 256x256)
- cam_pos + cam_mat non-trivial values
- 2D→3D backprojection 的 world coord 看起来合理 (落在桌面高度 z ~ 0.9-1.1m 范围)

- [ ] **Step 3: Commit**

```bash
git add scripts/probe_robocasa_depth.py
git commit -m "probe: test RoboCasa depth image and camera intrinsics"
git push
```

---

### Task 1.3: Phase 1 Decision Checkpoint

**This is a human review step.** User runs both probes, reviews logs, decides:

| Probe | Result | Decision |
|-------|--------|----------|
| VLM bbox | 至少一个 prompt 能解析出合理 bbox | **PATH A** (bbox mode) |
| VLM bbox | 所有 prompt 失败或 bbox 乱飞 | **PATH B** (geometry text mode) |
| RoboCasa depth | depth + K + extrinsic 都能拿 | 3D 投影可用 |
| RoboCasa depth | depth key 不存在 | 退化到 scene_describer 的"方向+距离"文本 |

**After decision, update this plan document Section "Phase 2 Path Selection" with the chosen path**, then proceed to Phase 2.

**User action: Post me both logs, I confirm path, then we start Phase 2.**

---

## Phase 2: VLM Grounding Module (5h)

> **NOTE:** Detailed bite-sized steps in Phase 2 will be filled in **after Phase 1 results decide Path A vs B**. Below is the architectural design covering both paths.

### Files

- Create: `src/vlm_grounding.py`
- Create: `prompts/vlm_grounding.txt`
- Create: `tests/test_vlm_grounding.py`

### Phase 2 Path Selection

**To be filled after Phase 1**: [PATH A | PATH B]

### Core API (unchanged across paths)

```python
# src/vlm_grounding.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class GroundedCandidate:
    """Single-view VLM grounding result for ONE candidate object."""
    label: str                      # VLM-given label, e.g. "peeler"
    confidence: float               # 0.0 - 1.0
    description: str                # Free-text description for downstream VLM description

    # Path A: bbox-based
    bbox_2d: Optional[tuple] = None  # (x1, y1, x2, y2) in pixels
    # Path B: geometry-text-based (from scene_describer's existing output)
    direction_zh: Optional[str] = None  # "左前方"
    distance_cm: Optional[float] = None

    # Optional
    color: Optional[str] = None
    material: Optional[str] = None


class VLMGrounder:
    """Single-view VLM grounding using Qwen2.5-VL.

    Wraps VLMBackend with a structured grounding prompt. Given one image
    and a user query, returns candidates with bbox OR direction/distance.
    """

    def __init__(self, vlm_backend, prompt_path: str):
        self.vlm = vlm_backend
        self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")

    def ground(
        self,
        image_path: str,
        user_query: str,
        scene_context: Optional[str] = None,
    ) -> list[GroundedCandidate]:
        """Ask VLM to ground user_query in image, return all candidate objects."""
        prompt = self._build_prompt(user_query, scene_context)
        raw = self.vlm.describe(image_path, prompt=prompt)
        return self._parse(raw)

    def _build_prompt(self, query: str, ctx: Optional[str]) -> str:
        # Fill in query, scene_context into template
        ...

    def _parse(self, raw: str) -> list[GroundedCandidate]:
        # Parse JSON or Qwen native format, yield GroundedCandidates
        ...
```

### Prompt Template (Path A example)

`prompts/vlm_grounding.txt`:

```text
You are a visual grounding assistant for an assistive robot serving
visually-impaired users.

User query: "{{ user_query }}"
Scene context (from prior perception): {{ scene_context }}

Your task:
1. Look at the image carefully.
2. Find ALL graspable task-relevant objects (peeler, bottle, cup, etc.).
3. For EACH object, output:
   - "label": English lowercase category
   - "bbox": [x1, y1, x2, y2] in pixels (image is 256x256)
   - "confidence": 0.0-1.0
   - "description": Chinese 1-sentence description
   - "matches_query": true if this object semantically matches '{{ user_query }}'
   - "color": main color
   - "material": plastic/glass/metal/ceramic/wood/fabric
4. Be STRICT about semantic matching:
   - "药瓶" means medicine/pill bottle, NOT condiment_bottle or water_bottle
   - "杯子" means cup/mug, NOT glass or bowl

Reply with ONLY a JSON object, no markdown:
{
  "objects": [
    {
      "label": "peeler",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.85,
      "description": "白色塑料削皮器, 带黑色握柄",
      "matches_query": false,
      "color": "white",
      "material": "plastic"
    }
  ]
}
```

### Test Strategy

```python
# tests/test_vlm_grounding.py

def test_parse_single_object_json():
    """VLM 返回 1 个物体的 JSON, 能正确解析为 GroundedCandidate."""
    raw = '{"objects": [{"label": "peeler", "bbox": [10,20,80,120], ...}]}'
    candidates = VLMGrounder._parse_static(raw)
    assert len(candidates) == 1
    assert candidates[0].label == "peeler"
    assert candidates[0].bbox_2d == (10, 20, 80, 120)

def test_parse_malformed_json_does_not_crash():
    """VLM 返回畸形 JSON, 不应崩溃, 返回空列表 + 警告日志."""
    ...

def test_matches_query_filter():
    """有 matches_query=True 的对象才是用户目标候选."""
    ...

def test_strict_semantic_rejection():
    """'药瓶' query 下, condiment_bottle 不应 matches_query=True.
    (这是 integration test, 需真 VLM; 在 mock 测里验证 filter 逻辑即可.)"""
    ...
```

### Bite-sized Task Breakdown (filled after Phase 1)

- [ ] Task 2.1: Define GroundedCandidate dataclass + failing test
- [ ] Task 2.2: Write VLMGrounder skeleton + failing test for `ground()` 方法签名
- [ ] Task 2.3: Write prompt template
- [ ] Task 2.4: Implement `_parse()` (Path A JSON bbox OR Path B JSON text)
- [ ] Task 2.5: Implement `_build_prompt()` with context injection
- [ ] Task 2.6: Integration test with real VLM on 1 sample image (服务器)
- [ ] Task 2.7: Commit

---

## Phase 3: Scene Model + 3D Fusion (6h)

### Files

- Create: `src/scene_model.py`
- Create: `tests/test_scene_model.py`
- Modify: `src/env_wrapper.py` (加 `get_depth_image()`, `get_camera_intrinsics()`, `get_camera_extrinsic()`)

### Core Data Model

```python
# src/scene_model.py

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GroundedObject:
    """Unified scene object aggregated across multi-view observations."""
    # Identity
    object_id: str                    # e.g. "obj_0", locally-unique
    label: str                        # "peeler"
    chinese_name: str                 # "削皮器"
    
    # Grounding (aggregated)
    position_m: tuple                 # 3D world, best estimate
    position_confidence: float        # how reliable is position_m
    
    # Sources
    observed_in_views: list[str]      # ["agentview_center", "frontview"]
    per_view_bbox: dict               # {"agentview_center": (x1,y1,x2,y2), ...}
    per_view_desc: dict               # {"agentview_center": "白色塑料...", ...}
    
    # Match to user query
    user_target_match_score: float    # 0-1 vs current user query
    match_reason: str                 # "exact category match" / "similar type"
    
    # Safety (filled by safety_gate.py)
    safety_risk: str = "unknown"      # "safe" / "fragile" / "hot" / "sharp" / "high"
    safety_reason: str = ""
    
    # Physical (from env, if available)
    body_name: Optional[str] = None   # "obj_main" etc, for grasp API
    category_gt: Optional[str] = None # "peeler" from ep_meta (ground truth)


class SceneModel:
    """Aggregates VLM multi-view observations into unified GroundedObjects."""
    
    def __init__(self, alignment_threshold_m: float = 0.15):
        self._objects: list[GroundedObject] = []
        self._threshold = alignment_threshold_m
    
    def add_view(
        self,
        viewpoint_name: str,
        candidates: list[GroundedCandidate],
        image_to_world_projector,  # callable (bbox) -> 3D pos
    ) -> None:
        """Add a single view's grounding candidates to the model.
        
        For each candidate:
            1. Project bbox center → 3D world via depth + intrinsics + extrinsic
            2. Find existing GroundedObject within threshold_m → merge
            3. Otherwise create new GroundedObject
        """
        ...
    
    def ground_user_query(self, query: str) -> list[GroundedObject]:
        """Return objects sorted by user_target_match_score desc."""
        ...
    
    def __len__(self) -> int:
        return len(self._objects)
```

### 3D Projection Helper (in env_wrapper or new module)

```python
def project_bbox_to_world(
    bbox_2d: tuple,
    depth_image: np.ndarray,
    K: np.ndarray,
    cam_pos_world: np.ndarray,
    cam_rot_world: np.ndarray,
) -> np.ndarray:
    """2D pixel bbox → 3D world coord of bbox center at depth surface."""
    x1, y1, x2, y2 = bbox_2d
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    # depth buffer value at center pixel
    z = depth_image[int(cy), int(cx)]
    # normalize buffer → real distance (mujoco specific)
    ...
    # back-project
    x_cam = (cx - K[0,2]) * z / K[0,0]
    y_cam = (cy - K[1,2]) * z / K[1,1]
    pt_cam = np.array([x_cam, -y_cam, -z])
    pt_world = cam_rot_world @ pt_cam + cam_pos_world
    return pt_world
```

### Test Strategy

```python
def test_add_single_view_creates_objects():
    sm = SceneModel()
    sm.add_view("v1", [candidate_peeler], mock_projector)
    assert len(sm) == 1
    assert sm._objects[0].label == "peeler"

def test_add_second_view_same_object_merges():
    sm = SceneModel(alignment_threshold_m=0.15)
    sm.add_view("v1", [c_peeler_at_pos1], mock_projector)
    sm.add_view("v2", [c_peeler_at_similar_pos], mock_projector)
    assert len(sm) == 1  # merged
    assert len(sm._objects[0].observed_in_views) == 2

def test_add_view_different_object_creates_new():
    ...

def test_ground_user_query_returns_sorted():
    ...
```

### Bite-sized Task Breakdown

- [ ] Task 3.1: `GroundedObject` dataclass + 1 minimal test
- [ ] Task 3.2: `SceneModel` skeleton + `__len__` test
- [ ] Task 3.3: `add_view` single-view test (no alignment)
- [ ] Task 3.4: `add_view` multi-view merge test (happy path)
- [ ] Task 3.5: `ground_user_query` sorting test
- [ ] Task 3.6: Add `env_wrapper.get_depth_image()`, `get_camera_intrinsics()`, `get_camera_extrinsic()` + smoke test
- [ ] Task 3.7: Implement `project_bbox_to_world()` + numerical test (simulated depth)
- [ ] Task 3.8: Integration test with real env on server
- [ ] Task 3.9: Commit

---

## Phase 4: Safety Gate (3h)

### Files

- Create: `src/safety_gate.py`
- Create: `configs/safety_rules.yaml`
- Create: `tests/test_safety_gate.py`

### Safety Rules YAML

```yaml
# configs/safety_rules.yaml
# RoboCasa 物体类别 → 风险等级 + 中文名 + 说明
# risk_level: safe / fragile / hot / sharp / high
# high = 视障场景下绝对危险 (例如锐器、开刃刀)
# hot  = 可能温热 (锅/壶)
# sharp = 有刃/尖角但一般场景安全 (削皮器)
# fragile = 易碎 (玻璃/陶瓷)
# safe = 无明显风险

categories:
  # 工具类
  peeler:
    risk_level: sharp
    zh_name: 削皮器
    reason: 刀片锋利, 握持时避免手部滑入刀口
  reamer:
    risk_level: safe
    zh_name: 榨汁器
    reason: 手动工具, 一般无风险
  knife:
    risk_level: high
    zh_name: 刀
    reason: 开刃刀具, 视障场景需人工确认
  
  # 容器类
  cup:
    risk_level: fragile
    zh_name: 杯子
    reason: 常为陶瓷/玻璃, 易碎
  mug:
    risk_level: fragile
    zh_name: 马克杯
    reason: 陶瓷易碎
  glass:
    risk_level: fragile
    zh_name: 玻璃杯
    reason: 玻璃易碎且锋利
  bowl:
    risk_level: fragile
    zh_name: 碗
    reason: 陶瓷易碎
  plate:
    risk_level: fragile
    zh_name: 盘子
    reason: 陶瓷易碎
  
  # 瓶类
  bottle:
    risk_level: safe
    zh_name: 瓶子
    reason: 一般塑料/玻璃, 小心易碎
  condiment_bottle:
    risk_level: safe
    zh_name: 调味瓶
    reason: 装调味品, 一般塑料/玻璃
  water_bottle:
    risk_level: safe
    zh_name: 水瓶
    reason: 塑料材质
  
  # 食物类
  apple:
    risk_level: safe
    zh_name: 苹果
    reason: 水果无风险
  banana:
    risk_level: safe
    zh_name: 香蕉
    reason: 水果无风险
  bread:
    risk_level: safe
    zh_name: 面包
    reason: 食物无风险
  
  # 烹饪类 (可能高温)
  pot:
    risk_level: hot
    zh_name: 锅
    reason: 可能温热, 请用手背先试温
  pan:
    risk_level: hot
    zh_name: 平底锅
    reason: 可能温热
  kettle:
    risk_level: hot
    zh_name: 水壶
    reason: 可能内有热水
  
  # Fallback (未匹配类别)
  _default:
    risk_level: unknown
    zh_name: 未知物体
    reason: 建议您手动确认

# 门控阈值
gates:
  # 一般执行门槛
  min_confidence: 0.75
  
  # 高风险类别需更高置信度才执行
  high_risk_min_confidence: 0.90
  high_risk_categories: [high, hot, sharp]
  
  # 绝对拒绝类别 (即使 conf=1.0 也不执行)
  never_execute_categories: []  # 当前场景无此类
```

### Safety Gate API

```python
# src/safety_gate.py

from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class SafetyDecision:
    allow_execute: bool
    risk_level: str
    confidence: float
    reason_user: str           # TTS 给用户的原因 (中文)
    reason_log: str            # 日志用 (英文技术细节)


class SafetyGate:
    """根据物体类别 + 置信度决定是否执行抓取."""
    
    def __init__(self, rules_path: str = "configs/safety_rules.yaml"):
        data = yaml.safe_load(Path(rules_path).read_text(encoding="utf-8"))
        self._categories = data.get("categories", {})
        self._gates = data.get("gates", {})
    
    def check(self, grounded_object) -> SafetyDecision:
        """Evaluate a GroundedObject for safety compliance.
        
        Inputs:
            grounded_object: has .label, .user_target_match_score, .position_confidence
        
        Outputs:
            SafetyDecision with allow_execute True/False
        """
        label = grounded_object.label
        conf = grounded_object.user_target_match_score
        
        # 查规则表
        rule = self._categories.get(label, self._categories["_default"])
        risk = rule["risk_level"]
        zh_name = rule["zh_name"]
        reason = rule["reason"]
        
        # 决策逻辑
        min_conf = self._gates["min_confidence"]
        if risk in self._gates["high_risk_categories"]:
            min_conf = max(min_conf, self._gates["high_risk_min_confidence"])
        
        if risk in self._gates.get("never_execute_categories", []):
            return SafetyDecision(
                allow_execute=False,
                risk_level=risk,
                confidence=conf,
                reason_user=f"检测到{zh_name}, 视障场景下不执行此类操作。{reason}",
                reason_log=f"[safety] reject: never_execute category '{label}'",
            )
        
        if conf < min_conf:
            return SafetyDecision(
                allow_execute=False,
                risk_level=risk,
                confidence=conf,
                reason_user=f"我不太确定眼前是否为{zh_name} (置信度 {conf:.0%}), 建议您手动确认。",
                reason_log=f"[safety] reject: conf {conf:.2f} < threshold {min_conf}",
            )
        
        # 通过
        return SafetyDecision(
            allow_execute=True,
            risk_level=risk,
            confidence=conf,
            reason_user=f"已定位{zh_name} (置信度 {conf:.0%})。{reason}" if reason else "",
            reason_log=f"[safety] pass: {label} risk={risk} conf={conf:.2f}",
        )
    
    def update_object_safety(self, grounded_object) -> None:
        """Inject safety info into GroundedObject (mutates in place)."""
        label = grounded_object.label
        rule = self._categories.get(label, self._categories["_default"])
        grounded_object.safety_risk = rule["risk_level"]
        grounded_object.safety_reason = rule["reason"]
```

### Bite-sized Task Breakdown

- [ ] **Task 4.1: Write `configs/safety_rules.yaml`** — copy content above

- [ ] **Task 4.2: Write failing test `test_safety_gate_pass_happy_path`**

```python
# tests/test_safety_gate.py
def test_gate_pass_when_safe_and_high_confidence():
    gate = SafetyGate("configs/safety_rules.yaml")
    obj = GroundedObject(
        object_id="o0", label="peeler", chinese_name="削皮器",
        position_m=(0,0,0), position_confidence=0.9,
        observed_in_views=[], per_view_bbox={}, per_view_desc={},
        user_target_match_score=0.95, match_reason="",
    )
    decision = gate.check(obj)
    assert decision.allow_execute is True
    assert decision.risk_level == "sharp"
```

- [ ] **Task 4.3: Run test, verify FAIL** (SafetyGate class doesn't exist yet)

- [ ] **Task 4.4: Implement `SafetyGate.__init__` + `check()` minimal**

- [ ] **Task 4.5: Re-run, verify PASS**

- [ ] **Task 4.6: Add test `test_gate_reject_low_confidence`** + implement

- [ ] **Task 4.7: Add test `test_gate_reject_high_risk_low_conf`** + implement

- [ ] **Task 4.8: Add test `test_gate_unknown_category_uses_default`** + implement

- [ ] **Task 4.9: Add `update_object_safety` + test**

- [ ] **Task 4.10: Commit**

```bash
git add src/safety_gate.py configs/safety_rules.yaml tests/test_safety_gate.py
git commit -m "feat(safety): add SafetyGate with YAML-driven risk rules"
git push
```

---

## Phase 5: Action Integration (4h)

### Files

- Modify: `src/scene_describer.py` — 输出 SceneModel 而非 StructuredDescription
- Modify: `src/action_executor.py` — 消费 SceneModel, 去除独立 grounding
- Modify: `src/pipeline.py` — 串接新数据流
- Update: `tests/test_env_wrapper_grasp.py` — 新接口兼容

### Key Changes

**scene_describer.py (high-level flow):**
```python
class SceneDescriber:
    def describe_multi_view(
        self,
        observations: list[Observation],
        user_query: str,
    ) -> SceneModel:
        """Run VLMGrounder on each view, aggregate into SceneModel, 
        inject safety info, return."""
        sm = SceneModel()
        for obs in observations:
            candidates = self.grounder.ground(
                obs.image_path,
                user_query,
                scene_context=sm.summarize_for_prompt(),
            )
            projector = self._make_projector(obs.viewpoint_name)
            sm.add_view(obs.viewpoint_name, candidates, projector)
        
        # Inject safety info
        for obj in sm._objects:
            self.safety_gate.update_object_safety(obj)
        
        return sm
```

**action_executor.py (high-level flow):**
```python
class ActionExecutor:
    def execute(
        self,
        plan: ActionPlan,
        scene_model: SceneModel,
        env: EnvWrapper,
    ) -> ActionResult:
        # 1) Ground user query through scene_model (不再调 env.ground_object)
        candidates = scene_model.ground_user_query(plan.target_object)
        if not candidates:
            return ActionResult(
                success=False, executed=False,
                message=f"场景中未找到 '{plan.target_object}'",
            )
        target = candidates[0]
        
        # 2) Safety gate
        decision = self.safety_gate.check(target)
        if not decision.allow_execute:
            return ActionResult(
                success=False, executed=False,
                message=decision.reason_user,
            )
        
        # 3) Path planning (unchanged)
        no_go = scene_model.get_no_go_zones()  # high-risk objects
        waypoints = self._plan_safe_path(env.get_eef_pos(), target.position_m, no_go)
        
        # 4) Execute grasp (unchanged, target.body_name may be None)
        ...
        grasp_ok = env.grasp_at(
            target.position_m,
            target_body=target.body_name or "obj_main",
            pre_grasp_verify=self._build_pre_grasp_verifier(plan.target_object),
        )
        ...
```

### Bite-sized Task Breakdown

- [ ] **Task 5.1: Modify `scene_describer.py`** — 加 `describe_multi_view` 方法, 保留旧 `describe` 做兼容

- [ ] **Task 5.2: Add test** `test_describe_multi_view_produces_scene_model`

- [ ] **Task 5.3: Modify `action_executor.py`** — 增加接受 `SceneModel` 参数的新 `execute_with_scene_model` 方法 (先不删旧 execute)

- [ ] **Task 5.4: Add test** `test_execute_with_scene_model_respects_safety_gate`

- [ ] **Task 5.5: Modify `pipeline.py`** — 新数据流串起来:

```python
# 在 pipeline.py 主循环里
observations = active_planner.plan_and_observe(subtasks, env)
scene_model = scene_describer.describe_multi_view(observations, query)
speech_text = speech_generator.render(scene_model)  # 可保留旧描述聚合
action_plan = action_decider.decide(speech_text, query, scene_model)
action_result = action_executor.execute_with_scene_model(
    action_plan, scene_model, env
)
```

- [ ] **Task 5.6: Manual test on server** — 跑 `帮我拿削皮器` (default scene), 验证:
  - `[obj_types] runtime object categories: {'obj_main': 'peeler', ...}`
  - Scene model aggregates VLM grounding across 3 views
  - Safety gate logs decision
  - `env.grasp_at` is called with `target_body='obj_main'`
  - Final result: `success=True`

- [ ] **Task 5.7: Commit**

```bash
git add src/scene_describer.py src/action_executor.py src/pipeline.py tests/
git commit -m "feat: integrate SceneModel + SafetyGate into action pipeline

Scene describer now produces unified SceneModel with grounded objects
(bbox, 3D position, safety, confidence). Action executor directly
consumes it without independent grounding. Legacy env.ground_object
preserved as fallback."
git push
```

---

## Phase 6: Active Planner Grounding-Aware Upgrade (5h)

### Files

- Modify: `src/active_planner.py`
- Modify: `prompts/active_planner.txt` (or new `prompts/active_planner_grounding_aware.txt`)
- Create: `tests/test_active_planner_grounding.py`
- Create: `docs/experiments/ablation_active_planner.md` (实验章节草稿)

### Upgrade Plan

当前 `active_planner` 按"覆盖率"选视角. 升级为按"能 ground 到 user_target 的可能性"选视角, 并记录"是否已 ground"作为早停条件.

### Prompt Upgrade Sketch

```text
# prompts/active_planner.txt (升级版)

You select the next best viewpoint for a visually-impaired assistance robot.

User query: {{ user_query }}
Current task subtasks: {{ subtasks }}

Already observed viewpoints: {{ observed }}
Grounded targets so far:
{{ #each scene_model.objects }}
  - {{ label }} (conf {{ user_target_match_score }}, risk {{ safety_risk }})
{{ /each }}

Candidate viewpoints:
{{ #each candidates }}
  - {{ name }}: {{ purpose }}
{{ /each }}

Strategy:
1. If user target NOT YET grounded with confidence > 0.8 → prioritize views likely to see it
2. If user target IS grounded but safety risks unclear → prioritize close-up/top views
3. If all info collected → return STOP

Reply JSON: {"choice": "<viewpoint_name>", "reason": "<why>"}
or {"choice": "STOP", "reason": "<why>"}
```

### Ablation Experiment Setup

创建 `src/eval.py` 里的新 `eval_grounding_driven_planning()` 函数:

| 条件 | 策略 |
|------|------|
| A. Baseline (exhaustive) | 穷举所有 6 个视角 |
| B. Coverage-only (当前 planner) | 按覆盖率早停 |
| C. Grounding-aware (new) | 按 grounding 状态早停 |

测量指标:
- 视角数 (planner 调用次数)
- LLM 调用次数
- Final SceneModel 置信度
- 是否 grasp 成功

跑 10 个 random scenes × 3 个 queries = 30 trial, 写进报告实验章节.

### Bite-sized Task Breakdown

- [ ] **Task 6.1: Copy `prompts/active_planner.txt` → `_grounding_aware.txt`** 以便消融对比

- [ ] **Task 6.2: Modify `active_planner.py` — add `grounding_mode` flag**

- [ ] **Task 6.3: Add test `test_planner_stops_when_target_grounded`**

- [ ] **Task 6.4: Run test, verify behavior**

- [ ] **Task 6.5: Write `src/eval.py::eval_grounding_driven_planning()`**

- [ ] **Task 6.6: Run 30 trials on server** — 提交 eval.log

- [ ] **Task 6.7: Draft `docs/experiments/ablation_active_planner.md`** with results table

- [ ] **Task 6.8: Commit**

```bash
git commit -m "feat(active_planner): grounding-aware viewpoint selection

New prompt variant + mode flag. Planner now stops early when user
target is grounded with sufficient confidence, saving VLM calls.
Ablation experiment eval script + initial results documented."
git push
```

---

## Success Criteria (End-to-End)

Phase 6 完成时, 应该满足:

- [ ] 跑 `python scripts/test_embodied.py --query "帮我拿削皮器"` 在默认场景下端到端成功 (success=True)
- [ ] 跑 `--query "帮我拿药瓶"` 在默认场景 (无真药瓶) 下**优雅拒绝** (success=False, 中文 TTS 说明原因)
- [ ] 跑 `--query "帮我拿杯子" --layout -1 --style -1` 在随机场景下正确 grounding 或优雅拒绝
- [ ] 消融实验数据: exhaustive vs coverage vs grounding-aware 的视角数对比
- [ ] 新代码测试覆盖率 > 70%
- [ ] 旧 `env.ground_object` 调用路径保留但被标记 `@deprecated`, 不在主 pipeline 上
- [ ] 所有 commit 都在 main 分支上推送, 服务器 `git pull` 就能跑

---

## Not in Scope (Future Work)

1. **Human-in-the-loop via ASR** — 当前用 safety gate 自动拒绝替代. Whisper 集成放到国赛后.
2. **Depth-based precise grasp pose (6DoF)** — 当前仍用 top-down grasp. 6DoF grasp 需要 GraspNet / AnyGrasp 集成.
3. **Multi-turn dialogue** — 当前每 query 独立. 上下文记忆放到国赛扩展.
4. **Sim2Real** — 国赛前再考虑, 真机有 camera intrinsic calibration + depth sensor 差异.
5. **Runtime aliases learning** — 用户说"那个红色的", 系统应能从上下文学. 当前必须 YAML 静态映射.

---

## Rollback Plan

每个 Phase 都有独立 commit. 任何 phase 出问题可回滚到前一 phase:

```bash
# 回滚 Phase 5 (假设 phase 5 commit hash 是 abc1234)
git revert abc1234  # 保留历史
# 或
git reset --hard <phase 4 last commit>  # 重写历史 (需推 --force-with-lease)
```

旧 `env.ground_object` + `action_executor.execute()` 路径**始终保留**, 最坏情况切回旧 pipeline。

---

## Self-Review Checklist

### Coverage Check
- [x] User spec: "安全第一" → Phase 4 SafetyGate
- [x] User spec: "准确识别" → Phase 2-3 VLM grounding + multi-view fusion
- [x] User spec: "精确抓取" → Phase 5 target position from 3D projection + pre-grasp verify (already exists)
- [x] User spec: "人类在环先不做" → AD-5 safety gate 自动拒绝替代
- [x] User spec: "完整版" → 6 个 Phase 全覆盖
- [x] 三大创新点延展: ① task_decomposer 保留 | ② active_planner 升级 (Phase 6) | ③ scene_describer 升级 (Phase 5)
- [x] 新增创新点: Zero-shot Open-vocabulary VLM grounding + Safety verification

### Placeholder Scan
- [x] No "TBD" or "TODO" in Phase 1 (完全 bite-sized)
- [⚠️] Phase 2 depends on Phase 1 decision (PATH A/B) — 明确标注, 不是遗漏
- [x] Phase 3-6 每个 Task 都有具体文件路径和代码草图
- [x] 配置文件内容完整给出 (safety_rules.yaml)

### Type Consistency
- [x] `GroundedCandidate` (Phase 2) → `GroundedObject` (Phase 3) 字段一致性: candidate 聚合为 object 时字段名对齐
- [x] `SceneModel.ground_user_query()` 返回 `list[GroundedObject]`, 被 `action_executor.execute_with_scene_model` 消费
- [x] `SafetyGate.check()` 返回 `SafetyDecision`, 被 action_executor 消费
- [x] `VLMGrounder.ground()` 返回 `list[GroundedCandidate]`, 被 `SceneModel.add_view()` 消费

### Risk Items
- **R-1**: Qwen2.5-VL-7B bbox 精度未知 → Phase 1 Task 1.1 直接探测, 决定 Path A/B
- **R-2**: RoboCasa depth API 可用性 → Phase 1 Task 1.2 直接探测
- **R-3**: 多视角物体 ID 对齐 (3D 距离阈值) 在 VLM 位置不准时可能失败 → Phase 3 Task 3.4 需调参
- **R-4**: Safety rules YAML 需维护 ~50 个 RoboCasa 类别, 不全 → 用 `_default` fallback + 逐步补充
- **R-5**: 新 pipeline 打破现有 scene_describer 输出格式 → Phase 5 保留旧方法做 backward compat

---

## Execution Handoff

Plan saved. Three execution options:

1. **Subagent-Driven** (recommended for high-stakes refactor):  
   - Fresh subagent per Phase + two-stage review  
   - Use `superpowers:subagent-driven-development`

2. **Inline Execution** (this session, phase-by-phase checkpoints):  
   - Execute Phase 1 in this session, review results, continue  
   - Use `superpowers:executing-plans`

3. **Human-Driven** (user runs probe themselves):  
   - User runs Phase 1 scripts on server, posts logs  
   - I analyze + update plan, then we continue  
   - **Most pragmatic given the server is not local**

**Recommended: Option 3 for Phase 1 → Option 1/2 for Phase 2+**


