# EmboSight Day 2 执行文档

> **前置条件**：Day 1 已全部完成（verify_robocasa.py 全 OK）
> **总耗时**：6-8 小时（一天）
> **目标**：跑通端到端最小可行 pipeline——视障者 query → 任务分解 → 多视角拍照 → VLM 描述 → 输出语音文本

---

## Day 2 总览

```
═══════════════════════════════════════════════════════
Phase 1: 实现 env_wrapper.py（仿真接入）       2 小时
Phase 2: LLM (DeepSeek) 集成 + 任务分解测试    1 小时
Phase 3: VLM (Qwen2.5-VL) 集成 + 描述测试      2 小时（含 16 GB 下载）
Phase 4: 端到端 pipeline 集成 + 录第一段 demo   1.5 小时
缓冲 + Debug                                    1 小时
═══════════════════════════════════════════════════════
合计                                            ~8 小时
```

---

# Phase 1：实现 env_wrapper.py（2 小时）

## 1.1 设计思路：多摄像头预设

**原计划**：用机械臂物理移动到 12 个视角拍照。
**实际困难**：移动机械臂需要 IK 求解、运动控制、防碰撞，工程量巨大。
**Day 2 简化方案**：在仿真中**同时部署多个固定摄像头**，"切换视角" = "切换摄像头"。

| 优势 | 代价 |
|---|---|
| 一次性渲染所有视角 | 演示视频里机械臂不动 |
| 没有 IK / 控制器问题 | 仿真感稍弱 |
| 渲染速度更快 | 但视障辅助叙事不受影响 |
| 校赛 100% 跑得通 | 国赛阶段再加机械臂运动 |

**Day 2 用 5 个 robosuite 内置摄像头**作为 5 个视角。

## 1.2 更新 configs/viewpoints.yaml

把 12 视角精简为 5 个（对应 robosuite 内置）：

```yaml
viewpoints:
  - name: agentview
    position: [0, 0, 60]
    orientation: [0, -45, 0]
    purpose: "全景前视，用于场景概览"

  - name: birdview
    position: [0, 0, 100]
    orientation: [0, -90, 0]
    purpose: "顶视图，用于俯瞰布局"

  - name: sideview
    position: [60, 0, 30]
    orientation: [0, -30, -90]
    purpose: "侧视图，用于深度判断"

  - name: frontview
    position: [0, -60, 30]
    orientation: [0, -30, 0]
    purpose: "正面视图，用于近距识别"

  - name: robot0_eye_in_hand
    position: [0, 0, 30]
    orientation: [0, -90, 0]
    purpose: "机械臂末端视角，用于物体特写"
```

## 1.3 重写 src/env_wrapper.py

完整代码（直接覆盖）：

```python
"""仿真环境封装（RoboCasa）

校赛 Day 2 实现:
    - 用 5 个固定摄像头作为离散视角
    - reset / observe / close 完整实现
    - move_arm_to 当前为 no-op (省赛/国赛阶段加 IK)
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    sim_type: str = "robocasa"
    env_name: str = "PickPlaceCounterToCabinet"
    robots: str = "PandaMobile"
    image_width: int = 256
    image_height: int = 256
    output_dir: str = "./results/observations"
    camera_names: tuple[str, ...] = (
        "agentview",
        "birdview",
        "sideview",
        "frontview",
        "robot0_eye_in_hand",
    )
    layout_ids: Optional[int] = None
    style_ids: Optional[int] = None


class EnvWrapper:
    """RoboCasa 仿真环境封装"""

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        self.config = config or EnvConfig()
        self._env = None
        self._latest_obs: dict[str, Any] = {}
        self._step = 0

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def reset(self) -> dict[str, Any]:
        if self._env is None:
            os.environ.setdefault("MUJOCO_GL", "egl")
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

            import robosuite as suite

            kwargs = dict(
                env_name=self.config.env_name,
                robots=self.config.robots,
                has_renderer=False,
                has_offscreen_renderer=True,
                use_camera_obs=True,
                camera_names=list(self.config.camera_names),
                camera_heights=self.config.image_height,
                camera_widths=self.config.image_width,
                control_freq=20,
            )
            if self.config.layout_ids is not None:
                kwargs["layout_ids"] = self.config.layout_ids
            if self.config.style_ids is not None:
                kwargs["style_ids"] = self.config.style_ids

            logger.info(f"创建仿真环境 {self.config.env_name}...")
            self._env = suite.make(**kwargs)

        self._latest_obs = self._env.reset()
        self._step = 0
        logger.info(f"环境重置完成 (cameras={list(self.config.camera_names)})")
        return self._latest_obs

    def move_arm_to(
        self,
        pose: tuple[float, float, float, float, float, float],
    ) -> bool:
        """校赛阶段为 no-op；省赛阶段实现真实 IK"""
        logger.debug(f"[move_arm_to] {pose} (校赛 Day 2: no-op)")
        return True

    def observe(self, viewpoint) -> "Observation":
        from .active_planner import Observation

        if not self._latest_obs:
            self.reset()

        camera_name = viewpoint.name
        img_key = f"{camera_name}_image"
        img = self._latest_obs.get(img_key)

        if img is None:
            logger.warning(f"未找到图像 {img_key}, 用 agentview 代替")
            img = self._latest_obs.get("agentview_image")

        self._step += 1
        image_path = os.path.join(
            self.config.output_dir,
            f"step_{self._step:03d}_{camera_name}.png",
        )

        if img is not None:
            try:
                import imageio.v2 as imageio
                imageio.imwrite(image_path, img)
                logger.debug(f"图像保存: {image_path}")
            except Exception as e:
                logger.warning(f"图像保存失败: {e}")

        return Observation(
            viewpoint=viewpoint,
            image_path=image_path,
        )

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
        logger.info("环境关闭")
```

## 1.4 测试 env_wrapper

```bash
cd ~/embodied/embodied-AI-one
git pull

python -c "
import logging
logging.basicConfig(level=logging.INFO)
import sys; sys.path.insert(0, '.')

from src.env_wrapper import EnvWrapper, EnvConfig
from src.active_planner import Viewpoint

env = EnvWrapper(EnvConfig(image_width=512, image_height=512))
env.reset()

vp_agent = Viewpoint('agentview', (0,0,60), (0,-45,0), 'agent view')
vp_bird = Viewpoint('birdview', (0,0,100), (0,-90,0), 'bird view')

obs1 = env.observe(vp_agent)
obs2 = env.observe(vp_bird)

print(f'观察 1: {obs1.image_path}')
print(f'观察 2: {obs2.image_path}')
env.close()
print('=== ENV_WRAPPER 测试完成 ===')
"
```

期望生成 2 张 PNG：`./results/observations/step_001_agentview.png` 和 `step_002_birdview.png`。

---

# Phase 2：LLM (DeepSeek) 集成（1 小时）

## 2.1 注册 + 拿 API Key

1. 访问 https://platform.deepseek.com/
2. 注册（手机号即可）+ 实名（如要求）
3. 充值 50 元（够用 2-3 个月）
4. https://platform.deepseek.com/api_keys → "Create new API key"
5. **复制 Key**（只显示一次，关闭后看不到）

## 2.2 配置服务器 .env

```bash
cd ~/embodied/embodied-AI-one
cp .env.example .env
nano .env
```

把 `DEEPSEEK_API_KEY=your_deepseek_api_key_here` 改成你拿到的 Key。`Ctrl+O` 回车 `Ctrl+X` 保存退出。

```bash
set -a; source .env; set +a
echo "Key 前 10 字符: ${DEEPSEEK_API_KEY:0:10}..."
```

## 2.3 测试 LLM 后端

```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.utils import load_dotenv
load_dotenv()

from src.llm_backend import LLMBackend

llm = LLMBackend()
print('调用 DeepSeek API...')
response = llm.generate(
    user_message='请用一句话介绍中国视障辅助现状。',
    system='你是一个无障碍科技专家。',
)
print('=' * 60)
print('响应:', response)
print('=' * 60)
"
```

期望看到 DeepSeek 中文回答（1-3 秒返回）。

## 2.4 跑 task_decomposer 端到端

```bash
python -c "
import sys; sys.path.insert(0, '.')
import logging
logging.basicConfig(level=logging.INFO)

from src.utils import load_dotenv
load_dotenv()

from src.llm_backend import LLMBackend
from src.task_decomposer import TaskDecomposer

llm = LLMBackend()
decomposer = TaskDecomposer(llm)

queries = [
    '我的药瓶在哪里？',
    '桌上有什么？',
    '帮我拿一下水杯',
]
for q in queries:
    print(f'\n查询: {q}')
    print('-' * 60)
    subtasks = decomposer.decompose(q)
    for i, t in enumerate(subtasks, 1):
        print(f'  {i}. {t}')
print('=' * 60)
print('=== TaskDecomposer 测试完成 ===')
"
```

期望每个查询输出 3-5 个结构化子任务（含 type / target / priority / blind_dimension）。

---

# Phase 3：VLM (Qwen2.5-VL) 集成（2 小时）

## 3.1 下载 Qwen2.5-VL-7B 权重（16 GB，约 15-30 分钟）

```bash
mkdir -p ~/embodied/embodied-AI-one/checkpoints

cd ~/embodied/embodied-AI-one
echo $HF_ENDPOINT

huggingface-cli download \
    Qwen/Qwen2.5-VL-7B-Instruct \
    --local-dir ./checkpoints/Qwen2.5-VL-7B-Instruct \
    --local-dir-use-symlinks False
```

下载期间你可以**继续 Phase 1/2 工作**（不冲突）。

## 3.2 测试 VLM 后端加载

```bash
python -c "
import sys; sys.path.insert(0, '.')
import logging
logging.basicConfig(level=logging.INFO)

from src.vlm_backend import VLMBackend

print('开始加载 Qwen2.5-VL-7B...（约 30-60 秒）')
vlm = VLMBackend(
    model_id='./checkpoints/Qwen2.5-VL-7B-Instruct',
    cache_dir='./checkpoints',
)
vlm._ensure_loaded()
print('=' * 60)
print('VLM 加载成功 ✓')

import torch
mem = torch.cuda.memory_allocated(0) / 1024**3
print(f'当前显存: {mem:.2f} GB')
print('=' * 60)
"
```

期望显存占用约 15-17 GB。

## 3.3 跑 scene_describer 端到端

```bash
python -c "
import sys; sys.path.insert(0, '.')
import logging
logging.basicConfig(level=logging.INFO)

from src.vlm_backend import VLMBackend
from src.scene_describer import SceneDescriber

vlm = VLMBackend(
    model_id='./checkpoints/Qwen2.5-VL-7B-Instruct',
    cache_dir='./checkpoints',
)
describer = SceneDescriber(vlm)

print('对 Phase 1 生成的图像做视障描述...')
desc = describer.describe('./results/observations/step_001_agentview.png')

print('=' * 60)
print('结构化描述:')
import json
print(json.dumps(desc.to_dict(), ensure_ascii=False, indent=2))
print()
print('视障语音输出:')
print(desc.to_speech())
print('=' * 60)
"
```

期望输出包含五维度（objects / positions / tactile / safety_alerts / actionable_advice）的 JSON。

---

# Phase 4：端到端 Pipeline 集成（1.5 小时）

## 4.1 检查 configs/default.yaml

```bash
cat configs/default.yaml
```

确认：
- `llm.base_url`: https://api.deepseek.com/v1
- `vlm.model_id`: ./checkpoints/Qwen2.5-VL-7B-Instruct（改本地路径）
- `viewpoints_path`: configs/viewpoints.yaml

## 4.2 跑 run_demo.py

```bash
mkdir -p logs results

python scripts/run_demo.py \
    --query "我面前桌上有什么？" \
    --config configs/default.yaml \
    --output results/demo_001.json
```

期望流程：
```
[task_decomposer]   分解出 3-5 个子任务
[active_planner]    选择 2-4 个视角
[env_wrapper]       渲染对应图像 (.png 保存到 results/observations/)
[scene_describer]   每个视角生成结构化描述
[aggregate]         合并五维度
[output]            视障友好语音文本
```

## 4.3 录第一段 mini demo（10-30 秒）

```bash
ls -1 results/observations/*.png | wc -l

cat results/demo_001.json | python -m json.tool | head -50
```

## 4.4 整理产出

```bash
mkdir -p results/day2

python -c "
import json
with open('results/demo_001.json', 'r') as f:
    data = json.load(f)

print('=' * 60)
print(f'查询: {data[\"query\"]}')
print(f'子任务数: {len(data[\"subtasks\"])}')
print(f'采集视角数: {len(data[\"observations\"])}')
print(f'最终输出长度: {len(data[\"speech\"])} 字符')
print('=' * 60)
print('视障语音输出:')
print(data['speech'])
print('=' * 60)
"
```

---

# 风险与备选

## 高风险

| 风险 | 概率 | 应对 |
|---|---|---|
| 5 个内置摄像头某个不存在 | 中 | reset 时报错就移除该摄像头，最少保留 agentview |
| Qwen2.5-VL 下载失败 | 低 | 改用 HF 命令行重试，或换 modelscope 镜像 |
| Qwen2.5-VL 显存不够 | 低 | 用 Qwen2.5-VL-3B 替代（只 7 GB） |
| VLM 输出非 JSON | 中 | scene_describer 已有兜底解析（找 {} 截取） |
| LLM 任务分解 schema 不准 | 低 | TaskDecomposer 已有验证逻辑 |

## 低优先功能（推迟到省赛）

- 真实机械臂运动 IK
- 视角的 12 个连续姿态
- 跨视角去重的精细聚合
- 几何后处理（深度图反推距离）

---

# Day 2 完成清单

```
✅ 完成标志:
  □ env_wrapper.py 重写完成
  □ DeepSeek API 配置 + LLM 测试通过
  □ Qwen2.5-VL 下载完成 + VLM 测试通过
  □ task_decomposer + scene_describer 单元测试通过
  □ scripts/run_demo.py 端到端跑通 1 个 query
  □ results/observations/ 下生成 5 张视角图
  □ results/demo_001.json 包含完整 pipeline 输出
  □ 视障语音文本可读且符合视障需求

🎯 Day 3 预告:
  □ 跑 5 类 seen 查询验证泛化
  □ 跑 5 个 unseen 查询验证零样本能力
  □ 录正式 demo 视频（60-90 秒）
```

---

# Phase Tips（按经验排雷）

## Phase 1 排雷

- **报错 "Camera xxx not found"**：删掉 EnvConfig.camera_names 里那个，最少保留 `agentview`
- **报错 "layout_ids must be int"**：传 None，自动随机布局
- **图像全黑**：检查 MUJOCO_GL=egl，重启 conda env

## Phase 2 排雷

- **API 401 Unauthorized**：检查 `echo $DEEPSEEK_API_KEY` 是否正确加载
- **API timeout**：检查代理 `env | grep -i proxy`
- **JSON 解析失败**：LLM 偶尔不输出严格 JSON，TaskDecomposer 有兜底，看日志

## Phase 3 排雷

- **下载超时**：`huggingface-cli download` 加 `--resume-download`
- **OOM 显存不够**：开 `device_map="auto"`，PyTorch 会自动分片
- **flash-attn 报错**：默认未装，会回退到 sdpa，无碍

## Phase 4 排雷

- **pipeline 整体失败**：先单独跑每个模块，确认各 OK
- **观察列表为空**：active_planner 早停了，检查 max_viewpoints 和 coverage
- **聚合后描述太短**：可能子任务覆盖率不够，临时增加 max_viewpoints