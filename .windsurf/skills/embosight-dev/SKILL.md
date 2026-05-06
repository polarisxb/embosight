---
name: embosight-dev
description: EmboSight 项目开发指南 — 具身AI视障辅助系统的架构、模块接口、编码规范。当涉及修改 src/ 下任何模块时自动调用。
---

## 项目概览

EmboSight 是一个零样本视障具身辅助智能体，部署在 RoboCasa 厨房仿真环境中。

## 技术栈

- **仿真**: RoboCasa (基于 robosuite) + MuJoCo
- **LLM**: DeepSeek API (OpenAI 兼容接口)
- **VLM**: Qwen2.5-VL-7B-Instruct (本地 GPU 推理)
- **机器人**: PandaMobile (Panda 机械臂 + Omron 移动底盘)
- **渲染**: EGL offscreen rendering (MUJOCO_GL=egl)

## 核心模块

| 模块 | 文件 | 职责 |
|---|---|---|
| TaskDecomposer | `src/task_decomposer.py` | 自然语言 → 结构化子任务 (五维度) |
| ActivePlanner | `src/active_planner.py` | LLM-NBV 主动视角规划 + 覆盖率早停 |
| SceneDescriber | `src/scene_describer.py` | VLM 图像 → 五维度结构化描述 + 安全分级 |
| EnvWrapper | `src/env_wrapper.py` | RoboCasa 环境封装 (reset/observe/move_arm_to/grasp) |
| Pipeline | `src/pipeline.py` | 端到端流程编排 |
| LLMBackend | `src/llm_backend.py` | DeepSeek API 封装 |
| VLMBackend | `src/vlm_backend.py` | Qwen2.5-VL 本地推理封装 |

## 五个视障核心维度

每个子任务必须归属以下之一：
1. **position** — 方位 (8方位词)
2. **distance** — 距离 (cm级)
3. **tactile** — 触觉 (材质/形状/温度/重量)
4. **safety** — 安全 (热源/锐器/易碎/不稳定)
5. **action** — 行动 (抓握方式/路径/注意事项)

## 6 个摄像头

```
robot0_agentview_center  → position, safety
robot0_agentview_left    → position, safety
robot0_agentview_right   → position, safety
robot0_frontview         → distance, tactile, position
robot0_robotview         → distance, action
robot0_eye_in_hand       → tactile, action, distance
```

## 编码规范

- Python 3.10+, type hints 必须
- 日志用 `logging.getLogger(__name__)`
- 数据结构用 `@dataclass`
- 所有模块底部有 `if __name__ == "__main__":` 自测
- 不在中间 import，所有 import 放文件顶部
- 配置走 `configs/default.yaml`，敏感信息走 `.env`

## 关键配置文件

- `configs/default.yaml` — 全局配置 (LLM/VLM/环境/路径)
- `configs/viewpoints.yaml` — 6 个离散视角定义
- `prompts/task_decompose.txt` — 任务分解 system prompt
- `prompts/active_planner.txt` — NBV 决策 system prompt
- `prompts/scene_describer.txt` — 场景描述 system prompt
- `prompts/blind_task_templates.json` — 模板库 (8 个模板)
