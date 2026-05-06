---
name: run-pipeline
description: EmboSight 运行与测试指南 — 如何在服务器上运行完整 pipeline、单模块测试、mock 测试，以及常见问题排查。
---

## 运行前置条件

1. `.env` 已配置 `DEEPSEEK_API_KEY`
2. 模型已下载到 `./checkpoints/Qwen2.5-VL-7B-Instruct/`
3. RoboCasa + robosuite 已安装
4. conda 环境: `embosight`

## 测试命令（按顺序）

### 1. Mock 测试（不需要 API/GPU）
```bash
python scripts/test_pipeline_mock.py
```
验证所有模块接口连通，不依赖外部资源。

### 2. LLM 集成测试（需要 API Key）
```bash
python scripts/test_real_llm.py
```
测试 DeepSeek API 连通 + TaskDecomposer + ActivePlanner NBV 决策。

### 3. VLM 集成测试（需要 GPU + 模型权重）
```bash
python scripts/test_real_vlm.py
```
测试 Qwen2.5-VL 推理 + SceneDescriber 五维度描述。

### 4. 端到端 Demo
```bash
python scripts/run_demo.py --query "我的药瓶在哪里？"
python scripts/run_demo.py --query "桌上有什么？"
python scripts/run_demo.py --query "帮我拿水杯"
```

## 输出文件

- `results/observations/step_XXX_*.png` — 各视角渲染图像
- `results/demo.json` — 完整 pipeline 输出
- `results/vlm_test_result.json` — VLM 测试结果

## 常见问题

| 问题 | 解决 |
|---|---|
| `ValueError: No "camera" with name xxx` | 检查 `configs/viewpoints.yaml` 和 `EnvConfig` 中的摄像头名 |
| `Qwen2VL vs Qwen2_5_VL` 类型不匹配 | `vlm_backend.py` 应使用 `Qwen2_5_VLForConditionalGeneration` |
| `device_map="cuda"` 报错 | 应使用 `device_map="auto"` |
| JSON 解析失败 | VLM 未输出纯 JSON，SceneDescriber 会使用 fallback 解析 |
| `MUJOCO_GL` 报错 | 确保设置 `MUJOCO_GL=egl` + `PYOPENGL_PLATFORM=egl` |

## 服务器环境

```bash
# 激活环境
conda activate embosight

# 设置渲染 (通常 env_wrapper.py 已自动设置)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```
