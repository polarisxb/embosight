---
name: enhance-prompts
description: 增强 EmboSight 的 LLM/VLM 提示词。当需要修改 prompts/ 目录下的文件或优化模型输出质量时自动调用。
---

## Prompt 文件列表

| 文件 | 用途 | 被谁调用 |
|---|---|---|
| `prompts/task_decompose.txt` | 任务分解 system prompt | TaskDecomposer → DeepSeek |
| `prompts/active_planner.txt` | NBV 视角选择 prompt | ActivePlanner → DeepSeek |
| `prompts/scene_describer.txt` | 图像描述 prompt | SceneDescriber → Qwen2.5-VL |
| `prompts/blind_task_templates.json` | 模板检索库 (8 模板) | TaskDecomposer.retrieve_similar() |

## 五维度覆盖规则

每个 prompt 必须确保输出覆盖全部 5 个维度：
1. **position** (方位) — 8 方位词：正前/左前/左/左后/正后/右后/右/右前
2. **distance** (距离) — cm 级整数，标注手臂可及范围 (≤60cm)
3. **tactile** (触觉) — 材质+形状+温度+重量+表面
4. **safety** (安全) — `[类别/等级]` 格式，等级：高风险/中风险/低风险
5. **action** (行动) — 手部方向+抓握方式+路径障碍

## Prompt 编写规范

1. **输出必须是纯 JSON** — 不要任何 markdown、注释、额外文字
2. **提供 few-shot 示例** — 至少 2 个完整 JSON 示例
3. **安全永远最高优先级** — priority=1, 放在第一位
4. **具体不模糊** — 不用"差不多""可能""那边"，用具体数值和方位
5. **视障者视角** — 不用纯颜色描述，用材质+形状+温度
6. **物体命名三段式** — "材质 + 形状 + 通用名" (如 "不锈钢圆筒锅")

## 模板库扩展规则

`blind_task_templates.json` 中每个模板：
- 必须有 `category` (find_object/describe_scene/fetch_object/alert_safety/navigate)
- 必须覆盖全部 5 个 blind_dimension
- `output_format` 要具体到数据格式
- `priority` 安全相关必须为 1

## 测试方法

修改 prompt 后：
```bash
# Mock 测试（检查格式）
python scripts/test_pipeline_mock.py

# 真实 LLM 测试（检查输出质量）
python scripts/test_real_llm.py
```
