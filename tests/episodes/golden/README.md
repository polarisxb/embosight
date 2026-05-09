# Golden Episodes

本目录存放 EpisodeLogger replay 测试的 golden 数据。
4 层契约 (F7) 验证决策路径在代码迭代中保持稳定。

## 5 个 query (设计稿 §10.3)

| # | query | 触发场景 | 期望关键 action |
|---|---|---|---|
| 01 | 拿苹果 | 基础 (label 唯一) | `observe` → `classify_safety` → `plan_grasp_candidates` → `grasp` |
| 02 | 我要那个削皮器 | zoom 消歧 | `observe` → `re_observe(zoom_in)` → `classify_safety` → `grasp` |
| 03 | 拿那个杯子 | safety 多次分类 | `observe` → `classify_safety` (×2) → `grasp` |
| 04 | 我要那个红色的 | ask_user 澄清 | `observe` → `ask_user` → `observe` → `grasp` |
| 05 | 帮我取碗, 避开刀 | constraint avoid | `decompose(constraint)` → `observe` → `grasp` |

## 4 层契约 (F7)

| 层 | 检查项 | 严格度 |
|---|---|---|
| L1 | `result.success` 一致 | 必过 |
| L2 | golden 中所有 `action.kind` 都出现在 replay (allow superset) | 必过 |
| L3 | `len(replay) ≤ 1.5x golden` (或 +3 步) | 必过 |
| L4 | golden 含 `re_observe(zoom_in)` 时 replay 也必须 | 必过 |

## v1 现状: 02-05 是 mock 模板

- `01_basic_apple.json` 是真实风格的最小 episode 模板, 用 `vlm_ground` `objects` 数组 + `llm_safety` dist 驱动 replay 通过 4 层契约。
- `02-05` 是用"手工编排 evidence 序列"模拟出 4 层契约必要的 action 序列, **不代表 sim 真实输出**。
- v1.1 演示前必须用 `scripts/record_golden_episode.py` 在真 sim 录制替换。

替换时关注:
1. **L2 契约 (action 集合) 不应变** — 见关键契约对照表
2. **L3 契约 (步数 1.5x)** 真 sim 通常更长, 必要时调宽阈值
3. **L4 契约 (zoom_in 命中)** 02 必须保留, 否则 demo 故事不成立

## 录制方式 (真 sim)

```bash
python scripts/record_golden_episode.py \
  --query "我要那个削皮器" \
  --output tests/episodes/golden/02_zoom_disambiguate_peeler.json \
  --user-mode fake_from_robocasa
```

依赖:
- `DEEPSEEK_API_KEY` 环境变量
- `./checkpoints/Qwen2.5-VL-7B-Instruct/` 模型
- RoboCasa + MuJoCo + GPU

## 重新录制流程

如果决策树或 prompt 改了导致 replay 测试 L1/L2 失败:
1. 跑 sim 重新录: `python scripts/record_golden_episode.py ...`
2. diff 新老 golden, 确认变化合理 (e.g. 多了一步 zoom_in)
3. 替换 golden 文件 + commit
