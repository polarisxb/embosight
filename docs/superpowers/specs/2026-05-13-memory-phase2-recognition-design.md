# EmboSight Memory Phase 2 — Recognition Hints 设计

> 日期: 2026-05-13
> 状态: draft
> 前序: `docs/superpowers/specs/2026-05-11-dual-store-memory-design.md` (Phase 1)
> 关键词: recognition memory, synonym hints, CLIP injection, LLM fallback, cross-episode learning

---

## 0. TL;DR

补齐 Phase 1 留下的 `recognition_hints` 写入端缺口：

- **触发**：CLIP 注入命中（`perception._inject_clip_scores`）、LLM semantic fallback 命中（`agent._llm_semantic_fallback`）实时写入 working memory
- **沉淀**：grasp 成功时 consolidate working → `memory/recognition_hints.yaml`
- **注入**：下次 episode 开始把 `effective_synonyms` 合并进 `belief.decomposed.primary_target_synonyms`，自动驱动现有 CLIP multi-query + label 匹配链路
- **范围**：仅 recognition domain；safety_knowledge 与 embedding 检索本轮 **不做**

解决问题：VLM 反复把 tangerine 叫成 orange 时，agent 第二次见到 tangerine 任务还要走完整 CLIP fallback 流程，浪费 1 个 viewpoint。

---

## 1. 动机

### 1.1 Phase 1 留下的缺口

Phase 1 实现了 `MemoryManager.get_recognition_hints()` 读路径（`src/memory_manager.py:113-127`），但：

- `MemoryManager.consolidate()` 只处理 grasp domain（`src/memory_manager.py:142-146`）
- `memory/recognition_hints.yaml` 永远为空 → 读路径返回 None
- recognition 类的 working memory event 完全没人 record

等于 Phase 1 写了一半。本设计补齐另一半。

### 1.2 现实数据驱动

batch_20260512_005101 中观察到的 recognition 难题：

| target | VLM 输出 | 命中方式 |
|---|---|---|
| tangerine | "orange" | CLIP 注入 (sim=0.31) |
| cake | "green vegetable" | CLIP 注入 (sim=0.24) |
| yogurt | "container" | LLM semantic fallback |

这些命中信息在当前架构下 **每个 episode 都要重新发现**。Phase 2 让它们跨 episode 复用。

### 1.3 范围裁剪 (YAGNI)

设计文档 §7 Phase 2 列了三块。本轮 **只做 recognition**，理由：

| 子项 | 决策 | 理由 |
|---|---|---|
| recognition_hints 读写闭环 | ✅ 本轮 | 补齐 Phase 1 缺口；真实数据驱动 |
| safety_knowledge | ❌ 暂缓 | batch eval 中 safety 重分类触发 ~0 次，无数据 |
| embedding 相似度检索 (Phase 3) | ❌ 暂缓 | 引入 sentence-transformers (~300MB) 是过度设计；当前 corpus 重叠不足 |
| 消融实验 | ❌ 本设计不含 | 属于实验跑数，Phase 2 代码完成后作为验证步骤 |

---

## 2. 架构

### 2.1 端到端数据流

```
┌────────────────── Episode N ──────────────────┐
│                                                │
│  belief.decomposed = TaskDecomposer.decompose │
│  → primary_target_synonyms = [LLM 生成]       │
│                                                │
│  ──── MERGE recognition hints ────             │
│  syns_from_memory = memory.get_recognition_   │
│       hints_synonyms(primary_target)          │
│  belief.decomposed.primary_target_synonyms =  │
│       dedupe(syns + syns_from_memory)         │
│  ───────────────────────────────────           │
│                                                │
│  ┌── agent loop ──────────────────────────┐   │
│  │  observe → perception.observe()         │   │
│  │    ├─ VLM call                          │   │
│  │    └─ if CLIP injection 命中:           │   │
│  │        evidence.raw_payload[           │   │
│  │          "clip_injected"] = {           │   │
│  │            target, synonym,             │   │
│  │            sim, vlm_label}              │   │
│  │                                          │   │
│  │  agent._merge_hypotheses_from_evidence  │   │
│  │    └─ if clip_injected present:         │   │
│  │        memory.record_event(             │   │
│  │          MemoryEntry(                   │   │
│  │            domain="recognition",        │   │
│  │            event="synonym_effective",   │   │
│  │            context={...}))              │   │
│  │                                          │   │
│  │  decide_next → _llm_semantic_fallback   │   │
│  │    └─ if 命中:                          │   │
│  │        memory.record_event(             │   │
│  │          MemoryEntry(                   │   │
│  │            domain="recognition",        │   │
│  │            event="label_corrected",     │   │
│  │            context={...}))              │   │
│  └─────────────────────────────────────────┘   │
│                                                │
│  if grasp success:                            │
│    memory.consolidate(success=True)           │
│    → _consolidate_recognition()               │
│      → merge working into                     │
│        memory/recognition_hints.yaml          │
└────────────────────────────────────────────────┘
```

### 2.2 关键设计决策

**决策 1：仅 grasp 成功才 consolidate recognition（与 Phase 1 grasp consolidate 独立）**

理由：CLIP/LLM 命中不代表 hypothesis 真的就是 target；只有 grasp 成功才证明整条识别链路正确。避免把误命中固化。

注意：Phase 1 的 `_consolidate_grasp` **无论成败都写入**（要记录 failure_mode 与计数），本决策只限制新增的 `_consolidate_recognition`。顶层 `consolidate(success, object_type)` 分发逻辑：
- `_consolidate_grasp`：Phase 1 原状，总是调用
- `_consolidate_recognition`：仅 `success=True` 时调用

**决策 2：注入侧只走 `synonyms` 一条路（方案 A）**

不改 VLM ground prompt、不改 LLM fallback prompt。理由：

- `primary_target_synonyms` 是现有数据通路，自动驱动 CLIP multi-query 和 `belief.target()` label 匹配
- 零 prompt 污染、零 token 成本上升
- 改 VLM prompt 风险高（hint 可能误导 VLM）

**决策 3：去重窗口 = 单 episode**

同一 episode 内同一 (target, synonym) 命中多次，working memory 只记一次。consolidate 时同一 (target, synonym) 在 yaml 中累加 count。

---

## 3. 数据结构

### 3.1 Working memory event

复用 Phase 1 的 `MemoryEntry`，按 event 类型区分：

```python
# CLIP 命中
MemoryEntry(
    step=current_step,
    domain="recognition",
    event="synonym_effective",
    context={
        "target": "tangerine",       # primary_target
        "synonym": "orange",          # CLIP best query (不可能等于 primary，见下)
        "sim": 0.31,                  # CLIP cosine similarity
        "vlm_label": "orange",        # 该 hypothesis VLM 原 label
    },
    lesson="tangerine: CLIP 命中 via 'orange' (sim=0.31)",
)

# 重要约束：CLIP best_q == primary_target 时不记录此 event
# (理由：synonym 字段存在的意义是“发现了一个新的表达式”；
#  如果 CLIP 是通过 primary 自己命中，只能说明 VLM 误命名，
#  不产生 synonym 知识；vlm_label 的价值在 vlm_common_labels 累加中体现)

# LLM fallback 命中
MemoryEntry(
    step=current_step,
    domain="recognition",
    event="label_corrected",
    context={
        "target": "yogurt",
        "detected_label": "container",   # LLM 选中的 VLM label
        "method": "llm",                 # "llm" | "user" (future)
    },
    lesson="yogurt: LLM matched detected 'container'",
)
```

### 3.2 Long-term YAML schema

`memory/recognition_hints.yaml`:

```yaml
entries:
  - target: tangerine
    vlm_common_labels: ["orange", "citrus"]   # set, append-only on label_corrected/synonym_effective
    effective_synonyms:                        # list of dicts, ordered by count desc
      - name: orange
        count: 3
        last_method: clip                     # "clip" | "llm" | "user"
    clip_helpful: true                         # any sample had method="clip" and count >= 1
    notes: ""
    last_updated: "2026-05-13"

  - target: yogurt
    vlm_common_labels: ["container", "dairy"]
    effective_synonyms:
      - name: container
        count: 2
        last_method: llm
    clip_helpful: false
    notes: ""
    last_updated: "2026-05-12"
```

字段语义：

- `vlm_common_labels`：set 语义，避免重复；用于未来 prompt hint（本轮不消费）
- `effective_synonyms[*].name`：注入侧消费的核心字段
- `effective_synonyms[*].count`：累计命中次数，用于排序 + 未来过滤低置信 hint
- `effective_synonyms[*].last_method`：用于 `clip_helpful` 推导
- `clip_helpful`：本轮不消费，但为未来"是否启用 CLIP"的开关留 schema

---

## 4. 模块改动清单

| 文件 | 改动 |
|---|---|
| `src/memory_manager.py` | 新增 `_consolidate_recognition()`，扩展 `consolidate()` 分发；新增 `get_recognition_hints_synonyms(target) -> list[str]` |
| `src/perception.py` | `observe()` 返回 Evidence 时，若 CLIP 注入命中，把 `clip_injected: {target, synonym, sim, vlm_label}` 放进 `raw_payload` |
| `src/agent.py` | (a) `_merge_hypotheses_from_evidence` 检测 `clip_injected` → `record_event(synonym_effective)` (b) `_llm_semantic_fallback` 命中时 `record_event(label_corrected)` (c) `run()` 起始处 merge `get_recognition_hints_synonyms` 进 `decomposed.primary_target_synonyms` |
| `tests/test_memory_manager.py` | 新增 recognition consolidate / get_synonyms 单元测试 |
| `tests/test_memory_integration.py` | 新增端到端：模拟 CLIP 命中 → consolidate → 下次 episode 加载到 synonyms |
| `tests/test_perception.py` | 新增 CLIP 注入时 evidence.raw_payload 含 `clip_injected` 的断言 |

不改：`prompts/`、`memory/index.yaml`、`world_belief.py`、其他模块。

---

## 5. 关键接口

### 5.1 MemoryManager 新增方法

```python
def get_recognition_hints_synonyms(self, target: str) -> list[str]:
    """返回该 target 历史上有效的 synonym 列表 (按 count 降序, 去重 lowercase)。

    Returns: e.g. ["orange", "citrus"] for tangerine
    Returns [] if no hints or load failed.
    """
```

```python
def _consolidate_recognition(self, events: list[MemoryEntry]) -> None:
    """把 recognition working events merge 进 long-term YAML。

    按 event.context["target"] 分组, 每个 target:
    - vlm_common_labels: union append (synonym_effective.vlm_label, label_corrected.detected_label)
    - effective_synonyms: 按 name 聚合, count += 该 episode 命中次数, last_method = 最新 method
    - clip_helpful: any last_method == "clip" → True
    - last_updated: today
    """
```

### 5.2 perception.observe() Evidence raw_payload 扩展

```python
# 现有字段 (不变)
raw_payload = {
    "viewpoint": ...,
    "hypotheses": [...],
    "image_path": ...,
    "raw_vlm_text": ...,
}

# 新增字段 (CLIP 注入命中且 best_q != primary_target 时, 否则不存在)
raw_payload["clip_injected"] = {
    "target": "tangerine",       # primary_target
    "synonym": "orange",          # best query (保证 != target)
    "sim": 0.31,                  # cosine similarity
    "vlm_label": "orange",        # 该 hypothesis 原 VLM label
}
```

LLM fallback 不经 evidence — 直接在 `agent._llm_semantic_fallback` 内部 `self.memory.record_event(...)`。

### 5.3 agent.run() 起始合并

```python
# 现有
self.memory.working_memory.clear()
prior = self.memory.load_for_task(...)

# 新增 — recognition hints 注入 synonyms
if belief.decomposed:
    hint_syns = self.memory.get_recognition_hints_synonyms(
        belief.decomposed.primary_target,
    )
    if hint_syns:
        existing = {s.lower() for s in belief.decomposed.primary_target_synonyms}
        added = [s for s in hint_syns if s.lower() not in existing]
        belief.decomposed.primary_target_synonyms.extend(added)
        if added:
            logger.info(
                "[memory] injected %d recognition synonym(s) for '%s': %s",
                len(added), belief.decomposed.primary_target, added,
            )
```

---

## 6. 测试计划

### 6.1 单元测试 (`tests/test_memory_manager.py`)

```python
def test_consolidate_recognition_creates_entry():
    """新 target → 新 yaml entry"""

def test_consolidate_recognition_merges_existing():
    """同 target 同 synonym → count += 1"""

def test_consolidate_recognition_appends_new_synonym():
    """同 target 不同 synonym → 新 entry in effective_synonyms"""

def test_consolidate_recognition_vlm_labels_dedupe():
    """同 target 重复 vlm_label → set 语义不重复"""

def test_get_recognition_hints_synonyms_returns_sorted():
    """count 降序返回"""

def test_get_recognition_hints_synonyms_empty():
    """no entry → []"""

def test_consolidate_recognition_clip_helpful_flag():
    """method=clip → clip_helpful=true"""
```

### 6.2 集成测试 (`tests/test_memory_integration.py`)

```python
def test_recognition_hints_persist_across_episodes(tmp_path):
    """
    Episode 1: 模拟 perception 返回 clip_injected → consolidate
    Episode 2: agent.run 开始时, get_recognition_hints_synonyms 返回非空
    """

def test_clip_injected_recorded_in_working_memory():
    """perception 返回 evidence with clip_injected
       → agent._merge_hypotheses_from_evidence
       → memory.working_memory 含 synonym_effective entry"""

def test_llm_fallback_recorded_in_working_memory():
    """_llm_semantic_fallback 命中 → working_memory 含 label_corrected entry"""

def test_episode_dedup_synonym_within():
    """同 episode 同 (target, synonym) 命中 2 次 → working_memory 只 1 entry"""
```

### 6.3 Perception 测试 (`tests/test_perception.py`)

```python
def test_observe_records_clip_injection_in_evidence():
    """mock vlm 返回 label='orange', mock clip 返回 sim>0.23 for primary='tangerine'
       → evidence.raw_payload['clip_injected']['synonym'] == 'tangerine' or 'orange'"""
```

---

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| YAML 文件并发写冲突（multi-GPU eval） | Phase 1 已有 `memory_dir` 隔离机制（`run_fixed.py --memory-dir`），长跑器已经按 seed 隔离，无新增并发风险 |
| 误命中污染 yaml | consolidate 仅在 grasp success 时触发 → 错误的 CLIP 命中（grasp 失败）不会写入 |
| Synonym 列表无限增长 | `effective_synonyms` 单 target 最多保留 top-5（按 count 降序裁剪） |
| 注入的 synonym 与已有冲突 | merge 时 lowercase 去重；不动 LLM 已生成的部分 |
| 旧版 recognition_hints.yaml schema 不兼容 | 当前文件是 `entries: []`，无历史数据，零成本升级 |

---

## 8. 不在本范围内（明确）

- safety_knowledge 写入与读取
- Phase 3 embedding-based 相似物体迁移
- VLM ground prompt 注入 hints（决策 2 已排除）
- LLM fallback prompt 注入 hints（决策 2 已排除）
- 消融实验执行（独立任务）
- recognition_hints 跨 target 的去重/合并（如 tangerine 和 mandarin 共享 hints）

---

## 9. 文件变更预估

| 文件 | 新增行数 | 修改行数 |
|---|---|---|
| `src/memory_manager.py` | ~70 | ~5 |
| `src/perception.py` | ~10 | ~5 |
| `src/agent.py` | ~25 | ~5 |
| `tests/test_memory_manager.py` | ~80 | 0 |
| `tests/test_memory_integration.py` | ~60 | 0 |
| `tests/test_perception.py` | ~30 | 0 |
| **总计** | **~275** | **~15** |

预计实现时间：**2-3 小时**（含测试）。
