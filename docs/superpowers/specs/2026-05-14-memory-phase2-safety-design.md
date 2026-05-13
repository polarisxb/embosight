# Memory Phase 2 — Safety Knowledge Design

- **Date**: 2026-05-14
- **Status**: Implemented (commits c9a9bf1 → 2e57c75)
- **Parent**: `docs/superpowers/specs/2026-05-11-dual-store-memory-design.md`
- **Sibling**: `docs/superpowers/specs/2026-05-13-memory-phase2-recognition-design.md`

---

## 0. TL;DR

补齐 Phase 1 双存储记忆的第三块（safety），让同一物体的安全分类先验跨 episode 复用：

- **Working memory event**: `safety_classified` — 每次 `SafetyClassifier.classify` 调用后写入
- **Long-term YAML**: `memory/safety_knowledge.yaml` — 仅 grasp 成功才 consolidate，跨 episode running-average 合并 dist
- **Re-use 通道**: `agent._execute_action("classify_safety")` 调用前从 memory 读 `safety_prior`, 格式化成 `prior_hint`, 通过 `SafetyClassifier.classify(hyp, prior_hint=...)` 注入 LLM prompt

zero schema 改动 / zero 模板文件改动 — 全部通过 SafetyClassifier 的 `prior_hint` 参数注入。

---

## 1. 动机

### 1.1 现有 SafetyClassifier 的浪费

`src/safety_gate.py:SafetyClassifier` 每次 classify 都是**完全冷启动**：LLM 仅看当前一帧的 label + features + pose 即给出 `dist`。即使同一台机器人在同一厨房见过 5 次刀子，第 6 次仍要 LLM 重新推理 "knife 是否锋利"。

### 1.2 现实数据驱动

`logs/long_generalization/` 中常见现象：

- 同一物体多 episode 反复 classify，输出 dist 方差大（LLM 不确定性）
- 高风险物体（knife, wine glass）首次 classify 偶尔 mis-label 为 safe，导致下游策略选择错误
- 安全分类失败后续 grasp 仍可能成功，但代价是多一轮 re-observe

### 1.3 范围裁剪 (YAGNI)

本轮 **不做**：
- safety_dist 的 entropy 阈值动态化（已经在 `WorldBelief._dynamic_thresholds` 中存在）
- LLM-aware ensemble（多次 classify 取平均）
- 跨用户共享 safety knowledge

---

## 2. 架构

### 2.1 端到端数据流

```
┌─────────────── Episode N ───────────────┐
│                                          │
│  perception observe                      │
│           │                              │
│           ▼                              │
│  agent._execute_action(classify_safety)  │
│           │                              │
│           │ ① get_safety_prior("knife")  │
│           │    → {dist, top_class, n}    │
│           │                              │
│           │ ② _format_safety_prior_hint  │
│           │    → "Historical: ..."       │
│           │                              │
│           ▼                              │
│  SafetyClassifier.classify(              │
│      hyp, prior_hint=hint)               │
│           │                              │
│           ▼                              │
│  ev = Evidence(source="llm_safety",      │
│                dist=..., entropy=...)    │
│           │                              │
│           ▼                              │
│  agent._record_safety_classified(        │
│      label, ev)                          │
│           │                              │
│           ▼                              │
│  working_memory += MemoryEntry(          │
│      domain="safety",                    │
│      event="safety_classified",          │
│      context={label, dist, entropy})     │
│                                          │
│            ... grasp ...                 │
│                                          │
│  agent._consolidate_memory(success)      │
│           │                              │
│           │ if success:                  │
│           ▼                              │
│  MemoryManager._consolidate_safety       │
│           │                              │
│           ▼                              │
│  safety_knowledge.yaml 持久化            │
└──────────────────────────────────────────┘
```

### 2.2 关键设计决策

**决策 1：仅 grasp 成功才 consolidate（与 recognition 同）**

理由：误分类（如把刀认成 safe）不应被固化为先验，否则下次 priors 更糟。`_consolidate_safety` 在 `MemoryManager.consolidate()` 中只在 `success=True` 分支被调用。

**决策 2：Running average 而非最新值**

同一 label 跨 episode 多次 classify，dist 按观察次数加权平均：`new = (old * n + obs) / (n+1)`。优点：早期噪声会被后续观察平滑。

**决策 3：Episode 内 dedup 取最新**

同 episode 内同一 label 可能 classify 多次（如 re_observe 后重判）。consolidate 时按 label 聚合，**保留最后一次** dist 视为该 episode 的最终判断，count 为 1 次 observation。

**决策 4：prior 作为软提示而非硬约束**

`_format_safety_prior_hint` 包含 "Treat this as a Bayesian prior — trust visual evidence if it strongly contradicts."，留给 LLM 否决空间。避免错误先验被反复强化的死锁。

---

## 3. 数据结构

### 3.1 Working memory event

```python
MemoryEntry(
    step=N,
    domain="safety",
    event="safety_classified",
    context={
        "label": "knife",
        "dist": {"sharp": 0.85, "safe": 0.10, "fragile": 0.05},
        "entropy": 0.40,
        "reasoning": "metal blade visible",  # optional
    },
    lesson="knife: safety classified",
)
```

### 3.2 Long-term YAML schema

`memory/safety_knowledge.yaml`:

```yaml
entries:
  - label: knife
    dist:
      sharp: 0.85
      safe: 0.10
      fragile: 0.05
    top_class: sharp           # argmax(dist)
    observations: 7            # 累计成功 episode 数
    last_updated: "2026-05-14"
```

---

## 4. 模块改动清单

| 文件 | 改动 |
|---|---|
| `src/memory_manager.py` | + `_consolidate_safety` (running average); + `get_safety_prior`; consolidate 分发 |
| `src/safety_gate.py` | `SafetyClassifier.classify(hyp, prior_hint=None)`; `_build_prompt` 接受 prior_hint |
| `src/agent.py` | + `_format_safety_prior_hint`; + `_record_safety_classified`; `_execute_action("classify_safety")` 接线 |
| `tests/test_memory_manager.py` | + `TestSafetyConsolidation` (4 cases); + `TestSafetyPriorReader` (4 cases) |
| `tests/test_safety_classifier.py` | + `TestSafetyPriorHint` (3 cases) |
| `tests/test_semantic_fallback.py` | + `TestAgentSafetyPriorAndRecording` (6 cases) |
| `tests/test_memory_integration.py` | + `TestSafetyRoundTrip` (3 cases — including failure-skips) |

---

## 5. 关键接口

### 5.1 MemoryManager

```python
def _consolidate_safety(self, events: list[MemoryEntry]) -> None:
    """Episode-scoped dedup (latest dist per label) →
       cross-episode running average → top_class = argmax(merged_dist)."""

def get_safety_prior(self, label: str) -> Optional[dict]:
    """Returns {dist, top_class, observations} or None."""
```

### 5.2 SafetyClassifier

```python
def classify(self, hyp: Hypothesis,
             prior_hint: Optional[str] = None) -> Evidence: ...
```

`prior_hint=None` 时行为完全等同 Phase 1（向后兼容）。

### 5.3 Agent

```python
def _format_safety_prior_hint(self, label: str) -> Optional[str]:
    """Renders prior into prompt-ready string.
       Returns None when no prior or empty label."""

def _record_safety_classified(self, label: str, ev: Evidence) -> None:
    """Skips empty label or empty dist (e.g. parse_failed)."""
```

接线点：

```python
elif action.kind == "classify_safety":
    hyp = action.target_hypothesis
    prior_hint = self._format_safety_prior_hint(hyp.label)
    ev = self.safety.classify(hyp, prior_hint=prior_hint)
    belief.evidence.append(ev)
    hyp.safety_dist = ev.raw_payload.get("dist", {})
    hyp.safety_entropy = ev.raw_payload.get("entropy", 1.0)
    self._record_safety_classified(hyp.label, ev)
```

---

## 6. 测试计划与结果

### 6.1 单元测试

- `TestSafetyConsolidation`: create entry / skip on failure / running average / dedup within episode ✓
- `TestSafetyPriorReader`: returns entry / case insensitive / unknown returns None / missing file returns None ✓
- `TestSafetyPriorHint`: no hint default prompt / injects when present / parsing not broken ✓
- `TestAgentSafetyPriorAndRecording`: hint format with prior / no prior returns None / empty label returns None / record writes event / skips empty label / skips empty dist ✓

### 6.2 集成测试

- `TestSafetyRoundTrip::test_safety_classify_persists_and_injects_next_episode` — Episode 1 record + consolidate, Episode 2 prior available + hint formatted ✓
- `TestSafetyRoundTrip::test_safety_dist_converges_across_episodes` — 3 episodes (0.80, 0.85, 0.90) → mean = 0.85 ✓
- `TestSafetyRoundTrip::test_failed_episode_does_not_persist_safety` — failed grasp → prior remains None ✓

### 6.3 全套回归

`pytest tests/ -q` → **262/262 passed**（含 +17 new safety tests）

---

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 错误先验被强化 | hint 文本强调 "Bayesian prior, trust visual evidence if contradicts"；仅 success 才合并 |
| label 不同导致 prior miss | Phase 2 recognition 已 normalize label，知识可跨 (vlm_label, target) 复用 |
| LLM 把 hint 当 ground truth | 测试 `test_with_prior_hint_injects_into_prompt` 验证拼接，不改 schema 输出格式 |
| YAML 损坏 | 沿用 Phase 1 graceful degradation 路径 |

---

## 8. 不在本范围内

- safety_knowledge 与 recognition_hints 的联合 prior（如 "tangerine 经常误标为 orange，且 orange 一般是 safe"）
- 用户反馈的 safety 修正（"这个不锋利，是塑料的"）
- 任何形式的 prompt 模板文件改动（`prompts/safety/classify.txt` 保持不变）

---

## 9. 文件变更统计

| 文件 | 新增行数 | 修改行数 |
|---|---|---|
| `src/memory_manager.py` | ~85 | 5 |
| `src/safety_gate.py` | ~15 | ~10 |
| `src/agent.py` | ~55 | 5 |
| `tests/test_memory_manager.py` | ~180 | 0 |
| `tests/test_safety_classifier.py` | ~40 | 0 |
| `tests/test_semantic_fallback.py` | ~95 | 0 |
| `tests/test_memory_integration.py` | ~115 | 0 |
| **TOTAL** | **~585** | **~20** |

---

## 10. Commits

| Hash | Summary |
|---|---|
| `c9a9bf1` | feat(memory): add safety consolidation + get_safety_prior reader |
| `6dd7134` | feat(safety): SafetyClassifier supports prior_hint injection |
| `7aa59d5` | feat(agent): inject safety prior + record safety_classified events |
| `2e57c75` | test(memory): end-to-end safety knowledge round-trip |
