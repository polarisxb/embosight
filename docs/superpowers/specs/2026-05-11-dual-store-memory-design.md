# EmboSight Dual-Store Episodic Memory 设计

> 日期: 2026-05-11
> 状态: approved
> 前序: `docs/superpowers/specs/2026-05-08-emboSight-belief-driven-agent-design.md`
> 关键词: episodic memory, dual-store, working memory, long-term memory, self-healing, strategy adaptation
> 对标: Reflexion (NeurIPS'23), Voyager (ICML'23), Claude Code three-layer memory

---

## 0. TL;DR

为 EmboSight agent 增加双存储记忆系统:

- **Working Memory**: 内存中，episode 内实时写入失败/修正事件，立刻影响后续决策
- **Long-term Memory**: 磁盘文件 (YAML pointer-index)，episode 结束后沉淀经验，下次任务加载

解决的问题类别: agent 反复犯同样的错误 (如重复选择失败的抓取策略)。
直接修复: 场景 001/007/009 的抓取策略回归 (geometric_centroid → ik_unreachable)。

---

## 1. 动机

### 1.1 当前问题

| 场景 | 失败原因 | 根因 |
|------|---------|------|
| 001 | geometric_centroid → ik_unreachable | agent 不记得此策略对 tupperware 无效 |
| 007 | vlm_top_grasp → hit_z_floor | agent 不知道此策略下降过深 |
| 009 | geometric_centroid → ik_unreachable | 同 001 |

共同模式: **agent 没有从失败中学习的能力**。每次 episode 从零开始，不记得上次犯的错。

### 1.2 设计灵感

- **Claude Code 泄漏架构**: 三层记忆 (context / memory.md pointer-index / CLAUDE.md)
- **Reflexion (NeurIPS'23)**: post-episode reflection，但只在 episode 结束后学习
- **Voyager (ICML'23)**: skill library 跨 episode 积累
- **认知科学双存储模型**: working memory (短期、容量有限) + long-term memory (持久、需要巩固)

### 1.3 我们的创新

相比 Reflexion 的改进:
- **实时写入**: 不等 episode 结束，action 失败后立即记录，同一 episode 内即可受益
- **结构化领域记忆**: 按 grasp/recognition/safety 分文件，按需加载，token 高效
- **Self-healing**: agent 自己维护记忆准确性 (read-before-write + merge)

---

## 2. 架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                        Episode 生命周期                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  [Episode Start]                                                   │
│       │                                                            │
│       ▼                                                            │
│  MemoryManager.load_for_task(target)                              │
│       │  读 index.yaml → 加载 grasp/recognition 相关条目           │
│       ▼                                                            │
│  seed into prompts (select_strategy, decide_next)                 │
│       │                                                            │
│       ▼                                                            │
│  ┌─── Agent Loop ─────────────────────────────────────────────┐   │
│  │  decide_next() ← reads working_memory                      │   │
│  │       │                                                     │   │
│  │       ▼                                                     │   │
│  │  execute_action()                                           │   │
│  │       │                                                     │   │
│  │       ├─ success → continue                                 │   │
│  │       └─ failure → MemoryManager.record_event(entry)        │   │
│  │              └─ writes to working_memory (in-place)         │   │
│  │                                                             │   │
│  │  (loop until success or MAX_STEPS)                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                            │
│       ▼                                                            │
│  [Episode End]                                                     │
│       │                                                            │
│       ▼                                                            │
│  MemoryManager.consolidate(working_memory, result)                │
│       │  working_memory 精华 → merge into long-term YAML          │
│       ▼                                                            │
│  memory/*.yaml updated (self-healing)                             │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. 数据结构

### 3.1 Working Memory Entry

```python
@dataclass
class MemoryEntry:
    step: int               # 发生在第几步
    domain: str             # "grasp" | "recognition" | "safety"
    event: str              # 事件类型 (见 3.2)
    context: dict           # 结构化上下文
    lesson: str             # 一句话总结 (注入 LLM prompt 用)
```

### 3.2 事件类型

| domain | event | context 内容 | 触发时机 |
|--------|-------|-------------|---------|
| grasp | strategy_failed | strategy, failure_mode, object_type | grasp 执行返回非 success |
| grasp | strategy_succeeded | strategy, object_type | grasp 成功 |
| recognition | label_corrected | old_label, new_label, method (clip/llm/user) | CLIP 注入或 LLM fallback 纠正标签 |
| recognition | synonym_effective | target, synonym, method | synonym 成功匹配到目标 |
| safety | risk_reclassified | object_type, old_risk, new_risk | 安全重分类（future） |

### 3.3 Long-term Memory 文件格式

**`memory/index.yaml`**
```yaml
version: 1
last_updated: "2026-05-11T00:00:00"
domains:
  grasp: memory/grasp_experience.yaml
  recognition: memory/recognition_hints.yaml
  # safety: memory/safety_knowledge.yaml      # future
  # environment: memory/environment_map.yaml  # future
```

**`memory/grasp_experience.yaml`**
```yaml
entries:
  - object_type: tupperware
    best_strategy: top_down
    failed:
      - strategy: geometric_centroid
        reason: ik_unreachable
        count: 2
      - strategy: vlm_top_grasp
        reason: hit_z_floor
        count: 1
    total_attempts: 5
    success_count: 3
    notes: "rectangular container, always prefer top_down approach"
    last_updated: "2026-05-11"

  - object_type: apple
    best_strategy: top_down
    failed: []
    total_attempts: 2
    success_count: 2
    notes: ""
    last_updated: "2026-05-10"
```

**`memory/recognition_hints.yaml`**
```yaml
entries:
  - target: tangerine
    vlm_common_labels: ["orange", "citrus", "fruit"]
    effective_synonyms: ["orange"]
    clip_helpful: true
    notes: "VLM rarely outputs 'tangerine' directly"
    last_updated: "2026-05-11"

  - target: yogurt
    vlm_common_labels: ["yogurt", "container", "dairy"]
    effective_synonyms: []
    clip_helpful: false
    notes: "VLM usually identifies correctly"
    last_updated: "2026-05-11"
```

---

## 4. 核心模块: MemoryManager

### 4.1 接口定义

```python
class MemoryManager:
    """Dual-store episodic memory manager.
    
    Manages working memory (in-session) and long-term memory (on-disk YAML).
    """

    def __init__(self, memory_dir: Path = Path("memory")):
        self.memory_dir = memory_dir
        self.working_memory: list[MemoryEntry] = []
        self._long_term: dict[str, dict] = {}  # domain → parsed YAML
        self._load_index()

    # ── Load (Episode 开始) ──

    def load_for_task(self, primary_target: str, object_type: str = "") -> str:
        """加载与当前任务相关的经验, 返回文本摘要 (可直接注入 prompt)。
        
        只加载相关条目, 不加载全部, token 高效。
        """

    def get_grasp_advice(self, object_type: str) -> Optional[str]:
        """获取特定物体的抓取建议文本。
        
        Returns: e.g. "tupperware: prefer top_down, avoid geometric_centroid (ik_unreachable ×2)"
        Returns None if no experience for this object_type.
        """

    def get_recognition_hints(self, target: str) -> Optional[str]:
        """获取识别提示。
        
        Returns: e.g. "tangerine: VLM often labels as 'orange', CLIP synonym effective"
        Returns None if no hints for this target.
        """

    # ── Record (Episode 中, 实时) ──

    def record_event(self, entry: MemoryEntry) -> None:
        """实时记录到 working_memory。"""
        self.working_memory.append(entry)

    def get_working_summary(self, domain: Optional[str] = None) -> str:
        """获取 working_memory 的文本摘要, 可按 domain 过滤。"""

    # ── Consolidate (Episode 结束) ──

    def consolidate(self, success: bool, object_type: str = "") -> None:
        """Episode 结束后, 将 working_memory 精华写入 long-term.
        
        原则:
        - read-before-write: 先读当前文件, merge 而非覆盖
        - 只写有意义的事件 (失败策略、成功策略)
        - 更新 last_updated 和计数
        """

    # ── Internal ──

    def _load_index(self) -> None: ...
    def _load_domain(self, domain: str) -> dict: ...
    def _save_domain(self, domain: str) -> None: ...
    def _merge_grasp_entry(self, existing: dict, events: list[MemoryEntry]) -> dict: ...
    def _merge_recognition_entry(self, existing: dict, events: list[MemoryEntry]) -> dict: ...
```

### 4.2 关键实现细节

**选择性加载**: `load_for_task("yogurt")` 只加载 grasp_experience 中 object_type="yogurt" 的条目 + recognition_hints 中 target="yogurt" 的条目。不加载整个文件到 prompt。

**Self-healing (read-before-write)**:
```python
def consolidate(self, ...):
    # 1. 读当前文件
    existing = self._load_domain("grasp")
    # 2. 找到对应条目 (或创建新条目)
    entry = find_or_create(existing, object_type)
    # 3. Merge: 更新 failed, success_count, best_strategy
    updated = self._merge_grasp_entry(entry, grasp_events)
    # 4. 写回
    self._save_domain("grasp")
```

**best_strategy 自动推导**:
```python
# 成功次数最多的策略 = best
strategies_success = Counter(
    e.context["strategy"] for e in events if e.event == "strategy_succeeded"
)
if strategies_success:
    entry["best_strategy"] = strategies_success.most_common(1)[0][0]
```

---

## 5. 集成点

### 5.1 GraspPlanner.select_strategy() — 注入经验

**修改**: `select_strategy(hyp, memory_advice: str = "")`

Prompt 模板新增:
```
Past experience with {label}: {memory_advice}
If past experience indicates a strategy failure, avoid repeating it.
```

`memory_advice` 来源:
1. Long-term: `MemoryManager.get_grasp_advice(hyp.label)`
2. Working: `MemoryManager.get_working_summary(domain="grasp")`
3. 合并为一段文本

### 5.2 Agent — 写入时机

```python
# agent.py _execute_action() 中 grasp 分支:

result = self.action_executor.act(...)
if result.success:
    self.memory.record_event(MemoryEntry(
        step=self._step, domain="grasp", event="strategy_succeeded",
        context={"strategy": strategy.strategy, "object": hyp.label},
        lesson=f"{hyp.label}: {strategy.strategy} succeeded",
    ))
else:
    self.memory.record_event(MemoryEntry(
        step=self._step, domain="grasp", event="strategy_failed",
        context={"strategy": strategy.strategy, "failure": result.failure_mode, "object": hyp.label},
        lesson=f"{hyp.label}: {strategy.strategy} failed ({result.failure_mode}), avoid this strategy",
    ))
```

### 5.3 Agent — Episode 生命周期

```python
# agent.py run() 中:

def run(self, ...):
    # Episode 开始: 加载 long-term 经验
    self.memory = MemoryManager()
    prior = self.memory.load_for_task(self.belief.decomposed.primary_target)
    # prior 注入初始 context

    # ... agent loop ...

    # Episode 结束: consolidate
    self.memory.consolidate(
        success=result["success"],
        object_type=hyp.label if hyp else "",
    )
```

### 5.4 WorldBelief — 持有 working_memory 引用

`WorldBelief` 不直接管理 MemoryManager，但 `working_memory` 列表存在 belief 中，方便 snapshot 和日志记录。

---

## 6. 具体示例: 修复场景 001

**无记忆 (当前行为):**
```
Step 1-4: observe, classify_safety, detect tupperware
Step 5:   select_strategy → LLM picks "geometric_centroid" (没经验)
Step 6:   grasp(geometric_centroid) → ik_unreachable
Step 7:   re_observe (浪费)
Step 8:   select_strategy → LLM 可能又选 geometric_centroid (无学习)
...
Step 13:  MAX_STEPS ✗
```

**有 Working Memory (Level 1):**
```
Step 1-4: observe, classify_safety, detect tupperware
Step 5:   select_strategy → LLM picks "geometric_centroid"
Step 6:   grasp(geometric_centroid) → ik_unreachable
          → working_memory += "geometric_centroid failed (ik_unreachable)"
Step 7:   select_strategy → prompt includes:
          "Past: geometric_centroid → ik_unreachable. Avoid."
          → LLM picks "top_down"
Step 8:   grasp(top_down) → SUCCESS ✓
```

**有 Long-term Memory (Level 2, 第二次遇到 tupperware):**
```
Step 1-4: observe, classify_safety, detect tupperware
          → load: "tupperware: prefer top_down, avoid geometric_centroid"
Step 5:   select_strategy → prompt includes prior experience
          → LLM directly picks "top_down"
Step 6:   grasp(top_down) → SUCCESS ✓ (省了 2 步)
```

---

## 7. 实现范围

### Phase 1 (本次实现 — 解决回归)

- [x] `MemoryEntry` dataclass
- [x] `MemoryManager` 核心类 (load/record/consolidate)
- [x] Working memory 写入 (grasp 失败/成功)
- [x] `select_strategy` prompt 注入经验
- [x] Long-term YAML 读写 (grasp_experience.yaml)
- [x] Episode 开始加载 / 结束 consolidate
- [x] 单元测试
- [x] Batch eval 验证成功率恢复

### Phase 2 (后续 — 扩展领域)

- [x] recognition_hints.yaml 写入 (CLIP/LLM 纠正时) — 见 `docs/superpowers/specs/2026-05-13-memory-phase2-recognition-design.md`
- [x] recognition hints 注入 perception 流程 (via primary_target_synonyms, 方案 A)
- [x] safety_knowledge.yaml 写入 + 注入 (跨 episode running average) — 见 `docs/superpowers/specs/2026-05-14-memory-phase2-safety-design.md`
- [ ] 消融实验 (w/ vs w/o memory)

### Phase 3 (论文级 — 语义检索)

- [ ] Embedding-based 相似物体经验迁移 (apple 经验 → pear)
- [ ] LLM 生成 consolidation summary
- [ ] Multi-agent memory sharing

---

## 8. 测试计划

### 单元测试

```python
class TestMemoryManager:
    def test_record_event_appends_to_working(): ...
    def test_get_grasp_advice_returns_text(): ...
    def test_get_grasp_advice_no_entry_returns_none(): ...
    def test_consolidate_creates_new_entry(): ...
    def test_consolidate_merges_existing(): ...
    def test_consolidate_updates_best_strategy(): ...
    def test_load_for_task_selective(): ...
    def test_self_healing_read_before_write(): ...

class TestMemoryIntegration:
    def test_strategy_prompt_includes_memory(): ...
    def test_failed_strategy_not_repeated(): ...
    def test_working_memory_survives_within_episode(): ...
```

### 集成验证

- Batch eval: 001/007/009 应从 ✗ → ✓
- 成功率目标: ≥ 8/11 (73%)，恢复 pre-synonym 基线

---

## 9. 风险与缓解

| 风险 | 缓解 |
|------|------|
| LLM 不听 memory advice | prompt 用强制语气 "You MUST NOT use..." |
| memory.yaml 损坏 | graceful degradation: 读取失败时忽略，不影响正常流程 |
| 记忆膨胀 | 每个 object_type 最多保留 5 条 failed 记录 |
| 跨 episode 经验错误泛化 | best_strategy 需要 ≥2 次成功才写入 |
| 并发写冲突 (多 GPU) | consolidate 用文件锁 (fcntl/msvcrt) |

---

## 10. 文件变更清单

| 文件 | 变更 |
|------|------|
| `src/memory_manager.py` | **新建** — MemoryManager + MemoryEntry |
| `src/agent.py` | 添加 memory 初始化、record_event 调用、consolidate |
| `src/grasp_planner.py` | select_strategy 接受 memory_advice 参数 |
| `prompts/grasp/select_strategy.txt` | 新增 Past experience 占位符 |
| `src/world_belief.py` | WorldBelief 新增 working_memory 字段 |
| `memory/index.yaml` | **新建** — 初始索引 |
| `memory/grasp_experience.yaml` | **新建** — 空初始 |
| `memory/recognition_hints.yaml` | **新建** — 空初始 |
| `tests/test_memory_manager.py` | **新建** — 单元测试 |
| `tests/test_memory_integration.py` | **新建** — 集成测试 |
