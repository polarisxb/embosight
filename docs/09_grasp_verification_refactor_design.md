# Grasp Verification Refactor — Phase 6 设计文档

**状态**: 设计中 · 待用户审阅  
**日期**: 2026-05-17  
**前置**: Phase 0-5 完成（navigate_base_to 已 GPU 验证 working, commit `5082fc6`）  
**关联**: 调研报告 `docs/08_grasp_verification_survey.md`

---

## TL;DR

修复 Phase 5 GPU Run 3 暴露的 false-positive grasp confirmation bug（tupperware lid 单侧 fingertip 擦到就触发 grasp confirmed，但物体没真夹住）。

**3 个阶段**：
- **Phase 6.1** 替换 `_finger_object_contact` 为 **bilateral fingerpad contact**（业界标准，robosuite 官方实现）
- **Phase 6.2** 在 `close_gripper` 之后立即做 **micro-lift verification**（升 2cm 查 object follow，失败立即 slipped_lift）
- **Phase 6.3** 加 **gripper width sanity check**（jaw 完全闭到 0 → empty grasp）

**预期效果**: tupperware seed=42 从 fail 变 success；wall-time 430s → ~250s；false-positive grasp 从 1 → 0。

**风险**: 纯增量改动，所有旧代码保留 fallback 路径；可单行 revert。

---

## 1. 背景

### 1.1 上一阶段成果（Phase 0-5）

Navigation refactor 已 GPU 验证 working：
- `navigate_base_to` teleport 把 base 精确放到 target 后方 0.43m
- anchor-local qpos 转换正确
- `ik_unreachable` 从 baseline 3 → 1 (剩下 1 个是不同 approach 角度的真实 IK 限制)
- ✅ Phase 4 commit `34e53f1`, anchor 修复 `5082fc6`

### 1.2 Phase 5 暴露的新问题

GPU Run 3 日志关键片段：

```
[close_gripper] contact at step 7, squeezing 10 more
[close_gripper] grasp confirmed + squeezed at step 16     ← 系统认为抓住了
[move_arm_to] max_steps reached, dist=0.0040m  ×4         ← lift 3 次 stall (23s)
[act] object NOT lifted: z_before=0.943 z_after=0.943 Δ=0.000  ← 实际没抓住
```

**问题陈述**：
- false-positive grasp 检测导致后续 lift_arm(0.10) 浪费 ~20s
- 浪费时间后才发现失败，触发 LLM 重新规划（再消耗 ~30s）
- 总损耗：~50s/false-positive × 多次 attempt → 430s/episode

### 1.3 根因

`@c:\all_project\embodied-AI-one\src\env_wrapper.py:1392-1426` 的 `_finger_object_contact`：

```python
finger_kw = ("finger", "fingertip", "finger_pad", "tip", "pad")
finger_geoms = set()  # ← 左右指 geom 合并!
for i in range(sim.model.ngeom):
    if any(kw in name.lower() for kw in finger_kw):
        finger_geoms.add(i)

for i in range(sim.data.ncon):
    if (g1 in obj_geoms and g2 in finger_geoms) or ...:
        return True  # ← 任意一指 contact 就 True
```

**问题**：左指 + 右指 + knuckle + tip 全部合并到一个 set。任意一个 finger geom 擦到物体（包括 fingertip 侧面、连接处）就返回 True，**与 grasp 是否真的形成无关**。

### 1.4 业界 Ground Truth

[robosuite 官方 `_check_grasp`](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/environments/manipulation/manipulation_env.py)：

```python
def _check_grasp(self, gripper, object_geoms):
    g_geoms = [gripper.important_geoms["left_fingerpad"],
               gripper.important_geoms["right_fingerpad"]]
    for g_group in g_geoms:
        if not self.check_contact(g_group, o_geoms):
            return False
    return True
```

**只用 `important_geoms["left_fingerpad"]` 和 `["right_fingerpad"]`**，且两组必须**同时**与 object 接触。这是 robosuite RL benchmark / PickPlace / Lift 任务的通用判定。

详细调研见 `docs/08_grasp_verification_survey.md`。

---

## 2. 目标与非目标

### 2.1 目标

- **G1**：消除 false-positive grasp confirmation（双侧 fingerpad 必须接触）
- **G2**：将 slip 检测从 lift 后 23s 提前到 close 后 2s
- **G3**：保留对 legacy mock / 异常 gripper 命名的兼容性（纯增量、有 fallback）
- **G4**：不改变 `grasp_failure_mode` 语义（slipped_lift 仍代表"夹后未跟随"）
- **G5**：单元测试覆盖所有新行为，336+ 个测试不破坏

### 2.2 非目标

- **N1**：不修 OSC stall（独立问题，Phase 7+ 处理）
- **N2**：不改 grasp_strategy 选择逻辑（GraspPlanner 不动）
- **N3**：不引入 tactile sensor / force feedback（sim-only，无硬件）
- **N4**：不再 bump GRASP_CODE_VERSION（v6.2 schema 保持，slipped_lift 标签语义不变）
- **N5**：不改 close_gripper 的 squeeze_steps 数（保留经验值）

---

## 3. 架构总览

### 3.1 当前 vs 目标 control flow

**Phase 5 当前**：

```
act()
 ├─ navigate_base_to            ← Phase 4 done
 ├─ move_to_pre_grasp
 ├─ approach / descend
 ├─ close_gripper
 │   └─ _close_gripper_until_grasp
 │       └─ _finger_object_contact  ← BUG: 单侧合并
 ├─ lift_arm(0.10)              ← 浪费 20s 才发现失败
 └─ verify Δz_obj > 0.05        ← 失败分类 slipped_lift
```

**Phase 6 目标**：

```
act()
 ├─ navigate_base_to
 ├─ move_to_pre_grasp
 ├─ approach / descend          ← 用新 bilateral 判定
 ├─ close_gripper               ← 用新 bilateral 判定
 ├─ ★ verify_grasp_by_micro_lift(0.02)   [NEW Phase 6.2]
 │   └─ 失败 → 立即 return slipped_lift
 ├─ lift_arm(0.10)              ← 只在 micro-lift 通过后才执行
 └─ verify Δz_obj > 0.05        ← 保留作为最终兜底
```

### 3.2 三层防御 (defense-in-depth)

| Layer | 检测 | 触发时机 | 失败处理 |
|-------|------|----------|----------|
| L1 | bilateral fingerpad contact | descend 时早停 / close_gripper 时确认 | descend 继续 / close 拒绝 confirm |
| L2 | micro-lift object follow (2cm) | close_gripper 之后立即 | 立即 return slipped_lift |
| L3 | full-lift Δz (10cm) | lift_arm 之后 | 现有 slipped_lift 分类（兜底） |
| **L4 (Phase 6.3)** | gripper width != 0 | close_gripper 内 | reject confirm（gripper 闭到空） |

---

## 4. Phase 6.1 — Bilateral Fingerpad Contact

### 4.1 接口设计

新增方法（保持 `_finger_object_contact` 名字不变，内部重构）：

```python
def _finger_object_contact(self, target_body: str) -> bool:
    """
    检查 left_fingerpad AND right_fingerpad 是否同时与 target_body 接触.
    
    业界标准 (robosuite._check_grasp): 两组 fingerpad geom 都必须与 object 有
    contact 才算 grasp. 防止 "一指擦边" 或 "fingertip 侧面碰" 的 false positive.
    
    优先路径: 调用 robosuite._env._check_grasp (业界 ground truth).
    Fallback: 本地 bilateral 实现 (用 geom name 的 'left'/'right' 关键字分组).
    Final fallback: 旧的合并判定 (legacy, 保 backward compat).
    """
    # Path 1: robosuite official API
    result = self._check_grasp_robosuite(target_body)
    if result is not None:
        return result
    
    # Path 2: local bilateral (基于 left/right 关键字)
    result = self._check_grasp_bilateral_local(target_body)
    if result is not None:
        return result
    
    # Path 3: legacy merged (兜底, 防新 gripper 命名不规范)
    return self._check_grasp_legacy_merged(target_body)
```

### 4.2 Path 1：调用 robosuite official API

```python
def _check_grasp_robosuite(self, target_body: str) -> Optional[bool]:
    """Returns None if API unavailable (caller falls through to Path 2)."""
    try:
        env = self._env
        if not hasattr(env, "_check_grasp"):
            return None
        robot = env.robots[0]
        gripper = robot.gripper
        # PandaMobile: gripper 可能是 dict {'right': GripperModel}
        if isinstance(gripper, dict):
            gripper = gripper.get("right") or next(iter(gripper.values()))
        if gripper is None or not hasattr(gripper, "important_geoms"):
            return None
        if "left_fingerpad" not in gripper.important_geoms:
            return None  # 非标准 gripper, fallback
        # object_geoms 用 geom NAME (robosuite API contract)
        obj_geom_ids = self._get_body_geom_ids(target_body)
        obj_geom_names = []
        for gid in obj_geom_ids:
            try:
                name = env.sim.model.geom_id2name(gid)
                if name:
                    obj_geom_names.append(name)
            except Exception:
                continue
        if not obj_geom_names:
            return None
        return bool(env._check_grasp(gripper, obj_geom_names))
    except Exception as e:
        logger.debug(f"[grasp_check] robosuite API failed: {e}")
        return None
```

### 4.3 Path 2：本地 bilateral 实现

```python
def _check_grasp_bilateral_local(self, target_body: str) -> Optional[bool]:
    """
    本地 bilateral 判定: 用 geom name 中 'left'/'right' 关键字分组.
    
    Returns None if 无法区分左右 (caller falls through to Path 3).
    """
    sim = self._env.sim
    obj_geoms = self._get_body_geom_ids(target_body)
    if not obj_geoms:
        return False
    
    left_geoms, right_geoms = set(), set()
    pad_kw = ("finger_pad", "fingerpad", "fingertip", "pad", "tip")
    
    for i in range(sim.model.ngeom):
        name = (sim.model.geom_id2name(i) or "").lower()
        if not any(kw in name for kw in pad_kw):
            continue
        if "left" in name or "_l_" in name or name.endswith("_l"):
            left_geoms.add(i)
        elif "right" in name or "_r_" in name or name.endswith("_r"):
            right_geoms.add(i)
    
    if not left_geoms or not right_geoms:
        return None  # 无法区分, fallback
    
    left_touch = right_touch = False
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 in obj_geoms:
            opp = g2
        elif g2 in obj_geoms:
            opp = g1
        else:
            continue
        if opp in left_geoms:
            left_touch = True
        if opp in right_geoms:
            right_touch = True
        if left_touch and right_touch:
            return True
    return False
```

### 4.4 Path 3：Legacy fallback

保留现有 `_finger_object_contact` 实现，改名为 `_check_grasp_legacy_merged` 作为最终 fallback。注释说明仅用于兼容不规范命名的 gripper。

### 4.5 测试计划

文件：`tests/test_env_wrapper_grasp_bilateral.py` (新)

| Test | 场景 | 期望 |
|------|------|------|
| `test_robosuite_api_path_uses_check_grasp` | mock `env._check_grasp` 返 True | 直接返回 True，不走 Path 2 |
| `test_local_bilateral_left_only_returns_false` | 仅左指 geom 接触 | False |
| `test_local_bilateral_right_only_returns_false` | 仅右指 geom 接触 | False |
| `test_local_bilateral_both_returns_true` | 左右指都接触 | True |
| `test_no_pad_geoms_falls_through_to_legacy` | gripper 没有 left/right 命名 | 走 Path 3，行为同旧版 |
| `test_target_body_not_found_returns_false` | target_body 无 geom | False（不抛异常） |
| `test_existing_descend_tests_still_pass` | 现有 descend / close_gripper 测试 | 全部通过（regression） |

---

## 5. Phase 6.2 — Micro-Lift Verification

### 5.1 接口设计

新增 EnvWrapper 方法：

```python
def verify_grasp_by_micro_lift(
    self,
    target_body: str,
    lift_m: float = 0.02,
    threshold: float = 0.5,
    max_steps: int = 80,
) -> bool:
    """
    关爪后立即做 lift_m 的微抬升, 检查 target_body 的 z 是否跟随至少
    threshold * lift_m 比例.
    
    业界标准 early slip detection: 在大 lift 之前快速验证 grasp 稳固性.
    若 micro-lift 失败, 立即放弃当前 attempt, 省下 ~20s lift_arm 浪费.
    
    Args:
        target_body: sim body name (e.g. "obj_main")
        lift_m: 微抬高度 (默认 2cm, 足以判断 slip 但不会触发 OSC stall)
        threshold: object Δz 必须 >= lift_m * threshold (默认 50%)
        max_steps: move_arm_to 步数上限
    
    Returns:
        True 若 object 跟随成功, False 若 slipped.
        无法读 obj z 时保守返 True (上游用 lift_arm 后的 Δz 兜底).
    """
```

### 5.2 实现

```python
def verify_grasp_by_micro_lift(
    self, target_body: str, lift_m: float = 0.02,
    threshold: float = 0.5, max_steps: int = 80,
) -> bool:
    try:
        obj_pos_before = self._get_body_pos(target_body)
        if obj_pos_before is None:
            logger.debug(
                f"[micro_lift] cannot read obj z for {target_body}, "
                "保守返 True"
            )
            return True
        obj_z_before = float(obj_pos_before[2])
        eef_before = self.get_eef_pos().copy()
        
        target = eef_before.copy()
        target[2] = float(eef_before[2]) + float(lift_m)
        
        # gripper_hold=1.0 保持夹爪闭合, 防止松开导致 slip
        self.move_arm_to(
            target,
            threshold_m=0.005,
            max_steps=max_steps,
            gripper_hold=1.0,
        )
        
        obj_pos_after = self._get_body_pos(target_body)
        if obj_pos_after is None:
            return True
        obj_z_after = float(obj_pos_after[2])
        obj_delta = obj_z_after - obj_z_before
        eef_delta = float(self.get_eef_pos()[2]) - float(eef_before[2])
        
        follows = obj_delta >= lift_m * threshold
        logger.info(
            f"[micro_lift] eef Δz={eef_delta:.4f} obj Δz={obj_delta:.4f} "
            f"follows={follows} (req >= {lift_m * threshold:.4f})"
        )
        return follows
    except Exception as e:
        logger.warning(f"[micro_lift] failed: {e}, 保守返 True")
        return True
```

### 5.3 ActionExecutor 集成

`@c:\all_project\embodied-AI-one\src\action_executor.py` `act()`：

```python
# 4. close gripper
if not env.close_gripper(target_label=target.label):
    return self._failed_result(candidate, "gripper_empty", {...}, env)

# 4.5 [NEW Phase 6.2] micro-lift early slip detection
if hasattr(env, "verify_grasp_by_micro_lift"):
    try:
        target_body = self._resolve_target_body(target, env)
        if target_body and not env.verify_grasp_by_micro_lift(
            target_body, lift_m=0.02, threshold=0.5
        ):
            return self._failed_result(
                candidate, "slipped_lift",
                {
                    "stage": "micro_lift_verify",
                    "reason": "object_not_following",
                    "threshold": 0.5,
                    "lift_m": 0.02,
                },
                env,
            )
    except Exception as e:
        logger.debug(f"[act] micro_lift verification error: {e}, continuing")

# 5. full lift_arm (现有逻辑)
ok, final_z = env.lift(height_m=0.10, approach_dir=...)
```

辅助方法 `_resolve_target_body(target, env)`：

```python
def _resolve_target_body(self, target, env) -> Optional[str]:
    """获取 Hypothesis.label 对应的 sim body name."""
    try:
        if hasattr(env, "_get_obj_type_map"):
            type_map = env._get_obj_type_map()
            for body, cat in type_map.items():
                if cat == target.label:
                    return body
        return None
    except Exception:
        return None
```

### 5.4 测试计划

文件：`tests/test_action_executor_phase6.py` (新)

| Test | 场景 | 期望 |
|------|------|------|
| `test_micro_lift_pass_continues_to_full_lift` | obj 跟随 2cm | act() 继续, 最终 success |
| `test_micro_lift_fail_returns_slipped_lift` | obj Δz=0 | act() 立即 return, failure_mode='slipped_lift', diagnostic.stage='micro_lift_verify' |
| `test_micro_lift_missing_method_falls_through` | env 无该方法 | act() 走老逻辑（backward compat） |
| `test_micro_lift_exception_falls_through` | verify 抛异常 | act() 继续到 full lift（保守） |
| `test_resolve_target_body_returns_body_name` | 正常 type_map | 返回 body name |
| `test_resolve_target_body_no_match_returns_none` | label 不匹配 | None，触发 fallback |

新 EnvWrapper 测试 (`tests/test_env_wrapper_grasp_bilateral.py`)：

| Test | 场景 | 期望 |
|------|------|------|
| `test_micro_lift_returns_true_when_obj_follows` | mock obj 跟随 | True |
| `test_micro_lift_returns_false_when_obj_stays` | obj z 不变 | False |
| `test_micro_lift_returns_true_when_body_not_found` | body 读不到 | True（保守） |
| `test_micro_lift_threshold_applied` | obj Δz < threshold*lift_m | False |

---

## 6. Phase 6.3 — Gripper Width Sanity Check

### 6.1 接口设计

```python
def _gripper_closed_on_empty(
    self, threshold_m: float = 0.005
) -> bool:
    """
    检查 gripper 是否完全闭合 (两指间距 < threshold_m).
    
    完全闭合 = jaw 撞在一起 = gripper 没夹住任何东西.
    用作 close_gripper_until_grasp 的额外验证, 防止 jaw closed empty
    false positive.
    
    Returns:
        True 若闭到空 (false positive 风险高).
        False 若 jaw 间还有空隙 (正常 grasp).
    """
```

### 6.2 实现

```python
def _gripper_closed_on_empty(self, threshold_m: float = 0.005) -> bool:
    try:
        # 优先用 obs (robosuite 标准 key)
        obs = self._latest_obs or {}
        for key in ("robot0_gripper_qpos", "gripper_qpos"):
            qpos = obs.get(key)
            if qpos is None:
                continue
            qpos = np.asarray(qpos, dtype=np.float32)
            # Panda parallel jaw: 前两维是两指位置, 相加 = 总 gap
            gap = float(np.abs(qpos[:2]).sum()) \
                if qpos.size >= 2 else float(qpos[0])
            return gap < threshold_m
        # Fallback: 直接读 sim.data.qpos at gripper joint addrs
        # ... (略, 见实现)
        return False
    except Exception:
        return False
```

### 6.3 集成到 close_gripper

`_close_gripper_until_grasp` 内部，confirm 之前加：

```python
if not confirmed and i >= min_close_steps and target_contact and generic_grasp:
    # [NEW Phase 6.3] 排除 jaw closed empty
    if self._gripper_closed_on_empty():
        logger.warning(
            f"[close_gripper] step {i}: contact detected but jaw closed empty, "
            "skipping confirm"
        )
        continue
    logger.info(f"[close_gripper] contact at step {i}, squeezing {squeeze_steps} more")
    confirmed = True
```

### 6.4 测试计划

| Test | 场景 | 期望 |
|------|------|------|
| `test_gripper_closed_on_empty_returns_true_when_gap_small` | gap=0.003m | True |
| `test_gripper_closed_on_empty_returns_false_when_gap_normal` | gap=0.020m | False |
| `test_gripper_closed_on_empty_missing_obs_returns_false` | obs 无该 key | False（保守不报告） |
| `test_close_gripper_skips_confirm_when_jaw_closed_empty` | mock empty jaw | confirmed 不触发 |

---

## 7. 实施顺序与 commit 策略

### 7.1 Phase 6.1 commit (~1h)

1. 重构 `_finger_object_contact` 为 3-path 实现
2. 添加 `_check_grasp_robosuite`, `_check_grasp_bilateral_local`, `_check_grasp_legacy_merged` 私有方法
3. 新增单测文件 `tests/test_env_wrapper_grasp_bilateral.py` (7 tests)
4. 跑 `pytest tests/` 确认全过 (341 → 348)
5. Commit: `fix(grasp): bilateral fingerpad contact (Phase 6.1)`

### 7.2 Phase 6.2 commit (~1.5h)

1. 添加 `verify_grasp_by_micro_lift` 到 EnvWrapper
2. 改 `ActionExecutor.act()` 注入 micro-lift 验证步骤
3. 添加 `_resolve_target_body` 辅助方法
4. 新增 `tests/test_action_executor_phase6.py` (6 tests)
5. 扩展 `test_env_wrapper_grasp_bilateral.py` (+4 tests)
6. 跑 `pytest tests/` 确认全过 (348 → 358)
7. Commit: `feat(act): micro-lift early slip detection (Phase 6.2)`

### 7.3 Phase 6.3 commit (~0.5h)

1. 添加 `_gripper_closed_on_empty` 到 EnvWrapper
2. 在 `_close_gripper_until_grasp` 内集成 jaw width check
3. 扩展测试 (+4 tests)
4. 跑 `pytest tests/` 确认全过 (358 → 362)
5. Commit: `feat(grasp): jaw closed-on-empty sanity check (Phase 6.3)`

### 7.4 Phase 6 GPU 验证 (~10min)

```bash
cd ~/projects/embodied-AI-one
git pull
rm -rf runs/after_phase6/memory
bash scripts/phase5_validation_gpu.sh
```

**期望日志变化**：
- 不再出现 `grasp confirmed + squeezed` 后 `object NOT lifted`
- 新增 `[micro_lift] eef Δz=0.020 obj Δz=0.018 follows=True`
- 或 `[micro_lift] eef Δz=0.020 obj Δz=0.000 follows=False` → 立即 `slipped_lift`
- wall-time 从 430s 降至 ~250s

---

## 8. 风险分析与回滚

### 8.1 风险矩阵

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| robosuite `_check_grasp` API 在 PandaMobile 失败 | 中 | 低 | 三层 fallback (Path 1 → 2 → 3) |
| `important_geoms["left_fingerpad"]` 不存在 | 中 | 低 | Path 1 内 check 后 fallback |
| `left_fingerpad` / `right_fingerpad` 命名 not standard | 低 | 低 | Path 2 用关键字模糊匹配 |
| micro-lift 2cm 触发 OSC stall | 中 | 中 | threshold=0.005 + max_steps=80 已校准 |
| micro-lift threshold=0.5 误杀 borderline grasp | 中 | 中 | 实测可调 → 0.3 / 0.4 |
| `verify_grasp_by_micro_lift` 抛异常阻断 act() | 低 | 高 | try/except + 保守返 True |
| gripper qpos key 在 PandaMobile 不同 | 中 | 低 | 多 key 尝试 + sim.data.qpos fallback |
| 新失败模式 `slipped_lift` 早期出现影响 memory | 低 | 低 | v6.2 schema 已覆盖, 不需要再 bump |

### 8.2 回滚策略

所有改动**纯增量**，可分级回滚：

**Level 1 — 关闭 micro-lift (1 行)**：  
`@c:\all_project\embodied-AI-one\src\action_executor.py` 删除 `if hasattr(env, "verify_grasp_by_micro_lift"):` 整个 block。

**Level 2 — 回滚 bilateral**：  
把 `_finger_object_contact` 改回直接调 `_check_grasp_legacy_merged`，注释掉 Path 1/2。

**Level 3 — full revert**：  
`git revert <phase6.1>..<phase6.3>`，3 个 commit 干净回滚到 `5082fc6`。

### 8.3 不变量

- 任何分支必须 termiante (不能 inf loop)
- 任何 path 返 False 不能抛异常
- 旧 `_finger_object_contact` callsite 行为不变（API 同名同签名）
- micro-lift 失败时 EEF 仍处于可恢复位置（不在物体下方）
- 所有 logger 调用都用 `[grasp_check]` / `[micro_lift]` / `[jaw_check]` 前缀便于 grep

---

## 9. 验证标准（DoD）

### 9.1 单元测试

- [ ] 7+ 个 Phase 6.1 bilateral 测试通过
- [ ] 6+ 个 Phase 6.2 micro-lift 集成测试通过
- [ ] 4+ 个 Phase 6.3 jaw width 测试通过
- [ ] 现有 341 个测试 0 regression

### 9.2 GPU 验证（tupperware seed=42）

- [ ] `[micro_lift]` 日志出现
- [ ] 不再出现 `object NOT lifted` 在 grasp confirmed 之后
- [ ] `slipped_lift` 在 micro-lift 阶段触发（若 grasp 不稳）
- [ ] wall-time < 350s (vs 430s baseline)
- [ ] success=True（grasp 真的形成 → 完整 lift 成功）

### 9.3 文档

- [ ] `docs/09_grasp_verification_refactor_design.md` (本文档) committed
- [ ] `docs/08_grasp_verification_survey.md` 引用本设计文档
- [ ] commit messages 引用本设计 §X.Y 章节

---

## 10. ADR — 关键架构决策

### ADR-1: 优先复用 robosuite API 而非自实现

**决策**：Path 1 调用 `env._check_grasp`，Path 2/3 仅作 fallback。

**理由**：
- 业界标准，与所有 robosuite RL benchmark 对齐
- 上游维护，未来 API 优化自动继承
- 我们只需保证 fallback 路径正确，不需要完全替代

**取舍**：依赖 robosuite 上游版本稳定性。已用 v1.5 验证 API 存在。

### ADR-2: micro-lift 失败立即 fail (Option A)

**决策**：Phase 6.2 检测到 obj 不跟随 → 立即 return `slipped_lift`，不在 EnvWrapper 层做 retry recovery。

**理由**：
- 职责清晰：EnvWrapper 负责检测，LLM/Memory 负责策略调整
- 双重 retry 会引入复杂的 stall stack
- LLM v6.2 已经学会从 `slipped_lift` 切到不同 grasp_strategy
- 单元测试简单（一个 control flow 分支）

**取舍**：有些可挽救的 case 也会被 fail。实测若误杀率高再升级到 Option B（重新 close + squeeze）。

### ADR-3: 不 bump GRASP_CODE_VERSION

**决策**：保持 `v6.2`，不升 `v6.3`。

**理由**：
- `slipped_lift` 失败模式定义没改（仍是"夹后未跟随"）
- 只是检测**时机**提前（从 lift 后 → close 后）
- v6.2 历史数据中的 `slipped_lift` 仍然语义有效
- bump 会触发 memory 重学，浪费已积累的策略经验

**取舍**：若实测发现 v6.2 数据中 slipped_lift 大量是 false-positive（即 Phase 6 之后该 case 实际能 success），再 bump 到 v6.3。

### ADR-4: 三层 fallback 而非两层

**决策**：bilateral contact 用 Path 1 (robosuite) → Path 2 (local bilateral) → Path 3 (legacy merged)。

**理由**：
- Path 1 失败原因可能是 API not in this version；Path 2 失败原因可能是 gripper 命名不规范
- 两种失败模式独立，各自需要 fallback
- 直接降到 Path 3（旧行为）保证 backward compat

**取舍**：实现 ~50 行 vs ~20 行。但 robustness 提升显著。

### ADR-5: 新 method 名 `verify_grasp_by_micro_lift` 而非扩展 `lift`

**决策**：micro-lift 是独立 API，不污染现有 `lift()`。

**理由**：
- `lift()` 语义是"完整抬起"，verify 是"验证抓取"，混在一起破坏单一职责
- `verify_grasp_by_micro_lift` 可独立单元测试
- 未来 perception layer 接 `verify_grasp` 时有清晰的 stub 点

**取舍**：略多 ~10 行接口代码。

---

## 11. 已知未决事项

### 11.1 important_geoms 在 RoboCasa PandaMobile 上是否存在？

设计上假定存在，未实测。Phase 6.1 实施前需要在 GPU 跑一次：

```python
robot = env.robots[0]
gripper = robot.gripper.get("right", robot.gripper)
print(gripper.important_geoms.keys())
```

预期：`{'left_fingerpad', 'right_fingerpad', 'left_finger', 'right_finger'}`。

**Fallback 已就位**：Path 2 / 3 兜底，即使 important_geoms 缺失也 work。

### 11.2 micro-lift 2cm 是否过保守？

业界范围 1-5cm。我们选 2cm 因为：
- < 1cm: OSC 在小步进 stall
- > 3cm: 可能触发 close_gripper 后的 lid 摩擦力衰减

实测若 false-negative 高（grasp 实际 OK 但 micro-lift 误杀），可降到 1.5cm。

### 11.3 Phase 6.3 gripper qpos key 验证

PandaOmron/PandaMobile 的 obs key 名是 `robot0_gripper_qpos` 还是其他？需要实测：

```python
print(env._latest_obs.keys())  # 寻找 *gripper_qpos*
```

**Fallback 已就位**：obs 读不到时直接返 False（保守不报告 empty）。

---

## 12. 验证后下一步

Phase 6 完成 + GPU 验证通过后，**可选** Phase 7+：
- OSC stall 优化（move_arm_to max_steps / KP 调优）
- arm home pose reset 在 navigate 后（若 stall 仍多）
- 5-seed sweep 验证泛化性

均不在本设计范围。
