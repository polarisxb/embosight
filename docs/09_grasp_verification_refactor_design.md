# Grasp Verification Refactor — Phase 6 设计文档

**状态**: 设计中 · 待用户审阅  
**日期**: 2026-05-17  
**前置**: Phase 0-5 完成（navigate_base_to 已 GPU 验证 working, commit `5082fc6`）  
**关联**: 调研报告 `docs/08_grasp_verification_survey.md`

---

## TL;DR

修复 Phase 5 GPU Run 3 暴露的 false-positive grasp confirmation bug（tupperware lid 单侧 fingertip 擦到就触发 grasp confirmed，但物体没真夹住）。

**3 个阶段**：
- **Phase 6.1** 给 `_finger_object_contact` 加 `bilateral: bool` 参数（descend 沿用 lenient 单侧、close_gripper 用 strict 双侧），双侧实现采用 robosuite 官方 `_check_grasp`
- **Phase 6.2** 在每个 `close_gripper + lift` 序列之间注入 **micro-lift verification**（升 2cm 查 object follow，失败立即 slipped_lift）；统一抽出 `_close_and_lift_with_verify` helper 覆盖 4 个分支
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
- **G5**：单元测试覆盖所有新行为，现有 341 个测试 0 regression

### 2.2 非目标

- **N1**：不修 OSC stall（独立问题，Phase 7+ 处理）
- **N2**：不改 grasp_strategy 选择逻辑（GraspPlanner 不动）
- **N3**：不引入 tactile sensor / force feedback（sim-only，无硬件）
- **N4**：不再 bump GRASP_CODE_VERSION（v6.2 schema 保持，slipped_lift 标签语义不变）
- **N5**：不改 close_gripper 的 squeeze_steps 数（保留经验值）

---

## 3. 架构总览

### 3.1 当前 vs 目标 control flow

**Phase 5 当前 — `act()` 实际有 4 个独立的 close+lift 分支**（不是单线）：

```
act()
 ├─ navigate_base_to                              ← Phase 4 done
 ├─ move_to_pre_grasp
 ├─ descend_until_contact / approach_along_dir
 │   └─ _finger_object_contact (gripper OPEN, 单侧合理早停)
 ├─ ─┬─ branch A: 正常路径 (descend ok)
 │   ├─ branch B: stall+contact+repositioned ok
 │   ├─ branch C: stall+contact+repositioned fail (现位置 grasp)
 │   └─ branch D: stall+contact 不可移动 (强行 grasp)
 │     └─ 每分支独立 close_gripper + lift + Δz 验证
 │         └─ close_gripper 内 _finger_object_contact  ← BUG: 单侧合并
 └─ release_and_retreat (失败时)
```

**Phase 6 目标 — 保留 4 个分支但抽出统一 helper**：

```
act()
 ├─ navigate_base_to
 ├─ move_to_pre_grasp
 ├─ descend / approach (gripper OPEN)
 │   └─ _finger_object_contact(bilateral=False)  ← 保持 lenient 单侧早停
 ├─ branch A/B/C/D → 都调 _close_and_lift_with_verify(env, target, candidate, approach_dir)
 │   ├─ env.close_gripper (gripper CLOSING)
 │   │   └─ _finger_object_contact(bilateral=True)  ← strict 双侧
 │   │   └─ (Phase 6.3) jaw_closed_on_empty check
 │   ├─ ★ verify_grasp_by_micro_lift(0.02)  [Phase 6.2]
 │   │   └─ 失败 → return slipped_lift (skip full lift, save 20s)
 │   ├─ env.lift(0.10)
 │   └─ Δz_obj 兜底验证 → slipped_lift
 └─ release_and_retreat
```

### 3.2 四层防御 (defense-in-depth)

| Layer | 检测 | 触发时机 | gripper 状态 | 失败处理 |
|-------|------|----------|-------------|----------|
| L1a | **lenient** finger contact (单侧 OK) | descend / approach 早停 | OPEN (~8cm) | continue descending |
| L1b | **strict** bilateral fingerpad contact | `_close_gripper_until_grasp` confirm | CLOSING | 拒绝 confirm，继续 squeeze |
| L2 (Phase 6.3) | gripper width > 5mm | close_gripper confirm 前 | CLOSING | 拒绝 confirm（jaw 闭到空） |
| L3 (Phase 6.2) | micro-lift object follow (2cm) | close_gripper 之后立即 | CLOSED | 立即 return slipped_lift |
| L4 | full-lift Δz (10cm) | lift_arm 之后（兜底） | CLOSED | 现有 slipped_lift 分类 |

**关键设计**：L1a 和 L1b 用**同一个** method `_finger_object_contact`，通过 `bilateral` 参数区分语义。这保证 descend 阶段（gripper OPEN，两指间距 ~8cm，物理上不可能同时碰一个 5cm 物体）继续走 lenient 单侧早停；只有 close_gripper 阶段切到 strict bilateral。

---

## 4. Phase 6.1 — Bilateral Fingerpad Contact (with mode switch)

### 4.1 接口设计

`_finger_object_contact` 名字保持，新增 `bilateral` 参数 (default=False 保 backward compat)：

```python
def _finger_object_contact(
    self,
    target_body: str,
    bilateral: bool = False,
) -> bool:
    """
    检查夹爪是否与 target_body 接触.

    - bilateral=False (default, lenient): 任意一指 (left/right/tip/pad) 与 object
      接触即返回 True. 用于 descend/approach 阶段的早停 (gripper OPEN, 两指张开
      ~8cm, 物理上不可能两侧同时碰一个 5cm 物体, 必须用单侧).
    - bilateral=True (strict): left_fingerpad AND right_fingerpad 都必须与 object
      接触. 用于 close_gripper 阶段的 grasp 确认, 防止 "fingertip 擦边" false
      positive (业界标准, robosuite._check_grasp 默认行为).

    Args:
        target_body: sim body name (e.g. "obj_main")
        bilateral: 若 True 走 strict 三层 fallback (Path 1 robosuite API →
                   Path 2 local left/right split → 降级 lenient 兜底).

    Returns:
        True 若满足对应模式的接触条件.
    """
    if not bilateral:
        return self._lenient_finger_contact(target_body)
    # Strict bilateral: 三层 fallback
    result = self._strict_grasp_via_robosuite(target_body)
    if result is not None:
        return result
    result = self._strict_grasp_bilateral_local(target_body)
    if result is not None:
        return result
    # Final fallback: lenient (保 backward compat, 即使无法 bilateral 也不假报 False)
    return self._lenient_finger_contact(target_body)
```

**关键决策**：
- 默认 `bilateral=False` → 现有 5 个 callsite (descend/approach) **不需要改任何代码**，行为完全保持
- 仅 `_close_gripper_until_grasp` 一处明确传 `bilateral=True`
- bilateral 实现 fail-through 时降级回 lenient（而非 False），避免 close_gripper 因 API 异常永远 reject confirm

### 4.2 Path 1：调用 robosuite official API

```python
def _strict_grasp_via_robosuite(self, target_body: str) -> Optional[bool]:
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
def _strict_grasp_bilateral_local(self, target_body: str) -> Optional[bool]:
    """
    本地 bilateral 判定: 用 geom name 中 'left'/'right' 关键字分组.
    
    Returns None if 无法区分左右 (caller falls through to lenient fallback).
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

### 4.4 Lenient (default mode + fallback)

把现有 `_finger_object_contact` 实现整体抽出为私有 `_lenient_finger_contact(target_body)`，逻辑不变（任意一个 finger geom 与 object contact 即 True）。

- `bilateral=False` 直接调用它（descend/approach 早停）
- `bilateral=True` 在 Path 1/2 都失败时降级调用它（保 close_gripper 不被 API 异常永久 block）

### 4.5 _close_gripper_until_grasp 内部更新

唯一需要改动的 callsite：

```python
target_contact = self._finger_object_contact(
    target_body, bilateral=True,   # ← Phase 6.1 新增 strict 模式
)
```

其他 5 个 callsite（descend/approach）**完全不变**，沿用默认 lenient 行为。

### 4.6 测试计划

文件：`tests/test_env_wrapper_grasp_bilateral.py` (新)

| Test | 场景 | 期望 |
|------|------|------|
| `test_lenient_default_returns_true_on_any_finger` | bilateral=False, 单侧 contact | True（保 backward compat） |
| `test_strict_robosuite_path_uses_check_grasp` | bilateral=True, mock `env._check_grasp` 返 True | True（不走 Path 2） |
| `test_strict_local_left_only_returns_false` | bilateral=True, 仅左指 contact | False |
| `test_strict_local_right_only_returns_false` | bilateral=True, 仅右指 contact | False |
| `test_strict_local_both_returns_true` | bilateral=True, 双侧 contact | True |
| `test_strict_falls_back_to_lenient_when_left_right_unrecognizable` | bilateral=True, geom 命名无 left/right | 走 lenient，单侧 True 时仍 True |
| `test_strict_target_body_not_found_returns_false` | bilateral=True, target body 无 geom | False（不抛异常） |
| `test_close_gripper_uses_strict_mode` | 集成测试：close_gripper 内部应该传 bilateral=True | mock 计数 `_finger_object_contact(bilateral=True)` 被调用 ≥1 次 |

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

### 5.3 ActionExecutor 集成 — 统一 helper 覆盖 4 个分支

`act()` 实际有 **4 个独立的 close→lift 分支**（base reposition 成功 / 失败 / stall+contact / 正常路径），每个都有自己的 `env.close_gripper(target_label=...)` + `env.lift(approach_dir=...)` 对。Phase 6.2 必须在**每一对**之间插入 micro-lift verification。

直接复制 4 次代码不可维护，抽出统一 helper：

```python
def _close_and_lift_with_verify(
    self,
    env,
    target,
    candidate,
    approach_dir: np.ndarray,
    obj_z_before: float,
) -> tuple[bool, Optional["GraspActionResult"]]:
    """
    Close gripper, run micro-lift verification (Phase 6.2),
    then full lift. 统一 4 个分支的逻辑.
    
    Returns:
        (success, early_failed_result)
        - success=True, early_failed_result=None: continue 到 post-lift Δz 验证
        - success=False, early_failed_result=<...>: micro-lift fail, 立即返回
        - success=True, early_failed_result=None 且 lift 失败: 走老 slipped 兜底
    
    EEF 状态语义:
        - micro-lift 成功 → EEF 已升 2cm, 继续 lift 8cm 即可凑到 height=0.10
        - micro-lift 失败 → EEF 已升 2cm, 但 gripper 空, 调用方负责 release_and_retreat
          (它本身就是 open + lift 10cm, 不依赖 EEF 已有偏移, 故 2cm 偏移无害)
    """
    label = getattr(target, "label", None)
    grasp_ok = env.close_gripper(target_label=label)
    
    # Phase 6.2: early micro-lift verification (only if grasp_ok)
    if grasp_ok and hasattr(env, "verify_grasp_by_micro_lift"):
        try:
            obj_body = self._resolve_target_body(target, env)
            if obj_body and not env.verify_grasp_by_micro_lift(
                obj_body, lift_m=0.02, threshold=0.5
            ):
                return False, self._failed_result(
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
            logger.debug(f"[act] micro_lift error: {e}, continuing to full lift")
    
    # 现有 full lift 逻辑 (与原 4 分支相同)
    lift_ok, final_z = env.lift(approach_dir=approach_dir)
    if not lift_ok:
        if not grasp_ok:
            return False, self._failed_result(
                candidate, "gripper_empty",
                {"stage": "lift", "grasp_ok": False}, env,
            )
        return False, self._failed_result(
            candidate, "slipped_lift",
            {"stage": "lift", "final_z": final_z}, env,
        )
    return True, None
```

**4 个分支统一调用**：

```python
# branch A/B/C/D 各处, 把原本的两行
#   grasp_ok = env.close_gripper(target_label=...)
#   lift_ok, final_z = env.lift(approach_dir=...)
# 换成:
ok, failed = self._close_and_lift_with_verify(
    env, target, candidate, approach_dir, obj_z_before,
)
if failed is not None:
    return failed
# 后续 post-lift Δz 验证保持不变
```

辅助方法 `_resolve_target_body(target, env)`：

```python
def _resolve_target_body(self, target, env) -> Optional[str]:
    """获取 Hypothesis.label 对应的 sim body name.
    
    通过 env._get_obj_type_map() 反查 (body_name -> category) 的字典.
    """
    try:
        label = getattr(target, "label", None)
        if not label:
            return None
        if hasattr(env, "_get_obj_type_map"):
            type_map = env._get_obj_type_map()
            for body, cat in type_map.items():
                if cat == label:
                    return body
        return None
    except Exception:
        return None
```

**注意**：
- Hypothesis 用 `getattr(target, "label", None)` 安全访问，与现有代码 (action_executor.py:166) 行为一致
- helper 返 `(success, failed_result)` tuple 而非异常，避免改变 `act()` 控制流复杂度
- 4 个分支保持完全独立（base reposition 等逻辑不动），仅替换其中 `close_gripper + lift` 这两行

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
        # Path 1: obs (robosuite 标准 key, 最快)
        obs = self._latest_obs or {}
        for key in ("robot0_gripper_qpos", "gripper_qpos"):
            qpos = obs.get(key)
            if qpos is None:
                continue
            qpos = np.asarray(qpos, dtype=np.float32)
            # Panda parallel jaw: 前两维是两指位置, 相加 = 总 gap
            gap = float(np.abs(qpos[:2]).sum()) \
                if qpos.size >= 2 else float(qpos[0])
            logger.debug(f"[jaw_check] gap={gap:.4f}m (key={key})")
            return gap < threshold_m
        
        # Path 2: 直接读 sim.data.qpos at gripper joint addrs
        sim = getattr(self._env, "sim", None)
        if sim is None:
            return False
        # robosuite 1.5: robot.gripper_joints 是 list of joint names
        try:
            robot = self._env.robots[0]
            gripper = robot.gripper
            if isinstance(gripper, dict):
                gripper = gripper.get("right") or next(iter(gripper.values()))
            joint_names = getattr(gripper, "joints", None) or []
        except Exception:
            joint_names = []
        if not joint_names:
            return False
        total_gap = 0.0
        for jname in joint_names[:2]:  # 两指
            try:
                jid = sim.model.joint_name2id(jname)
                addr = int(sim.model.jnt_qposadr[jid])
                total_gap += float(abs(sim.data.qpos[addr]))
            except Exception:
                continue
        logger.debug(f"[jaw_check] gap={total_gap:.4f}m (sim.data fallback)")
        return total_gap < threshold_m
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

1. 把现有 `_finger_object_contact` 实现整体抽出为 `_lenient_finger_contact`
2. 给 `_finger_object_contact` 加 `bilateral: bool = False` 参数；strict 时走三层 fallback
3. 添加 `_strict_grasp_via_robosuite`, `_strict_grasp_bilateral_local` 私有方法
4. `_close_gripper_until_grasp` 内 callsite 改为 `bilateral=True`，其他 5 处不动
5. 新增单测文件 `tests/test_env_wrapper_grasp_bilateral.py` (8 tests)
6. 跑 `pytest tests/` 确认全过 (341 → 349)
7. Commit: `fix(grasp): bilateral fingerpad contact mode (Phase 6.1)`

### 7.2 Phase 6.2 commit (~1.5h)

1. 添加 `verify_grasp_by_micro_lift` 到 EnvWrapper
2. 添加 `ActionExecutor._close_and_lift_with_verify` helper
3. 把 act() 内 4 个 close+lift 分支都替换为调用 helper（branch A/B/C/D）
4. 添加 `_resolve_target_body` 辅助方法
5. 新增 `tests/test_action_executor_phase6.py` (6 tests)
6. 扩展 `test_env_wrapper_grasp_bilateral.py` (+4 micro-lift tests)
7. 跑 `pytest tests/` 确认全过 (349 → 359)
8. Commit: `feat(act): micro-lift early slip detection helper (Phase 6.2)`

### 7.3 Phase 6.3 commit (~0.5h)

1. 添加 `_gripper_closed_on_empty` 到 EnvWrapper
2. 在 `_close_gripper_until_grasp` 内集成 jaw width check（confirmed 前）
3. 扩展测试 (+4 tests)
4. 跑 `pytest tests/` 确认全过 (359 → 363)
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
在 `_close_and_lift_with_verify` helper 内删除 `if grasp_ok and hasattr(env, "verify_grasp_by_micro_lift"):` 整个 block。

**Level 2 — 回滚 bilateral**：  
`_close_gripper_until_grasp` 内的 `bilateral=True` 改回不传（默认 False），等于回到 lenient 行为。

**Level 3 — full revert**：  
`git revert <phase6.1>..<phase6.3>`，3 个 commit 干净回滚到 `5082fc6`。

### 8.3 不变量

- 任何分支必须 terminate（不能 inf loop）
- 任何 path 返 False 不能抛异常（异常被 try/except 吞掉后保守降级，而非传播）
- 现有 5 个 `_finger_object_contact` callsite **行为完全不变**（默认 bilateral=False，等价于 lenient）
- micro-lift 失败时 EEF 偏移 2cm 不影响后续 `release_and_retreat`
- 所有新 logger 调用用 `[grasp_check]` / `[micro_lift]` / `[jaw_check]` 前缀便于 grep

---

## 9. 验证标准（DoD）

### 9.1 单元测试

- [ ] 8 个 Phase 6.1 bilateral 测试通过
- [ ] 6 个 Phase 6.2 micro-lift 集成测试 + 4 个 EnvWrapper 测试通过
- [ ] 4 个 Phase 6.3 jaw width 测试通过
- [ ] 现有 341 个测试 0 regression（最终 363 测试全过）

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

### ADR-2: micro-lift 失败立即 fail (Option A，用户已选)

**决策**：Phase 6.2 检测到 obj 不跟随 → 立即 return `slipped_lift`，不在 EnvWrapper 层做 retry recovery。

**用户已选 Option A**（vs Option B 重新 close+squeeze / Option C 重新 descend / Option D 让 LLM 决策）。

**理由**：
- 职责清晰：EnvWrapper 负责检测，LLM/Memory 负责策略调整
- 双重 retry 会引入复杂的 stall stack
- LLM v6.2 已经学会从 `slipped_lift` 切到不同 grasp_strategy
- 单元测试简单（一个 control flow 分支）

**取舍**：有些可挽救的 case 也会被 fail。实测若误杀率高（>20%）再升级到 Option B（重新 close + squeeze）。

### ADR-3: 不 bump GRASP_CODE_VERSION

**决策**：保持 `v6.2`，不升 `v6.3`。

**理由**：
- `slipped_lift` 失败模式定义没改（仍是"夹后未跟随"）
- 只是检测**时机**提前（从 lift 后 → close 后）
- v6.2 历史数据中的 `slipped_lift` 仍然语义有效
- bump 会触发 memory 重学，浪费已积累的策略经验

**取舍**：若实测发现 v6.2 数据中 slipped_lift 大量是 false-positive（即 Phase 6 之后该 case 实际能 success），再 bump 到 v6.3。

### ADR-4: bilateral 三层 fallback（API → local → lenient）

**决策**：bilateral=True 时走 Path 1 (`_strict_grasp_via_robosuite`) → Path 2 (`_strict_grasp_bilateral_local`) → 降级 lenient（`_lenient_finger_contact`）。

**理由**：
- Path 1 失败原因可能是 API not in this version；Path 2 失败原因可能是 gripper 命名不规范
- 两种失败模式独立，各自需要 fallback
- **最后降级到 lenient 而非 strict False**：避免 close_gripper 因 API 异常永远 reject confirm（会导致 episode 永远不能 grasp 任何物体）
- lenient 兜底保证 Phase 6.1 在最坏情况下退化为 Phase 5 行为（不会 regression）

**取舍**：实现 ~50 行 vs ~20 行。但 robustness 提升显著。Lenient 降级意味着不规范命名的 gripper 上 6.1 等于无效（不会假阴），需要靠 6.2 micro-lift 兜底。

### ADR-5: 新 method 名 `verify_grasp_by_micro_lift` 而非扩展 `lift`

**决策**：micro-lift 是独立 API，不污染现有 `lift()`。

**理由**：
- `lift()` 语义是"完整抬起"，verify 是"验证抓取"，混在一起破坏单一职责
- `verify_grasp_by_micro_lift` 可独立单元测试
- 未来 perception layer 接 `verify_grasp` 时有清晰的 stub 点

**取舍**：略多 ~10 行接口代码。

### ADR-6: 不做 contact force 检测 (调研 L3)

**决策**：跳过 contact force threshold check，仅依赖 bilateral contact + micro-lift。

**理由**：
- MuJoCo contact force 在 robosuite obs 不直接暴露，需要遍历 `sim.data.contact[i].efc_pos` 或加 force sensor，实现复杂度高
- bilateral fingerpad 已经隐式排除"轻触 hover"（hover 时一指悬空，不可能两侧同时 contact）
- micro-lift 是端到端验证（物体跟随就 OK，不跟随就 fail），比 force threshold 更鲁棒
- YAGNI：先验证 6.1+6.2 是否解决问题，不够再加

**取舍**：极端 case（双侧轻触但握力不足）会漏过 6.1 但被 6.2 兜住；如果 6.2 也漏（lift 2cm 内仍粘在 lid），最后 L4 兜底（lift 10cm Δz 不到 0.05）。三层防御足够。

---

## 11. 已知未决事项

### 11.1 important_geoms 在 RoboCasa PandaMobile 上是否存在？

**策略**：不预先 probe，直接实施 + 在代码中加 INFO 级日志记录走了哪条 path。

Phase 6.1 实施时 `_strict_grasp_via_robosuite` 会输出：
- 若 `_check_grasp` API 可用且有 `important_geoms["left_fingerpad"]` → `[grasp_check] using robosuite API path`
- 若 API 缺失或 keys 不存在 → `[grasp_check] robosuite API unavailable, fallback to local`
- 若 local left/right 分组失败 → `[grasp_check] left/right unrecognizable, fallback to lenient`

GPU 验证脚本 (`scripts/phase5_validation_gpu.sh`) 跑完看 log 一行就知道哪条 path 触发。**Fallback 已就位**：Path 2 / lenient 兜底，即使 important_geoms 缺失也 work。

### 11.2 micro-lift 2cm 是否过保守？

业界范围 1-5cm（Pin-pression Gripper paper 用 30cm，但那是 final test；早期 verify 通常 1-3cm）。我们选 2cm 因为：
- < 1cm: OSC 在小步进 stall（Phase 5 显示 0.005-0.01m stall 非常频繁）
- > 3cm: lift 失败时 EEF 偏移过大，恢复成本高（虽然 release_and_retreat 能处理，但物体位置感知会失真）

实测若 false-negative 高（grasp 实际 OK 但 micro-lift 误杀），可降到 1.5cm；若 OSC stall 严重则升到 2.5cm。

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
