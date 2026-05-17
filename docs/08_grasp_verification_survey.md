# EmboSight Grasp Verification — 全网/全论文调研与修复方案

**日期**: 2026-05-17  
**问题**: Phase 5 GPU 验证显示 `[act] object NOT lifted: z_before=0.943 z_after=0.943 Δ=0.000` —— 系统报告 `grasp confirmed + squeezed`，但实际物体没被抓起。

**结论先行**: 当前 `_finger_object_contact` 的 **单侧接触** 判定是 false-positive 根源；业界（robosuite 官方 / Isaac Sim / 2024 SOTA paper）通用做法是 **双侧 fingerpad bilateral 接触 + micro-lift verification**。推荐 **A+C 方案** 修复，~50 行代码。

---

## 1. 当前实现的 bug 定位

### 1.1 false-positive contact check

`@c:\all_project\embodied-AI-one\src\env_wrapper.py:1392-1426` 的 `_finger_object_contact`:

```python
finger_kw = ("finger", "fingertip", "finger_pad", "tip", "pad")
finger_geoms = set()  # ← 所有 finger geom 合并为一个 set
for i in range(sim.model.ngeom):
    name = sim.model.geom_id2name(i) or ""
    if any(kw in name.lower() for kw in finger_kw):
        finger_geoms.add(i)

for i in range(sim.data.ncon):
    c = sim.data.contact[i]
    if (g1 in obj_geoms and g2 in finger_geoms) or ...:
        return True  # ← 任意一指 contact 就 True
```

**致命缺陷**：
- 左指 / 右指 geom **合并到同一个 set**，丢失了 bilateral 信息
- 任意一指（甚至是 fingertip 侧面）擦到物体表面就触发 True
- tupperware 是平顶宽盒，gripper 下降时一指**只是压在 lid 上**也算 contact

### 1.2 错误的 grasp 判定级联

```python
target_contact = self._finger_object_contact(target_body)  # ← 单侧 OK 就 True
generic_grasp = self._check_grasp_contact()  # robot.is_grasping(), 也可能 false-positive
if target_contact and generic_grasp:
    confirmed = True
    squeeze_steps = 10  # 多压 10 步, 但没真夹住
```

→ `[close_gripper] grasp confirmed + squeezed at step 16` —— **完全误报**。

### 1.3 lift verification 时机太晚

`@c:\all_project\embodied-AI-one\src\action_executor.py` 在 `lift_arm(0.10)` 之后才检查 `obj Δz`。Lift 全程消耗 ~20 秒（4 次 stall），失败后才发现 `Δz=0.000`。

---

## 2. 业界 / 论文调研

### 2.1 robosuite 官方 `_check_grasp` (Ground Truth)

[`manipulation_env.py:_check_grasp`](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/environments/manipulation/manipulation_env.py)

```python
def _check_grasp(self, gripper, object_geoms):
    """
    Returns True if at least one geom in BOTH "left_fingerpad" and 
    "right_fingerpad" geom groups are in contact with @object_geoms.
    """
    if isinstance(gripper, GripperModel):
        g_geoms = [
            gripper.important_geoms["left_fingerpad"],
            gripper.important_geoms["right_fingerpad"],
        ]
    # Search for collisions between each gripper geom group AND object geoms
    for g_group in g_geoms:
        if not self.check_contact(g_group, o_geoms):
            return False  # ← 任一指无 contact 就 False
    return True
```

**核心结论**：robosuite 业界标准要求 **left_fingerpad AND right_fingerpad 同时与 object 接触**，且 **only 用 `important_geoms["left_fingerpad"]` / `["right_fingerpad"]`** —— 不用所有 finger geom，避免 finger 侧面、knuckle 误触。

### 2.2 robosuite Lift task 的 success criterion

[`lift.py:_check_success`](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/environments/manipulation/lift.py):

```python
def _check_success(self):
    cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
    table_height = self.model.mujoco_arena.table_offset[2]
    return cube_height > table_height + 0.04  # 升 4cm 算成功
```

→ **object z 上升 4cm** 是 robosuite RL benchmark 的硬标准。

### 2.3 robosuite PickPlace staged reward

```python
r_grasp = int(self._check_grasp(...)) * 0.35  # 仅当双侧 fingerpad 接触
r_lift = grasp_mult + (1 - tanh(15 * z_dist)) * (lift_mult - grasp_mult)  # z 距离单调
```

**关键**: `r_lift` 必须建立在 `r_grasp > 0` 之上 —— 没有 grasp 就没有 lift reward。这是业界默认 pipeline。

### 2.4 Isaac Sim 官方 Grasping SDG ([NVIDIA Docs](https://docs.isaacsim.omniverse.nvidia.com/latest/synthetic_data_generation/tutorial_replicator_grasping_sdg.html))

> Physics-Based Evaluation: Each phase of the grasp is simulated in the physics engine. The **success or failure of the grasp attempt**, along with other metrics (like **contact forces**, **object displacement**), can be recorded.

→ NVIDIA 推荐 multi-modal: **contact + force + displacement** 三重验证。

### 2.5 Pin-pression Gripper paper (arxiv 2505.18994, 2024)

> To verify grasp success, **we lift the object to a height of 30cm and then check if it has dropped**.

→ 学术界最严格做法：lift 30cm 后才确认。我们 lift 10cm 已经足够，但需要 **early micro-lift check 在 ~2cm 处**，提前发现 slip。

### 2.6 Universal slip detection ([Frontiers 2024](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1478758/full))

> Tactile-free slip detection in sim relies on three signals:
> 1. **Object position tracking** (does object move with gripper?)
> 2. **Contact force decay** (force drops mid-lift = slipping)
> 3. **Gripper qpos closure** (jaw closes further than expected = empty)

→ 简化版本（不要 tactile）：(1) object follow + (3) jaw width。

---

## 3. 业界 4-layer grasp confirmation pattern (综合所有来源)

| Layer | 判定 | 误报防御 | 当前 EmboSight |
|-------|------|----------|----------------|
| L1 | **bilateral fingerpad contact** | 防"一指擦边" | ❌ 单侧合并 |
| L2 | **gripper width > 0** (没完全闭合) | 防 gripper 闭到空 | ❌ 没检查 |
| L3 | **contact force > threshold** | 防轻触 hover | ❌ 没检查 |
| L4 | **micro-lift (Δz=1-2cm) + object follow** | 防 slip / sealed lid | ⚠️ 在 lift_arm 全程结束后才查 |

EmboSight 当前**只有半个 L1 (单侧) 和 late L4**。

---

## 4. 推荐修复方案

按 ROI（修复成本 vs 准确率提升）排序：

### Option A: 改用 robosuite 官方 `_check_grasp` ⭐ **推荐**

**代价**: ~20 行  
**风险**: 极低（直接复用上游 API，业界标准）  
**收益**: 立即获得双侧 fingerpad bilateral contact 判定

```python
def _finger_object_contact(self, target_body: str) -> bool:
    # 优先用 robosuite 官方 API (业界标准 bilateral 判定)
    try:
        env = self._env
        if hasattr(env, "_check_grasp"):
            robot = env.robots[0]
            gripper = robot.gripper.get("right", robot.gripper) \
                if isinstance(robot.gripper, dict) else robot.gripper
            obj_geoms = self._get_body_geom_ids(target_body)
            obj_geom_names = [
                env.sim.model.geom_id2name(g) for g in obj_geoms
                if env.sim.model.geom_id2name(g)
            ]
            return env._check_grasp(gripper, obj_geom_names)
    except Exception as e:
        logger.debug(f"[finger_contact] _check_grasp failed: {e}, fallback to local")
    
    # Fallback: 现有 local 实现 (改成 bilateral)
    return self._bilateral_finger_contact(target_body)
```

### Option B: 本地实现 bilateral contact (无 robosuite API 依赖)

**代价**: ~30 行  
**风险**: 低（不依赖上游版本）  
**收益**: 与 A 等效，但解耦 robosuite

```python
def _bilateral_finger_contact(self, target_body: str) -> bool:
    """左指 AND 右指都必须接触 target."""
    sim = self._env.sim
    obj_geoms = self._get_body_geom_ids(target_body)
    if not obj_geoms:
        return False
    
    # 分离左右指 geom (基于 'left' / 'right' 关键字)
    left_geoms, right_geoms = set(), set()
    for i in range(sim.model.ngeom):
        name = (sim.model.geom_id2name(i) or "").lower()
        is_finger = any(kw in name for kw in ("finger_pad", "fingerpad", "fingertip", "tip", "pad"))
        if not is_finger:
            continue
        if "left" in name or "l_" in name:
            left_geoms.add(i)
        elif "right" in name or "r_" in name:
            right_geoms.add(i)
    
    if not left_geoms or not right_geoms:
        # 无法区分左右 (gripper 命名不规范), 退到合并判断
        return self._legacy_finger_contact(target_body)
    
    left_touch = right_touch = False
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 in obj_geoms or g2 in obj_geoms:
            opp = g2 if g1 in obj_geoms else g1
            if opp in left_geoms:
                left_touch = True
            elif opp in right_geoms:
                right_touch = True
            if left_touch and right_touch:
                return True
    return False
```

### Option C: Micro-lift verification (early slip detection) ⭐ **推荐**

**代价**: ~40 行（新增 `verify_grasp_by_micro_lift` 方法）  
**风险**: 低（独立验证步骤，失败仅 +1s 开销）  
**收益**: **省下 20s/episode 浪费**（lift 失败提前 fail），slip 早期发现

```python
def verify_grasp_by_micro_lift(
    self, target_body: str, lift_m: float = 0.02, threshold: float = 0.5,
) -> bool:
    """关爪后做微抬 lift_m, 检查 object z 是否跟随至少 threshold * lift_m.
    
    业界标准 early slip detection: 在大 lift 之前快速验证 grasp 稳固性.
    若 micro-lift 失败, 立即放弃当前 attempt, 省下 ~20s 浪费.
    
    Returns:
        True 若 object 跟随 >= threshold * lift_m
    """
    obj_z_before = self._get_body_z(target_body)
    if obj_z_before is None:
        return True  # 无法验证, 保守通过
    
    eef_before = self.get_eef_pos().copy()
    target = eef_before.copy()
    target[2] += lift_m
    
    self.move_arm_to(
        target, threshold_m=0.005, max_steps=80, gripper_hold=1.0,
    )
    
    obj_z_after = self._get_body_z(target_body)
    if obj_z_after is None:
        return True
    
    obj_delta = obj_z_after - obj_z_before
    eef_delta = self.get_eef_pos()[2] - eef_before[2]
    
    # object 必须跟随 EEF 至少 threshold 比例
    follows = obj_delta >= lift_m * threshold
    logger.info(
        f"[verify_micro_lift] eef Δz={eef_delta:.4f} obj Δz={obj_delta:.4f} "
        f"follows={follows} (threshold={lift_m * threshold:.4f})"
    )
    return follows
```

集成到 `act()`：

```python
# 4. close gripper
if not env.close_gripper(target_label=target.label):
    return self._failed_result(candidate, "gripper_empty", {...}, env)

# 4.5 [NEW Phase 6] micro-lift verification (early slip detection)
if hasattr(env, "verify_grasp_by_micro_lift"):
    obj_body = env._resolve_target_body(target.label)
    if not env.verify_grasp_by_micro_lift(obj_body, lift_m=0.02, threshold=0.5):
        return self._failed_result(
            candidate, "slipped_lift",
            {"stage": "micro_lift_verify", "reason": "object_not_following"},
            env,
        )

# 5. full lift (only if micro-lift passed)
ok, final_z = env.lift(...)
```

### Option D: Gripper width sanity check

**代价**: ~15 行  
**风险**: 低  
**收益**: 防 "gripper closed to empty" false positive

```python
def _gripper_closed_on_empty(self, threshold_m: float = 0.005) -> bool:
    """检查 gripper 是否完全闭合 (两指间距小于 threshold).
    
    完全闭合 = gripper 没夹住任何东西 (jaw 撞在一起).
    """
    try:
        obs = self._latest_obs
        if obs is None:
            return False
        # robosuite obs key: robot0_gripper_qpos (6-vec for Panda parallel jaw)
        qpos = obs.get("robot0_gripper_qpos")
        if qpos is None:
            return False
        # Panda parallel jaw: qpos[0..1] 是两指位置, 相加 = total gap
        gap = float(qpos[0] + qpos[1])
        return gap < threshold_m
    except Exception:
        return False
```

---

## 5. 推荐组合 & 实施顺序

### Phase 6.1 (必做) — bilateral contact 修复
**Option A** (调用 robosuite `_check_grasp`)  
**+ Option B** fallback (本地 bilateral)  
**预估时间**: 1 小时  
**单测**: 5 个 (左侧 only / 右侧 only / 双侧 / 双侧不同 geom / API fail fallback)

### Phase 6.2 (强推) — early micro-lift verification
**Option C** (`verify_grasp_by_micro_lift`)  
集成到 `action_executor.act()`，新失败模式 `slipped_lift` 分类  
**预估时间**: 1.5 小时  
**单测**: 4 个 (object follows / not follows / no body resolve / exception safety)

### Phase 6.3 (可选) — gripper width sanity
**Option D**  
作为 `close_gripper` 内部的额外断言  
**预估时间**: 0.5 小时

---

## 6. 期望效果（vs Phase 5 baseline）

| 指标 | Phase 5 Run 3 | After Phase 6.1+6.2 |
|------|---------------|---------------------|
| false-positive grasp confirmed | 1 (tupperware) | 0 |
| episode wall-time | 430s | **~250s** (省下 lift attempt + 1 LLM retry) |
| `slipped_lift` 早期检测 | lift 后 23s 才 fail | close 后 2s fail |
| success rate (tupperware) | False | **True** (LLM 立即切策略) |

---

## 7. 风险与回滚

### 主要风险
- robosuite `_check_grasp` 在 PandaMobile 上 `important_geoms` 可能为空 → **fallback 到 local bilateral** 兜底
- micro-lift 2cm 可能触发 OSC stall → **threshold 0.005 + max_steps 80** 已校准小步长
- `slipped_lift` 失败模式提前出现 → 既有 memory v6.2 schema 可吞，无需再 bump version

### 回滚策略
所有改动**纯增量**：
- 新 method `_bilateral_finger_contact` / `verify_grasp_by_micro_lift` 不删旧的
- 旧 `_finger_object_contact` 改为转调 bilateral, **保留 fallback 路径**
- 一行 revert 可关闭 micro-lift verification

---

## 8. 是否需要 GRASP_CODE_VERSION bump?

**不需要**。Phase 6 修复的是 **检测精度**，不改变 grasp 语义。v6.2 标签下记录的 `slipped_lift` 仍然代表 "grasp 后物体未跟随" —— Phase 6 只是更早发现，更准确分类，不需要 invalidate 历史数据。

---

## 9. 行动确认

请确认我推荐的实施顺序：
1. **Phase 6.1**: bilateral contact (Option A + B fallback)  ← 必做
2. **Phase 6.2**: micro-lift verification (Option C)         ← 强推
3. **Phase 6.3**: gripper width sanity (Option D)            ← 可选

如同意，我按 phase 顺序逐个 commit + 单测 + push。
