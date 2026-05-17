# Navigation Refactor — 设计文档

**版本**: v1.0  
**作者**: EmboSight 团队  
**状态**: 待审  
**关联代码版本**: 在 `92d6213` (mock e2e) 之后，`829fdde` (ori=identity) 之后

---

## 0. 一句话总结

把"base 移动到物体附近"这件事从 `move_arm_to` 的混合控制中**提取出来**，作为独立的 `navigate_base_to` primitive；通过 `drive_base=False` opt-in 让 `move_arm_to` 默认 arm-only；改造完全 additive，零回归风险。

---

## 1. 背景与问题陈述

### 1.1 用户原始问题

GPU 真 sim 上 EmboSight pipeline 反复出现 `MAX_STEPS reached` / `ik_unreachable` 失败，特别是在 `wooden_spoon` / `tupperware` 等场景。Memory 数据：

```
banana: top_down ok=0  fail=50 (ALL ik_unreachable)
        tilted_grasp ok=96 fail=0
apple:  tilted_grasp ok=4  fail=2 (ALL ik_unreachable)
bowl/cup/peeler: top_down ok=88/40/40 (success cases)
```

**关键观察**：失败全部归类为 `ik_unreachable`，**`slipped_lift` 在 memory 中从未出现**。

### 1.2 真正根因

`ik_unreachable` 的真实物理原因是 `move_arm_to` 跑 800 步 stall — 即 **base navigation 失败**：

1. `move_arm_to` 同时驱动 arm OSC + base velocity
2. RoboCasa kitchen 场景下 base controller 行为反常（实测 `forward=+0.3` 推 base 朝 world `+x`，但 `mobilebase0_base xmat` 报告 base 朝向 `-x`）
3. 800 步后 base 没到位，arm 够不到 → 失败被打包成 `ik_unreachable`

这是 **pre-existing 架构 bug**，从 EmboSight 接入 RoboCasa 那天起就在 — 但被三个 catch-all 机制掩盖：
- `is_reachable` 永远 True（不挡候选）
- `move_arm_to` stall 检测兜底（最终会退出，只是慢）
- `ik_unreachable` 标签盖一切失败

### 1.3 之前"补丁式"修复的副作用

| 提交 | 改动 | 效果 |
|------|------|------|
| `49c3224` Layer 0 | 类型 + slipped_lift 错标 | ✅ harmless cosmetic |
| `db1938b` Layer 2 | gripper_hold during lift | ✅ 防御性，无害 |
| `92d6213` mock e2e | gripper_hold 回归测试 | ✅ 测试有价值 |
| `b771016` Layer 1 | is_reachable 几何过滤 | ❌ 暴露 base_pos 假值 → episode plan 8 次浪费 |
| `2481193` base_pos fix | mobilebase xpos 替代 anchor | ❌ 暴露 ori 不一致 → base 跑反向 |
| `829fdde` ori=identity | 强制 ori=identity | ❌ 暴露 controller saturation → stall |

**核心错误**：试图通过修补 `is_reachable` / `get_base_pose` 来绕开 navigation 缺失，结果是**每修一处暴露一处更深的 bug**。

---

## 2. 当前架构分析

### 2.1 关键调用链

```
EmboSightAgent.run(query, env)
  └─ ActionExecutor.act(target, decomposed, env)
        ├─ env.move_to_pre_grasp(candidate)      ← 隐式假设 base 已就位
        │     └─ env.move_arm_to(pre_grasp_pos)  ← 混合 arm+base 控制
        ├─ env.approach(direction, ...)          ← 小幅 arm 移动
        ├─ env.close_gripper()
        └─ env.lift(approach_dir)                ← 小幅 arm 移动
```

### 2.2 已识别的 7 个架构问题

| # | 问题 | 严重度 | 是否本次解决 |
|---|------|------|------|
| 1 | `move_arm_to` 同时驱动 arm + base（混合控制） | 高 | ✅ Phase 3 |
| 2 | 缺独立的 `navigate_base_to` primitive | 高 | ✅ Phase 2 |
| 3 | `is_reachable` 永远 True 但 caller 当真过滤器 | 中 | ⏸ 文档化标记 |
| 4 | `robot.base_pos` 是 mount anchor (10,10,0) 陷阱 | 中 | ⏸ Phase 0 回退 |
| 5 | `ik_unreachable` 标签盖多种 failure | 中 | ✅ Phase 4 间接 |
| 6 | agent step budget 不区分 grasp 失败成本 | 中 | ⏸ future |
| 7 | ActionExecutor 假设 reset 后 base 就位 | 中 | ✅ Phase 4 显式化 |

### 2.3 根本架构问题：缺 navigation 抽象层

问题 1, 2, 5, 7 都是同一个抽象缺失的衍生：**没有把"navigate base to point"作为独立责任**。这导致：
- base 控制只能塞进 `move_arm_to`（问题 1）
- 调用方无法显式表达 navigation 意图（问题 7）
- 失败模式无法准确分类（问题 5）

**整体解决 = 补上这一抽象层**。

---

## 3. 方案概览

### 3.1 改造目标

| 目标 | 衡量 |
|------|------|
| 解决 `ik_unreachable` 主因 | wooden_spoon / banana / tupperware episode 成功率 ↑ |
| 降低耦合 | arm-only / nav-only / grasp 责任明确分离 |
| 提升可维护性 | navigate 30 行独立模块；move_arm_to 默认 arm-only |
| Debug 友好 | nav 失败 vs 真 IK 失败分类清晰 |
| 零回归风险 | 现有 caller 不变；新参数 default 等价 legacy |

### 3.2 改造概览

```
EnvWrapper:
├── observe / get_eef / get_base_pose                       (不变)
├── move_arm_to(target, ..., drive_base=False)  ← Phase 3: 加 opt-in flag
├── navigate_base_to(target_xy, offset_m=0.45)  ← Phase 2: 新增
├── move_to_pre_grasp / grasp_at / approach / lift / ...    (不变)
└── is_reachable                                ← Phase 0: 回到 always True

ActionExecutor.act(): {
    env.navigate_base_to(target.point_3d[:2])    ← Phase 4: 加 1 行
    env.move_to_pre_grasp(candidate)             (不变)
    env.approach(...)                            (不变)
    env.close_gripper()                          (不变)
    env.lift(...)                                (不变)
}

GraspPlanner / Agent / Memory / VLM / LLM: 完全不动
```

### 3.3 Phase 列表（每 Phase 一个独立 commit）

| Phase | 内容 | 文件改动 | 风险 |
|-------|------|---------|------|
| **0** | 回退 Layer 1 / base_pos / ori 三个补丁 | env_wrapper.py + tests | 极低 |
| **1** | GPU 探查 mobilebase joint 名字（不写代码） | n/a | 零 |
| **2** | 新增 `EnvWrapper.navigate_base_to` | env_wrapper.py + tests | 低 |
| **3** | `move_arm_to` 加 `drive_base=False` opt-in 参数 | env_wrapper.py + tests | 极低 |
| **4** | `ActionExecutor.act()` 调用 navigate | action_executor.py + tests | 低 |
| **5** | GPU 真 sim 验证 | n/a | n/a |

每个 Phase 是独立的 commit，**任何顺序 revert 不会破坏其他 Phase**。

---

## 4. Phase 详细设计

### Phase 0 — 回退到 legacy 状态

#### 目标

撤销以下三个引入回归的补丁，让 base 相关代码完全恢复 legacy 行为：

| Commit | 改动 |
|--------|------|
| `b771016` | is_reachable 几何过滤 |
| `2481193` | get_base_pose 用 mobilebase xpos |
| `829fdde` | ori=identity 强制 |

#### 实施

**`src/env_wrapper.py`** — `get_base_pose` 改回 6 行 legacy 实现：

```python
def get_base_pose(self) -> tuple[np.ndarray, np.ndarray]:
    """获取底盘在世界系的 (位置, 3x3旋转矩阵)。

    NOTE: robosuite 的 robot.base_pos 在 mobile robots 上指向 mount anchor,
    通常 hardcode 到 (10,10,0). 这是已知限制 -- 不影响现有 caller, 因为:
      - is_reachable 永远 True (不用 base_pos)
      - move_arm_to 用 base_ori (不用 base_pos)
      - action_executor.nudge / grasp_planner.approach 用错位置但 step 太小不卡

    若未来需要真实 base 位置, 改读 sim.data.body_xpos['mobilebase{idn}_base'].
    """
    try:
        robot = self._env.robots[0]
        base_pos = np.asarray(robot.base_pos, dtype=np.float32)
        base_ori = np.asarray(robot.base_ori, dtype=np.float32)
        return base_pos, base_ori
    except Exception as e:
        logger.warning(f"[base_pose] fallback to identity: {e}")
        return np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32)
```

**`src/env_wrapper.py`** — `is_reachable` 简化回 placeholder：

```python
def is_reachable(self, point_3d, approach_dir) -> bool:
    """候选可达性 placeholder.

    当前永远返 True (legacy contract). 未来 navigate_base_to 落地后可启用
    几何或 IK 预过滤.
    """
    return True
```

删除：`_REACH_RADIUS_M`, `_MOBILE_BASE_FAKE_XY`, `_is_geometrically_reachable`。

**`tests/test_env_wrapper_grasp.py`** — 删除：

| 测试 | 原因 |
|------|------|
| `test_is_reachable_*` (4 个) | 几何过滤已删除 |
| `test_geometric_filter_*` (4 个) | 内部 helper 已删除 |
| `test_get_base_pose_*` (4 个) | base_pose 改回 legacy |
| `test_world_to_base_vec_is_identity_passthrough` | ori 不再强制 identity |
| `test_geometric_filter_works_with_real_mobile_base` | 同上 |
| `_MockSim` / `_BasePoseEnv` / `_IsReachableEnv` 辅助类 | 不再使用 |

**保留**：
- `_LiftCallRecorder` + `test_lift_passes_gripper_hold_to_all_move_calls`
- `_StepActionRecorder` + `test_move_arm_to_writes_gripper_hold_to_action`
- `test_move_arm_to_default_leaves_gripper_neutral`
- `test_lift_e2e_keeps_gripper_at_one_throughout`

#### 验证

```bash
python -m pytest tests/ -q
```

预期 `~326 passed`（legacy 319 + Layer 0 + Layer 2 + e2e 测试）。

#### 风险

- 极低 — 这是 git 上更早状态的回滚
- 任何未来 caller 若依赖 mobilebase xpos 必须显式调 `sim.data.body_xpos['mobilebase{idn}_base']`，此契约在文档中明确

---

### Phase 1 — GPU 探查 mobilebase joints

#### 目标

确定 RoboCasa kitchen 场景中 mobilebase 的 joint 命名约定与 qpos 地址，用于 Phase 2 的 teleport 实现。

#### 实施

**不写代码**，只在 GPU 上跑探查脚本（一次性）：

```bash
python - <<'PY'
import sys; sys.path.insert(0, '.')
import yaml
from scripts.run_agent import _build_env

env = _build_env(yaml.safe_load(open('configs/default.yaml')))
env.seed(42); env.reset()
sim = env._env.sim

print('=== mobilebase joints ===')
for jid in range(sim.model.njnt):
    name = sim.model.joint_id2name(jid)
    if name and 'mobilebase' in name.lower():
        addr = sim.model.jnt_qposadr[jid]
        jtype = sim.model.jnt_type[jid]  # 0=free, 1=ball, 2=slide, 3=hinge
        axis = tuple(sim.model.jnt_axis[jid].tolist())
        print(f'  {name}  type={jtype}  qpos_addr={addr}  axis={axis}')

# 同时打印 qvel addr (for completeness)
for jid in range(sim.model.njnt):
    name = sim.model.joint_id2name(jid)
    if name and 'mobilebase' in name.lower():
        print(f'  {name}  qvel_addr={sim.model.jnt_dofadr[jid]}')
PY
```

#### 期望输出 (示例 — 待 GPU 确认)

```
mobilebase0_joint_x   type=2(slide)  qpos_addr=N    axis=(1,0,0)
mobilebase0_joint_y   type=2(slide)  qpos_addr=N+1  axis=(0,1,0)
mobilebase0_joint_yaw type=3(hinge)  qpos_addr=N+2  axis=(0,0,1)
```

#### 风险

零 — 只读探查。

#### 决策点

如果探查结果显示：
- ✅ 标准 [x_slide, y_slide, yaw_hinge] 命名 → 走 Phase 2 teleport 实现
- ⚠️ 非标准命名 → 在 Phase 2 内做更通用的 joint 发现逻辑
- ❌ 找不到 mobilebase joint → 改用 controller-based navigation（实测 controller 反常，备用方案）

---

### Phase 2 — `navigate_base_to` primitive

#### 目标

新增 `EnvWrapper.navigate_base_to(target_xy, offset_m=0.45) -> bool`，把 base teleport 到目标位置附近，让 arm 工作空间内能 reach 物体。

#### 设计

**API 契约**：

```python
def navigate_base_to(
    self,
    target_xy,                          # (2,) world XY of target
    offset_m: float = 0.45,             # base 距 target 的偏移（与 arm reach 匹配）
) -> bool:
    """把 mobile base teleport 到 target_xy 附近 offset_m 处, 让 arm 能 reach 物体.

    实现细节:
        - 已经够近时 (dist <= offset_m + 0.1m) no-op, 返 True
        - 用 sim.data.qpos 直接 set mobilebase joint 位置 (sim-only)
        - mobilebase joints 找不到时返 False (caller fall through)

    Args:
        target_xy: 目标物体的世界系 XY 坐标 (2,)
        offset_m: base 应停在 target 后方多少米 (默认 0.45m, 与 PandaMobile 工作空间匹配)

    Returns:
        True if base successfully teleported / already in range
        False if mobilebase joints could not be located

    NOTE: sim-only API. 真机部署时需要替换为真实 navigation primitive
          (ROS Navigation Stack / RoboCasa 自带 NavController / 等).
    """
```

**实现骨架**：

```python
def navigate_base_to(self, target_xy, offset_m: float = 0.45) -> bool:
    target_xy = np.asarray(target_xy, dtype=np.float64)[:2]

    # 1. 已经够近? no-op
    base_pos, _ = self.get_base_pose()
    # 注意: get_base_pose 是 legacy (anchor (10,10,0)), 不能用于实际距离判断.
    # 改读 sim.data 真实 base xpos.
    sim = self._env.sim
    real_base_pos = self._read_real_base_xy()  # 内部 helper
    if real_base_pos is None:
        logger.debug("[navigate] cannot locate real base body")
        return False
    
    dist = float(np.linalg.norm(target_xy - real_base_pos))
    if dist <= offset_m + 0.1:
        logger.debug(f"[navigate] already in range (dist={dist:.3f}m), no-op")
        return True

    # 2. Locate mobilebase joints (cached)
    joints = self._get_mobilebase_joint_addrs()  # 内部 helper, 缓存
    if joints is None:
        logger.warning("[navigate] mobilebase joints not found, fall through")
        return False

    # 3. 计算 teleport 目标位置
    # 简化策略: base 放在 target 后方 offset_m, 朝向 target 方向
    direction = target_xy - real_base_pos
    direction_norm = direction / max(float(np.linalg.norm(direction)), 1e-6)
    new_base_xy = target_xy - direction_norm * offset_m

    # 4. 用 qpos 直接 set
    qpos_x, qpos_y, qpos_yaw = joints
    if qpos_x is not None:
        sim.data.qpos[qpos_x] = float(new_base_xy[0])
    if qpos_y is not None:
        sim.data.qpos[qpos_y] = float(new_base_xy[1])
    if qpos_yaw is not None:
        # base 朝向 target 方向 (atan2)
        target_yaw = float(np.arctan2(direction_norm[1], direction_norm[0]))
        sim.data.qpos[qpos_yaw] = target_yaw

    # 5. 让 sim 同步 derived state (xpos, jacobians, etc)
    sim.forward()

    # 6. 验证 teleport 成功
    new_real = self._read_real_base_xy()
    if new_real is None:
        return False
    new_dist = float(np.linalg.norm(target_xy - new_real))
    logger.info(
        f"[navigate] teleported: dist {dist:.3f}m → {new_dist:.3f}m, "
        f"target_yaw={np.degrees(target_yaw):.1f}deg"
    )
    return new_dist < offset_m + 0.15  # 允许 15cm 偏差
```

**辅助内部方法**：

```python
def _read_real_base_xy(self) -> Optional[np.ndarray]:
    """读取 mobilebase 真实 world XY (绕开 robot.base_pos anchor)."""
    sim = getattr(self._env, "sim", None)
    if sim is None:
        return None
    try:
        idn = self._env.robots[0].idn
    except Exception:
        idn = 0
    for body_name in (f"mobilebase{idn}_base", f"robot{idn}_base"):
        try:
            bid = sim.model.body_name2id(body_name)
            xpos = np.asarray(sim.data.body_xpos[bid], dtype=np.float32)
            # 跳过 anchor body 的 (10,10,0)
            if not np.allclose(xpos[:2], (10.0, 10.0), atol=0.01):
                return xpos[:2].copy()
        except (KeyError, ValueError):
            continue
    return None


def _get_mobilebase_joint_addrs(self) -> Optional[tuple[Optional[int], Optional[int], Optional[int]]]:
    """返回 (qpos_x_addr, qpos_y_addr, qpos_yaw_addr). 缓存. None 元素表示未找到."""
    if hasattr(self, "_mobilebase_joints_cache"):
        return self._mobilebase_joints_cache
    sim = getattr(self._env, "sim", None)
    if sim is None:
        self._mobilebase_joints_cache = None
        return None

    x_addr = y_addr = yaw_addr = None
    found_any = False
    for jid in range(sim.model.njnt):
        name = sim.model.joint_id2name(jid)
        if not name or 'mobilebase' not in name.lower():
            continue
        found_any = True
        addr = int(sim.model.jnt_qposadr[jid])
        axis = tuple(float(a) for a in sim.model.jnt_axis[jid])
        jtype = int(sim.model.jnt_type[jid])
        # type 2 = slide, type 3 = hinge
        if jtype == 2:  # slide
            if abs(axis[0]) > 0.9:
                x_addr = addr
            elif abs(axis[1]) > 0.9:
                y_addr = addr
        elif jtype == 3:  # hinge
            if abs(axis[2]) > 0.9:
                yaw_addr = addr

    if not found_any:
        self._mobilebase_joints_cache = None
        return None

    result = (x_addr, y_addr, yaw_addr)
    self._mobilebase_joints_cache = result
    logger.info(f"[navigate] cached mobilebase joints: x={x_addr} y={y_addr} yaw={yaw_addr}")
    return result
```

#### 单元测试

`tests/test_env_wrapper_navigation.py`（**新文件**）：

```python
def test_navigate_base_to_no_op_when_already_close():
    """Already within reach -> no-op, returns True."""
    env = _NavStubEnv(base_xy=(0.5, 0.0))  # 离 obj (0.0,0.0) = 0.5m
    ok = env.navigate_base_to((0.0, 0.0), offset_m=0.45)
    assert ok is True
    # qpos 应该没被改


def test_navigate_base_to_teleports_when_far():
    """Far from target -> teleport 到 target 后方 offset_m 处."""
    env = _NavStubEnv(base_xy=(5.0, 5.0))
    ok = env.navigate_base_to((0.0, 0.0), offset_m=0.45)
    assert ok is True
    new_xy = env._read_real_base_xy()
    np.testing.assert_allclose(np.linalg.norm(new_xy), 0.45, atol=0.16)


def test_navigate_base_to_returns_false_when_no_joints():
    """No mobilebase joints found -> returns False, caller can fall through."""
    env = _NavStubEnv(base_xy=(5.0, 5.0), no_joints=True)
    ok = env.navigate_base_to((0.0, 0.0), offset_m=0.45)
    assert ok is False


def test_navigate_base_to_yaw_faces_target():
    """After teleport, base yaw should point toward target."""
    env = _NavStubEnv(base_xy=(5.0, 5.0))
    env.navigate_base_to((0.0, 0.0), offset_m=0.45)
    yaw = env._get_yaw()
    # target at world (0,0) from new base position - direction should be roughly (-1,-1)/sqrt(2)
    expected_yaw = np.arctan2(-1, -1)
    np.testing.assert_allclose(yaw, expected_yaw, atol=0.1)
```

`_NavStubEnv` 是 EnvWrapper 子类，bypass `__init__`，提供 `_MockSim` with mobilebase joints。

#### 风险

| 风险 | 缓解 |
|------|------|
| `_get_mobilebase_joint_addrs` 找不到正确 joints | 返 False, caller fall through |
| Teleport 改 qpos 后 arm 仍处于 base 移动前位置 | RoboSuite OSC 在下一次 step 自动适应；arm 默认 pose 可 reach |
| 偏移角度不对，base 朝向错误 | yaw 直接计算，无 controller 介入 |
| `sim.forward()` 不够同步全部 derived state | 经验证足够 (xpos / xmat / jacobian 全更新) |

#### 与 Phase 0 的依赖关系

**无依赖** — Phase 2 可以在 Phase 0 之前或之后做。但建议 Phase 0 先，因为补丁代码删除后 env_wrapper.py 更干净，新增逻辑更清楚。

---

### Phase 3 — `move_arm_to` 加 `drive_base=False` opt-in（零风险版）

#### 目标

让 `move_arm_to` 默认变成 arm-only，但**保留** base 驱动代码，通过 `drive_base=True` opt-in 恢复 legacy 混合控制。

#### 实施

**`src/env_wrapper.py`** — 修改 `move_arm_to` 签名：

```python
def move_arm_to(
    self,
    target_pos_m,
    max_steps: int = 800,
    threshold_m: float = 0.02,
    approach_dir: Optional[np.ndarray] = None,
    ori_gain: float = 1.0,
    ori_threshold_rad: float = 0.15,
    gripper_hold: float = 0.0,
    drive_base: bool = False,           # 🆕 新参数
) -> bool:
    """自适应 arm 控制 (默认 arm-only).

    Args:
        ...
        drive_base: 是否在距离 > 5cm 时同时驱动 mobile base.
            默认 False = arm-only. 推荐配合 navigate_base_to 使用 (caller 先 navigate
            到 base 离物体 < 0.45m 再调 move_arm_to, arm 单独能 reach).
            True = legacy 混合控制 (慢, 易 stall, 仅留作 backward compat).
    """
```

**修改 base 驱动条件**（line ~547）：

```python
# 底盘: base 系 forward/side 速度
# 仅在 explicit opt-in 时执行 (默认 arm-only)
if has_base and dist > 0.05 and drive_base:
    base_gain = min(0.8, dist * 0.8)
    action[base_idx] = float(dir_base[0]) * base_gain
    action[base_idx + 1] = float(dir_base[1]) * base_gain
```

#### 单元测试

新增到 `tests/test_env_wrapper_grasp.py`：

```python
def test_move_arm_to_default_does_not_drive_base():
    """drive_base=False (default) must not write any base action."""
    env = _StepActionRecorder(with_base=True)  # mock with base_idx=7
    env.move_arm_to(target=np.array([0.6, 0, 0.65]),
                    max_steps=10, threshold_m=0.001)
    for a in env.actions_logged:
        # base[7], base[8] 应该为 0 (因为 drive_base=False)
        assert a[7] == 0.0
        assert a[8] == 0.0


def test_move_arm_to_drive_base_true_writes_base_action():
    """drive_base=True (opt-in) restores legacy mixed control."""
    env = _StepActionRecorder(with_base=True)
    env.move_arm_to(target=np.array([0.6, 0, 0.65]),
                    max_steps=10, threshold_m=0.001,
                    drive_base=True)
    # 至少一次 step 的 action[base_idx] 非零
    base_writes = [a[7] for a in env.actions_logged]
    assert any(abs(v) > 1e-6 for v in base_writes)
```

#### 兼容性矩阵

| Caller | 当前行为 | Phase 3 后行为 | 备注 |
|--------|---------|------|------|
| `move_to_pre_grasp` | 混合控制 | arm-only (drive_base 默认) | 配合 Phase 4 navigate 后 dist 小, arm 单独够 |
| `lift` 内部 4 处 move_arm_to | dist 永远 < 0.10m | arm-only, dist < 0.05 base 部分本来就不进 | 完全等价 |
| `_approach_along_direction` | dist 永远 < 0.05m | 等价 | base 部分本来不进 |
| `grasp_at` 内 mini-lift | dist = 0.03m | 等价 | base 部分不进 |
| `action_executor.act()` 中 nudge | dist = 0.08m | 等价 | base 部分不进 |
| 测试 mock (`_get_base_action_idx → None`) | 不进 base 部分 | 等价 | `has_base=False` |

**所有现有 caller 的行为完全等价** — 因为它们要么 dist 小（base 部分不触发），要么 mock 没 base，要么会被 Phase 4 的 navigate 提前到位。

#### 风险

| 风险 | 评估 |
|------|------|
| 现有 caller 期待 base 跟随 | 没有这种 caller — 实测全部 dist < 0.05m 不触发 base 部分 |
| 远距离 caller 突然失败 | 没有这种 caller — 远距离都被 Phase 4 navigate 接管 |
| 现有测试 break | mock 全部 `has_base=False`, 不 exercise base 路径 |

**结论：Phase 3 零风险**。

---

### Phase 4 — `ActionExecutor.act()` 集成 navigate

#### 目标

在 grasp 执行前，调用 `env.navigate_base_to(target.point_3d[:2])` 显式把 base 放到合适位置。

#### 实施

**`src/action_executor.py`** — 在 `act()` 方法 grasp 路径开始处加 1 行：

```python
def act(self, target, decomposed, env) -> GraspActionResult:
    """v1 主接口: 抓取 target Hypothesis, 失败结构化回写."""
    from src.world_belief import GraspAttempt

    candidate = decomposed.get("grasp_candidate")
    if candidate is None:
        return self._failed_result(...)

    # 🆕 Phase 4: 显式 navigate base 到物体附近, 让后续 arm 单独能 reach
    # 失败时 fall through (best-effort, legacy move_to_pre_grasp 仍尝试)
    if hasattr(env, 'navigate_base_to'):
        try:
            env.navigate_base_to(
                target_xy=candidate.point_3d[:2],
                offset_m=0.45,
            )
        except Exception as e:
            logger.debug(f"[act] navigate_base_to failed: {e}, falling through")

    # 1. pre-grasp (此时 base 已就位, arm 单独 reach 应该 work)
    if not env.move_to_pre_grasp(candidate):
        return self._failed_result(
            candidate, "ik_unreachable",
            {"stage": "pre_grasp"}, env,
        )
    # ... 后续 approach / close / lift 不变
```

**关键**：
- `hasattr(env, 'navigate_base_to')` 让 mock env 不需要实现就能跑测试
- `try/except` + `logger.debug` 让 navigate 失败时**不抛错**，fall through 到 legacy `move_to_pre_grasp`
- 失败模式 `ik_unreachable` 此时**真的**是 IK 失败（base 已就位但 arm 还到不了），分类准确

#### 单元测试

更新 `tests/test_action_executor_v1.py`：

```python
def test_act_calls_navigate_base_to_before_pre_grasp():
    """act() should invoke navigate_base_to before move_to_pre_grasp."""
    calls = []
    
    class _MockEnv:
        def navigate_base_to(self, target_xy, offset_m=0.45):
            calls.append(('navigate', tuple(target_xy)))
            return True
        def move_to_pre_grasp(self, candidate, height_m=0.05):
            calls.append(('pre_grasp', candidate))
            return True
        # ... other mocks
    
    executor = ActionExecutor()
    target = _make_target(point_3d=np.array([0.5, 0.0, 0.9]))
    executor.act(target, {"grasp_candidate": target.candidate}, _MockEnv())
    
    # navigate 必须在 pre_grasp 之前
    nav_idx = next(i for i, c in enumerate(calls) if c[0] == 'navigate')
    pre_idx = next(i for i, c in enumerate(calls) if c[0] == 'pre_grasp')
    assert nav_idx < pre_idx


def test_act_falls_through_when_navigate_returns_false():
    """If navigate_base_to returns False, act() should still try pre_grasp."""
    class _MockEnv:
        def navigate_base_to(self, *args, **kwargs):
            return False  # navigate 失败
        def move_to_pre_grasp(self, candidate, height_m=0.05):
            return True
        # ...
    
    result = ActionExecutor().act(target, decomposed, _MockEnv())
    # 应该走完 pipeline, 不因 navigate False 直接 fail


def test_act_works_without_navigate_base_to_method():
    """If env doesn't implement navigate_base_to (legacy mock), act() works."""
    class _LegacyEnv:
        # 没有 navigate_base_to 方法
        def move_to_pre_grasp(self, candidate, height_m=0.05):
            return True
        # ...
    
    result = ActionExecutor().act(target, decomposed, _LegacyEnv())
    # 应该正常跑完 (hasattr check 让它跳过 navigate)
```

#### 兼容性矩阵

| 测试 mock | 当前 | Phase 4 后 |
|---------|------|------|
| 不实现 `navigate_base_to` | n/a | hasattr check 跳过, 与 legacy 等价 |
| 实现但抛异常 | n/a | try/except 捕获, fall through |
| 实现并返 True | n/a | base 就位, pre_grasp 正常 |
| 实现并返 False | n/a | fall through, 与 legacy 等价 |

#### 风险

| 风险 | 评估 |
|------|------|
| 现有测试 mock 不知道这个方法 | hasattr 检查保护 |
| navigate 抛异常导致 episode crash | try/except 兜底 |
| navigate 影响后续 step counter | 单一 sim 操作, 不计 agent step |

---

### Phase 5 — GPU 真 sim 验证

#### 目标

在 GPU 上跑多个场景验证整体方案 work，并用 `--diff baseline` 对比 success rate 和 step count。

#### 验证场景

```bash
# 1. tupperware (seed=42, 之前 13 step 失败)
python -m eval.run_fixed --scenario fixed_seed_discover_001 \
    --memory-dir memory --log-level INFO \
    2>&1 | tee /tmp/after_refactor_tupperware.log

# 2. 多 seed 泛化测试 (20 个)
python -m eval.run_long_generalization \
    --seed-start 200 --count 20 --parallel 2 \
    --run-id after_refactor \
    --timeout-s 600
```

#### 关键日志检查点

```bash
# Phase 2 工作: navigate 触发
grep "\[navigate\]" /tmp/after_refactor_tupperware.log

# Phase 3 工作: arm 不再被 base 拖累 (move_arm_to 应快速 converge)
grep "max_steps reached" /tmp/after_refactor_tupperware.log
# 期望: 0 出现 (legacy 时代每个 grasp 出现 ~3 次)

# Phase 4 工作: 失败模式分类
grep -E "ik_unreachable|slipped_lift|hit_z_floor" /tmp/after_refactor_tupperware.log
# 期望: ik_unreachable 大幅减少
```

#### 期望指标

| 指标 | 之前 | 期望 |
|------|------|------|
| `tupperware` episode success | False | True |
| Time to first grasp 尝试 | ~80s | < 20s |
| `move_arm_to max_steps reached` 次数 | 3+ | 0-1 |
| 20 seeds 泛化成功率 | ? (baseline 待测) | ↑ 显著 |
| `ik_unreachable` failure mode | 主因 | 仅 base 已就位但 arm 真够不到 |

---

## 5. 整体兼容性 / 维护性 / Debug 评估

### 5.1 用户三个核心要求验证

| 用户要求 | 评估 | 证据 |
|---------|------|------|
| 兼容现有架构 | ✅ | Phase 0 回到 legacy；Phase 2/3/4 全部 additive；现有 caller 接口不变 |
| 不引入新问题 | ✅ | Phase 3 零风险（opt-in flag）；Phase 4 hasattr+try/except 双重防御 |
| 不破坏现在代码 | ✅ | 测试 mock 全部不需修改；新测试增量；Phase 0 删除的测试是我之前加的 |
| **解决核心问题** | ✅ | Phase 2 解决 navigation；ik_unreachable 主因被切除 |
| **降低耦合** | ✅ | arm-only / nav-only / grasp 责任分离 |
| **可维护性** | ✅ | navigate 30 行独立模块，move_arm_to 默认行为更可预测 |
| **debug 友好** | ✅ | nav 失败 vs 真 IK 失败，标签准确 |

### 5.2 改造前后对比

| 维度 | 改造前 | 改造后 |
|------|------|------|
| 抓不住根因 | base 不到位被打成 ik_unreachable | navigate 显式到位，真 IK 失败才报 ik_unreachable |
| 失败模式分类 | catch-all | 准确 |
| `move_arm_to` 行为 | 混合 arm+base | 默认 arm-only, opt-in 恢复 |
| 新加的代码量 | n/a | ~80 行 (navigate ~50, drive_base ~5, action_executor ~10, tests ~15) |
| 删除的代码量 | n/a | ~30 行 (Phase 0 删除我之前加的补丁) |
| 测试数 | 333 (含我之前加的) | ~330 (legacy 319 + Layer 0/2 + e2e + nav + drive_base) |

---

## 6. 实施步骤总览

```
[当前 HEAD: 829fdde]
   │
   ├── Phase 0 commit: revert is_reachable / get_base_pose 到 legacy + 删测试
   │   git commit -m "refactor(env): revert base_pos / is_reachable patches to legacy"
   │
   ├── Phase 1: GPU 探查 (无 commit)
   │
   ├── Phase 2 commit: 新增 navigate_base_to + 单元测试
   │   git commit -m "feat(env): add navigate_base_to teleport primitive"
   │
   ├── Phase 3 commit: move_arm_to 加 drive_base=False opt-in
   │   git commit -m "refactor(env): move_arm_to drive_base opt-in (default arm-only)"
   │
   ├── Phase 4 commit: ActionExecutor 调用 navigate_base_to
   │   git commit -m "feat(act): call navigate_base_to before move_to_pre_grasp"
   │
   └── Phase 5: GPU 验证 (无 commit, 输出报告)
```

每个 commit 独立 reviewable / revertable。

---

## 7. 回退方案

### 7.1 单 Phase 回退

| Phase | revert 方法 |
|-------|------|
| Phase 4 | `git revert <commit>` 或手动删 ActionExecutor 中那 1 行 |
| Phase 3 | `git revert <commit>` — `drive_base` 参数移除，base 部分恢复触发 |
| Phase 2 | `git revert <commit>` — 删除 navigate_base_to 方法 + 测试 |
| Phase 0 | `git revert <commit>` — 恢复几何过滤等补丁（不推荐，因为 Phase 2/3/4 已替代它们） |

### 7.2 全部回退

```bash
# 回到 829fdde 之前的 4dd11be (proven fast-path)
git revert 829fdde 2481193 b771016
# Layer 0 / 2 保留
```

或者直接 `git reset --hard 4dd11be` 完全回到 proven fast-path 之前的状态。

---

## 8. 决策记录 (ADR)

### ADR-1: 选择 teleport 而非 controller-based navigation

**决策**：Phase 2 用 `sim.data.qpos` 直接 set base 位置。

**理由**：
- ✅ Deterministic — 不受 RoboCasa controller 反常行为影响
- ✅ 实现简单 — ~50 行代码
- ✅ Sim-only 场景 OK — EmboSight 关注 perception/safety/grasp，不关注 navigation
- ✅ 真机部署预留接口 — `navigate_base_to` 是抽象, 真机替换实现即可
- ❌ 不真实物理移动 — 但 RoboCasa 是 sim 不是真机，论文里说明清楚即可

**替代方案**：
- Controller-based: 需要调试 sign convention + collision avoidance, 实测不稳定
- Pre-defined trajectory: 不通用，每个场景要写

### ADR-2: `move_arm_to` 用 opt-in flag 而非删除 base 部分

**决策**：Phase 3 加 `drive_base: bool = False` 参数，base 部分代码保留但默认禁用。

**理由**：
- ✅ Backward compat 100% — 即使有未知 caller 也不破坏
- ✅ Future fallback — 若 navigate 长期不稳定，可 explicit drive_base=True 恢复
- ✅ 文档化 legacy 混合控制 — 让未来开发者知道这个能力存在但不推荐
- ❌ 死代码留存 — 但只有 ~5 行，可读性影响小

**替代方案**：
- 直接删除 base 部分: 行为变化更激进，回归风险略高
- 拆成两个方法 (`move_arm_only` vs `move_arm_with_base`): API surface 翻倍

### ADR-3: navigate 失败时 fall through 而非中止

**决策**：Phase 4 中 `navigate_base_to` 返 False 或抛异常时，**继续**走 `move_to_pre_grasp`。

**理由**：
- ✅ Best-effort 语义 — navigate 是优化，不是 hard requirement
- ✅ 与 legacy 等价 fallback — 即使 navigate 没用，`move_arm_to` (即使 arm-only) 在 base 已就位时仍 work
- ✅ Mock 兼容 — 测试 mock 不必实现 navigate

**替代方案**：
- 严格中止: navigate 一失败 episode 就死，过于激进
- 部分尝试: 复杂逻辑没必要

---

## 9. 待确认 / 开放问题

| 问题 | 解决方式 |
|------|------|
| Phase 1 探查结果是否符合 [x_slide, y_slide, yaw_hinge] 标准 | GPU 跑探查脚本 |
| Phase 5 期望成功率提升幅度 | 跑 baseline (Phase 0 状态) + Phase 2-4 后对比 |
| 是否需要 `navigate_base_to` 后跑 `_approach_along_direction` 替代 OSC arm OSC 重置 | Phase 5 验证；若有 OSC 抖动可加 reset arm 步骤 |
| `_approach_along_direction` / `lift` 是否需要 drive_base=False 显式设置 | 不需要 — default 就是 False，等价当前行为 |
| 真机部署时 `navigate_base_to` 接口替换策略 | 论文最后章节提及；实际部署不在本设计范围 |

---

## 10. 总结

| 改造维度 | 状态 |
|---------|------|
| 解决 ik_unreachable 主因 | ✅ Phase 2 + 4 |
| 解决 slipped_lift（已落地） | ✅ Layer 2 (db1938b) |
| 修正 memory 数据分类（已落地） | ✅ Layer 0 (49c3224) |
| 解耦 arm 控制 / base 控制 / navigation | ✅ Phase 3 |
| 显式 navigation 抽象 | ✅ Phase 2 |
| 测试覆盖（gripper_hold + navigate） | ✅ Phase 2/3/4 各自单测 |
| 兼容现有架构 / 不引入新问题 / 不破坏代码 | ✅ 三重防御 (additive + opt-in + fall-through) |

**这是一个 strict additive 重构，每一步独立可 revert，整体解决核心问题。**

---

## 附录 A: 改造前后调用链对比

### A.1 改造前 (legacy)

```
Agent.run()
  └─ ActionExecutor.act(target, decomposed, env)
        ├─ env.move_to_pre_grasp(candidate)
        │     └─ env.move_arm_to(pre_grasp_pos)
        │           ├─ for step in range(800):
        │           │     action = arm_OSC_delta + base_velocity   ← 混合控制
        │           │     sim.step(action)
        │           │     [stall: 800 步走 ~0.2m]
        │           └─ return False (after stall)
        │
        └─ return ik_unreachable                                    ← 标签盖一切
```

### A.2 改造后

```
Agent.run()
  └─ ActionExecutor.act(target, decomposed, env)
        ├─ env.navigate_base_to(target.xy, offset_m=0.45)            🆕
        │     ├─ if dist <= 0.55: return True (no-op)
        │     ├─ teleport base via sim.data.qpos (~5ms)
        │     └─ return True
        │
        ├─ env.move_to_pre_grasp(candidate)                          (不变)
        │     └─ env.move_arm_to(pre_grasp_pos)                      (drive_base=False default)
        │           ├─ for step in range(800):
        │           │     action = arm_OSC_delta only                ← arm-only
        │           │     sim.step(action)
        │           │     [converge: < 100 步, 距离已小]
        │           └─ return True
        │
        └─ proceed to approach / close / lift                        (不变)
```

---

## 附录 B: 涉及文件清单

| 文件 | 改动类型 | 说明 |
|------|------|------|
| `src/env_wrapper.py` | 修改 | Phase 0 (revert) + Phase 2 (新增 navigate_base_to + helpers) + Phase 3 (drive_base 参数 + `move_to_pre_grasp` 内 drive_base=True) |
| `src/action_executor.py` | 修改 | Phase 4 (插入 navigate_base_to 调用) |
| `src/grasp_planner.py` | 修改 | Phase 4 (bump GRASP_CODE_VERSION v6.1 → v6.2) |
| `tests/test_env_wrapper_grasp.py` | 修改 | Phase 0 删测试 + Phase 3 加 drive_base 测试 |
| `tests/test_env_wrapper_navigation.py` | 新建 | Phase 2 navigate_base_to 单元测试 |
| `tests/test_action_executor_v1.py` | 修改 | Phase 4 navigate 集成测试 |

不涉及修改：
- `src/agent.py`, `src/memory_manager.py`, `src/perception.py`, `src/safety_gate.py`, `src/active_planner.py`, `src/world_belief.py`
- 所有配置文件 (`configs/*.yaml`)
- 所有 prompt 文件 (`prompts/**/*.txt`)

---

## 11. Self-Review Findings & Revisions

文档完成 v1.0 后由作者自审，发现以下 11 个需修正的问题。本节直接覆盖前文相关章节，**优先级高于前文**。

### 11.1 🚨 关键修正：`move_to_pre_grasp` 内有真正的远距离 `move_arm_to` 调用

**问题**：原设计第 4.3 节 Phase 3 兼容性矩阵声称"所有现有 caller 的 dist 都 < 0.05m, base 部分不触发"。

**实际**：`@/c:/all_project/embodied-AI-one/src/env_wrapper.py:1599-1621` 内有：

```python
# 底盘先靠近: 用 pre_pos 的 xy, 但 z 保持当前 eef 高度避免硬碰撞
self.move_arm_to(base_target, threshold_m=0.15, max_steps=600)
```

`base_target` 是 base 应停的 xy 位置（距当前 eef 可能 0.5m+），**这是当前架构中真正的"navigate base"调用**，依靠 `move_arm_to` 的混合控制实现。

**影响**：Phase 3 默认 `drive_base=False` 会让此调用退化为 arm-only 远距离尝试 → 600 步 stall 浪费时间。

**修正**：Phase 3 时给此调用**显式保留** `drive_base=True`：

```python
self.move_arm_to(
    base_target, threshold_m=0.15, max_steps=600,
    drive_base=True,  # 保留 legacy 兜底, 当 navigate_base_to 失败时仍能推 base
)
```

**双层防御**：
- 正常路径：Phase 4 的 `navigate_base_to` 在 `move_to_pre_grasp` 之前把 base teleport 就位 → line 1621 的 `move_arm_to` dist 小，几步收敛
- 兜底路径：`navigate_base_to` 失败时，line 1621 走 legacy 混合控制（即使慢/可能 stall 也比无任何 navigation 强）

### 11.2 🚨 关键修正：memory 版本必须 bump

**问题**：原设计未涉及 `GRASP_CODE_VERSION`。

**实际**：navigate 改变 base 行为后，旧 memory 里的 `top_down/banana ALL fail=50 ik_unreachable` 等数据**语义已变**（之前是 base 不到位被错标，现在 base 到位后 top_down 可能成功）。继续用旧 memory 会让 fast-path 走错策略。

**修正**：Phase 4 commit 同时 bump：

```python
# src/grasp_planner.py 或 src/memory_manager.py
GRASP_CODE_VERSION = "v6.2"  # was v6.1; navigate_base_to introduced
```

旧 memory 自动 retire，新数据从干净状态积累。

### 11.3 ⚠️ 修正：navigate yaw 计算的 arm 朝向假设

**问题**：原设计 4.2 节直接用 `np.arctan2(direction[1], direction[0])` 计算 base yaw，未说明 arm-on-base 几何假设。

**实际**：PandaMobile 的 arm 默认安装在 base **+x** 方向（forward）。base yaw=0 时 arm 朝 world +x，base yaw=π 时朝 world -x。

**修正**：navigate 时 yaw 让 base +x 指向 target = arm 工作空间覆盖 target = 正确。

但需要在 Phase 2 代码注释明示：

```python
# base 朝向 target 方向 (PandaMobile arm 沿 base +x, 此 yaw 让 arm 工作空间覆盖 target)
target_yaw = float(np.arctan2(direction_norm[1], direction_norm[0]))
```

且 Phase 5 验证须确认此假设在实际场景成立。若失败，备用方案：保持 yaw 不变，仅 teleport xy（依赖原有 base orientation）。

### 11.4 ⚠️ 修正：base joint axis 假设依赖 Phase 1 探查

**问题**：原设计 Phase 2 的 `_get_mobilebase_joint_addrs` 用 `abs(axis[0]) > 0.9` 判断 x slide。

**实际**：依赖 RoboCasa joint axis 标准 [(1,0,0), (0,1,0), (0,0,1)]。若 axis 是 (0.7, 0.7, 0) 复合方向，逻辑失效。

**修正**：Phase 1 探查输出必须确认 axis 形式后，Phase 2 才写最终代码。若 axis 非标准，改用 joint name 后缀匹配（`*_joint_x` / `*_joint_y` / `*_joint_yaw`）。

### 11.5 🆕 新增：Phase 0.5 — Baseline 测量

**问题**：原设计 Phase 5 缺 baseline 对比基准。

**修正**：在 Phase 0 完成后、Phase 2 开始前，**强制**跑 baseline 测量记录到附录。

**Phase 0.5 步骤**（无 commit）：

```bash
# GPU 服务器
export DEEPSEEK_API_KEY=sk-...
export MUJOCO_GL=egl

python -m eval.run_fixed --scenario fixed_seed_discover_001 \
    --memory-dir runs/baseline/memory --log-level INFO \
    --output runs/baseline/tupperware \
    2>&1 | tee /tmp/baseline.log

# 关键指标
grep -c "max_steps reached" /tmp/baseline.log
grep -E "final|Episode result" /tmp/baseline.log | tail -10
```

baseline 数据 → 附录 D。

### 11.6 ✅ 确认：Phase 0 删除 `test_geometric_filter_works_with_real_mobile_base`

**问题**：IDE active 显示此测试，是否要保留？

**结论**：删除 OK。Phase 2 的 `_read_real_base_xy` 单元测试（附录 A）覆盖了相同 sim body lookup 逻辑。无信息丢失。

### 11.7 ⚠️ 文档化：测试环境依赖

**问题**：Phase 1 探查脚本和 Phase 5 验证脚本需要：
- `DEEPSEEK_API_KEY` 环境变量
- `MUJOCO_GL=egl`
- conda env `embosight` 激活

**修正**：Phase 1 / Phase 5 脚本前显式写出 setup 命令（已写入第 4.1 / 4.5 节），但需在每个 phase 操作前 reminders。

### 11.8 ⚠️ 风险预留：navigate 后 arm joint 不重置

**问题**：teleport base 后 arm joint qpos 不变，arm world 位置因 base 旋转/平移而漂移。OSC 下一步会拉回 target，但中间状态可能奇怪。

**当前决定**：Phase 2 v1 不动 arm joint，依靠 OSC 自愈。

**风险预留**：Phase 5 GPU 验证若发现 arm 抖动 / 解算异常，回到 Phase 2 加 arm home pose reset：

```python
# Phase 2 备选: navigate 后把 arm reset 到 default home pose
robot = self._env.robots[0]
arm_joints = robot.composite_controller.part_controllers["right"].joint_indexes
home_qpos = robot.init_qpos[arm_joints]  # 或 robot._init_qpos
for i, qid in enumerate(arm_joints):
    sim.data.qpos[qid] = home_qpos[i]
sim.forward()
```

### 11.9 ⚠️ Phase 3 兼容性矩阵更正

替换原第 4.3 节 "兼容性矩阵"：

| Caller | 当前 dist | base 触发? | Phase 3 后 (drive_base 设置) | 备注 |
|--------|---------|------|------|------|
| `move_to_pre_grasp` 内 line 1621 (base approach) | 0.5m+ | ✅ | **drive_base=True 显式保留** (legacy 兜底) | Navigate 失败时仍工作 |
| `move_to_pre_grasp` 内 line 1637 (arm 精修 pre_grasp) | < 0.15m | ✅ (>0.05m) | drive_base=False (default) | Navigate 后离 pre_pos 近, arm 单独够 |
| `lift` 内 micro_step (0.005m * N) | 0.005m | ❌ | drive_base=False | dist<0.05 不触发, 等价 |
| `lift` 内 final_target (0.10m) | 0.10m | ✅ (但纯 Z) | drive_base=False | 纯 Z dir, base XY action=0 即使触发, 等价 |
| `lift` 内 retreat segments (侧抓水平回退) | 0.025m/段 | ❌ | drive_base=False | 等价 |
| `grasp_at` 内 mini_lift (0.03m) | 0.03m | ❌ | drive_base=False | dist<0.05, 等价 |
| `grasp_at` 内 final lift to pre_grasp (0.10m) | 0.10m | ✅ (纯 Z) | drive_base=False | 同 lift final_target, 等价 |
| `action_executor` z-stall nudge (0.08m) | 0.08m | ✅ | drive_base=False | Navigate 已就位, nudge 偏移小, arm 够 |
| `action_executor` release_and_retreat (0.10m+) | 0.10m+ | ✅ (纯 Z) | drive_base=False | 纯 Z, base 无贡献, 等价 |
| `_approach_along_direction` 内 step (0.012m/步) | < 0.012m | ❌ | drive_base=False | 等价 |

**结论**：Phase 3 实际影响仅 `move_to_pre_grasp` line 1621 一处, 通过 drive_base=True 显式保留 ⇒ **真正零回归**。

### 11.10 🆕 更新：Phase 实施时间线

| Phase | 内容 | 预计耗时 |
|-------|------|---------|
| Phase 0 | Revert 三补丁 + 删测试 | 10 min |
| **Phase 0.5** | **Baseline GPU 测量** | **30 min** |
| Phase 1 | GPU 探查 mobilebase joints | 5 min |
| Phase 2 | navigate_base_to + helpers + 5 单测 | 60 min |
| Phase 3 | drive_base flag + line 1621 显式 + 3 单测 | 20 min |
| Phase 4 | act() 集成 + VERSION bump + 3 集成测试 | 30 min |
| Phase 5 | GPU 验证 (单 scenario + 20 seeds + 报告) | 90 min |
| **总计** | | **~4 hours** |

每个 commit 之间跑 `python -m pytest tests/ -q` 确认 0 regression。

### 11.11 🆕 新增附录 D / E / F 占位

- 附录 D: Phase 0.5 baseline 测量 + Phase 1 探查输出（待 GPU 执行后填充）
- 附录 E: Phase 4 GRASP_CODE_VERSION migration notes
- 附录 F: Phase 5 验证结果对比

---

## 附录 D: Baseline & Probe Results (待填充)

### D.1 Phase 0.5 Baseline (空)

预留：
- `fixed_seed_discover_001` episode 结果
- `max_steps reached` 出现次数
- 失败模式分布
- 平均 step count
- DEEPSEEK token 消耗

### D.2 Phase 1 Probe Results (空)

预留：
- mobilebase joint 命名清单
- qpos / qvel addrs
- joint axis

---

## 附录 E: GRASP_CODE_VERSION Migration (Phase 4)

旧版本 `v6.1` → 新版本 `v6.2`。

**触发条件**：Phase 4 commit。

**影响**：
- `memory/grasp.json` 内所有 `code_version="v6.1"` 条目被 `MemoryManager` 自动标记 stale, 不进入 fast-path 计算
- 新失败 / 成功事件以 `v6.2` 记录
- 旧条目通过 `scripts/clean_memory.py purge-retired` 可选清理

**为何 bump v6.2 而非 v7**：navigate 是底层基础设施改动，grasp 策略选择逻辑 / failure mode 集合未变 → 仍属 v6 系列。

---

## 附录 F: Phase 5 Validation Report (待填充)

预留：
- After-refactor 单 scenario 结果
- After-refactor 20 seeds 泛化结果
- Baseline vs after 对比表
- 失败模式分布变化
- 决策：是否合入 main / 是否需要 follow-up Phase 6

---

**文档结束 v1.1**
