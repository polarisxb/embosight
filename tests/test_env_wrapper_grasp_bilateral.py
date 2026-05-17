"""Phase 6.1 unit tests: bilateral fingerpad contact mode.

Tests for the new `bilateral: bool` parameter on
`EnvWrapper._finger_object_contact` and its three-level fallback
(robosuite API → local bilateral → lenient).

See: docs/09_grasp_verification_refactor_design.md §4.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.env_wrapper import EnvWrapper  # noqa: E402


# ----------------------------------------------------------------------
# Minimal sim stub for contact-checking tests
# ----------------------------------------------------------------------


def _build_contact_stub(
    obj_geom_names: list[str],
    finger_geom_names: list[str],
    active_contacts: list[tuple[str, str]],
    obj_body_name: str = "obj_main",
) -> "_StubEnv":
    """Build a fake env whose sim reports contact pairs.

    active_contacts is a list of (geom_a_name, geom_b_name) tuples
    representing detected mujoco contacts. The stub maps names <-> ids.
    """
    all_geoms = list(dict.fromkeys(obj_geom_names + finger_geom_names))
    geom_name_to_id = {n: i for i, n in enumerate(all_geoms)}

    class _Model:
        ngeom = len(all_geoms)

        @staticmethod
        def geom_id2name(gid: int):
            for n, i in geom_name_to_id.items():
                if i == gid:
                    return n
            return None

        @staticmethod
        def body_name2id(name: str):
            if name == obj_body_name:
                return 0
            raise ValueError(name)

        @staticmethod
        def geom_bodyid(gid):  # not used but defensive
            return 0

        # geom_bodyid array surrogate (some env_wrapper methods may use)
        geom_bodyid = np.zeros(len(all_geoms), dtype=np.int32)

    # Sim data: data.ncon + contact[i].geom1/geom2
    class _Contact:
        def __init__(self, g1, g2):
            self.geom1 = g1
            self.geom2 = g2

    contact_objs = [
        _Contact(geom_name_to_id[a], geom_name_to_id[b])
        for a, b in active_contacts
    ]

    class _Data:
        ncon = len(contact_objs)
        contact = contact_objs

    class _Sim:
        model = _Model()
        data = _Data()

    class _Robot:
        idn = 0
        gripper = None  # default; tests can override

    class _StubEnv:
        sim = _Sim()
        robots = [_Robot()]

        def __init__(self):
            # Track geoms mapped to obj body via _get_body_geom_ids stub
            self._obj_geom_ids = {
                geom_name_to_id[n] for n in obj_geom_names
            }

    return _StubEnv()


class _BilateralStubWrapper(EnvWrapper):
    """EnvWrapper subclass that bypasses __init__ and uses contact stub."""

    def __init__(self, stub_env, obj_geom_ids: set[int]):
        self._env = stub_env
        self._obj_geom_ids_override = obj_geom_ids

    def _get_body_geom_ids(self, target_body: str) -> set[int]:
        return self._obj_geom_ids_override


# ======================================================================
# Tests
# ======================================================================


def test_lenient_default_returns_true_on_any_finger():
    """bilateral=False (default) -> 任意一指 contact 即 True (backward compat)."""
    env = _build_contact_stub(
        obj_geom_names=["obj_geom"],
        finger_geom_names=["right_fingerpad", "left_fingerpad"],
        active_contacts=[("obj_geom", "right_fingerpad")],  # 仅右指
    )
    w = _BilateralStubWrapper(env, obj_geom_ids={0})
    # 默认 lenient
    assert w._finger_object_contact("obj_main") is True


def test_strict_local_left_only_returns_false():
    """bilateral=True, 仅左指 contact -> False (本地 bilateral path)."""
    env = _build_contact_stub(
        obj_geom_names=["obj_geom"],
        finger_geom_names=["right_fingerpad", "left_fingerpad"],
        active_contacts=[("obj_geom", "left_fingerpad")],
    )
    w = _BilateralStubWrapper(env, obj_geom_ids={0})
    assert w._finger_object_contact("obj_main", bilateral=True) is False


def test_strict_local_right_only_returns_false():
    """bilateral=True, 仅右指 contact -> False."""
    env = _build_contact_stub(
        obj_geom_names=["obj_geom"],
        finger_geom_names=["right_fingerpad", "left_fingerpad"],
        active_contacts=[("obj_geom", "right_fingerpad")],
    )
    w = _BilateralStubWrapper(env, obj_geom_ids={0})
    assert w._finger_object_contact("obj_main", bilateral=True) is False


def test_strict_local_both_returns_true():
    """bilateral=True, 左右指都 contact -> True."""
    env = _build_contact_stub(
        obj_geom_names=["obj_geom"],
        finger_geom_names=["right_fingerpad", "left_fingerpad"],
        active_contacts=[
            ("obj_geom", "left_fingerpad"),
            ("obj_geom", "right_fingerpad"),
        ],
    )
    w = _BilateralStubWrapper(env, obj_geom_ids={0})
    assert w._finger_object_contact("obj_main", bilateral=True) is True


def test_strict_robosuite_path_uses_check_grasp():
    """bilateral=True 优先调用 env._check_grasp (Path 1).
    
    Mock 一个有 _check_grasp + important_geoms 的 env, 验证它被调用.
    """
    env = _build_contact_stub(
        obj_geom_names=["obj_geom"],
        finger_geom_names=["right_fingerpad", "left_fingerpad"],
        active_contacts=[],  # 本地路径检查不到, 但 robosuite mock 返 True
    )
    # 注入 _check_grasp + gripper.important_geoms
    check_grasp_mock = MagicMock(return_value=True)
    env._check_grasp = check_grasp_mock
    fake_gripper = MagicMock()
    fake_gripper.important_geoms = {
        "left_fingerpad": ["left_fingerpad"],
        "right_fingerpad": ["right_fingerpad"],
    }
    env.robots[0].gripper = fake_gripper

    w = _BilateralStubWrapper(env, obj_geom_ids={0})
    result = w._finger_object_contact("obj_main", bilateral=True)
    assert result is True
    # 验证 robosuite API 被调用了
    assert check_grasp_mock.call_count == 1
    # 验证传入的 gripper 是 fake_gripper, geom names 是 obj geom names
    args, _ = check_grasp_mock.call_args
    assert args[0] is fake_gripper
    assert "obj_geom" in args[1]


def test_strict_falls_back_to_lenient_when_left_right_unrecognizable():
    """bilateral=True, 无 left/right 命名 + 无 robosuite API -> lenient 兜底.
    
    设计 ADR-4: 降级回 lenient 而非 strict False, 避免 close_gripper 因 API
    异常永远 reject.
    """
    env = _build_contact_stub(
        obj_geom_names=["obj_geom"],
        # finger geom 没有 left/right 关键字
        finger_geom_names=["finger_a", "finger_b"],
        active_contacts=[("obj_geom", "finger_a")],  # 单侧 contact
    )
    # 不注入 _check_grasp, 强制走 Path 1 -> None -> Path 2 -> None -> lenient

    w = _BilateralStubWrapper(env, obj_geom_ids={0})
    # Lenient 看任意一指, finger_a 含 "finger" 关键字, 应该返 True
    assert w._finger_object_contact("obj_main", bilateral=True) is True


def test_strict_target_body_not_found_returns_false():
    """bilateral=True, target body 无 geom -> False (不抛异常)."""
    env = _build_contact_stub(
        obj_geom_names=["other_geom"],
        finger_geom_names=["left_fingerpad", "right_fingerpad"],
        active_contacts=[],
    )
    w = _BilateralStubWrapper(env, obj_geom_ids=set())  # 空, 触发 False
    result = w._finger_object_contact("nonexistent_body", bilateral=True)
    assert result is False


def test_close_gripper_uses_strict_bilateral_mode():
    """集成: _close_gripper_until_grasp 内调用应该传 bilateral=True.
    
    通过源码 grep 验证 callsite. (运行时已被 strict 路径覆盖,
    但这个测试守护未来重构不会意外丢失 bilateral 参数.)
    """
    import inspect
    from src.env_wrapper import EnvWrapper
    src = inspect.getsource(EnvWrapper._close_gripper_until_grasp)
    assert "_finger_object_contact(" in src
    assert "bilateral=True" in src, (
        "close_gripper_until_grasp must call _finger_object_contact with "
        "bilateral=True (Phase 6.1 contract)"
    )


# ======================================================================
# Edge cases
# ======================================================================


def test_strict_robosuite_path_returns_none_when_important_geoms_missing():
    """Path 1: gripper 无 important_geoms['left_fingerpad'] -> 返 None, 触发 fallback."""
    env = _build_contact_stub(
        obj_geom_names=["obj_geom"],
        finger_geom_names=["right_fingerpad", "left_fingerpad"],
        active_contacts=[
            ("obj_geom", "left_fingerpad"),
            ("obj_geom", "right_fingerpad"),
        ],
    )
    env._check_grasp = MagicMock(return_value=True)
    fake_gripper = MagicMock()
    fake_gripper.important_geoms = {}  # 缺 left_fingerpad
    env.robots[0].gripper = fake_gripper

    w = _BilateralStubWrapper(env, obj_geom_ids={0})
    # Path 1 None -> fallback 到 Path 2 local (左右都 contact -> True)
    assert w._finger_object_contact("obj_main", bilateral=True) is True
    # robosuite._check_grasp 不应该被调用 (因为 important_geoms 缺失)
    assert env._check_grasp.call_count == 0


def test_strict_robosuite_path_handles_dict_gripper():
    """Path 1: gripper 是 dict (PandaMobile 等) 时正确取 'right' key."""
    env = _build_contact_stub(
        obj_geom_names=["obj_geom"],
        finger_geom_names=["right_fingerpad", "left_fingerpad"],
        active_contacts=[],
    )
    check_grasp_mock = MagicMock(return_value=True)
    env._check_grasp = check_grasp_mock
    fake_right_gripper = MagicMock()
    fake_right_gripper.important_geoms = {
        "left_fingerpad": ["left_fingerpad"],
        "right_fingerpad": ["right_fingerpad"],
    }
    env.robots[0].gripper = {"right": fake_right_gripper}

    w = _BilateralStubWrapper(env, obj_geom_ids={0})
    assert w._finger_object_contact("obj_main", bilateral=True) is True
    # 传入的应该是 fake_right_gripper, 不是整个 dict
    assert check_grasp_mock.call_args[0][0] is fake_right_gripper


# ======================================================================
# Phase 6.2: verify_grasp_by_micro_lift unit tests
# ======================================================================


class _MicroLiftStubWrapper(EnvWrapper):
    """EnvWrapper subclass that stubs out get_eef_pos / _get_body_pos /
    move_arm_to for micro-lift testing."""

    def __init__(
        self,
        eef_z_start: float = 0.95,
        obj_z_start: float = 0.94,
        obj_z_delta_after_lift: float = 0.02,  # how much obj follows
        eef_z_delta_after_lift: float = 0.02,  # how much EEF actually rose
        get_body_pos_returns_none: bool = False,
        raise_on_move: bool = False,
    ):
        self._eef_z = eef_z_start
        self._obj_z = obj_z_start
        self._obj_z_delta = obj_z_delta_after_lift
        self._eef_z_delta_actual = eef_z_delta_after_lift
        self._return_none = get_body_pos_returns_none
        self._raise_on_move = raise_on_move
        # provide a fake _env / _latest_obs so move_arm_to doesn't crash
        # (but we override move_arm_to entirely)
        self._env = MagicMock()
        self._latest_obs = {}

    def get_eef_pos(self) -> np.ndarray:
        return np.array([0.0, 0.0, self._eef_z], dtype=np.float32)

    def _get_body_pos(self, body_name: str):
        if self._return_none:
            return None
        return np.array([0.0, 0.0, self._obj_z], dtype=np.float32)

    def move_arm_to(self, target, **kwargs):
        if self._raise_on_move:
            raise RuntimeError("simulated move failure")
        # Simulate partial completion based on stub config
        self._eef_z += self._eef_z_delta_actual
        self._obj_z += self._obj_z_delta
        return True


def test_micro_lift_returns_true_when_obj_follows():
    """obj 跟随 lift_m * threshold 比例 -> True."""
    w = _MicroLiftStubWrapper(
        obj_z_delta_after_lift=0.02,  # full 2cm follow
        eef_z_delta_after_lift=0.02,
    )
    assert w.verify_grasp_by_micro_lift(
        "obj_main", lift_m=0.02, threshold=0.5,
    ) is True


def test_micro_lift_returns_false_when_obj_stays():
    """obj Δz=0 -> False (slipped)."""
    w = _MicroLiftStubWrapper(
        obj_z_delta_after_lift=0.0,  # obj doesn't move
        eef_z_delta_after_lift=0.02,
    )
    assert w.verify_grasp_by_micro_lift(
        "obj_main", lift_m=0.02, threshold=0.5,
    ) is False


def test_micro_lift_threshold_applied():
    """obj Δz < eef_delta * threshold (with 5mm floor) -> False.

    Phase 6.2 v2: 用 eef_delta 而非 lift_m 作为基准. 这里 stub 让 EEF
    完整升 2cm 所以 eef_delta = lift_m, 等价行为.
    """
    # threshold=0.5, eef_delta=0.02 -> required = max(0.005, 0.01) = 0.01
    # 给 obj_delta=0.005 -> 0.005 < 0.01 -> False
    w = _MicroLiftStubWrapper(
        obj_z_delta_after_lift=0.005,
        eef_z_delta_after_lift=0.02,
    )
    assert w.verify_grasp_by_micro_lift(
        "obj_main", lift_m=0.02, threshold=0.5,
    ) is False
    # threshold=0.5 但 obj_delta=0.012 > 0.01 -> True
    w2 = _MicroLiftStubWrapper(
        obj_z_delta_after_lift=0.012,
        eef_z_delta_after_lift=0.02,
    )
    assert w2.verify_grasp_by_micro_lift(
        "obj_main", lift_m=0.02, threshold=0.5,
    ) is True


def test_micro_lift_handles_osc_stall_without_false_negative():
    """Phase 6.2 v2 regression: OSC stall 让 EEF 只升 5mm, object 也跟着
    升 5mm -> 应判 True (正常 grasp), 不能误杀 slipped.

    旧逻辑 (基准 lift_m * threshold = 0.01) 会判 False = 误杀.
    新逻辑 (基准 max(0.005, eef_delta * threshold) = max(0.005, 0.0025) =
    0.005) 判 0.005 >= 0.005 -> True.
    """
    w = _MicroLiftStubWrapper(
        obj_z_delta_after_lift=0.006,   # object 跟着升 6mm (噪声 + 安全 margin)
        eef_z_delta_after_lift=0.005,   # OSC stall, EEF 只升 5mm
    )
    assert w.verify_grasp_by_micro_lift(
        "obj_main", lift_m=0.02, threshold=0.5,
    ) is True


def test_micro_lift_min_required_floor_prevents_zero_zero_false_positive():
    """Phase 6.2 v2: EEF 完全未动 (eef_delta=0) 时 5mm 底限保护.

    没有底限 required = 0, obj_delta=0 >= 0 -> 假报成功 (实际 grasp 可能 fail).
    有底限 required = 0.005, obj_delta=0 < 0.005 -> 正确 False.
    """
    w = _MicroLiftStubWrapper(
        obj_z_delta_after_lift=0.0,   # object 没动
        eef_z_delta_after_lift=0.0,   # EEF 也没动 (彻底 stall)
    )
    assert w.verify_grasp_by_micro_lift(
        "obj_main", lift_m=0.02, threshold=0.5,
    ) is False


def test_micro_lift_returns_true_when_body_not_found():
    """无法读 obj pos -> 保守 True (上游用 post-lift Δz 兜底)."""
    w = _MicroLiftStubWrapper(get_body_pos_returns_none=True)
    assert w.verify_grasp_by_micro_lift(
        "nonexistent_body", lift_m=0.02, threshold=0.5,
    ) is True


def test_micro_lift_returns_true_on_exception():
    """move_arm_to 抛异常 -> 保守 True (不阻断后续 lift)."""
    w = _MicroLiftStubWrapper(raise_on_move=True)
    assert w.verify_grasp_by_micro_lift(
        "obj_main", lift_m=0.02, threshold=0.5,
    ) is True


# ======================================================================
# Phase 6.3: _gripper_closed_on_empty unit tests
# ======================================================================


class _JawCheckStubWrapper(EnvWrapper):
    """Minimal wrapper for jaw-width testing."""

    def __init__(
        self,
        obs: dict | None = None,
        sim_qpos: dict[str, float] | None = None,
        joint_names: list[str] | None = None,
    ):
        self._latest_obs = obs
        if sim_qpos is not None:
            # Build a minimal sim with joint_name2id + jnt_qposadr
            joint_to_addr = {}
            qpos_list = []
            for jname, jval in sim_qpos.items():
                joint_to_addr[jname] = len(qpos_list)
                qpos_list.append(jval)

            class _Model:
                @staticmethod
                def joint_name2id(name):
                    if name not in joint_to_addr:
                        raise ValueError(name)
                    return list(joint_to_addr.keys()).index(name)

                @property
                def jnt_qposadr(self):
                    return np.array(list(joint_to_addr.values()), dtype=np.int32)

            class _Data:
                qpos = np.array(qpos_list, dtype=np.float32)

            class _Sim:
                model = _Model()
                data = _Data()

            class _Robot:
                idn = 0

                class _G:
                    joints = joint_names or list(joint_to_addr.keys())

                gripper = _G()

            class _Env:
                sim = _Sim()
                robots = [_Robot()]

            self._env = _Env()
        else:
            self._env = MagicMock()


def test_gripper_closed_on_empty_returns_true_when_obs_gap_small():
    """obs.robot0_gripper_qpos 显示 gap < 5mm -> True (jaw 闭到空)."""
    w = _JawCheckStubWrapper(obs={"robot0_gripper_qpos": [0.001, 0.001]})
    # gap = 0.002m < 0.005m default
    assert w._gripper_closed_on_empty() is True


def test_gripper_closed_on_empty_returns_false_when_obs_gap_normal():
    """obs.gripper_qpos 显示 gap >= 5mm -> False (正常 grasp)."""
    w = _JawCheckStubWrapper(obs={"robot0_gripper_qpos": [0.01, 0.01]})
    # gap = 0.02m > 0.005m
    assert w._gripper_closed_on_empty() is False


def test_gripper_closed_on_empty_returns_false_when_obs_missing():
    """obs 无该 key + sim 不可用 -> False (保守不报告 empty)."""
    w = _JawCheckStubWrapper(obs={})
    # 没有 sim 走 fallback, 也没有, 最终 False
    assert w._gripper_closed_on_empty() is False


def test_gripper_closed_on_empty_sim_fallback_path():
    """obs 不可用时走 sim.data.qpos fallback."""
    w = _JawCheckStubWrapper(
        obs=None,
        sim_qpos={"finger_left_joint": 0.001, "finger_right_joint": 0.001},
    )
    # 两指 qpos 总和 0.002 < 0.005 -> True
    assert w._gripper_closed_on_empty() is True


def test_close_gripper_skips_confirm_when_jaw_closed_empty():
    """集成: _close_gripper_until_grasp 内调用 _gripper_closed_on_empty.
    
    通过源码 grep 验证 callsite (运行时已被 jaw_check 覆盖).
    """
    import inspect
    from src.env_wrapper import EnvWrapper
    src = inspect.getsource(EnvWrapper._close_gripper_until_grasp)
    assert "_gripper_closed_on_empty" in src, (
        "_close_gripper_until_grasp must consult _gripper_closed_on_empty "
        "(Phase 6.3 contract)"
    )
    # 也要确认 skip 逻辑使用 continue (不是 return)
    assert "continue" in src
