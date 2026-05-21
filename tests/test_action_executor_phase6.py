"""Phase 6.2 integration tests: micro-lift verification in ActionExecutor.act().

Tests that:
- act() calls env.verify_grasp_by_micro_lift after close_gripper
- when micro-lift fails (obj doesn't follow), act() returns slipped_lift
  with stage=micro_lift_verify
- when env doesn't have the method (legacy mock), act() still works
- when target body can't be resolved, act() falls through (defer to L4)
- when micro-lift raises exception, act() falls through

See: docs/09_grasp_verification_refactor_design.md §5.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Reuse the existing FakeEnv test fixtures from test_action_executor_v1
from tests.test_action_executor_v1 import FakeEnv, _hyp_with_candidate  # noqa: E402

from src.action_executor import ActionExecutor  # noqa: E402
from src.world_belief import DecomposedTask  # noqa: E402


class _MicroLiftEnv(FakeEnv):
    """FakeEnv extension that supports verify_grasp_by_micro_lift.

    micro_lift_result controls verify_grasp_by_micro_lift's return:
      True  -> object follows (grasp OK)
      False -> slipped (failure)
      "raise" -> raises RuntimeError (test exception path)
    """

    def __init__(
        self,
        micro_lift_result=True,
        type_map: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._micro_lift_result = micro_lift_result
        self.micro_lift_calls: list[dict] = []
        self._type_map_override = type_map

    def _get_obj_type_map(self):
        if self._type_map_override is not None:
            return self._type_map_override
        return super()._get_obj_type_map()

    def verify_grasp_by_micro_lift(
        self,
        target_body: str,
        lift_m: float = 0.02,
        threshold: float = 0.5,
    ) -> bool:
        self.micro_lift_calls.append({
            "target_body": target_body,
            "lift_m": float(lift_m),
            "threshold": float(threshold),
        })
        self.calls.append("micro_lift")
        if self._micro_lift_result == "raise":
            raise RuntimeError("simulated micro_lift failure")
        return bool(self._micro_lift_result)


class TestMicroLiftIntegration:
    def test_micro_lift_called_between_close_and_lift_on_normal_path(self):
        """正常路径: close_gripper -> micro_lift -> lift 顺序."""
        env = _MicroLiftEnv(micro_lift_result=True)
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)
        assert result.success is True
        # 验证顺序
        idx_close = env.calls.index("close")
        idx_micro = env.calls.index("micro_lift")
        idx_lift = env.calls.index("lift")
        assert idx_close < idx_micro < idx_lift, (
            f"Expected close < micro_lift < lift, got order: {env.calls}"
        )
        # 验证 args
        assert len(env.micro_lift_calls) == 1
        call = env.micro_lift_calls[0]
        assert call["target_body"] == "obj_main"  # FakeEnv maps obj_main -> apple
        assert call["lift_m"] == 0.02
        assert call["threshold"] == 0.5

    def test_micro_lift_fail_returns_slipped_lift_before_full_lift(self):
        """micro_lift 失败 -> 立即 return slipped_lift, 不调 env.lift."""
        env = _MicroLiftEnv(micro_lift_result=False)
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)

        assert result.success is False
        assert result.attempt.failure_mode == "slipped_lift"
        diag = result.attempt.diagnostic
        assert diag["stage"] == "micro_lift_verify"
        assert diag["reason"] == "object_not_following"
        assert diag["threshold"] == 0.5
        assert diag["lift_m"] == 0.02
        # 关键: env.lift 不应该被调用 (early-out 省时间)
        assert "lift" not in env.calls or env.calls.count("lift") == 0 or (
            # release_and_retreat 走 move 不走 lift, 所以应该没有 lift
            env.calls.index("micro_lift") == len(env.calls) - 1 - len([
                c for c in env.calls if c == "open" or c.startswith("move")
            ])
        )

    def test_micro_lift_exception_falls_through_to_full_lift(self):
        """micro_lift 抛异常 -> 保守 continue, 走完整 lift (不阻断流程)."""
        env = _MicroLiftEnv(micro_lift_result="raise")
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)
        # 流程继续到 lift, FakeEnv 默认 lift_ok=True -> success
        assert result.success is True
        # 验证 micro_lift 被调过且 lift 也被调过
        assert "micro_lift" in env.calls
        assert "lift" in env.calls

    def test_legacy_env_without_micro_lift_works(self):
        """无 verify_grasp_by_micro_lift 方法的 env -> backward compat."""
        # 普通 FakeEnv (没有 verify_grasp_by_micro_lift)
        env = FakeEnv()
        assert not hasattr(env, "verify_grasp_by_micro_lift")
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)
        # 与 Phase 6.2 前行为一致
        assert result.success is True
        assert "lift" in env.calls

    def test_micro_lift_skipped_when_target_body_unresolvable(self):
        """无法 resolve target body (type_map 找不到 label) -> skip micro_lift."""
        # type_map 中没有 "apple" 类别 → _resolve_target_body 返 None → skip
        env = _MicroLiftEnv(
            micro_lift_result=False,  # 如果调了就会 fail
            type_map={"obj_main": "banana"},  # 不匹配 "apple"
        )
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()  # label="apple"
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)
        # 因为 micro_lift 被 skip, 走老逻辑 -> FakeEnv default lift_ok=True -> success
        assert result.success is True
        # micro_lift 不应该被调用 (因为 target body 解析失败)
        assert len(env.micro_lift_calls) == 0

    def test_micro_lift_not_called_when_grasp_failed(self):
        """grasp_ok=False -> 跳过 micro_lift (没夹住就没必要 verify)."""
        env = _MicroLiftEnv(
            micro_lift_result=False,  # 不会到这里, 因为 grasp_ok=False
            grasp_ok=False,
        )
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)
        # micro_lift 不应该被调用 (grasp 没成功)
        assert len(env.micro_lift_calls) == 0
        # 走老逻辑: grasp_ok=False + lift_ok=True (default) -> still success on FakeEnv
        # (FakeEnv 的 lift 不依赖 grasp_ok)


class TestResolveTargetBody:
    def test_resolves_via_type_map(self):
        """正常的 type_map 反查."""
        from src.world_belief import Hypothesis
        env = _MicroLiftEnv(type_map={
            "obj_main": "apple",
            "distr_main": "banana",
        })
        target = Hypothesis(
            object_id="o0", label="apple",
            label_alternatives=[("apple", 1.0)], label_entropy=0.0,
            position_3d=np.zeros(3), position_std_m=0.0,
        )
        result = ActionExecutor._resolve_target_body(target, env)
        assert result == "obj_main"

    def test_returns_none_when_label_missing(self):
        """target 没 label -> None."""
        class _Stub:
            label = None
        env = _MicroLiftEnv()
        assert ActionExecutor._resolve_target_body(_Stub(), env) is None

    def test_returns_none_when_type_map_no_match(self):
        """type_map 中无该 label -> None."""
        from src.world_belief import Hypothesis
        env = _MicroLiftEnv(type_map={"obj_main": "orange"})
        target = Hypothesis(
            object_id="o0", label="apple",
            label_alternatives=[("apple", 1.0)], label_entropy=0.0,
            position_3d=np.zeros(3), position_std_m=0.0,
        )
        assert ActionExecutor._resolve_target_body(target, env) is None

    def test_returns_none_on_exception(self):
        """env._get_obj_type_map 抛异常 -> None (defensive)."""
        class _BrokenEnv:
            def _get_obj_type_map(self):
                raise RuntimeError("oops")
        from src.world_belief import Hypothesis
        target = Hypothesis(
            object_id="o0", label="apple",
            label_alternatives=[("apple", 1.0)], label_entropy=0.0,
            position_3d=np.zeros(3), position_std_m=0.0,
        )
        assert ActionExecutor._resolve_target_body(target, _BrokenEnv()) is None


def test_micro_lift_failure_records_execution_failure_diagnostics():
    env = _MicroLiftEnv(micro_lift_result=False)
    exe = ActionExecutor(
        scene_describer=None,
        grasp_policy_config={
            "mode": "profiled",
            "execution_recovery_diagnostics": True,
        },
    )
    h, first = _hyp_with_candidate()
    first.source = "vlm_top_grasp"

    result = exe.act(h, DecomposedTask(primary_target="apple"), env)

    diag = result.attempt.diagnostic
    assert result.attempt.failure_mode == "slipped_lift"
    assert diag["execution_failure_stage"] == "micro_lift_verify"
    assert diag["execution_failure_reason"] == "object_not_following"
    assert diag["execution_branch"] == "lift"
    assert diag["execution_failure_recoverable"] is True
    assert diag["execution_recovery_applied"] is False


class _MicroLiftRecoveryEnv(_MicroLiftEnv):
    def __init__(self):
        super().__init__(micro_lift_result=False)
        self._candidate_source = "vlm_top_grasp"
        self.retreat_opens = 0

    def move_to_pre_grasp_diagnostic(self, candidate, height_m=0.05):
        from types import SimpleNamespace

        self._candidate_source = str(candidate.source)
        return SimpleNamespace(
            ok=True,
            reason="strict_ok",
            total_error_m=0.0,
            lateral_error_m=0.0,
            axis_error_m=0.0,
            approach_gap_m=0.0,
            lateral_limit_m=0.02,
            handoff_ok=False,
            needs_recovery=False,
        )

    def verify_grasp_by_micro_lift(self, target_body, lift_m=0.02, threshold=0.5):
        self.micro_lift_calls.append({
            "target_body": target_body,
            "lift_m": float(lift_m),
            "threshold": float(threshold),
            "source": self._candidate_source,
        })
        self.calls.append("micro_lift")
        return self._candidate_source != "vlm_top_grasp"

    def open_gripper(self) -> bool:
        self.retreat_opens += 1
        return super().open_gripper()


def test_execution_recovery_skips_micro_lift_failure_candidate():
    from src.world_belief import GraspCandidate

    env = _MicroLiftRecoveryEnv()
    exe = ActionExecutor(
        scene_describer=None,
        grasp_policy_config={
            "mode": "profiled",
            "execution_recovery_diagnostics": True,
            "execution_recovery_gate": True,
            "execution_recovery_max_attempts": 1,
        },
    )
    h, first = _hyp_with_candidate()
    first.source = "vlm_top_grasp"
    second = GraspCandidate(
        point_3d=np.array([0.5, 0.0, 0.9], dtype=np.float32),
        approach_dir=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        finger_width_m=0.04,
        score=0.7,
        source="strategy_top_down",
    )
    h.grasp_candidates = [first, second]

    result = exe.act(h, DecomposedTask(primary_target="apple"), env)

    assert result.success is True
    assert result.attempt.candidate.source == "strategy_top_down"
    assert result.attempt.diagnostic["execution_recovery_applied"] is True
    assert result.attempt.diagnostic["execution_recovery_skip_count"] == 1
    assert result.attempt.diagnostic["execution_recovery_skipped_sources"] == [
        "vlm_top_grasp",
    ]
    assert env.retreat_opens >= 1


def test_micro_lift_failure_remains_terminal_when_recovery_disabled():
    env = _MicroLiftRecoveryEnv()
    exe = ActionExecutor(
        scene_describer=None,
        grasp_policy_config={
            "mode": "profiled",
            "execution_recovery_diagnostics": True,
            "execution_recovery_gate": False,
        },
    )
    h, first = _hyp_with_candidate()
    first.source = "vlm_top_grasp"

    result = exe.act(h, DecomposedTask(primary_target="apple"), env)

    assert result.success is False
    assert result.attempt.failure_mode == "slipped_lift"
    assert result.attempt.diagnostic["execution_recovery_applied"] is False
