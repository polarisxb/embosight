"""EmboSightAgent.run 集成测试 (mock 全部依赖, 验证 5 种场景 + verify_mismatch)。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import numpy as np
import pytest

from tests._mocks import MockLLM, MockVLM


@pytest.fixture
def tmp_image(tmp_path):
    from PIL import Image
    p = tmp_path / "img.png"
    Image.new("RGB", (256, 256), (200, 100, 50)).save(p)
    return str(p)


def _make_full_agent(decompose_response, vlm_responses, safety_response,
                     vp_count=3, user_response="apple", image_path="/dev/null"):
    from src.action_executor import ActionExecutor
    from src.active_planner import ActiveViewpointSelector
    from src.agent import EmboSightAgent
    from src.grasp_planner import GraspPlanner
    from src.perception import QueryAwareGrounder
    from src.safety_gate import SafetyClassifier
    from src.task_decomposer import TaskDecomposer
    from src.user_channel import FakeUserChannel
    from src.vlm_cache import VLMCache

    class FakeVPLib:
        def __init__(self, n):
            self.viewpoints = [
                type("VP", (), {"name": f"v{i}"})() for i in range(n)
            ]

        def __len__(self):
            return len(self.viewpoints)

        def __getitem__(self, i):
            return self.viewpoints[i]

        def __iter__(self):
            return iter(self.viewpoints)

    class FakeEnv:
        def observe(self, vp):
            return type("Obs", (), {"image_path": image_path})()

        def viewpoint_intrinsics(self, vp):
            return None

        def is_reachable(self, p, d):
            return True

        def move_to_pre_grasp(self, c):
            return True

        def descend(self, p, target_label=None, **kwargs):
            return True, float(p[2])

        def close_gripper(self, target_label=None):
            return True

        def open_gripper(self):
            return True

        def lift(self):
            return True, 0.05

        def get_eef_pos(self):
            return np.array([0.5, 0, 0.95])

        def move_arm_to(self, p, **kw):
            return True

        def eye_in_hand_viewpoint(self):
            return type("VP", (), {"name": "eye_in_hand"})()

    decompose_llm = MockLLM(responses=[decompose_response])
    nbv_llm = MockLLM(responses=["1", "2", "-1"] * 10)
    safety_llm = MockLLM(responses=[safety_response] * 30)
    user_llm = MockLLM(responses=[user_response] * 30)

    vlm = MockVLM(responses=vlm_responses)
    cache = VLMCache()

    agent = EmboSightAgent(
        task_decomposer=TaskDecomposer(decompose_llm),
        perception=QueryAwareGrounder(
            vlm=vlm, llm=decompose_llm,
            cache=cache, label_temperature=1.0,
        ),
        safety_classifier=SafetyClassifier(llm=safety_llm),
        grasp_planner=GraspPlanner(vlm=vlm, env=FakeEnv()),
        action_executor=ActionExecutor(scene_describer=None),
        nbv_selector=ActiveViewpointSelector(
            llm=nbv_llm, viewpoint_lib=FakeVPLib(vp_count),
        ),
        user_channel=FakeUserChannel.from_explicit(user_llm, user_response),
        episode_logger=None,
        viewpoint_lib=FakeVPLib(vp_count),
        llm=decompose_llm,
        vlm=vlm,
    )
    return agent, FakeEnv()


class TestRun:
    def test_basic_success_path(self, tmp_image):
        """1 frame 看到 confident 的苹果 → grasp success。"""
        decomp = json.dumps({"primary_target": "apple", "constraints": []})
        vlm_resp = json.dumps({"objects": [
            {"bbox_2d": [50, 50, 100, 100], "label": "apple",
             "alternatives": [["apple", 0.95], ["other", 0.05]],
             "confidence": 0.9, "visible_features": "red round"},
        ]})
        safety = json.dumps({"dist": {"safe": 0.98, "fragile": 0.02},
                             "reasoning": "fruit"})
        agent, env = _make_full_agent(decomp, [vlm_resp] * 5, safety,
                                      image_path=tmp_image)
        result = agent.run("拿苹果", env)
        assert result.success is True
        assert result.target.label == "apple"

    def test_no_target_runs_until_max_or_ask_user(self, tmp_image):
        """全场没苹果 → 应触发 ask_user 或 give_up。"""
        decomp = json.dumps({"primary_target": "apple", "constraints": []})
        vlm_resp = json.dumps({"objects": [
            {"bbox_2d": [10, 10, 20, 20], "label": "banana",
             "alternatives": [["banana", 0.9], ["other", 0.1]],
             "confidence": 0.9, "visible_features": "yellow"},
        ]})
        safety = json.dumps({"dist": {"safe": 1.0}, "reasoning": "?"})
        agent, env = _make_full_agent(decomp, [vlm_resp] * 15, safety,
                                      vp_count=2, image_path=tmp_image)
        result = agent.run("拿苹果", env)
        assert result.speech != ""

    def test_ask_user_branch_runs(self, tmp_image):
        """场景里 2 个 ambiguous 苹果 → target=None → ask_user。"""
        decomp = json.dumps({"primary_target": "apple", "constraints": []})
        vlm_resp = json.dumps({"objects": [
            {"bbox_2d": [10, 10, 30, 30], "label": "apple",
             "alternatives": [["apple", 0.5], ["pear", 0.5]],
             "confidence": 0.9, "visible_features": "red"},
            {"bbox_2d": [80, 80, 100, 100], "label": "apple",
             "alternatives": [["apple", 0.5], ["pear", 0.5]],
             "confidence": 0.9, "visible_features": "red"},
        ]})
        safety = json.dumps({"dist": {"safe": 1.0}, "reasoning": "?"})
        agent, env = _make_full_agent(decomp, [vlm_resp] * 15, safety,
                                      image_path=tmp_image)
        result = agent.run("拿苹果", env)
        assert any(a.kind == "ask_user" for a in result.action_history)

    def test_decompose_with_constraint(self, tmp_image):
        """avoid:knife 通过到 perception."""
        decomp = json.dumps({
            "primary_target": "bowl",
            "constraints": [
                {"kind": "avoid", "target_label": "knife", "reason": "用户避开"},
            ],
        })
        vlm_resp = json.dumps({"objects": [
            {"bbox_2d": [50, 50, 80, 80], "label": "bowl",
             "alternatives": [["bowl", 0.95]],
             "confidence": 0.9, "visible_features": "round"},
        ]})
        safety = json.dumps({"dist": {"safe": 0.95, "fragile": 0.05}})
        agent, env = _make_full_agent(decomp, [vlm_resp] * 5, safety,
                                      user_response="bowl",
                                      image_path=tmp_image)
        result = agent.run("拿碗, 避开刀", env)
        assert result.target is not None
        assert result.target.label == "bowl"

    def test_max_steps_stops(self, tmp_image):
        """场景持续模糊 → MAX_STEPS 后 give_up。"""
        decomp = json.dumps({"primary_target": "apple", "constraints": []})
        vlm_resp = json.dumps({"objects": [
            {"bbox_2d": [10, 10, 20, 20], "label": "banana",
             "alternatives": [["banana", 0.5], ["other", 0.5]],
             "confidence": 0.5, "visible_features": "yellow"},
        ]})
        safety = json.dumps({"dist": {"safe": 1.0}})
        agent, env = _make_full_agent(decomp, [vlm_resp] * 30, safety,
                                      vp_count=2, image_path=tmp_image)
        result = agent.run("拿苹果", env)
        assert result.success is False
        assert result.failure_reason is not None
        assert len(result.action_history) <= agent.MAX_STEPS + 2


class TestVerifyMismatchFlow:
    """F6 / Edge 9.6: post-grasp verify 失败时的完整恢复流程。

    覆盖契约:
    1. result.attempt.failure_mode 改为 "verify_mismatch"
    2. target.label_entropy 拉到 ≥ 0.6
    3. target.times_re_observed += 1
    4. executor.release_and_retreat 被调一次 (避免夹爪遮挡死锁)
    """

    def _make_agent_with_failing_verify(
        self, decompose_response, vlm_resp, safety, image_path,
    ):
        agent, env = _make_full_agent(
            decompose_response, [vlm_resp] * 10, safety, image_path=image_path,
        )
        # monkey-patch executor.verify_grasp + release_and_retreat
        orig_release = agent.executor.release_and_retreat
        env._release_call_count = 0

        def fail_verify(target, e):
            return False, 0.4

        def count_release(e, retreat_height_m=0.10):
            env._release_call_count += 1
            return orig_release(e, retreat_height_m)

        agent.executor.verify_grasp = fail_verify
        agent.executor.release_and_retreat = count_release
        return agent, env

    def test_verify_mismatch_marks_failure_and_retreats(self, tmp_image):
        """物理 grasp 成功但 verify 说不对 → failure_mode 改写, release 被调。"""
        decomp = json.dumps({"primary_target": "apple", "constraints": []})
        vlm_resp = json.dumps({"objects": [
            {"bbox_2d": [50, 50, 100, 100], "label": "apple",
             "alternatives": [["apple", 0.95], ["other", 0.05]],
             "confidence": 0.9, "visible_features": "red round"},
        ]})
        safety = json.dumps({"dist": {"safe": 0.98, "fragile": 0.02}})
        agent, env = self._make_agent_with_failing_verify(
            decomp, vlm_resp, safety, tmp_image,
        )
        result = agent.run("拿苹果", env)
        attempts = result.target.grasp_attempts if result.target else []
        assert any(a.failure_mode == "verify_mismatch" for a in attempts), \
            "verify 失败必须改写 failure_mode 为 verify_mismatch"
        n_mismatch = sum(1 for a in attempts if a.failure_mode == "verify_mismatch")
        assert env._release_call_count >= n_mismatch, \
            f"release_and_retreat 调用 {env._release_call_count} < verify_mismatch {n_mismatch}"

    def test_verify_mismatch_raises_label_entropy(self, tmp_image):
        """verify_mismatch 后 label_entropy 必须 ≥ 0.6 (触发下一轮 zoom_in)。"""
        decomp = json.dumps({"primary_target": "apple", "constraints": []})
        vlm_resp = json.dumps({"objects": [
            {"bbox_2d": [50, 50, 100, 100], "label": "apple",
             "alternatives": [["apple", 0.95], ["other", 0.05]],
             "confidence": 0.9, "visible_features": "red"},
        ]})
        safety = json.dumps({"dist": {"safe": 0.98, "fragile": 0.02}})
        agent, env = self._make_agent_with_failing_verify(
            decomp, vlm_resp, safety, tmp_image,
        )
        result = agent.run("拿苹果", env)
        h = result.target
        if h is not None and any(a.failure_mode == "verify_mismatch"
                                 for a in h.grasp_attempts):
            assert h.label_entropy >= 0.6 - 1e-6, \
                f"verify_mismatch 后 label_entropy={h.label_entropy} 未提升到 ≥ 0.6"

    def test_verify_mismatch_increments_re_observed(self, tmp_image):
        """verify_mismatch 后 times_re_observed += 1 (标"已扰动")。"""
        decomp = json.dumps({"primary_target": "apple", "constraints": []})
        vlm_resp = json.dumps({"objects": [
            {"bbox_2d": [50, 50, 100, 100], "label": "apple",
             "alternatives": [["apple", 0.95]],
             "confidence": 0.9, "visible_features": "red"},
        ]})
        safety = json.dumps({"dist": {"safe": 0.98, "fragile": 0.02}})
        agent, env = self._make_agent_with_failing_verify(
            decomp, vlm_resp, safety, tmp_image,
        )
        result = agent.run("拿苹果", env)
        h = result.target
        if h is not None:
            n_mismatch = sum(1 for a in h.grasp_attempts
                             if a.failure_mode == "verify_mismatch")
            assert h.times_re_observed >= n_mismatch, \
                f"times_re_observed={h.times_re_observed} 应 ≥ mismatch 次数 {n_mismatch}"
