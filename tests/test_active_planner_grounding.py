"""ActivePlanner Grounding-Aware 单元测试.

测试 plan_with_grounding() 的早停逻辑和 prompt 构建.
用 mock 对象, 不需要 GPU/VLM/仿真环境.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.active_planner import ActivePlanner, Observation, Viewpoint, ViewpointLibrary
from src.vlm_grounding import GroundedCandidate
from src.scene_model import SceneModel, GroundedObject


# ============================================================
# Mock Helpers
# ============================================================

class MockLLM:
    """模拟 LLM, 总是返回第一个未使用视角."""
    def __init__(self, stop_after: int = 3):
        self._call_count = 0
        self._stop_after = stop_after

    def generate(self, user_message="", system="", json_mode=False, **kw):
        self._call_count += 1
        if self._call_count >= self._stop_after:
            return '{"viewpoint_idx": -1, "reason": "enough"}'
        # 从 prompt 中解析可用视角索引
        import re
        indices = re.findall(r'^\s*(\d+):.*(?!\[已用\])', user_message, re.MULTILINE)
        used = set(re.findall(r'(\d+):.*\[已用\]', user_message))
        for idx in indices:
            if idx not in used:
                return f'{{"viewpoint_idx": {idx}, "reason": "next available"}}'
        return '{"viewpoint_idx": -1, "reason": "all used"}'


class MockEnv:
    """模拟 EnvWrapper."""
    def __init__(self):
        self._step = 0

    def observe(self, viewpoint):
        self._step += 1
        return Observation(
            viewpoint=viewpoint,
            image_path=f"/tmp/test_step_{self._step}.png",
        )


class MockSceneDescriber:
    """模拟 SceneDescriber, 在指定视角 'ground' 到目标."""
    def __init__(self, ground_on_view: str = "robot0_agentview_center", match_score: float = 0.9):
        self.ground_on_view = ground_on_view
        self.match_score = match_score
        self.grounder = MagicMock()
        self.safety_gate = MagicMock()

        # grounder.ground 返回一个候选
        def _ground(img_path):
            c = GroundedCandidate("apple", 0.9, (100, 100, 150, 150), "red round")
            return [c]
        self.grounder.ground = _ground

        # grounder.match_query 根据视角决定 score
        def _match(candidates, query, gt=None):
            for c in candidates:
                if self.ground_on_view in (img_path_hint := ""):
                    c.query_match_score = self.match_score
                else:
                    c.query_match_score = self.match_score
                c.matched_category = "apple"
                c.match_method = "exact"
            return candidates
        self.grounder.match_query = _match

        # safety_gate.update_object_safety 只设置字段
        def _update(obj):
            obj.safety_risk = "safe"
            obj.safety_reason = ""
        self.safety_gate.update_object_safety = _update


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def vp_lib():
    return ViewpointLibrary()


@pytest.fixture
def mock_subtasks():
    """创建简单 mock subtasks (覆盖率立即达标)."""
    class MockSubtask:
        def __init__(self, target, dim):
            self.target = target
            self.type = MagicMock(value="locate")
            self.blind_dimension = MagicMock(value=dim)
            self.priority = 1
            self.coverage_status = False

    return [
        MockSubtask("苹果", "position"),
        MockSubtask("苹果", "safety"),
    ]


# ============================================================
# Tests
# ============================================================

class TestPlanWithGrounding:
    def test_stops_when_target_grounded_and_covered(self, vp_lib, mock_subtasks):
        """目标已 grounded (score>=0.8) + 覆盖率达标 → 早停."""
        llm = MockLLM(stop_after=10)  # LLM 不会主动停
        planner = ActivePlanner(
            llm_client=llm,
            viewpoint_lib=vp_lib,
            max_viewpoints=6,
            coverage_threshold=0.85,
            grounding_confidence_threshold=0.8,
        )
        env = MockEnv()
        describer = MockSceneDescriber(match_score=0.9)

        observations, scene_model = planner.plan_with_grounding(
            mock_subtasks, env, "帮我拿苹果", describer
        )

        # 应该早停 (初始视角 center 就覆盖了 position+safety 且 grounded)
        assert len(observations) <= 2  # 最多 2 个 (初始 + 可能 1 个确认)
        assert scene_model is not None
        assert len(scene_model) > 0

    def test_continues_when_not_grounded(self, vp_lib, mock_subtasks):
        """目标未 grounded → 继续拍更多视角."""
        llm = MockLLM(stop_after=4)
        planner = ActivePlanner(
            llm_client=llm,
            viewpoint_lib=vp_lib,
            max_viewpoints=6,
            grounding_confidence_threshold=0.8,
        )
        env = MockEnv()
        # match_score=0.3 → 不够 grounding threshold
        describer = MockSceneDescriber(match_score=0.3)

        observations, scene_model = planner.plan_with_grounding(
            mock_subtasks, env, "帮我拿苹果", describer
        )

        # 覆盖率 1 个视角就达标 (position+safety), 但 grounding 不够
        # 应该继续拍
        assert len(observations) >= 2

    def test_no_scene_describer_falls_back(self, vp_lib, mock_subtasks):
        """没有 scene_describer → 等同于普通 plan (按覆盖率)."""
        llm = MockLLM(stop_after=10)
        planner = ActivePlanner(
            llm_client=llm,
            viewpoint_lib=vp_lib,
            max_viewpoints=6,
        )
        env = MockEnv()

        observations, scene_model = planner.plan_with_grounding(
            mock_subtasks, env, "帮我拿苹果", None
        )

        assert scene_model is None
        assert len(observations) >= 1

    def test_returns_scene_model(self, vp_lib, mock_subtasks):
        """返回的 scene_model 包含 grounded objects."""
        llm = MockLLM(stop_after=2)
        planner = ActivePlanner(
            llm_client=llm,
            viewpoint_lib=vp_lib,
            max_viewpoints=3,
        )
        env = MockEnv()
        describer = MockSceneDescriber(match_score=0.9)

        _, scene_model = planner.plan_with_grounding(
            mock_subtasks, env, "帮我拿苹果", describer
        )

        assert scene_model is not None
        best = scene_model.get_best_match()
        assert best is not None
        assert best.label == "apple"


class TestGroundingPromptBuild:
    def test_prompt_includes_grounding_status(self, vp_lib, mock_subtasks):
        """Grounding-aware prompt 包含 grounding 状态."""
        llm = MockLLM()
        planner = ActivePlanner(llm_client=llm, viewpoint_lib=vp_lib)

        # 创建一个有物体的 scene_model
        scene = SceneModel()
        c = GroundedCandidate("apple", 0.9, (100, 100, 150, 150), "red")
        c.query_match_score = 0.85
        c.matched_category = "apple"
        scene.add_view("center", [c], lambda b: np.array([0.5, 0.3, 0.95]))

        obs = [Observation(
            viewpoint=Viewpoint("center", (0, 0, 0), (0, 0, 0)),
            image_path="/tmp/test.png"
        )]

        prompt = planner._build_grounding_nbv_prompt(
            mock_subtasks, obs, {0}, "帮我拿苹果", scene
        )

        assert "Grounding" in prompt
        assert "apple" in prompt
        assert "0.85" in prompt

    def test_prompt_no_scene_model(self, vp_lib, mock_subtasks):
        """无 scene_model 时 prompt 显示目标未被发现."""
        llm = MockLLM()
        planner = ActivePlanner(llm_client=llm, viewpoint_lib=vp_lib)

        obs = [Observation(
            viewpoint=Viewpoint("center", (0, 0, 0), (0, 0, 0)),
            image_path="/tmp/test.png"
        )]

        prompt = planner._build_grounding_nbv_prompt(
            mock_subtasks, obs, {0}, "帮我拿苹果", None
        )

        assert "尚未被发现" in prompt


class TestPromptFileExists:
    def test_grounding_aware_prompt_exists(self):
        """prompts/active_planner_grounding_aware.txt 文件存在."""
        p = Path("prompts/active_planner_grounding_aware.txt")
        assert p.exists(), f"Prompt file not found: {p}"
