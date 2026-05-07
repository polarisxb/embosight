"""SceneModel + 3D 投影 单元测试.

所有测试用 mock 数据, 不需要 GPU 或仿真环境.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.vlm_grounding import GroundedCandidate
from src.scene_model import (
    GroundedObject,
    SceneModel,
    compute_intrinsics,
    depth_buffer_to_meters,
    project_bbox_to_world,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def scene_model():
    return SceneModel(alignment_threshold_m=0.15)


@pytest.fixture
def candidate_apple():
    c = GroundedCandidate("red apple", 0.9, (235, 65, 256, 85), "round, red, shiny")
    c.query_match_score = 0.9
    c.matched_category = "apple"
    c.match_method = "exact"
    return c


@pytest.fixture
def candidate_bottle():
    c = GroundedCandidate("plastic bottle", 0.8, (50, 100, 90, 180), "cylindrical, plastic")
    c.query_match_score = 0.85
    c.matched_category = "bottle"
    c.match_method = "alias"
    return c


def _mock_projector(offset=None):
    """返回一个 mock projector, 根据 bbox 中心生成伪 3D 坐标."""
    if offset is None:
        offset = np.zeros(3)

    def _proj(bbox_2d):
        x1, y1, x2, y2 = bbox_2d
        u = (x1 + x2) / 2.0 / 256.0  # normalize to [0,1]
        v = (y1 + y2) / 2.0 / 256.0
        return np.array([u, v, 0.95]) + np.asarray(offset)

    return _proj


# ============================================================
# GroundedObject Tests
# ============================================================

class TestGroundedObject:
    def test_create(self):
        obj = GroundedObject(
            object_id="obj_0",
            label="apple",
            position_3d=np.array([0.5, 0.3, 0.95]),
        )
        assert obj.object_id == "obj_0"
        assert obj.label == "apple"
        np.testing.assert_array_almost_equal(obj.position_3d, [0.5, 0.3, 0.95])

    def test_to_dict(self):
        obj = GroundedObject(
            object_id="obj_0",
            label="apple",
            position_3d=np.array([0.5, 0.3, 0.95]),
        )
        d = obj.to_dict()
        assert d["object_id"] == "obj_0"
        assert d["position_3d"] == [0.5, 0.3, 0.95]

    def test_default_fields(self):
        obj = GroundedObject("obj_0", "x", np.zeros(3))
        assert obj.safety_risk == "unknown"
        assert obj.body_name is None
        assert obj.observed_in_views == []


# ============================================================
# SceneModel Tests
# ============================================================

class TestSceneModel:
    def test_empty(self, scene_model):
        assert len(scene_model) == 0

    def test_add_single_view_creates_objects(self, scene_model, candidate_apple):
        proj = _mock_projector()
        added = scene_model.add_view("center", [candidate_apple], proj)
        assert added == 1
        assert len(scene_model) == 1
        obj = scene_model.objects[0]
        assert obj.label == "red apple"
        assert "center" in obj.observed_in_views

    def test_add_second_view_same_position_merges(self, scene_model, candidate_apple):
        proj = _mock_projector()
        scene_model.add_view("center", [candidate_apple], proj)

        # 第二个视角 bbox 不同但投影到几乎同一位置
        c2 = GroundedCandidate("apple", 0.85, (230, 60, 255, 82), "red fruit")
        c2.query_match_score = 0.88
        scene_model.add_view("left", [c2], proj)

        # 应该合并为同一物体 (center 接近)
        assert len(scene_model) == 1
        obj = scene_model.objects[0]
        assert len(obj.observed_in_views) == 2
        assert "center" in obj.observed_in_views
        assert "left" in obj.observed_in_views
        # 多视角确认 → 置信度提升
        assert obj.position_confidence > 0.9

    def test_add_view_different_object_creates_new(
        self, scene_model, candidate_apple, candidate_bottle
    ):
        proj = _mock_projector()
        scene_model.add_view("center", [candidate_apple, candidate_bottle], proj)
        # apple 和 bottle bbox 中心距离远 → 两个不同物体
        assert len(scene_model) == 2

    def test_add_view_no_projector(self, scene_model, candidate_apple):
        """没有 projector 时仍能添加 (position 默认 [0,0,0])."""
        added = scene_model.add_view("center", [candidate_apple], None)
        assert added == 1
        obj = scene_model.objects[0]
        np.testing.assert_array_equal(obj.position_3d, [0, 0, 0])
        assert obj.position_confidence == 0.1  # low confidence without 3D

    def test_clear(self, scene_model, candidate_apple):
        scene_model.add_view("c", [candidate_apple], _mock_projector())
        assert len(scene_model) == 1
        scene_model.clear()
        assert len(scene_model) == 0

    def test_get_best_match(self, scene_model, candidate_apple, candidate_bottle):
        scene_model.add_view("c", [candidate_apple, candidate_bottle], _mock_projector())
        best = scene_model.get_best_match(min_score=0.5)
        assert best is not None
        assert best.query_match_score == 0.9
        assert best.label == "red apple"

    def test_get_best_match_below_threshold(self, scene_model):
        c = GroundedCandidate("unknown", 0.3, (10, 10, 50, 50), "small dark")
        c.query_match_score = 0.1
        scene_model.add_view("c", [c], _mock_projector())
        best = scene_model.get_best_match(min_score=0.5)
        assert best is None

    def test_get_sorted_matches(self, scene_model, candidate_apple, candidate_bottle):
        scene_model.add_view("c", [candidate_apple, candidate_bottle], _mock_projector())
        matches = scene_model.get_sorted_matches()
        assert len(matches) == 2
        assert matches[0].query_match_score >= matches[1].query_match_score

    def test_multi_view_position_weighted_average(self, scene_model):
        """多视角融合: 3D 位置应该是加权平均."""
        c1 = GroundedCandidate("cup", 0.9, (128, 128, 140, 140), "white")

        # 第一个视角投影到 (0.5, 0.5, 0.95)
        proj1 = lambda bbox: np.array([0.5, 0.5, 0.95])
        scene_model.add_view("v1", [c1], proj1)

        # 第二个视角投影到 (0.52, 0.48, 0.96) — 距离 < 0.15m → 合并
        c2 = GroundedCandidate("cup", 0.7, (125, 130, 138, 142), "white ceramic")
        proj2 = lambda bbox: np.array([0.52, 0.48, 0.96])
        scene_model.add_view("v2", [c2], proj2)

        assert len(scene_model) == 1
        pos = scene_model.objects[0].position_3d
        # 加权平均: (0.9*0.5 + 0.7*0.52) / 1.6, (0.9*0.5 + 0.7*0.48) / 1.6
        expected_x = (0.9 * 0.5 + 0.7 * 0.52) / (0.9 + 0.7)
        assert abs(pos[0] - expected_x) < 0.01


# ============================================================
# 3D 投影 Tests
# ============================================================

class TestDepthConversion:
    def test_depth_buffer_near(self):
        """z_buffer=0 → 应该接近 near."""
        d = depth_buffer_to_meters(0.0, extent=20.0, znear_ratio=0.001, zfar_ratio=50.0)
        assert abs(d - 0.02) < 0.001  # near = 0.001 * 20 = 0.02

    def test_depth_buffer_far(self):
        """z_buffer=1.0 → 应该等于 far."""
        d = depth_buffer_to_meters(1.0, extent=20.0, znear_ratio=0.001, zfar_ratio=50.0)
        assert d == 1000.0  # far = 50 * 20

    def test_depth_buffer_mid(self):
        """中间 z_buffer 应该给出合理距离."""
        d = depth_buffer_to_meters(0.98, extent=19.594, znear_ratio=0.001, zfar_ratio=50.0)
        # 应在 1-2m 范围 (桌面距离)
        assert 0.5 < d < 5.0


class TestComputeIntrinsics:
    def test_fovy_45_256(self):
        """fovy=45°, 256x256 → fx=fy≈309.02."""
        K = compute_intrinsics(45.0, 256, 256)
        assert abs(K[0, 0] - 309.02) < 0.1
        assert abs(K[1, 1] - 309.02) < 0.1
        assert K[0, 2] == 128.0
        assert K[1, 2] == 128.0

    def test_fovy_60_256(self):
        """fovy=60°, 256x256 → fx=fy≈221.7."""
        K = compute_intrinsics(60.0, 256, 256)
        assert abs(K[0, 0] - 221.7) < 0.1


class TestProjectBboxToWorld:
    def test_center_pixel_identity_camera(self):
        """相机在原点朝 -z, bbox 中心在图像中心 → world (0, 0, -z)."""
        bbox = (126, 126, 130, 130)  # center at (128, 128)
        # fake depth: all 0.99 → ~ 2m
        depth = np.full((256, 256), 0.99, dtype=np.float32)
        K = compute_intrinsics(45.0, 256, 256)
        cam_pos = np.array([0.0, 0.0, 0.0])
        cam_rot = np.eye(3)

        pt = project_bbox_to_world(
            bbox, depth, K, cam_pos, cam_rot,
            extent=20.0, znear_ratio=0.001, zfar_ratio=50.0,
        )
        assert pt is not None
        # 中心像素, 反投影后 x≈0, y≈0
        assert abs(pt[0]) < 0.01
        assert abs(pt[1]) < 0.01
        # z 应该是负的 (looks along -z)
        assert pt[2] < 0

    def test_far_plane_returns_none(self):
        """深度在 far plane (z_buffer=1.0) 时返回 None."""
        bbox = (126, 126, 130, 130)
        depth = np.ones((256, 256), dtype=np.float32)  # all at far
        K = compute_intrinsics(45.0, 256, 256)

        pt = project_bbox_to_world(
            bbox, depth, K, np.zeros(3), np.eye(3),
            extent=20.0, znear_ratio=0.001, zfar_ratio=50.0,
        )
        assert pt is None

    def test_3d_depth_squeezed(self):
        """depth shape (H,W,1) 自动 squeeze."""
        bbox = (126, 126, 130, 130)
        depth = np.full((256, 256, 1), 0.98, dtype=np.float32)
        K = compute_intrinsics(45.0, 256, 256)

        pt = project_bbox_to_world(
            bbox, depth, K, np.zeros(3), np.eye(3),
            extent=20.0, znear_ratio=0.001, zfar_ratio=50.0,
        )
        assert pt is not None

    def test_bbox_at_edge(self):
        """bbox 在图像边缘也能正确投影 (不越界)."""
        bbox = (240, 240, 256, 256)  # bottom-right corner
        depth = np.full((256, 256), 0.98, dtype=np.float32)
        K = compute_intrinsics(45.0, 256, 256)

        pt = project_bbox_to_world(
            bbox, depth, K, np.zeros(3), np.eye(3),
            extent=20.0, znear_ratio=0.001, zfar_ratio=50.0,
        )
        assert pt is not None
        # 右下角 → x 应该是正的, y 应该是负的 (v>cy → y_cam<0)
        assert pt[0] > 0


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_candidates(self, scene_model):
        added = scene_model.add_view("c", [], _mock_projector())
        assert added == 0
        assert len(scene_model) == 0

    def test_projector_returns_none(self, scene_model):
        """projector 返回 None 时, 物体仍被创建 (position 为 0)."""
        c = GroundedCandidate("x", 0.5, (10, 10, 50, 50), "")

        def fail_proj(bbox):
            return None

        added = scene_model.add_view("c", [c], fail_proj)
        assert added == 1
        assert scene_model.objects[0].position_confidence == 0.1
