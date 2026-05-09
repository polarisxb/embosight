"""src/projection.py 单元测试 (从老 tests/test_scene_model.py 迁移的纯函数部分)。

仅覆盖 3 个公开函数:
- depth_buffer_to_meters
- compute_intrinsics
- project_bbox_to_world (with simple identity rotation)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.projection import (
    compute_intrinsics,
    depth_buffer_to_meters,
    project_bbox_to_world,
)


class TestDepthBuffer:
    def test_zero_buffer_returns_near(self):
        """z_buffer=0 时应返回近平面距离。"""
        result = depth_buffer_to_meters(0.0, extent=10.0,
                                        znear_ratio=0.001, zfar_ratio=50.0)
        assert result == pytest.approx(0.01, abs=1e-6)

    def test_one_buffer_returns_far(self):
        """z_buffer=1 时返回远平面 (clamp)。"""
        result = depth_buffer_to_meters(1.0, extent=10.0,
                                        znear_ratio=0.001, zfar_ratio=50.0)
        assert result == pytest.approx(500.0, abs=1e-3)

    def test_mid_buffer(self):
        """z_buffer=0.5 时介于 near 和 far 之间。"""
        result = depth_buffer_to_meters(0.5, extent=10.0,
                                        znear_ratio=0.001, zfar_ratio=50.0)
        assert 0.01 < result < 500.0


class TestIntrinsics:
    def test_basic_intrinsics(self):
        """fovy=90, 256x256 → cx=cy=128, fx=fy=128。"""
        K = compute_intrinsics(90.0, 256, 256)
        assert K.shape == (3, 3)
        assert K[0, 2] == pytest.approx(128.0)
        assert K[1, 2] == pytest.approx(128.0)
        assert K[0, 0] == pytest.approx(K[1, 1])  # 正方形像素

    def test_narrow_fov_yields_larger_focal(self):
        """fovy 越小, focal length 越大。"""
        K_wide = compute_intrinsics(90.0, 256, 256)
        K_narrow = compute_intrinsics(45.0, 256, 256)
        assert K_narrow[0, 0] > K_wide[0, 0]


class TestProjectBboxToWorld:
    def test_center_pixel_identity_camera(self):
        """相机位姿为 identity, bbox 中心点投影到 -z 方向。"""
        # 256x256 depth 全部为 0.5 (mid range)
        depth = np.full((256, 256), 0.5, dtype=np.float32)
        K = compute_intrinsics(90.0, 256, 256)
        cam_pos = np.zeros(3, dtype=np.float32)
        cam_rot = np.eye(3, dtype=np.float32)

        bbox = (120, 120, 136, 136)  # 中心 (128, 128)
        result = project_bbox_to_world(
            bbox, depth, K, cam_pos, cam_rot,
            extent=10.0, znear_ratio=0.001, zfar_ratio=50.0,
            image_size=256,
        )
        assert result is not None
        # 中心像素 → 相机系 (0, 0, -depth) → world 系不变
        assert result[0] == pytest.approx(0.0, abs=1e-3)
        assert result[1] == pytest.approx(0.0, abs=1e-3)
        assert result[2] < 0.0  # 沿 -z

    def test_far_plane_returns_none(self):
        """深度 = 1.0 (far plane) 时返回 None。"""
        depth = np.full((256, 256), 1.0, dtype=np.float32)
        K = compute_intrinsics(90.0, 256, 256)
        cam_pos = np.zeros(3, dtype=np.float32)
        cam_rot = np.eye(3, dtype=np.float32)

        result = project_bbox_to_world(
            (120, 120, 136, 136), depth, K, cam_pos, cam_rot,
            extent=10.0, znear_ratio=0.001, zfar_ratio=50.0,
            image_size=256,
        )
        assert result is None

    def test_offset_camera_translates_world_pos(self):
        """相机位置 = (1, 2, 3) 时, world 位置应 + (1, 2, 3)。"""
        depth = np.full((256, 256), 0.5, dtype=np.float32)
        K = compute_intrinsics(90.0, 256, 256)
        cam_pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cam_rot = np.eye(3, dtype=np.float32)

        result_origin = project_bbox_to_world(
            (120, 120, 136, 136), depth, K,
            np.zeros(3, dtype=np.float32), cam_rot,
            extent=10.0, znear_ratio=0.001, zfar_ratio=50.0,
            image_size=256,
        )
        result_offset = project_bbox_to_world(
            (120, 120, 136, 136), depth, K, cam_pos, cam_rot,
            extent=10.0, znear_ratio=0.001, zfar_ratio=50.0,
            image_size=256,
        )
        diff = result_offset - result_origin
        assert diff[0] == pytest.approx(1.0, abs=1e-3)
        assert diff[1] == pytest.approx(2.0, abs=1e-3)
        assert diff[2] == pytest.approx(3.0, abs=1e-3)
