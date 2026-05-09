"""3D 投影工具函数 (从 scene_model.py 拆出, 算法不变)。

保留范围:
- depth_buffer_to_meters
- project_bbox_to_world
- compute_intrinsics

删除 (v1 用 Hypothesis + WorldBelief.merge_hypothesis 替代):
- @dataclass GroundedObject
- class SceneModel (add_view / merge / get_best_match / ...)

依赖关系:
- src/env_wrapper.py 通过 lazy import 使用 project_bbox_to_world / compute_intrinsics
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def depth_buffer_to_meters(
    z_buffer: float,
    extent: float,
    znear_ratio: float,
    zfar_ratio: float,
) -> float:
    """MuJoCo depth buffer [0,1] → 真实距离 (m)。

    MuJoCo uses inverted perspective: z_buf = (far - d) / (far - near)
    所以: d = near / (1 - z_buf * (1 - near/far))
    """
    near = znear_ratio * extent
    far = zfar_ratio * extent
    if z_buffer >= 1.0:
        return far
    return near / (1.0 - z_buffer * (1.0 - near / far))


def project_bbox_to_world(
    bbox_2d: tuple[int, int, int, int],
    depth_image: np.ndarray,
    K: np.ndarray,
    cam_pos_world: np.ndarray,
    cam_rot_world: np.ndarray,
    extent: float,
    znear_ratio: float,
    zfar_ratio: float,
    image_size: int = 256,
) -> Optional[np.ndarray]:
    """2D bbox 中心 → 深度采样 → 3D 世界坐标。

    MuJoCo 相机坐标系: x→right, y→up, z→back (looks along -z)。
    图像坐标: u→right, v→down。所以 y_cam = -(v - cy) * z / fy。
    """
    x1, y1, x2, y2 = bbox_2d
    u = (x1 + x2) / 2.0
    v = (y1 + y2) / 2.0

    u = max(0, min(image_size - 1, u))
    v = max(0, min(image_size - 1, v))

    depth = depth_image
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    z_buffer = float(depth[int(v), int(u)])
    if z_buffer >= 1.0:
        logger.warning(
            f"[project] depth at ({u:.0f},{v:.0f}) is at far plane, skipping",
        )
        return None

    real_z = depth_buffer_to_meters(z_buffer, extent, znear_ratio, zfar_ratio)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy_k = float(K[0, 2]), float(K[1, 2])

    x_cam = (u - cx) * real_z / fx
    y_cam = -(v - cy_k) * real_z / fy
    z_cam = -real_z

    pt_cam = np.array([x_cam, y_cam, z_cam])
    pt_world = cam_rot_world @ pt_cam + cam_pos_world

    return pt_world.astype(np.float32)


def compute_intrinsics(fovy_deg: float, height: int, width: int) -> np.ndarray:
    """从 fovy 计算 3x3 内参矩阵 K。"""
    fy = 0.5 * height / np.tan(0.5 * np.radians(fovy_deg))
    fx = fy  # 正方形像素
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
