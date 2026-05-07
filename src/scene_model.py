"""SafeQuery-VLM Phase 3: 多视角 3D 场景模型

聚合多个视角的 VLM grounding 结果, 通过 depth 反投影融合到统一的 3D 世界坐标.

核心设计:
    1. GroundedObject: 多视角融合后的统一物体表示
    2. SceneModel: 增量式 add_view, 自动空间对齐与合并
    3. project_bbox_to_world: 2D bbox → 深度采样 → 相机坐标 → 世界坐标

探测参数 (Phase 1 probes):
    - fovy=45°, fx=fy=309.02, 256x256
    - depth buffer [0,1] → near/(1 - z*(1-near/far))
    - MuJoCo cam: x→right, y→up, z→back

使用示例:
    >>> from src.scene_model import SceneModel, project_bbox_to_world
    >>> model = SceneModel()
    >>> model.add_view("center", candidates, projector_fn)
    >>> best = model.get_best_match("帮我拿苹果")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .vlm_grounding import GroundedCandidate

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class GroundedObject:
    """多视角融合后的统一物体表示.

    由 SceneModel.add_view() 逐步构建, 每增加一个视角就更新.
    """
    # Identity
    object_id: str                              # "obj_0", 局部自增
    label: str                                  # 最佳英文名 (来自最高 confidence 视角)

    # 3D Grounding
    position_3d: np.ndarray                     # 3D world coord (best estimate)
    position_confidence: float = 0.0            # 置信度 (多视角确认提升)

    # 多视角来源
    observed_in_views: list[str] = field(default_factory=list)
    per_view_bbox: dict[str, tuple] = field(default_factory=dict)
    per_view_position: dict[str, np.ndarray] = field(default_factory=dict)
    per_view_features: dict[str, str] = field(default_factory=dict)
    per_view_confidence: dict[str, float] = field(default_factory=dict)

    # Query 匹配结果 (由 match_query 填充)
    query_match_score: float = 0.0
    matched_category: str = ""
    match_method: str = ""

    # Safety (Phase 4 填充)
    safety_risk: str = "unknown"
    safety_reason: str = ""

    # Sim ground truth (可选)
    body_name: Optional[str] = None
    category_gt: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "label": self.label,
            "position_3d": self.position_3d.tolist(),
            "position_confidence": self.position_confidence,
            "observed_in_views": self.observed_in_views,
            "per_view_bbox": {k: list(v) for k, v in self.per_view_bbox.items()},
            "query_match_score": self.query_match_score,
            "matched_category": self.matched_category,
            "safety_risk": self.safety_risk,
        }


# ============================================================
# 3D 投影
# ============================================================

def depth_buffer_to_meters(
    z_buffer: float,
    extent: float,
    znear_ratio: float,
    zfar_ratio: float,
) -> float:
    """MuJoCo depth buffer [0,1] → 真实距离 (m).

    MuJoCo uses inverted perspective: z_buf = (far - d) / (far - near)
    所以: d = near / (1 - z_buf * (1 - near/far))

    Args:
        z_buffer: 深度缓冲值 [0,1]
        extent: sim.model.stat.extent
        znear_ratio: sim.model.vis.map.znear (默认 0.001)
        zfar_ratio: sim.model.vis.map.zfar (默认 50.0)

    Returns:
        真实距离 (m), 在相机光轴方向
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
    """2D bbox 中心 → 深度采样 → 3D 世界坐标.

    MuJoCo 相机坐标系: x→right, y→up, z→back (looks along -z).
    图像坐标: u→right, v→down. 所以 y_cam = -(v - cy) * z / fy.

    Args:
        bbox_2d: (x1, y1, x2, y2) 像素坐标
        depth_image: HxW 或 HxWx1 深度缓冲 [0,1]
        K: 3x3 内参矩阵
        cam_pos_world: (3,) 相机世界位置
        cam_rot_world: (3,3) 相机世界旋转矩阵
        extent, znear_ratio, zfar_ratio: MuJoCo 深度参数
        image_size: 图像边长

    Returns:
        (3,) 世界坐标, 或 None (深度无效时)
    """
    x1, y1, x2, y2 = bbox_2d
    u = (x1 + x2) / 2.0
    v = (y1 + y2) / 2.0

    # 确保在图像范围内
    u = max(0, min(image_size - 1, u))
    v = max(0, min(image_size - 1, v))

    # 深度采样
    depth = depth_image
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    z_buffer = float(depth[int(v), int(u)])
    if z_buffer >= 1.0:
        logger.warning(f"[project] depth at ({u:.0f},{v:.0f}) is at far plane, skipping")
        return None

    real_z = depth_buffer_to_meters(z_buffer, extent, znear_ratio, zfar_ratio)

    # pinhole 反投影: 像素 → 相机系
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy_k = float(K[0, 2]), float(K[1, 2])

    x_cam = (u - cx) * real_z / fx
    y_cam = -(v - cy_k) * real_z / fy   # v→down → y→up
    z_cam = -real_z                       # looks along -z

    pt_cam = np.array([x_cam, y_cam, z_cam])
    pt_world = cam_rot_world @ pt_cam + cam_pos_world

    return pt_world.astype(np.float32)


def compute_intrinsics(fovy_deg: float, height: int, width: int) -> np.ndarray:
    """从 fovy 计算 3x3 内参矩阵 K."""
    fy = 0.5 * height / np.tan(0.5 * np.radians(fovy_deg))
    fx = fy  # 正方形像素
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


# ============================================================
# 核心类: SceneModel
# ============================================================

class SceneModel:
    """多视角 VLM 检测结果的 3D 融合模型.

    用法:
        1. 每拍一个视角, 调 add_view() 加入候选
        2. 自动空间对齐 + 合并相近物体
        3. 调 get_best_match() 获取最佳匹配
    """

    def __init__(self, alignment_threshold_m: float = 0.15):
        """
        Args:
            alignment_threshold_m: 两个 3D 点间距小于此值则认为是同一物体
        """
        self._objects: list[GroundedObject] = []
        self._threshold = alignment_threshold_m
        self._next_id = 0

    def __len__(self) -> int:
        return len(self._objects)

    @property
    def objects(self) -> list[GroundedObject]:
        return self._objects

    def clear(self) -> None:
        """清空场景模型 (新 episode)."""
        self._objects.clear()
        self._next_id = 0

    def add_view(
        self,
        viewpoint_name: str,
        candidates: list[GroundedCandidate],
        projector: Optional[Callable[[tuple], Optional[np.ndarray]]] = None,
    ) -> int:
        """将单个视角的 VLM 检测结果加入模型.

        Args:
            viewpoint_name: 视角名 (e.g. "robot0_agentview_center")
            candidates: VLMGrounder.ground() 返回的候选列表
            projector: bbox → 3D world coord 的投影函数
                       签名: projector(bbox_2d) -> np.ndarray(3,) or None

        Returns:
            新增物体数量
        """
        added = 0

        for c in candidates:
            # 投影到 3D
            pos_3d = None
            if projector is not None:
                try:
                    pos_3d = projector(c.bbox_2d)
                except Exception as e:
                    logger.warning(f"[scene_model] projection failed for {c.label}: {e}")

            # 尝试与已有物体对齐合并
            merged = False
            if pos_3d is not None:
                for obj in self._objects:
                    dist = np.linalg.norm(obj.position_3d - pos_3d)
                    if dist < self._threshold:
                        # 合并: 更新已有物体
                        self._merge_into(obj, viewpoint_name, c, pos_3d)
                        merged = True
                        break

            if not merged:
                # 创建新物体
                obj_id = f"obj_{self._next_id}"
                self._next_id += 1

                new_obj = GroundedObject(
                    object_id=obj_id,
                    label=c.label,
                    position_3d=pos_3d if pos_3d is not None else np.zeros(3),
                    position_confidence=c.confidence if pos_3d is not None else 0.1,
                    observed_in_views=[viewpoint_name],
                    per_view_bbox={viewpoint_name: c.bbox_2d},
                    per_view_position={viewpoint_name: pos_3d} if pos_3d is not None else {},
                    per_view_features={viewpoint_name: c.visible_features},
                    per_view_confidence={viewpoint_name: c.confidence},
                    query_match_score=c.query_match_score,
                    matched_category=c.matched_category,
                    match_method=c.match_method,
                )
                self._objects.append(new_obj)
                added += 1

        logger.info(
            f"[scene_model] add_view '{viewpoint_name}': "
            f"{len(candidates)} candidates → {added} new, "
            f"{len(candidates) - added} merged. total={len(self._objects)}"
        )
        return added

    @staticmethod
    def _merge_into(
        obj: GroundedObject,
        viewpoint_name: str,
        candidate: GroundedCandidate,
        pos_3d: np.ndarray,
    ) -> None:
        """将新视角的候选合并到已有 GroundedObject."""
        obj.observed_in_views.append(viewpoint_name)
        obj.per_view_bbox[viewpoint_name] = candidate.bbox_2d
        obj.per_view_position[viewpoint_name] = pos_3d
        obj.per_view_features[viewpoint_name] = candidate.visible_features
        obj.per_view_confidence[viewpoint_name] = candidate.confidence

        # 更新 3D 位置: 加权平均 (置信度权重)
        total_weight = 0.0
        weighted_pos = np.zeros(3)
        for vn, p in obj.per_view_position.items():
            w = obj.per_view_confidence.get(vn, 0.5)
            weighted_pos += w * p
            total_weight += w
        if total_weight > 0:
            obj.position_3d = (weighted_pos / total_weight).astype(np.float32)

        # 更新置信度: 多视角确认 → boost
        n_views = len(obj.observed_in_views)
        max_conf = max(obj.per_view_confidence.values())
        obj.position_confidence = min(1.0, max_conf + 0.1 * (n_views - 1))

        # 更新 label: 取最高置信度视角的 label
        best_view = max(obj.per_view_confidence, key=obj.per_view_confidence.get)
        obj.label = obj.per_view_features.get(best_view, obj.label)
        # 不, label 应该保持语义名, features 才是描述
        # 重新从候选中取 label
        if candidate.confidence >= max_conf:
            obj.label = candidate.label

        # 更新 query match (取最高)
        if candidate.query_match_score > obj.query_match_score:
            obj.query_match_score = candidate.query_match_score
            obj.matched_category = candidate.matched_category
            obj.match_method = candidate.match_method

    def get_best_match(self, min_score: float = 0.3) -> Optional[GroundedObject]:
        """返回 query_match_score 最高的物体 (大于 min_score).

        Args:
            min_score: 最低匹配分, 低于此值不返回

        Returns:
            最佳匹配物体, 或 None
        """
        valid = [o for o in self._objects if o.query_match_score >= min_score]
        if not valid:
            return None
        return max(valid, key=lambda o: o.query_match_score)

    def get_sorted_matches(self, min_score: float = 0.0) -> list[GroundedObject]:
        """返回所有物体按 query_match_score 降序."""
        result = [o for o in self._objects if o.query_match_score >= min_score]
        result.sort(key=lambda o: o.query_match_score, reverse=True)
        return result


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("[SceneModel] 模块加载测试")

    # 模拟候选
    c1 = GroundedCandidate("apple", 0.9, (235, 65, 256, 85), "red, round")
    c1.query_match_score = 0.9
    c1.matched_category = "apple"

    # 模拟投影
    def mock_proj(bbox):
        return np.array([0.5, 0.3, 0.95])

    model = SceneModel()
    model.add_view("center", [c1], mock_proj)
    print(f"Objects: {len(model)}")
    for o in model.objects:
        print(f"  {o.object_id}: {o.label} at {o.position_3d} conf={o.position_confidence}")
