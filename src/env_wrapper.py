"""仿真环境封装（RoboCasa）

校赛 Day 2: 离散视角 + reset/observe/close
省赛增强: 真实 OSC 手臂控制 + observe 实时刷新 + 可视化支持
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """单次视角下的观察结果 (env.observe 返回)。"""
    viewpoint: Any
    image_path: str
    depth_map_path: Optional[str] = None
    description: str = ""


@dataclass
class EnvConfig:
    """环境配置"""

    sim_type: str = "robocasa"
    env_name: str = "PickPlaceCounterToCabinet"
    robots: str = "PandaMobile"
    image_width: int = 256
    image_height: int = 256
    output_dir: str = "./results/observations"
    camera_names: tuple[str, ...] = (
        "robot0_agentview_center",
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_frontview",
        "robot0_robotview",
        "robot0_eye_in_hand",
    )
    layout_ids: Optional[int] = None
    style_ids: Optional[int] = None
    seed: Optional[int] = None
    has_renderer: bool = False
    has_offscreen_renderer: bool = True


@dataclass
class ObjectGrounding:
    """用户目标 → 仿真物体的 grounding 结果"""

    user_target: str
    canonical_name: str
    sim_body_name: str
    position_m: tuple[float, float, float]
    confidence: float
    source: str  # "alias_map" | "fuzzy_match"


class EnvWrapper:
    """RoboCasa 仿真环境封装"""

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        self.config = config or EnvConfig()
        self._env = None
        self._latest_obs: dict[str, Any] = {}
        self._step = 0

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def reset(self) -> dict[str, Any]:
        """重置环境（首次调用时创建 robosuite 环境）

        Returns:
            初始观察字典
        """
        if self._env is None:
            if not self.config.has_renderer:
                os.environ.setdefault("MUJOCO_GL", "egl")
                os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

            import robocasa  # noqa: F401 — 注册 RoboCasa 环境到 robosuite
            import robosuite as suite

            kwargs = dict(
                env_name=self.config.env_name,
                robots=self.config.robots,
                has_renderer=self.config.has_renderer,
                has_offscreen_renderer=self.config.has_offscreen_renderer,
                use_camera_obs=True,
                camera_names=list(self.config.camera_names),
                camera_heights=self.config.image_height,
                camera_widths=self.config.image_width,
                camera_depths=True,
                control_freq=20,
            )
            if self.config.layout_ids is not None:
                kwargs["layout_ids"] = int(self.config.layout_ids)
            if self.config.style_ids is not None:
                kwargs["style_ids"] = int(self.config.style_ids)
            if self.config.seed is not None:
                kwargs["seed"] = int(self.config.seed)

            logger.info(f"创建仿真环境 {self.config.env_name}...")
            self._env = suite.make(**kwargs)

        if self.config.seed is not None:
            self._apply_seed(self.config.seed)
        self._latest_obs = self._env.reset()
        self._step = 0
        # 清除上一 episode 的物体类型缓存 (新 episode 可能随机出不同物体)
        if hasattr(self, "_obj_type_cache"):
            self._obj_type_cache = {}
        logger.info(f"环境重置完成 (cameras={list(self.config.camera_names)})")
        return self._latest_obs

    def seed(self, seed: int) -> None:
        self.config.seed = int(seed)
        self._apply_seed(self.config.seed)

    def _apply_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        if self._env is None:
            return
        seed_fn = getattr(self._env, "seed", None)
        if callable(seed_fn):
            seed_fn(seed)

    # ------------------------------------------------------------------
    # 深度图 + 相机参数 (Phase 3: 3D 投影)
    # ------------------------------------------------------------------

    def get_depth_image(self, camera_name: str = "robot0_agentview_center") -> Optional[np.ndarray]:
        """获取指定相机的深度图.

        Returns:
            HxW float32 深度缓冲 [0,1], 或 None
        """
        depth_key = f"{camera_name}_depth"
        depth = self._latest_obs.get(depth_key)
        if depth is None:
            logger.warning(f"[depth] key '{depth_key}' not in obs")
            return None
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        return depth

    def get_camera_intrinsics(self, camera_name: str = "robot0_agentview_center") -> Optional[np.ndarray]:
        """获取 3x3 内参矩阵 K.

        从 MuJoCo sim.model.cam_fovy 计算 fx, fy, cx, cy.
        """
        if self._env is None:
            return None
        try:
            sim = self._env.sim
            cam_id = sim.model.camera_name2id(camera_name)
            fovy_deg = float(sim.model.cam_fovy[cam_id])
            h, w = self.config.image_height, self.config.image_width
            fy = 0.5 * h / np.tan(0.5 * np.radians(fovy_deg))
            fx = fy
            cx, cy = w / 2.0, h / 2.0
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        except Exception as e:
            logger.error(f"[intrinsics] failed for {camera_name}: {e}")
            return None

    def get_camera_extrinsic(self, camera_name: str = "robot0_agentview_center") -> Optional[tuple[np.ndarray, np.ndarray]]:
        """获取相机世界位姿 (position, 3x3 rotation matrix).

        Returns:
            (cam_pos(3,), cam_rot(3,3)) 或 None
        """
        if self._env is None:
            return None
        try:
            sim = self._env.sim
            cam_id = sim.model.camera_name2id(camera_name)
            cam_pos = sim.data.cam_xpos[cam_id].copy().astype(np.float64)
            cam_rot = sim.data.cam_xmat[cam_id].reshape(3, 3).copy().astype(np.float64)
            return cam_pos, cam_rot
        except Exception as e:
            logger.error(f"[extrinsic] failed for {camera_name}: {e}")
            return None

    def get_depth_params(self) -> tuple[float, float, float]:
        """获取 MuJoCo depth buffer 归一化参数.

        Returns:
            (extent, znear_ratio, zfar_ratio)
        """
        sim = self._env.sim
        extent = float(sim.model.stat.extent)
        znear = float(sim.model.vis.map.znear)
        zfar = float(sim.model.vis.map.zfar)
        return extent, znear, zfar

    def make_projector(self, camera_name: str = "robot0_agentview_center"):
        """创建 bbox → 3D world 投影函数 (供 SceneModel.add_view 使用).

        Returns:
            callable: projector(bbox_2d) -> np.ndarray(3,) or None
        """
        depth = self.get_depth_image(camera_name)
        K = self.get_camera_intrinsics(camera_name)
        ext = self.get_camera_extrinsic(camera_name)

        if depth is None or K is None or ext is None:
            logger.warning(f"[projector] cannot build for {camera_name}: missing data")
            return None

        cam_pos, cam_rot = ext
        extent, znear, zfar = self.get_depth_params()
        img_size = self.config.image_width

        from .projection import project_bbox_to_world

        def _projector(bbox_2d: tuple) -> Optional[np.ndarray]:
            return project_bbox_to_world(
                bbox_2d, depth, K, cam_pos, cam_rot,
                extent, znear, zfar, img_size,
            )

        return _projector

    # ------------------------------------------------------------------
    # 手臂控制 (Phase 1)
    # ------------------------------------------------------------------

    def get_eef_pos(self) -> np.ndarray:
        """获取末端执行器当前世界坐标 (单位: m)"""
        if not self._latest_obs:
            self.reset()
        pos = self._latest_obs.get("robot0_eef_pos")
        if pos is None:
            raise RuntimeError("robot0_eef_pos not in observation keys")
        return np.asarray(pos, dtype=np.float32)

    def _get_eef_quat(self) -> np.ndarray:
        """获取末端执行器当前世界系四元数 (xyzw)。

        Used for orientation control in move_arm_to. Reads robot0_eef_quat
        from the latest observation. Robosuite Panda exposes the gripper
        quaternion in world frame in this key by default.
        """
        if not self._latest_obs:
            self.reset()
        q = self._latest_obs.get("robot0_eef_quat")
        if q is None:
            raise RuntimeError("robot0_eef_quat not in observation keys")
        return np.asarray(q, dtype=np.float64)

    # robosuite mobile robots 上, Robot.base_pos / Robot.base_ori 指向
    # mount anchor body (e.g. robot0_base, hardcoded 到 (10,10,0)) 而非
    # 实际 mobile base. 真实位置在 sim 中以 'mobilebase{idn}_base' 命名.
    # 优先级: mobilebase{idn}_base → robot{idn}_base (若非 fake) → property fallback
    _MOBILE_BASE_FAKE_XY: tuple[float, float] = (10.0, 10.0)

    def get_base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """获取底盘在世界系的 (位置, 3x3旋转矩阵)

        手臂 OSC 用 input_ref_frame='base', 底盘也是 base 系 JointVelocity,
        因此所有 action 都是 base 系增量, 需要把世界系 delta 旋转回 base 系.

        实现说明: robosuite 的 Robot.base_pos 在 PandaOmron / PandaMobile 等
        mobile 机器人上指向 anchor body (xpos = (10,10,0)), 不是真实底盘位置.
        先按 sim body name 优先级取真实 xpos/xmat, 实在拿不到才回退到 property.
        """
        sim = getattr(self._env, "sim", None)
        if sim is not None:
            try:
                robot = self._env.robots[0]
                idn = getattr(robot, "idn", 0)
            except Exception:
                idn = 0
            for body_name in (
                f"mobilebase{idn}_base", f"robot{idn}_base",
            ):
                try:
                    bid = sim.model.body_name2id(body_name)
                except (KeyError, ValueError):
                    continue
                pos = np.asarray(sim.data.body_xpos[bid], dtype=np.float32)
                # 跳过 anchor body 的 fake (10,10,0) 位置
                fake_xy = np.asarray(self._MOBILE_BASE_FAKE_XY, dtype=np.float32)
                if np.allclose(pos[:2], fake_xy, atol=0.01):
                    continue
                mat = np.asarray(
                    sim.data.body_xmat[bid], dtype=np.float32,
                ).reshape(3, 3)
                return pos, mat

        # Fallback: robosuite property (legacy behavior, may be inaccurate)
        try:
            robot = self._env.robots[0]
            base_pos = np.asarray(robot.base_pos, dtype=np.float32)
            base_ori = np.asarray(robot.base_ori, dtype=np.float32)  # 3x3
            return base_pos, base_ori
        except Exception as e:
            logger.warning(f"[base_pose] fallback to identity: {e}")
            return np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32)

    def world_to_base_vec(self, vec_world: np.ndarray) -> np.ndarray:
        """世界系向量 → 底盘局部系 (R.T @ v)"""
        _, base_ori = self.get_base_pose()
        return base_ori.T @ np.asarray(vec_world, dtype=np.float32)

    def render(self) -> None:
        """如果 has_renderer 则刷新可视化窗口"""
        if self.config.has_renderer and self._env is not None:
            try:
                self._env.render()
            except Exception as e:
                logger.warning(f"[render] {e}")

    ARM_STEP_CAP = 0.15  # 手臂 OSC 单步增量上限 (base 系)
    BASE_XY_THRESHOLD = 0.25  # 底盘主导 → 手臂主导的切换距离 (m)

    def _get_base_action_idx(self) -> Optional[int]:
        """动态获取 base controller 在 action vector 中的起始 index"""
        if hasattr(self, "_base_idx_cache"):
            return self._base_idx_cache
        try:
            robot = self._env.robots[0]
            idx = 0
            parts_info = []
            base_idx = None
            base_dim = 0
            for part_name, ctrl in robot.composite_controller.part_controllers.items():
                dim = ctrl.control_dim
                parts_info.append(f"{part_name}[{idx}:{idx+dim}]")
                pn = part_name.lower()
                if base_idx is None and ("base" in pn or "mobile" in pn):
                    base_idx = idx
                    base_dim = dim
                idx += dim
            logger.info(
                f"[base] action layout: {', '.join(parts_info)} "
                f"| total={idx}"
            )
            if base_idx is not None:
                self._base_idx_cache = base_idx
                logger.info(
                    f"[base] detected index={base_idx} dim={base_dim}"
                )
                return base_idx
        except Exception as e:
            logger.warning(f"[base] auto-detect failed ({e})")
        self._base_idx_cache = None
        return None

    # ------------------------------------------------------------------
    # Orientation control helpers (Tasks 2-4 of orientation-aware-grasping)
    # ------------------------------------------------------------------

    @staticmethod
    def _approach_dir_to_quat(approach_dir: np.ndarray) -> np.ndarray:
        """Compute world-frame gripper quaternion so its local +z axis points
        in `approach_dir` (i.e., the gripper "looks at" the object).

        The gripper's home orientation in robosuite Panda has local +z pointing
        forward along the kinematic chain. For top-down grasping, approach_dir
        = [0, 0, -1] makes the gripper face downward. For side approach from
        +x, approach_dir = [1, 0, 0] makes it face +x.

        Args:
            approach_dir: 3D vector in world frame (will be normalized).

        Returns:
            Quaternion (xyzw, unit length, float64) representing the target
            gripper rotation in world frame.
        """
        from scipy.spatial.transform import Rotation as R

        v = np.asarray(approach_dir, dtype=np.float64)
        norm = float(np.linalg.norm(v))
        if norm < 1e-9:
            return np.array([0.0, 0.0, 0.0, 1.0])
        v = v / norm
        z_axis = np.array([0.0, 0.0, 1.0])
        cross = np.cross(z_axis, v)
        dot = float(np.dot(z_axis, v))
        if dot > 1.0 - 1e-9:
            # Already aligned with +z
            return np.array([0.0, 0.0, 0.0, 1.0])
        if dot < -1.0 + 1e-9:
            # Anti-parallel: rotate 180° around any axis orthogonal to z.
            # Pick +x as a convention.
            return R.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0])).as_quat()
        s = float(np.sqrt(2.0 * (1.0 + dot)))
        q_xyz = cross / s
        q_w = s / 2.0
        return np.array([q_xyz[0], q_xyz[1], q_xyz[2], q_w])

    @staticmethod
    def _quat_delta_to_axis_angle(
        q_current: np.ndarray, q_target: np.ndarray,
    ) -> np.ndarray:
        """Compute axis-angle (rotation vector) to rotate from `q_current` to
        `q_target`, taking the shortest path.

        Both quaternions are in xyzw convention. The output is a 3-vector
        suitable for OSC_POSE `action[3:6]` (after frame conversion if needed).

        Args:
            q_current: current orientation quaternion (xyzw).
            q_target:  target orientation quaternion (xyzw).

        Returns:
            3-vector rotation in world frame (same frame as both quats).
        """
        from scipy.spatial.transform import Rotation as R

        q_c = np.asarray(q_current, dtype=np.float64)
        q_t = np.asarray(q_target, dtype=np.float64)
        # Pick shortest path: if dot < 0, negate target
        if np.dot(q_c, q_t) < 0:
            q_t = -q_t
        r_c = R.from_quat(q_c)
        r_t = R.from_quat(q_t)
        # q_delta = q_target * inv(q_current)
        r_delta = r_t * r_c.inv()
        return r_delta.as_rotvec().astype(np.float64)

    def move_arm_to(
        self,
        target_pos_m,
        max_steps: int = 800,
        threshold_m: float = 0.02,
        approach_dir: Optional[np.ndarray] = None,
        ori_gain: float = 1.0,
        ori_threshold_rad: float = 0.15,
        gripper_hold: float = 0.0,
    ) -> bool:
        """自适应控制: 世界系目标 → base 系增量 → 手臂+底盘协同

        关键修正 (相比之前):
            - 手臂 OSC `input_ref_frame='base'`: action[0:3] 是 base 系增量
            - 底盘 JointVelocity (forward/side): 也是 base 系速度
            - 底盘 action index 通过 _get_base_action_idx() 动态检测
            - 因此世界系 delta 必须先旋转到 base 系才能用作 action

        策略:
            每步重读 base_ori (因为底盘可能旋转), 把世界系 delta
            旋转到当前 base 系, 同时驱动手臂和底盘. 步数按距离动态分配.
            底盘增益 0.8 (OmronMobileBase frictionloss=250, kv=1000).

        Args:
            target_pos_m: 3D target position in world frame.
            approach_dir: optional unit vector; if provided, gripper orientation
                is driven so its local +z axis aligns with approach_dir.
            ori_gain: scaling factor for orientation action delta.
            ori_threshold_rad: orientation convergence threshold (radians).
                When approach_dir is given, loop continues until BOTH
                position AND orientation converge (or max_steps hit).
            gripper_hold: per-step gripper command (0 = neutral, 1.0 = keep
                closed). Default 0 preserves legacy behavior. Set to 1.0
                during lift / retreat after a successful close_gripper to
                prevent the object from slipping out — RoboCasa's gripper
                releases force when the action position is 0.

        Returns:
            True if converged within threshold
        """
        if not self._latest_obs:
            self.reset()

        target = np.asarray(target_pos_m, dtype=np.float32)
        if target.shape[0] > 3:
            target = target[:3]
        action_dim = self._env.action_dim
        base_idx = self._get_base_action_idx()
        has_base = base_idx is not None

        # ── Orientation target (Task 5) ──
        target_quat: Optional[np.ndarray] = None
        if approach_dir is not None:
            ad = np.asarray(approach_dir, dtype=np.float64)
            if np.linalg.norm(ad) > 1e-6:
                target_quat = self._approach_dir_to_quat(ad)

        init_dist = float(np.linalg.norm(target - self.get_eef_pos()))
        if init_dist < threshold_m and target_quat is None:
            return True
        if init_dist > 0.5:
            max_steps = max(max_steps, int(init_dist * 1500))
        logger.debug(
            f"[move] target={target}, init_dist={init_dist:.3f}m, "
            f"max_steps={max_steps}, base_idx={base_idx}, "
            f"approach_dir={approach_dir}"
        )

        prev_dist = float("inf")
        prev_ori_err = float("inf")
        stall = 0
        check_interval = 120  # 每 N 步检查一次 stall
        stall_limit = 6
        max_ori_step_per_iter = 0.5  # 单步朝向 axis-angle 模长上限 (rad)

        for step in range(max_steps):
            current = self.get_eef_pos()
            delta_world = target - current
            dist = float(np.linalg.norm(delta_world))

            # ── 朝向误差 ──
            ori_err = 0.0
            if target_quat is not None:
                try:
                    q_cur = self._get_eef_quat()
                    ori_delta_world = self._quat_delta_to_axis_angle(
                        q_cur, target_quat
                    )
                    ori_err = float(np.linalg.norm(ori_delta_world))
                except Exception:
                    ori_err = float("inf")

            # ── 双重收敛检查 ──
            pos_ok = dist < threshold_m
            ori_ok = ori_err < ori_threshold_rad if target_quat is not None else True
            if pos_ok and ori_ok:
                logger.debug(
                    f"[move] converged step={step} dist={dist:.4f}m "
                    f"ori_err={ori_err:.4f}rad"
                )
                return True

            # stall 检测: 位置 OR 朝向有进展就不算 stall
            if step > 0 and step % check_interval == 0:
                pos_progress = prev_dist - dist
                ori_progress = prev_ori_err - ori_err if target_quat is not None else 0.0
                making_progress = pos_progress > 0.005 or ori_progress > 0.01
                if not making_progress:
                    stall += 1
                    if stall >= stall_limit:
                        logger.warning(
                            f"[move_arm_to] stalled at step={step}, "
                            f"dist={dist:.4f}m ori_err={ori_err:.4f}rad"
                        )
                        return pos_ok and ori_ok
                else:
                    stall = max(0, stall - 1)
                prev_dist = dist
                prev_ori_err = ori_err
                if step % (check_interval * 3) == 0:
                    logger.debug(
                        f"[move] step={step} dist={dist:.3f}m "
                        f"ori={ori_err:.3f}rad stall={stall}/{stall_limit}"
                    )

            # 世界系 → base 系 (核心修正)
            _, base_ori = self.get_base_pose()
            delta_base = base_ori.T @ delta_world  # 3D vector in base frame

            dir_base = delta_base / max(dist, 1e-6)
            step_size = min(self.ARM_STEP_CAP, dist)

            action = np.zeros(action_dim, dtype=np.float32)
            # 手臂位置: base 系增量 (只有还没到位时才驱动)
            if not pos_ok:
                action[0:3] = dir_base * step_size

            # 手臂朝向: 把世界系 axis-angle 转到 base 系, 再 clamp + scale
            if target_quat is not None and ori_err > ori_threshold_rad:
                try:
                    # ori_delta_world 已在上面计算
                    ori_delta_base = base_ori.T @ ori_delta_world
                    # Clamp per-step magnitude
                    if ori_err > max_ori_step_per_iter:
                        ori_delta_base = ori_delta_base * (
                            max_ori_step_per_iter / ori_err
                        )
                    action[3:6] = (ori_delta_base * ori_gain).astype(np.float32)
                except Exception as e:
                    logger.debug(f"[move_arm_to] ori control skipped: {e}")

            # 底盘: base 系 forward/side 速度
            if has_base and dist > 0.05:
                base_gain = min(0.8, dist * 0.8)
                action[base_idx] = float(dir_base[0]) * base_gain      # forward
                action[base_idx + 1] = float(dir_base[1]) * base_gain  # side

            # 夹爪保持: 抬升期间持续施力, 防止物体滑落
            if gripper_hold != 0.0:
                action[self._get_gripper_idx()] = gripper_hold

            try:
                obs, _, _, _ = self._env.step(action)
                self._latest_obs = obs
                self.render()
            except Exception as e:
                logger.warning(f"[move_arm_to] step {step} failed: {e}")
                return False

        logger.warning(
            f"[move_arm_to] max_steps reached, dist={dist:.4f}m "
            f"ori_err={ori_err:.4f}rad"
        )
        return False

    # ------------------------------------------------------------------
    # Object Grounding (Phase 2)
    # ------------------------------------------------------------------

    def _load_aliases(self) -> dict[str, list[str]]:
        import yaml

        repo_root = Path(__file__).resolve().parent.parent
        path = repo_root / "configs" / "object_aliases.yaml"
        if not path.exists():
            logger.warning(f"[grounding] alias map not found: {path}")
            return {}
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("aliases", {})

    def _get_body_pos(self, body_name: str) -> Optional[np.ndarray]:
        sim = self._env.sim
        try:
            body_id = sim.model.body_name2id(body_name)
            return np.asarray(sim.data.body_xpos[body_id], dtype=np.float32)
        except (KeyError, ValueError):
            return None

    def _get_body_subtree_ids(self, body_name: str) -> set[int]:
        sim = self._env.sim
        body_id = sim.model.body_name2id(body_name)
        ids = {body_id}
        changed = True
        while changed:
            changed = False
            for i in range(sim.model.nbody):
                parent = int(sim.model.body_parentid[i])
                if parent in ids and i not in ids:
                    ids.add(i)
                    changed = True
        return ids

    def _get_body_geom_ids(self, body_name: str) -> set[int]:
        sim = self._env.sim
        body_ids = self._get_body_subtree_ids(body_name)
        return {
            i for i in range(sim.model.ngeom)
            if int(sim.model.geom_bodyid[i]) in body_ids
        }

    def _get_body_aabb(
        self, body_name: str
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """获取 body 所有 geom 在世界系的 AABB (轴对齐包围盒)

        累加该 body 下所有 geom 的位置 ± half-extent, 取并集.
        对 sphere/cylinder 用 size[0] 作半径近似.

        Returns:
            (min_xyz, max_xyz) 或 None
        """
        sim = self._env.sim
        try:
            geom_ids = self._get_body_geom_ids(body_name)
            if not geom_ids:
                return None

            mins, maxs = [], []
            for gid in geom_ids:
                pos = np.asarray(sim.data.geom_xpos[gid], dtype=np.float32)
                size = np.asarray(sim.model.geom_size[gid], dtype=np.float32)
                geom_type = int(sim.model.geom_type[gid])

                if geom_type == 7 and hasattr(sim.model, "mesh_vert"):
                    mesh_id = int(sim.model.geom_dataid[gid])
                    if mesh_id >= 0:
                        adr = int(sim.model.mesh_vertadr[mesh_id])
                        num = int(sim.model.mesh_vertnum[mesh_id])
                        verts = np.asarray(
                            sim.model.mesh_vert[adr: adr + num], dtype=np.float32
                        )
                        xmat = np.asarray(
                            sim.data.geom_xmat[gid], dtype=np.float32
                        ).reshape(3, 3)
                        world = pos + verts @ xmat.T
                        mins.append(np.min(world, axis=0))
                        maxs.append(np.max(world, axis=0))
                        continue

                if geom_type == 2:
                    local_half = np.array([size[0], size[0], size[0]], dtype=np.float32)
                elif geom_type == 3:
                    local_half = np.array(
                        [size[0], size[0], size[1] + size[0]], dtype=np.float32
                    )
                elif geom_type == 5:
                    local_half = np.array([size[0], size[0], size[1]], dtype=np.float32)
                elif geom_type in (4, 6):
                    local_half = size.copy()
                else:
                    r = float(sim.model.geom_rbound[gid]) if hasattr(
                        sim.model, "geom_rbound"
                    ) else float(np.max(size))
                    local_half = np.array([r, r, r], dtype=np.float32)

                xmat = np.asarray(sim.data.geom_xmat[gid], dtype=np.float32).reshape(3, 3)
                world_half = np.abs(xmat) @ local_half
                mins.append(pos - world_half)
                maxs.append(pos + world_half)

            aabb_min = np.min(np.array(mins), axis=0)
            aabb_max = np.max(np.array(maxs), axis=0)
            return aabb_min.astype(np.float32), aabb_max.astype(np.float32)
        except Exception as e:
            logger.debug(f"[aabb] {body_name} failed: {e}")
            return None

    # 抓取相关常量
    FINGERTIP_OFFSET_Z = 0.0  # eef_pos 已在指尖中点 (Panda gripper 默认)
    GRASP_HEIGHT_RATIO = 0.55  # 指尖目标在物体高度 55% 处 (中部偏上, 重心稳定)

    def _compute_grasp_pose(
        self, body_name: str, fallback_pos: np.ndarray
    ) -> np.ndarray:
        """根据物体几何计算 wrist 目标位置 (世界系)

        策略 (top-down grasp):
            - AABB 给出物体真实顶/底高度
            - 指尖目标 z = z_bot + GRASP_HEIGHT_RATIO * height
            - wrist 目标 = (cx, cy, fingertip_z + FINGERTIP_OFFSET_Z)

        Returns:
            wrist 目标位置 (3D world)
        """
        fallback = np.asarray(fallback_pos, dtype=np.float32)
        aabb = self._get_body_aabb(body_name)
        if aabb is None:
            logger.debug(f"[grasp_pose] no AABB for {body_name}, fallback to body_xpos")
            return fallback

        aabb_min, aabb_max = aabb
        cx = 0.5 * (aabb_min[0] + aabb_max[0])
        cy = 0.5 * (aabb_min[1] + aabb_max[1])
        z_top = float(aabb_max[2])
        z_bot = float(aabb_min[2])
        height = z_top - z_bot

        fingertip_z = z_bot + self.GRASP_HEIGHT_RATIO * height
        wrist_z = fingertip_z + self.FINGERTIP_OFFSET_Z

        target = np.array([cx, cy, wrist_z], dtype=np.float32)
        logger.info(
            f"[grasp_pose] '{body_name}' AABB z=[{z_bot:.3f},{z_top:.3f}] "
            f"h={height:.3f}m → wrist_z={wrist_z:.3f} (was {fallback[2]:.3f})"
        )
        return target

    def _get_task_objects(self) -> list[str]:
        """获取 RoboCasa 环境中的任务物体 body name 列表

        RoboCasa 用 obj_main / distr_*_main 作为任务物体的 body name，
        每次 reset 物体种类随机。
        """
        sim_body_names = list(self._env.sim.model.body_names)
        task_objs = [
            b for b in sim_body_names
            if b.startswith("obj_") or b.startswith("distr_")
        ]
        if not task_objs:
            # fallback: 排除 robot / wall / floor / cab / stack 等固定结构
            skip_prefixes = (
                "world", "robot0", "gripper", "mobilebase",
                "manipulator", "wall", "floor", "counter",
                "cab_", "stack_", "hood_", "shelves_", "stool",
                "fridge", "dishwasher", "microwave", "sink",
                "outlet", "light_switch", "window", "left_eef",
                "right_eef", "bottom_", "top_", "box_",
                "right_corner", "micro_housing", "utensil",
                "coffee_machine", "knife_block", "plant",
                "toaster", "paper_towel",
            )
            task_objs = [
                b for b in sim_body_names
                if not any(b.startswith(p) or b.startswith(p.lower()) for p in skip_prefixes)
            ]
        logger.debug(f"[grounding] task objects: {task_objs}")
        return task_objs

    def _get_obj_type_map(self) -> dict[str, str]:
        """从 RoboCasa 的 ep_meta 提取 {body_name: object_cat} 映射.

        RoboCasa 每次 reset 后, 真实物体类型只能从 env.get_ep_meta() 取到:
            object_cfgs[i].info.cat ∈ {peeler, condiment_bottle, reamer, ...}
            object_cfgs[i].name     ∈ {obj, distr_counter, distr_cab}
            body_name = f"{name}_main"  (即 obj_main, distr_counter_main, ...)

        这是 grounding 的关键信息源, 因为 body name 本身 (obj_main) 不含类型.

        Returns:
            {'obj_main': 'peeler', 'distr_counter_main': 'condiment_bottle', ...}
            失败时返回空 dict (不阻塞其他 grounding 策略).
        """
        if not hasattr(self, "_obj_type_cache"):
            self._obj_type_cache = {}

        if self._obj_type_cache:
            return self._obj_type_cache

        try:
            meta = self._env.get_ep_meta()
        except Exception as e:
            logger.debug(f"[obj_types] get_ep_meta failed: {e}")
            return {}

        result: dict[str, str] = {}
        for cfg in meta.get("object_cfgs", []) or []:
            name = cfg.get("name")
            info = cfg.get("info", {}) or {}
            cat = info.get("cat")
            if name and cat:
                body_name = f"{name}_main"
                result[body_name] = str(cat)

        if result:
            logger.info(f"[obj_types] runtime object categories: {result}")
        self._obj_type_cache = result
        return result

    def ground_object(
        self, user_target: str, *, allow_fallback: bool = True
    ) -> Optional[ObjectGrounding]:
        """将用户目标名 grounding 到仿真物体

        搜索策略 (按优先级):
        1. alias_map 别名匹配 → confidence=0.9
        2. body name 子串模糊匹配 → confidence=0.6
        3. LLM 语义匹配 (解析 body name 中的物体类型) → confidence=0.75
        4. 回退: 返回 obj_main (仅 allow_fallback=True 时) → confidence=0.5

        Args:
            user_target: 用户目标物体名 (中文或英文)
            allow_fallback: 是否允许 fallback 到 obj_main.
                用于区分主目标 (False) 和辅助查询如危险物体 (True).

        Returns:
            ObjectGrounding with meter coordinates, or None
        """
        if not self._latest_obs:
            self.reset()

        if not hasattr(self, "_aliases"):
            self._aliases = self._load_aliases()

        sim_body_names = list(self._env.sim.model.body_names)
        task_objs = self._get_task_objects()

        # 1) 别名精确匹配 (全 body)
        candidates = self._aliases.get(user_target, [])
        for canonical in candidates:
            for body in sim_body_names:
                if canonical.lower() in body.lower():
                    pos = self._get_body_pos(body)
                    if pos is not None:
                        return ObjectGrounding(
                            user_target=user_target,
                            canonical_name=canonical,
                            sim_body_name=body,
                            position_m=tuple(pos.tolist()),
                            confidence=0.9,
                            source="alias_map",
                        )

        # 2) body name 模糊匹配 (英文输入)
        for body in sim_body_names:
            if user_target.lower() in body.lower():
                pos = self._get_body_pos(body)
                if pos is not None:
                    return ObjectGrounding(
                        user_target=user_target,
                        canonical_name=body,
                        sim_body_name=body,
                        position_m=tuple(pos.tolist()),
                        confidence=0.6,
                        source="fuzzy_match",
                    )

        # 3) LLM 语义匹配 (解析 body name 中的物体类型)
        llm_result = self._llm_semantic_grounding(user_target, task_objs)
        if llm_result is not None:
            return llm_result

        # 诊断: alias/fuzzy/LLM 全部失败时, 打印 task_objs 让用户/开发者看清
        # 场景实际有什么 body, 是否真的没有目标物体
        logger.info(
            f"[ground_object] all strategies failed for '{user_target}'. "
            f"task_objs ({len(task_objs)}): {task_objs[:30]}"
        )

        # 4) 回退: 返回 obj_main (仅辅助查询时允许)
        if allow_fallback and "obj_main" in sim_body_names:
            pos = self._get_body_pos("obj_main")
            if pos is not None:
                logger.info(
                    f"[ground_object] '{user_target}' → fallback to obj_main at {pos}"
                )
                return ObjectGrounding(
                    user_target=user_target,
                    canonical_name="obj_main",
                    sim_body_name="obj_main",
                    position_m=tuple(pos.tolist()),
                    confidence=0.5,
                    source="fallback_task_obj",
                )

        logger.warning(f"[ground_object] failed to ground '{user_target}'")
        return None

    def _llm_semantic_grounding(
        self, user_target: str, task_objs: list[str]
    ) -> Optional[ObjectGrounding]:
        """LLM 语义匹配: 解析 body name 中的物体类型, 匹配用户目标

        优势 (相比 VLM):
            - body name 解析是纯语言任务, LLM 天然擅长
            - API 调用 ~1s, 比 VLM 推理 ~15s 快一个数量级
            - 不依赖视觉, 不需要硬编码负面例子
            - 能正确区分: Juice≠cup, Turmeric≠药瓶, Banana≠碗

        Returns:
            ObjectGrounding if LLM found a semantic match, else None
        """
        if not task_objs:
            return None

        if not hasattr(self, "_llm") or self._llm is None:
            try:
                from .llm_backend import LLMBackend
                self._llm = LLMBackend(max_tokens=256, temperature=0.0)
            except Exception as e:
                logger.debug(f"[llm_grounding] LLM unavailable: {e}")
                return None

        try:
            # 运行时查真实物体类型 (RoboCasa ep_meta)
            type_map = self._get_obj_type_map()
            obj_lines = []
            for i, b in enumerate(task_objs[:20]):
                cat = type_map.get(b)
                if cat:
                    obj_lines.append(f"  {i+1}. {b} (type: {cat})")
                else:
                    obj_lines.append(f"  {i+1}. {b}")
            obj_list = "\n".join(obj_lines)

            prompt = (
                f"Task: Match a user's target object to simulation body names.\n\n"
                f"User target: '{user_target}'\n"
                f"(If Chinese: 杯子=cup/mug, 碗=bowl, 药瓶=medicine bottle, "
                f"锅=pot/pan, 盘子=plate, 瓶子=bottle, 刀=knife, 勺=spoon, "
                f"苹果=apple, 面包=bread, 水壶=kettle, 罐头=can, "
                f"削皮器=peeler, 榨汁器=reamer, 调味瓶=condiment_bottle)\n\n"
                f"Available body names:\n{obj_list}\n\n"
                f"Instructions:\n"
                f"1. Use 'type:' annotation if present (most reliable), "
                f"otherwise parse the body name string.\n"
                f"2. Find which object semantically matches '{user_target}'.\n"
                f"3. Be STRICT about semantic equivalence:\n"
                f"   - peeler≠knife (both tools but different function)\n"
                f"   - condiment_bottle≠medicine_bottle (both bottles but different content)\n"
                f"   - reamer≠juicer-as-cup (reamer is manual tool not a cup)\n"
                f"   - juice≠cup, turmeric≠medicine, banana≠bowl\n"
                f"4. If no good match exists, return NONE. Do NOT force a match.\n\n"
                f"Reply in JSON: {{\"match\": \"<exact body name>\"}} "
                f"or {{\"match\": \"NONE\"}} if no semantic match."
            )

            raw = self._llm.generate(
                user_message=prompt,
                system="You are a precise object-matching assistant. "
                       "Only match when the object types are semantically equivalent.",
                json_mode=True,
                temperature=0.0,
            )

            import json as _json
            if not raw or not raw.strip():
                logger.warning("[llm_grounding] LLM returned empty response")
                return None
            # 尝试从可能的 markdown 代码块中提取 JSON
            text = raw.strip()
            if "```" in text:
                # 去掉 ```json ... ``` 包裹
                import re
                m = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL)
                if m:
                    text = m.group(1)
            # 尝试提取第一个 JSON 对象
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]
            data = _json.loads(text)
            matched = data.get("match", "NONE").strip()
            logger.info(f"[llm_grounding] LLM match: '{matched}'")

            if not matched or matched.upper() == "NONE":
                return None

            # 验证 LLM 返回的 body name 确实在列表中
            best_body = None
            matched_lower = matched.lower()
            for body in task_objs:
                if body.lower() == matched_lower:
                    best_body = body
                    break
            if best_body is None:
                for body in task_objs:
                    if matched_lower in body.lower() or body.lower() in matched_lower:
                        best_body = body
                        break

            if best_body is not None:
                pos = self._get_body_pos(best_body)
                if pos is not None:
                    logger.info(
                        f"[llm_grounding] '{user_target}' → {best_body} at {pos}"
                    )
                    return ObjectGrounding(
                        user_target=user_target,
                        canonical_name=matched,
                        sim_body_name=best_body,
                        position_m=tuple(pos.tolist()),
                        confidence=0.75,
                        source="llm_grounding",
                    )
            else:
                logger.info(
                    f"[llm_grounding] LLM returned '{matched}' "
                    f"but not in task_objs, skipping"
                )
        except Exception as e:
            logger.warning(f"[llm_grounding] failed: {e}")
        return None

    # ------------------------------------------------------------------
    # 观察 (Phase 1)
    # ------------------------------------------------------------------

    def observe(self, viewpoint) -> Observation:
        """采集指定视角的 RGB 图像

        Args:
            viewpoint: Viewpoint 对象 (来自 active_planner)

        Returns:
            Observation 对象 (含图像路径和元数据)
        """
        if not self._latest_obs:
            self.reset()

        # 做一次 zero-action step 让所有摄像头渲染当前帧
        try:
            zero_action = np.zeros(self._env.action_dim, dtype=np.float32)
            obs, _, _, _ = self._env.step(zero_action)
            self._latest_obs = obs
            self.render()
        except Exception as e:
            logger.warning(f"[observe] zero-step failed, using stale obs: {e}")

        camera_name = viewpoint.name
        img_key = f"{camera_name}_image"
        img = self._latest_obs.get(img_key)

        if img is None:
            logger.warning(f"未找到图像 {img_key}, 用 robot0_agentview_center 代替")
            img = self._latest_obs.get("robot0_agentview_center_image")

        self._step += 1
        image_path = os.path.join(
            self.config.output_dir,
            f"step_{self._step:03d}_{camera_name}.png",
        )

        if img is not None:
            try:
                import imageio.v2 as imageio

                imageio.imwrite(image_path, img)
                logger.debug(f"图像保存: {image_path}")
            except Exception as e:
                logger.warning(f"图像保存失败: {e}")

        # 深度图保存 (Phase 3)
        depth_map_path = None
        depth_key = f"{camera_name}_depth"
        if depth_key in self._latest_obs:
            depth_path = os.path.join(
                self.config.output_dir,
                f"step_{self._step:03d}_{camera_name}_depth.npy",
            )
            try:
                np.save(depth_path, self._latest_obs[depth_key])
                depth_map_path = depth_path
            except Exception as e:
                logger.warning(f"深度图保存失败: {e}")

        return Observation(
            viewpoint=viewpoint,
            image_path=image_path,
            depth_map_path=depth_map_path,
        )

    # ------------------------------------------------------------------
    # 抓取 (Phase 3)
    # ------------------------------------------------------------------

    def _get_gripper_idx(self) -> int:
        """动态获取 gripper 在 action vector 中的 index"""
        if hasattr(self, "_gripper_idx_cache"):
            return self._gripper_idx_cache
        # composite controller: 累加 arm controller 输出维度
        try:
            robot = self._env.robots[0]
            idx = 0
            for part_name, controller in robot.composite_controller.part_controllers.items():
                if "gripper" in part_name.lower():
                    self._gripper_idx_cache = idx
                    logger.info(f"[gripper] detected index={idx} (part={part_name})")
                    return idx
                idx += controller.control_dim
        except Exception as e:
            logger.warning(f"[gripper] auto-detect failed ({e}), fallback to idx=6")
        self._gripper_idx_cache = 6
        return 6

    def _gripper_action(self, gripper_value: float, n_steps: int = 10) -> None:
        """控制夹爪 (gripper_value: -1 开, +1 关)"""
        action = np.zeros(self._env.action_dim, dtype=np.float32)
        # PandaOmron composite: indices detected dynamically via part_controllers
        gripper_idx = self._get_gripper_idx()
        action[gripper_idx] = gripper_value
        for _ in range(n_steps):
            try:
                obs, _, _, _ = self._env.step(action)
                self._latest_obs = obs
                self.render()
            except Exception as e:
                logger.warning(f"[gripper] step failed: {e}")
                break

    def _check_grasp_contact(self) -> bool:
        """通过 robosuite 的物理接触检测判断是否真的夹住了物体"""
        try:
            robot = self._env.robots[0]
            if hasattr(robot, "is_grasping"):
                # robosuite ≥1.5: 直接用 robot API
                return robot.is_grasping() != 0
            # fallback: 检查 gripper 两指间距 (夹住时间距小)
            if hasattr(robot, "gripper") and hasattr(robot.gripper.get("right", robot), "current_action"):
                pass
        except Exception as e:
            logger.debug(f"[grasp_contact] check failed: {e}")
        # 无法检测时保守返回 True (不因检测失败阻止后续流程)
        return True

    def _finger_object_contact(self, target_body: str) -> bool:
        """检查夹爪指尖是否与指定物体的 geom 处于接触

        通过遍历 `sim.data.contact[0:ncon]`, 判断每个接触对中:
        是否一边是 target body 的 geom, 另一边是 finger geom.

        相比 `_check_grasp_contact` (`is_grasping` 任意物体即 True),
        此方法**特定到目标物体**, 用于下降阶段的早停判断.
        """
        sim = self._env.sim
        try:
            obj_geoms = self._get_body_geom_ids(target_body)
            if not obj_geoms:
                return False

            finger_kw = ("finger", "fingertip", "finger_pad", "tip", "pad")
            finger_geoms = set()
            for i in range(sim.model.ngeom):
                try:
                    name = sim.model.geom_id2name(i) or ""
                except Exception:
                    name = ""
                lname = name.lower()
                if any(kw in lname for kw in finger_kw):
                    finger_geoms.add(i)

            for i in range(sim.data.ncon):
                c = sim.data.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                if (g1 in obj_geoms and g2 in finger_geoms) or \
                   (g2 in obj_geoms and g1 in finger_geoms):
                    return True
        except Exception as e:
            logger.debug(f"[finger_contact] {target_body}: {e}")
        return False

    def _descend_until_contact(
        self,
        target_pos: np.ndarray,
        target_body: str,
        step_z: float = 0.01,
        max_steps: int = 35,
    ) -> tuple[bool, float]:
        """步进式下降, 指尖接触目标物体即停 (避免硬撞或停太高)

        Args:
            target_pos: wrist 最终目标 (世界系)
            target_body: 用于接触检测的物体 body name
            step_z: 每步下降量 (m)
            max_steps: 最大步数

        Returns:
            (contact_ok, final_z): 是否接触到目标; 末端 Z
        """
        target = np.asarray(target_pos, dtype=np.float32)
        start_z = float(self.get_eef_pos()[2])
        logger.info(
            f"[descend] start z={start_z:.3f} → target z={target[2]:.3f} "
            f"(Δ={start_z - target[2]:.3f}m), step_z={step_z:.3f}m"
        )

        prev_z = None
        stall_count = 0
        contact_streak = 0
        # 至少下降 50% 路程或 3cm 才信任接触检测 (防止瞬时擦碰误报)
        min_descend_m = min(0.03, (start_z - target[2]) * 0.5)
        for i in range(max_steps):
            curr = self.get_eef_pos()
            descended = start_z - curr[2]
            if self._finger_object_contact(target_body):
                contact_streak += 1
                # 需要: (a) 已经下降足够距离 AND (b) 连续 2 帧接触
                #        AND (c) 已接近目标 z (1.5cm 内)
                # 若离目标仍远, 接触可能是碗沿/宽部擦碰, 应继续下降
                near_target = (curr[2] - target[2]) < 0.015
                if descended >= min_descend_m and contact_streak >= 2 and near_target:
                    logger.info(
                        f"[descend] contact at step {i}, "
                        f"z={curr[2]:.3f} (target {target[2]:.3f}, "
                        f"descended {descended:.3f}m)"
                    )
                    return True, float(curr[2])
            else:
                contact_streak = 0

            curr = self.get_eef_pos()
            if curr[2] <= target[2] + 0.001:
                # 已到目标 z (含 margin), 即使没接触也视为成功
                contact = self._finger_object_contact(target_body)
                logger.info(
                    f"[descend] reached target z={curr[2]:.3f}, contact={contact}"
                )
                return True, float(curr[2])

            # 收敛检测: 连续 5 步 z 几乎没下降才算 stall (放宽至 0.5mm)
            if prev_z is not None and abs(prev_z - curr[2]) < 0.0005:
                stall_count += 1
                if stall_count >= 5:
                    gap = curr[2] - target[2]
                    contact = self._finger_object_contact(target_body)
                    # 关键: stall + contact → 工作空间极限处已接触到目标
                    # 必须接受当前位置, 不能 reposition (横向移动 EEF 会推开物体)
                    if contact:
                        logger.info(
                            f"[descend] z stalled at {curr[2]:.3f} WITH contact "
                            f"(Δ={gap:.3f}m above target). Accepting "
                            f"(workspace limit + in contact with target)."
                        )
                        return True, float(curr[2])
                    # 如果已足够接近目标 z (< 1.5cm), 视为成功下降
                    close_enough = gap < 0.015
                    logger.warning(
                        f"[descend] z stalled at {curr[2]:.3f} for {stall_count} steps "
                        f"(Δ={gap:.3f}m above target). contact={contact}, "
                        f"close_enough={close_enough}"
                    )
                    return close_enough, float(curr[2])
            else:
                stall_count = 0
            prev_z = float(curr[2])

            # 下降一小步, XY 同步对齐 target
            next_z = max(curr[2] - step_z, target[2])
            next_target = np.array(
                [target[0], target[1], next_z], dtype=np.float32
            )
            # 增大 max_steps (120→200) 给收敛更多时间
            self.move_arm_to(
                next_target, threshold_m=0.005, max_steps=200
            )
            after_z = float(self.get_eef_pos()[2])
            if i % 3 == 0 or i < 5:
                logger.debug(
                    f"[descend] step {i}: z {curr[2]:.4f}→{after_z:.4f} "
                    f"(target_substep={next_z:.4f}, final_target={target[2]:.3f})"
                )

        curr = self.get_eef_pos()
        contact = self._finger_object_contact(target_body)
        logger.debug(f"[descend] max_steps reached, contact={contact}, z={curr[2]:.3f}")
        return contact, float(curr[2])

    def _close_gripper_until_grasp(
        self, target_body: str, max_steps: int = 30, min_close_steps: int = 6,
        squeeze_steps: int = 10,
    ) -> bool:
        """力闭环关爪: 关闭直到检测到夹持, 然后继续施力确保稳固

        Args:
            target_body: 目标物体 body name
            max_steps: 最大关爪步数 (上限保护)
            min_close_steps: 最少关爪步数 (给夹爪初始关闭时间)
            squeeze_steps: 接触确认后继续闭合的步数 (让夹爪完全包裹物体)

        Returns:
            True 若检测到稳定夹持, False 若超时无夹持
        """
        action = np.zeros(self._env.action_dim, dtype=np.float32)
        action[self._get_gripper_idx()] = 1.0
        confirmed = False
        for i in range(max_steps):
            try:
                obs, _, _, _ = self._env.step(action)
                self._latest_obs = obs
                self.render()
            except Exception as e:
                logger.warning(f"[close_gripper] step {i} failed: {e}")
                return False
            target_contact = self._finger_object_contact(target_body)
            generic_grasp = self._check_grasp_contact()
            if not confirmed and i >= min_close_steps and target_contact and generic_grasp:
                logger.info(f"[close_gripper] contact at step {i}, squeezing {squeeze_steps} more")
                confirmed = True
                squeeze_remaining = squeeze_steps
            if confirmed:
                squeeze_remaining -= 1
                if squeeze_remaining <= 0:
                    logger.info(f"[close_gripper] grasp confirmed + squeezed at step {i}")
                    return True
        if confirmed:
            logger.info(f"[close_gripper] grasp confirmed (squeeze partial) at step {max_steps}")
            return True
        logger.warning(f"[close_gripper] no grasp after {max_steps} steps")
        return False

    def grasp_at(
        self,
        target_pos_m,
        pre_grasp_height_m: float = 0.10,
        target_body: str = "obj_main",
        pre_grasp_verify=None,
    ) -> bool:
        """闭环自适应抓取流程

        范式:
            1. 几何感知 wrist 目标 (AABB → 中部偏上抓取点)
            2. 预抓取 (容差宽松, 用于粗对位)
            2.5 [可选] pre-grasp 语义验证 (执行前闭环, 创新点⑥)
            3. 接触式下降 (步进 + 指尖-物体接触早停, 避免硬撞)
            4. 力闭环关爪 (检测到夹持即停, 不固定步数)
            5. 微抬验证 (升 3cm 看物体是否跟随 → 跟随才确认)
            6. 失败重试一次 (重新计算 grasp_pose, 物体可能被推动)
            7. 最终提升

        每个环节都有反馈, 而非开环位置控制.

        Args:
            target_pos_m: fallback 目标位置 (世界系米制)
            pre_grasp_height_m: 预抓取高度
            target_body: sim body name, 用于物理验证
            pre_grasp_verify: 可选的 pre-grasp 验证回调.
                签名: (image_path: str) -> tuple[bool, str]
                返回 (是否通过, 原因). 通过则继续, 不通过则放弃此次尝试.

        Returns:
            True 若物体被成功抓起 (最终抬起 ≥5cm)
        """
        fallback_target = np.asarray(target_pos_m, dtype=np.float32)

        # 记录物体初始 Z (用于多次验证)
        def _obj_z() -> Optional[float]:
            p = self._get_body_pos(target_body)
            return float(p[2]) if p is not None else None
        obj_z_before = _obj_z()

        def _target_seed() -> np.ndarray:
            p = self._get_body_pos(target_body)
            return p if p is not None else fallback_target

        def _attempt(label: str) -> tuple[bool, bool, bool, bool]:
            """单次抓取尝试

            Returns: (pre_ok, descend_contact, grasp_confirmed, mini_lift_ok)
            """
            # (a) 几何感知 wrist 目标
            wrist_target = self._compute_grasp_pose(target_body, _target_seed())
            pre_grasp = wrist_target + np.array(
                [0.0, 0.0, pre_grasp_height_m], dtype=np.float32
            )

            logger.info(f"[grasp:{label}] open gripper")
            self._gripper_action(-1.0, n_steps=8)

            logger.info(f"[grasp:{label}] pre-grasp → {pre_grasp}")
            pre_ok = self.move_arm_to(pre_grasp, threshold_m=0.03)
            if not pre_ok:
                logger.warning(f"[grasp:{label}] pre-grasp unreachable, abort attempt")
                return False, False, False, False

            # 创新点⑥: pre-grasp 语义验证闭环 (事前主动验证)
            if pre_grasp_verify is not None:
                try:
                    eih = self.observe(self.eye_in_hand_viewpoint())
                    ok, reason = pre_grasp_verify(eih.image_path)
                    if not ok:
                        logger.warning(
                            f"[grasp:{label}] pre-grasp verify FAILED: {reason}"
                        )
                        return False, False, False, False
                    logger.info(
                        f"[grasp:{label}] pre-grasp verify PASSED: {reason}"
                    )
                except Exception as e:
                    logger.warning(
                        f"[grasp:{label}] pre-grasp verify error, "
                        f"continuing without it: {e}"
                    )

            logger.info(f"[grasp:{label}] descend → {wrist_target}")
            descend_contact, final_z = self._descend_until_contact(
                wrist_target, target_body, step_z=0.01, max_steps=25
            )

            logger.info(f"[grasp:{label}] close gripper (force loop)")
            grasp_confirmed = self._close_gripper_until_grasp(
                target_body, max_steps=30, min_close_steps=6
            )

            # 微抬验证: 升 3cm 看物体是否跟随 (持续夹紧防滑)
            curr = self.get_eef_pos()
            mini_target = curr + np.array([0.0, 0.0, 0.03], dtype=np.float32)
            self.move_arm_to(
                mini_target, threshold_m=0.01, max_steps=120,
                gripper_hold=1.0,
            )
            mini_lift_ok = False
            obj_z_now = _obj_z()
            if obj_z_now is not None and obj_z_before is not None:
                dz = obj_z_now - obj_z_before
                mini_lift_ok = dz > 0.01  # 物体跟随 ≥1cm
                logger.info(
                    f"[grasp:{label}] mini-lift: obj Δz={dz:.3f}m "
                    f"→ ok={mini_lift_ok}"
                )

            return True, descend_contact, grasp_confirmed, mini_lift_ok

        # ===== 第一次尝试 =====
        p1, d1, g1, m1 = _attempt("try1")
        if not p1:
            logger.info(
                "[grasp] done: pre_grasp=False, descend_contact=False, "
                "grasp_confirmed=False, mini_lift=False, lift_ok=False, "
                "final_lifted=False → False"
            )
            return False
        attempt_ok = m1  # 微抬验证为黄金标准

        # ===== 失败则重试一次 (物体可能被推动, 重算 pose) =====
        if not attempt_ok:
            logger.warning("[grasp] try1 failed, retrying...")
            p2, d2, g2, m2 = _attempt("try2")
            if not p2:
                logger.info(
                    f"[grasp] done: descend_contact={d1 or d2}, "
                    f"grasp_confirmed={g1 or g2}, mini_lift=False, "
                    "lift_ok=False, final_lifted=False → False"
                )
                return False
            attempt_ok = m2
            d1, g1 = d1 or d2, g1 or g2  # 累计

        # ===== 最终提升到 pre_grasp 高度 =====
        wrist_now = self._compute_grasp_pose(target_body, _target_seed())
        final_pre_grasp = wrist_now + np.array(
            [0.0, 0.0, pre_grasp_height_m], dtype=np.float32
        )
        logger.info(f"[grasp] final lift → {final_pre_grasp}")
        ok_lift = self.move_arm_to(final_pre_grasp, threshold_m=0.03)

        # ===== 最终物理验证: 物体是否真的被抬起 =====
        obj_lifted = False
        obj_z_after = _obj_z()
        if obj_z_after is not None and obj_z_before is not None:
            z_delta = obj_z_after - obj_z_before
            obj_lifted = z_delta > 0.05  # 跟随 ≥5cm
            logger.info(
                f"[grasp] final z: {obj_z_before:.3f} → {obj_z_after:.3f} "
                f"(Δ={z_delta:.3f}m, lifted={obj_lifted})"
            )

        # 只有真正抬起 (≥5cm) 才算抓取成功. mini_lift (1cm) 不够,
        # 因为 gripper 可能在 final lift 途中松开.
        # 如需放宽, 可降低 0.05m 阈值, 但不能用 mini_lift 替代.
        result = obj_lifted
        logger.info(
            f"[grasp] done: descend_contact={d1}, grasp_confirmed={g1}, "
            f"mini_lift={attempt_ok}, lift_ok={ok_lift}, "
            f"final_lifted={obj_lifted} → {result}"
        )
        return result

    def eye_in_hand_viewpoint(self):
        """快速获取 eye_in_hand viewpoint 对象"""
        from .active_planner import Viewpoint

        return Viewpoint(
            name="robot0_eye_in_hand",
            position=(0, 0, 30),
            orientation=(0, -90, 0),
            purpose="抓取后视觉验证",
        )

    # ------------------------------------------------------------------
    # v1 ActionExecutor / GraspPlanner 适配接口 (Phase 8.5 真 sim 集成)
    # ------------------------------------------------------------------

    # PandaMobile 工作空间约 base 周围 0.65m 半径 (实测).
    # 阈值 0.75m 略宽, 给 base 二次靠近留余地; 超过此距离 move_arm_to
    # 几乎肯定 stall 跑满 800 步, 提前剔除候选可省 ~30s/episode budget.
    _REACH_RADIUS_M: float = 0.75

    def is_reachable(self, point_3d, approach_dir) -> bool:
        """几何预过滤候选可达性。

        当前仅基于 base→point 水平距离: 超过 _REACH_RADIUS_M 视为不可达,
        避免 GraspPlanner 把不可达候选送进 move_to_pre_grasp 浪费步数。
        approach_dir 暂未用 (留作后续 IK pose 检查接口).

        Returns False 仅在能确证不可达时; 不确定时保守返 True 让
        move_arm_to 自己判断 (与改动前等价行为)。
        """
        try:
            base_pos, _ = self.get_base_pose()
            horiz = float(np.linalg.norm(
                np.asarray(point_3d, dtype=np.float32)[:2] - base_pos[:2]
            ))
            if horiz > self._REACH_RADIUS_M:
                logger.info(
                    "[is_reachable] False: base→point horiz=%.2fm > %.2fm",
                    horiz, self._REACH_RADIUS_M,
                )
                return False
            return True
        except Exception as e:
            logger.debug("[is_reachable] geometry check failed: %s", e)
            return True

    def move_to_pre_grasp(self, candidate, height_m: float = 0.05) -> bool:
        """移动到 candidate 的 pre-grasp 位置, 朝向对齐 candidate.approach_dir。

        Pre-grasp 位置 = point_3d - approach_dir * height_m
            - top_down (approach_dir=[0,0,-1]): 物体上方 height_m
            - 侧抓 (approach_dir=[1,0,0]): 物体后方 height_m (沿 -x)

        先移动底盘到 pre-grasp 附近 (xy 平面), 再带朝向控制移到精确 pre-grasp 位。
        """
        # 解析 approach_dir (默认 top_down)
        approach_dir = np.asarray(
            getattr(candidate, "approach_dir", [0.0, 0.0, -1.0]),
            dtype=np.float32,
        )
        ad_norm = float(np.linalg.norm(approach_dir))
        if ad_norm < 1e-6:
            approach_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            ad_norm = 1.0
        ad_unit = approach_dir / ad_norm

        target_pos = np.asarray(candidate.point_3d, dtype=np.float32)
        pre_pos = target_pos - ad_unit * float(height_m)

        # 底盘先靠近: 用 pre_pos 的 xy, 但 z 保持当前 eef 高度避免硬碰撞
        try:
            eef = self.get_eef_pos()
            # 底盘前置点: pre_pos.xy 再后退 0.4m 沿 -approach_dir.xy
            xy_approach = np.array(
                [ad_unit[0], ad_unit[1], 0.0], dtype=np.float32,
            )
            xy_norm = float(np.linalg.norm(xy_approach))
            if xy_norm > 0.1:
                xy_unit = xy_approach / xy_norm
                base_target = np.array([
                    float(pre_pos[0]) - xy_unit[0] * 0.10,
                    float(pre_pos[1]) - xy_unit[1] * 0.10,
                    float(eef[2]),
                ], dtype=np.float32)
            else:
                # top_down 等纯垂直接近: 维持原行为, 底盘前置 0.4m
                base_target = np.array([
                    float(target_pos[0]) - 0.4,
                    float(target_pos[1]),
                    float(eef[2]),
                ], dtype=np.float32)
            self.move_arm_to(base_target, threshold_m=0.15, max_steps=600)
        except Exception as e:
            logger.debug(f"[pre_grasp] base approach failed: {e}")

        # 张爪
        try:
            self._gripper_action(-1.0, n_steps=8)
        except Exception:
            pass

        # 移到 pre-grasp 位置 + 朝向对齐 approach_dir
        # 侧抓工作空间精度低于 top-down, 放宽阈值 (approach 步骤会补偿)
        is_top_down = (
            ad_unit[2] < -0.9 and abs(ad_unit[0]) < 0.1 and abs(ad_unit[1]) < 0.1
        )
        pre_thresh = 0.06 if is_top_down else 0.12
        return self.move_arm_to(
            pre_pos, threshold_m=pre_thresh, approach_dir=ad_unit,
        )

    def _approach_along_direction(
        self,
        target_pos: np.ndarray,
        target_body: str,
        approach_dir: np.ndarray,
        step_m: float = 0.012,
        max_steps: int = 30,
    ) -> tuple[bool, float]:
        """沿任意方向步进接近, 指尖接触目标即停。

        与 _descend_until_contact 类似, 但适用于任意角度 (倾斜/侧面)。
        每步沿 approach_dir 前进 step_m, 同时保持夹爪朝向对齐。

        Args:
            target_pos: 最终接近目标 (含 margin)
            target_body: 接触检测用的物体 body name
            approach_dir: 归一化接近方向
            step_m: 每步前进距离
            max_steps: 最大步数

        Returns:
            (contact_ok, final_eef_z)
        """
        ad = np.asarray(approach_dir, dtype=np.float32)
        target = np.asarray(target_pos, dtype=np.float32)
        start = self.get_eef_pos().copy()
        start_dist = float(np.linalg.norm(target - start))

        logger.info(
            f"[approach_dir] start dist={start_dist:.3f}m along {ad}, "
            f"step={step_m:.3f}m, max_steps={max_steps}"
        )

        contact_streak = 0
        min_travel_m = min(0.02, start_dist * 0.3)
        prev_pos = start.copy()
        stall_count = 0

        for i in range(max_steps):
            curr = self.get_eef_pos()
            traveled = float(np.linalg.norm(curr - start))
            remaining = float(np.linalg.norm(target - curr))

            # 接触检测
            if self._finger_object_contact(target_body):
                contact_streak += 1
                if traveled >= min_travel_m and contact_streak >= 2:
                    logger.info(
                        f"[approach_dir] contact at step {i}, "
                        f"traveled={traveled:.3f}m, remaining={remaining:.3f}m"
                    )
                    return True, float(curr[2])
            else:
                contact_streak = 0

            # 到达目标
            if remaining < 0.008:
                logger.info(
                    f"[approach_dir] reached target at step {i}, "
                    f"remaining={remaining:.3f}m"
                )
                return True, float(curr[2])

            # 收敛检测 (stall)
            move_since_last = float(np.linalg.norm(curr - prev_pos))
            if move_since_last < 0.001:
                stall_count += 1
                if stall_count >= 5:
                    near_target = remaining < 0.02
                    logger.warning(
                        f"[approach_dir] stalled at step {i}, "
                        f"remaining={remaining:.3f}m, near={near_target}"
                    )
                    return near_target, float(curr[2])
            else:
                stall_count = 0
            prev_pos = curr.copy()

            # 步进: 沿 approach_dir 前进一步 (仅位置, 不强制朝向)
            # 朝向由 pre_grasp 决定; 这里只沿倾斜路径推进位置,
            # 避免手臂因朝向无法收敛而 stall.
            step_dist = min(step_m, remaining)
            next_pos = curr + ad * step_dist
            self.move_arm_to(
                next_pos, threshold_m=0.005, max_steps=150,
            )

        # max_steps 用尽
        final = self.get_eef_pos()
        remaining = float(np.linalg.norm(target - final))
        logger.warning(
            f"[approach_dir] max_steps={max_steps} reached, "
            f"remaining={remaining:.3f}m"
        )
        return remaining < 0.02, float(final[2])

    def approach(
        self,
        point_3d,
        approach_dir: np.ndarray,
        target_label: Optional[str] = None,
        step_z: float = 0.01,
        max_steps: int = 35,
        margin_m: float = 0.015,
    ) -> tuple[bool, float]:
        """方向感知的抓取接近原语。沿 approach_dir 接近 point_3d。

        路径选择:
        - top_down (approach_dir≈[0,0,-1]): 接触式逐步下降
        - 任意方向 + 有接触检测: 沿方向逐步接近 (contact-aware)
        - 任意方向 + 无接触检测: 单次 move_arm_to (fallback)

        Args:
            point_3d: 抓取点 3D 坐标 (世界系)
            approach_dir: 单位向量, 从接近起点指向物体
            target_label: 接触检测用的物体 label; None 仅靠位置收敛
            margin_m: 沿 approach_dir 额外推进的距离 (补偿估计误差)

        Returns:
            (success, final_eef_z)
        """
        ad = np.asarray(approach_dir, dtype=np.float32)
        ad_norm = float(np.linalg.norm(ad))
        if ad_norm < 1e-6:
            ad = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            ad_norm = 1.0
        ad_unit = ad / ad_norm

        target = np.asarray(point_3d, dtype=np.float32).copy()
        if margin_m > 0:
            target = target + ad_unit * margin_m
            logger.info(
                f"[approach] margin={margin_m:.3f}m along {ad_unit}, "
                f"adjusted target={target}"
            )

        # top_down 路径: 用接触检测式下降 (保持原有行为)
        is_top_down = (
            ad_unit[2] < -0.9 and abs(ad_unit[0]) < 0.1 and abs(ad_unit[1]) < 0.1
        )

        target_body: Optional[str] = None
        if target_label:
            try:
                type_map = self._get_obj_type_map()
                for body, cat in type_map.items():
                    if cat == target_label:
                        target_body = body
                        break
            except Exception as e:
                logger.debug(f"[approach] type_map lookup failed: {e}")

        if is_top_down and target_body:
            return self._descend_until_contact(
                target, target_body, step_z=step_z, max_steps=max_steps,
            )

        # 非 top_down + 有接触检测: 逐步方向接近 (tilted/side 通用)
        if target_body:
            ok, z = self._approach_along_direction(
                target, target_body, ad_unit,
                step_m=0.012, max_steps=max_steps,
            )
            # 倾斜路径因工作空间限制无法到达时, 退化为垂直下降
            if not ok:
                curr = self.get_eef_pos()
                remaining = float(np.linalg.norm(target - curr))
                if remaining > 0.03:
                    logger.info(
                        "[approach] tilted path stalled (remaining=%.3fm), "
                        "falling back to vertical descent", remaining,
                    )
                    return self._descend_until_contact(
                        target, target_body,
                        step_z=step_z, max_steps=max_steps,
                    )
            return ok, z

        # fallback: 无接触检测时单次 move_arm_to
        ok = self.move_arm_to(
            target, threshold_m=0.01, max_steps=400,
            approach_dir=ad_unit,
        )
        return bool(ok), float(self.get_eef_pos()[2])

    def descend(
        self, point_3d, target_label: Optional[str] = None,
        step_z: float = 0.01, max_steps: int = 25,
        margin_m: float = 0.015,
    ) -> tuple[bool, float]:
        """[向后兼容] 顶部下降, 委托给 approach() with [0,0,-1]。

        旧调用方仍可用 descend(); 新代码应直接调用 approach() 并显式传 approach_dir.
        """
        return self.approach(
            point_3d,
            approach_dir=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            target_label=target_label,
            step_z=step_z, max_steps=max_steps, margin_m=margin_m,
        )

    def close_gripper(self, target_label: Optional[str] = None) -> bool:
        """关爪。有 target_label 时走力闭环 (检测物体接触)。"""
        if target_label:
            try:
                type_map = self._get_obj_type_map()
                target_body = next(
                    (b for b, c in type_map.items() if c == target_label),
                    None,
                )
                if target_body:
                    return self._close_gripper_until_grasp(
                        target_body, max_steps=30, min_close_steps=6,
                    )
            except Exception as e:
                logger.debug(f"[close_gripper] type_map lookup failed: {e}")
        # fallback: 简单关爪
        self._gripper_action(1.0, n_steps=15)
        return True

    def open_gripper(self) -> None:
        """开爪。"""
        self._gripper_action(-1.0, n_steps=10)

    def lift(
        self,
        height_m: float = 0.10,
        approach_dir: Optional[np.ndarray] = None,
    ) -> tuple[bool, float]:
        """从当前抓取位回退/抬升, 分阶段控速以避免物体滑落。

        - top_down (approach_dir≈[0,0,-1] 或 None): 直接竖直抬 height_m
        - 侧抓 (e.g. [1,0,0]): 先沿 -approach_dir 水平回退 height_m,
          再竖直抬升 height_m → 物体先脱离台面边缘再升空

        Args:
            height_m: 抬升 / 回退高度
            approach_dir: 抓取时的接近方向; None 视为 top_down

        Returns:
            (success, final_eef_z)
        """
        # 解析方向: None / 接近 [0,0,-1] 时视为 top_down
        if approach_dir is None:
            ad = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            ad = np.asarray(approach_dir, dtype=np.float32)
            n = float(np.linalg.norm(ad))
            ad = ad / n if n > 1e-6 else np.array([0.0, 0.0, -1.0], dtype=np.float32)
        is_top_down = (
            ad[2] < -0.9 and abs(ad[0]) < 0.1 and abs(ad[1]) < 0.1
        )

        try:
            curr = self.get_eef_pos().copy()
            start_z = float(curr[2])

            # ── 侧抓: 先水平回退 height_m 沿 -approach_dir ──
            # 注意: lift 全程 gripper_hold=1.0 维持夹爪闭合, 防止
            # close_gripper 后物体在移动期间因夹爪松开而滑落.
            if not is_top_down:
                retreat = -ad * float(height_m)
                # 慢退分 4 段 (类似慢起, 防止物体因惯性脱落)
                n_micro = 4
                for k in range(n_micro):
                    frac = (k + 1) / n_micro
                    target = np.array(
                        [curr[0] + retreat[0] * frac,
                         curr[1] + retreat[1] * frac,
                         curr[2] + retreat[2] * frac],
                        dtype=np.float32,
                    )
                    self.move_arm_to(
                        target, threshold_m=0.005, max_steps=120,
                        gripper_hold=1.0,
                    )
                # 更新当前位置作为后续 lift 起点
                curr = self.get_eef_pos().copy()

            # ── 竖直抬升 height_m ──
            # 阶段1: 慢起 2cm, 每次 5mm
            gentle_total = min(0.02, height_m)
            micro_step = 0.005
            n_micro = max(1, int(gentle_total / micro_step))
            for k in range(n_micro):
                target_z = float(curr[2]) + (k + 1) * micro_step
                target = np.array(
                    [curr[0], curr[1], target_z], dtype=np.float32
                )
                self.move_arm_to(
                    target, threshold_m=0.003, max_steps=80,
                    gripper_hold=1.0,
                )

            # 阶段2: 剩余高度正常速度
            remaining = height_m - gentle_total
            if remaining > 0.005:
                final_target = np.array(
                    [curr[0], curr[1], float(curr[2]) + height_m],
                    dtype=np.float32,
                )
                self.move_arm_to(
                    final_target, threshold_m=0.02, max_steps=200,
                    gripper_hold=1.0,
                )

            final_z = float(self.get_eef_pos()[2])
            # 至少升了一半 (top_down 比较 start_z, 侧抓比较回退后的 curr[2])
            base_z = start_z if is_top_down else float(curr[2])
            ok = final_z > base_z + height_m * 0.5
            return ok, final_z
        except Exception as e:
            logger.warning(f"[lift] failed: {e}")
            return False, 0.0

    def viewpoint_intrinsics(self, viewpoint):
        """v1 perception 用的相机内参占位。RoboCasa 内参由 mujoco 算出, 此处返 None
        让 perception 走 fallback (单视角 prior pos)。
        """
        return None

    def close(self) -> None:
        """清理资源"""
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
        logger.info("环境关闭")


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="启动仿真并测试手臂控制")
    parser.add_argument("--visualize", action="store_true", help="开 MuJoCo viewer")
    args = parser.parse_args()

    cfg = EnvConfig(has_renderer=args.visualize)
    print("[EnvWrapper] 配置:")
    print(f"  env_name:     {cfg.env_name}")
    print(f"  robots:       {cfg.robots}")
    print(f"  camera_names: {cfg.camera_names}")
    print(f"  image_size:   {cfg.image_width}x{cfg.image_height}")
    print(f"  has_renderer: {cfg.has_renderer}")

    if args.sim:
        env = EnvWrapper(cfg)
        env.reset()

        # Test: get_eef_pos
        pos = env.get_eef_pos()
        print(f"\n[Test] eef_pos = {pos}")

        # Test: move_arm_to (向上移 5cm)
        target = pos + np.array([0.0, 0.0, 0.05])
        print(f"[Test] move_arm_to target={target}")
        ok = env.move_arm_to(target)
        end = env.get_eef_pos()
        print(f"[Test] result: ok={ok}, end={end}, dist={np.linalg.norm(end - target):.4f}m")

        env.close()
    else:
        print("注: 加 --sim 启动仿真测试手臂控制")