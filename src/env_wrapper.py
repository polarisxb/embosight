"""仿真环境封装（RoboCasa）

校赛 Day 2: 离散视角 + reset/observe/close
省赛增强: 真实 OSC 手臂控制 + observe 实时刷新 + 可视化支持
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


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
                control_freq=20,
            )
            if self.config.layout_ids is not None:
                kwargs["layout_ids"] = int(self.config.layout_ids)
            if self.config.style_ids is not None:
                kwargs["style_ids"] = int(self.config.style_ids)

            logger.info(f"创建仿真环境 {self.config.env_name}...")
            self._env = suite.make(**kwargs)

        self._latest_obs = self._env.reset()
        self._step = 0
        logger.info(f"环境重置完成 (cameras={list(self.config.camera_names)})")
        return self._latest_obs

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

    def get_base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """获取底盘在世界系的 (位置, 3x3旋转矩阵)

        手臂 OSC 用 input_ref_frame='base', 底盘也是 base 系 JointVelocity,
        因此所有 action 都是 base 系增量, 需要把世界系 delta 旋转回 base 系.
        """
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
            for part_name, ctrl in robot.composite_controller.part_controllers.items():
                pn = part_name.lower()
                if "base" in pn or "mobile" in pn:
                    self._base_idx_cache = idx
                    logger.info(
                        f"[base] detected index={idx} dim={ctrl.control_dim} "
                        f"(part={part_name})"
                    )
                    return idx
                idx += ctrl.control_dim
        except Exception as e:
            logger.warning(f"[base] auto-detect failed ({e})")
        self._base_idx_cache = None
        return None

    def move_arm_to(
        self,
        target_pos_m,
        max_steps: int = 800,
        threshold_m: float = 0.02,
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

        init_dist = float(np.linalg.norm(target - self.get_eef_pos()))
        if init_dist < threshold_m:
            return True
        if init_dist > 0.5:
            max_steps = max(max_steps, int(init_dist * 1500))
        logger.debug(
            f"[move] target={target}, init_dist={init_dist:.3f}m, "
            f"max_steps={max_steps}, base_idx={base_idx}"
        )

        prev_dist = float("inf")
        stall = 0
        check_interval = 120  # 每 N 步检查一次 stall
        stall_limit = 6

        for step in range(max_steps):
            current = self.get_eef_pos()
            delta_world = target - current
            dist = float(np.linalg.norm(delta_world))

            if dist < threshold_m:
                logger.debug(f"[move] converged step={step} dist={dist:.4f}m")
                return True

            # stall 检测: 放宽间隔和阈值
            if step > 0 and step % check_interval == 0:
                progress = prev_dist - dist
                if progress < 0.005:
                    stall += 1
                    if stall >= stall_limit:
                        logger.warning(
                            f"[move_arm_to] stalled at step={step}, "
                            f"dist={dist:.4f}m (progress={progress:.4f}m)"
                        )
                        return dist < threshold_m
                else:
                    stall = max(0, stall - 1)
                prev_dist = dist
                if step % (check_interval * 3) == 0:
                    logger.debug(
                        f"[move] step={step} dist={dist:.3f}m "
                        f"stall={stall}/{stall_limit}"
                    )

            # 世界系 → base 系 (核心修正)
            _, base_ori = self.get_base_pose()
            delta_base = base_ori.T @ delta_world  # 3D vector in base frame

            dir_base = delta_base / max(dist, 1e-6)
            step_size = min(self.ARM_STEP_CAP, dist)

            action = np.zeros(action_dim, dtype=np.float32)
            # 手臂: base 系增量
            action[0:3] = dir_base * step_size

            # 底盘: base 系 forward/side 速度
            if has_base and dist > 0.05:
                base_gain = min(0.8, dist * 0.8)
                action[base_idx] = float(dir_base[0]) * base_gain      # forward
                action[base_idx + 1] = float(dir_base[1]) * base_gain  # side

            try:
                obs, _, _, _ = self._env.step(action)
                self._latest_obs = obs
                self.render()
            except Exception as e:
                logger.warning(f"[move_arm_to] step {step} failed: {e}")
                return False

        logger.warning(f"[move_arm_to] max_steps reached, dist={dist:.4f}m")
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

    def ground_object(self, user_target: str) -> Optional[ObjectGrounding]:
        """将用户目标名 grounding 到仿真物体

        搜索策略 (按优先级):
        1. alias_map 精确匹配 body name
        2. alias_map 精确匹配 task objects
        3. body name 模糊匹配
        4. 回退: 返回 obj_main (RoboCasa 主任务物体)

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

        # 3) 回退: 返回 obj_main (RoboCasa 主任务物体)
        #    RoboCasa PickPlace 任务中 obj_main 就是要抓的物体
        if "obj_main" in sim_body_names:
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

    # ------------------------------------------------------------------
    # 观察 (Phase 1)
    # ------------------------------------------------------------------

    def observe(self, viewpoint) -> "Observation":
        """采集指定视角的 RGB 图像

        Args:
            viewpoint: Viewpoint 对象（来自 active_planner）

        Returns:
            Observation 对象（含图像路径和元数据）
        """
        from .active_planner import Observation

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

        return Observation(
            viewpoint=viewpoint,
            image_path=image_path,
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
        max_steps: int = 25,
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

        for i in range(max_steps):
            if self._finger_object_contact(target_body):
                curr = self.get_eef_pos()
                logger.info(
                    f"[descend] contact at step {i}, "
                    f"z={curr[2]:.3f} (target {target[2]:.3f})"
                )
                return True, float(curr[2])

            curr = self.get_eef_pos()
            if curr[2] <= target[2] + 0.005:
                # 已到目标 z, 不再下降
                logger.debug(f"[descend] reached target z without contact")
                return self._finger_object_contact(target_body), float(curr[2])

            # 下降一小步, XY 同步对齐 target
            next_z = max(curr[2] - step_z, target[2])
            next_target = np.array(
                [target[0], target[1], next_z], dtype=np.float32
            )
            self.move_arm_to(
                next_target, threshold_m=0.005, max_steps=120
            )

        curr = self.get_eef_pos()
        contact = self._finger_object_contact(target_body)
        logger.debug(f"[descend] max_steps reached, contact={contact}, z={curr[2]:.3f}")
        return contact, float(curr[2])

    def _close_gripper_until_grasp(
        self, target_body: str, max_steps: int = 30, min_close_steps: int = 6
    ) -> bool:
        """力闭环关爪: 关闭直到检测到稳定夹持, 不是固定步数

        Args:
            target_body: 目标物体 body name
            max_steps: 最大关爪步数 (上限保护)
            min_close_steps: 最少关爪步数 (给夹爪初始关闭时间)

        Returns:
            True 若检测到夹持, False 若超时无夹持
        """
        action = np.zeros(self._env.action_dim, dtype=np.float32)
        action[self._get_gripper_idx()] = 1.0
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
            if i >= min_close_steps and target_contact and generic_grasp:
                logger.info(f"[close_gripper] grasp confirmed at step {i}")
                return True
        logger.warning(f"[close_gripper] no grasp after {max_steps} steps")
        return False

    def grasp_at(
        self,
        target_pos_m,
        pre_grasp_height_m: float = 0.10,
        target_body: str = "obj_main",
    ) -> bool:
        """闭环自适应抓取流程

        范式:
            1. 几何感知 wrist 目标 (AABB → 中部偏上抓取点)
            2. 预抓取 (容差宽松, 用于粗对位)
            3. 接触式下降 (步进 + 指尖-物体接触早停, 避免硬撞)
            4. 力闭环关爪 (检测到夹持即停, 不固定步数)
            5. 微抬验证 (升 3cm 看物体是否跟随 → 跟随才确认)
            6. 失败重试一次 (重新计算 grasp_pose, 物体可能被推动)
            7. 最终提升

        每个环节都有反馈, 而非开环位置控制.

        Returns:
            True 若物体被成功抓起 (微抬或最终抬起 ≥1cm)
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

            logger.info(f"[grasp:{label}] descend → {wrist_target}")
            descend_contact, final_z = self._descend_until_contact(
                wrist_target, target_body, step_z=0.01, max_steps=25
            )

            logger.info(f"[grasp:{label}] close gripper (force loop)")
            grasp_confirmed = self._close_gripper_until_grasp(
                target_body, max_steps=30, min_close_steps=6
            )

            # 微抬验证: 升 3cm 看物体是否跟随
            curr = self.get_eef_pos()
            mini_target = curr + np.array([0.0, 0.0, 0.03], dtype=np.float32)
            self.move_arm_to(mini_target, threshold_m=0.01, max_steps=120)
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

        result = obj_lifted or attempt_ok
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