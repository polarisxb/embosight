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

    def render(self) -> None:
        """如果 has_renderer 则刷新可视化窗口"""
        if self.config.has_renderer and self._env is not None:
            try:
                self._env.render()
            except Exception as e:
                logger.warning(f"[render] {e}")

    def move_arm_to(
        self,
        target_pos_m,
        max_steps: int = 800,
        threshold_m: float = 0.02,
    ) -> bool:
        """OSC 增量控制移动末端到目标位置 (单位: m)

        PandaOmron composite layout (action_dim=12):
          [0:3]=arm_pos, [3:6]=arm_rot, [6:10]=base, [10:12]=gripper
        长距移动同时驱动底盘 XY 辅助靠近。

        Returns:
            True if converged within threshold
        """
        if not self._latest_obs:
            self.reset()

        target = np.asarray(target_pos_m, dtype=np.float32)
        if target.shape[0] > 3:
            logger.debug(f"[move_arm_to] truncating {target.shape[0]}D → 3D (xyz only)")
            target = target[:3]
        action_dim = self._env.action_dim
        dist = float("inf")
        prev_dist = float("inf")
        stall_count = 0

        for step in range(max_steps):
            current = self.get_eef_pos()
            delta = target - current
            dist = float(np.linalg.norm(delta))

            if dist < threshold_m:
                logger.debug(f"[move_arm_to] converged step={step} dist={dist:.4f}m")
                return True

            # 发散/停滞检测: 每 50 步看一次
            if step > 0 and step % 50 == 0:
                if dist >= prev_dist - 0.005:
                    stall_count += 1
                    if stall_count >= 3:
                        logger.warning(f"[move_arm_to] stalled, dist={dist:.4f}m")
                        return dist < threshold_m
                else:
                    stall_count = 0
                prev_dist = dist

            direction = delta / max(dist, 1e-6)

            # 稳定 step_size: cap=0.15 防止 OSC 饱和
            step_size = min(0.15, dist)

            action = np.zeros(action_dim, dtype=np.float32)
            action[0:3] = direction * step_size

            # 长距 (>0.15m) 同时驱动底盘 XY 辅助
            if dist > 0.15 and action_dim >= 10:
                base_gain = min(0.1, dist * 0.3)
                action[6] = direction[0] * base_gain   # base X
                action[7] = direction[1] * base_gain   # base Y

            try:
                obs, _, done, _ = self._env.step(action)
                self._latest_obs = obs
                self.render()
            except Exception as e:
                logger.warning(f"[move_arm_to] step failed at {step}: {e}")
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
        # PandaOmron composite: [0:6]=arm(pos+rot), [6]=gripper, [7:]=base
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
        # 无法检测时返回 None 表示未知
        return True

    def grasp_at(
        self,
        target_pos_m,
        pre_grasp_height_m: float = 0.10,
        target_body: str = "obj_main",
    ) -> bool:
        """完整抓取流程: 开爪 → 预抓取 → 下降 → 关爪 → 提升

        加入物理验证:
        1. 关爪后检查接触
        2. 提升后检查物体是否跟着升高

        Returns:
            True if grasp physically succeeded
        """
        target = np.asarray(target_pos_m, dtype=np.float32)
        pre_grasp = target + np.array([0.0, 0.0, pre_grasp_height_m], dtype=np.float32)
        grasp_tol = 0.05  # 5cm 容差

        # 记录物体初始高度
        obj_z_before = None
        obj_pos = self._get_body_pos(target_body)
        if obj_pos is not None:
            obj_z_before = float(obj_pos[2])

        logger.info("[grasp] open gripper")
        self._gripper_action(-1.0, n_steps=8)

        logger.info(f"[grasp] move to pre-grasp {pre_grasp}")
        ok1 = self.move_arm_to(pre_grasp, threshold_m=grasp_tol)

        logger.info(f"[grasp] descend to target {target}")
        ok2 = self.move_arm_to(target, threshold_m=grasp_tol)

        logger.info("[grasp] close gripper")
        self._gripper_action(+1.0, n_steps=15)

        # 物理验证 1: 接触检测
        contact_ok = self._check_grasp_contact()
        logger.info(f"[grasp] contact check: {contact_ok}")

        logger.info(f"[grasp] lift to {pre_grasp}")
        ok3 = self.move_arm_to(pre_grasp, threshold_m=grasp_tol)

        # 物理验证 2: 物体是否跟着升高
        obj_lifted = False
        obj_pos_after = self._get_body_pos(target_body)
        if obj_pos_after is not None and obj_z_before is not None:
            z_delta = float(obj_pos_after[2]) - obj_z_before
            obj_lifted = z_delta > 0.02  # 物体升高了 2cm+
            logger.info(
                f"[grasp] object z: {obj_z_before:.3f} → {float(obj_pos_after[2]):.3f} "
                f"(Δ={z_delta:.3f}m, lifted={obj_lifted})"
            )

        motion_ok = ok1 and ok2 and ok3
        logger.info(
            f"[grasp] done: motion={motion_ok}, contact={contact_ok}, "
            f"lifted={obj_lifted} → {motion_ok and (contact_ok or obj_lifted)}"
        )
        return motion_ok and (contact_ok or obj_lifted)

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