"""仿真环境封装（RoboCasa）

校赛 Day 2: 离散视角 + reset/observe/close
省赛增强: 真实 OSC 手臂控制 + observe 实时刷新 + 可视化支持
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
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
        max_steps: int = 200,
        threshold_m: float = 0.02,
    ) -> bool:
        """OSC 增量控制移动末端到目标位置 (单位: m)

        Returns:
            True if converged within threshold
        """
        if not self._latest_obs:
            self.reset()

        target = np.asarray(target_pos_m, dtype=np.float32)
        action_dim = self._env.action_dim

        for step in range(max_steps):
            current = self.get_eef_pos()
            delta = target - current
            dist = float(np.linalg.norm(delta))

            if dist < threshold_m:
                logger.debug(f"[move_arm_to] converged step={step} dist={dist:.4f}m")
                return True

            step_size = min(0.05, dist)
            direction = delta / max(dist, 1e-6)

            action = np.zeros(action_dim, dtype=np.float32)
            action[0:3] = direction * step_size

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

        path = Path("configs/object_aliases.yaml")
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

    def ground_object(self, user_target: str) -> Optional[ObjectGrounding]:
        """将用户目标名 grounding 到仿真物体

        Returns:
            ObjectGrounding with meter coordinates, or None
        """
        if not self._latest_obs:
            self.reset()

        if not hasattr(self, "_aliases"):
            self._aliases = self._load_aliases()

        sim_body_names = list(self._env.sim.model.body_names)

        # 1) 别名精确匹配
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

    def _gripper_action(self, gripper_value: float, n_steps: int = 10) -> None:
        """控制夹爪 (gripper_value: -1 开, +1 关)"""
        action = np.zeros(self._env.action_dim, dtype=np.float32)
        # PandaMobile OSC_POSE: action layout 取决于 controller
        # 一般 [0:3]=pos, [3:6]=rot, [6]=gripper, [7:]=base
        # 首次运行请用 print(env._env.action_spec) 校验
        gripper_idx = 6
        action[gripper_idx] = gripper_value
        for _ in range(n_steps):
            try:
                obs, _, _, _ = self._env.step(action)
                self._latest_obs = obs
                self.render()
            except Exception as e:
                logger.warning(f"[gripper] step failed: {e}")
                break

    def grasp_at(
        self,
        target_pos_m,
        pre_grasp_height_m: float = 0.10,
    ) -> bool:
        """完整抓取流程: 开爪 → 预抓取 → 下降 → 关爪 → 提升

        Returns:
            True if all motion steps converged
        """
        target = np.asarray(target_pos_m, dtype=np.float32)
        pre_grasp = target + np.array([0.0, 0.0, pre_grasp_height_m], dtype=np.float32)

        logger.info("[grasp] open gripper")
        self._gripper_action(-1.0, n_steps=8)

        logger.info(f"[grasp] move to pre-grasp {pre_grasp}")
        if not self.move_arm_to(pre_grasp):
            return False

        logger.info(f"[grasp] descend to target {target}")
        if not self.move_arm_to(target):
            return False

        logger.info("[grasp] close gripper")
        self._gripper_action(+1.0, n_steps=15)

        logger.info(f"[grasp] lift to {pre_grasp}")
        if not self.move_arm_to(pre_grasp):
            return False

        return True

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