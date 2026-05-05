"""仿真环境封装（RoboCasa）

封装 RoboCasa 环境以提供统一接口供 Pipeline 调用。

校赛阶段实现的最小接口:
    - reset()            重置环境
    - move_arm_to(pose)  移动机械臂到目标位姿
    - observe(viewpoint) 采集当前视角的 RGB + 深度图
    - close()            清理资源

TODO 校赛阶段实现:
    - RoboCasa 厨房场景加载
    - Franka Panda 机械臂控制
    - RGB-D 摄像头数据采集
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    """环境配置"""

    sim_type: str = "robocasa"
    scene_name: str = "kitchen_default"
    image_width: int = 512
    image_height: int = 512
    output_dir: str = "./results/observations"


class EnvWrapper:
    """RoboCasa 仿真环境封装

    TODO: Day 2-3 实现具体逻辑
    """

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        self.config = config or EnvConfig()
        self._env = None
        self._step = 0

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def reset(self) -> dict[str, Any]:
        """重置环境

        Returns:
            初始观察 (occupancy/state info)
        """
        # TODO: 调用 robocasa 加载场景
        # import robocasa
        # self._env = robocasa.make(self.config.scene_name)
        # obs = self._env.reset()
        self._step = 0
        logger.info(f"[EnvWrapper] 环境重置 (TODO: 接入 RoboCasa)")
        return {}

    def move_arm_to(
        self,
        pose: tuple[float, float, float, float, float, float],
    ) -> bool:
        """移动机械臂到目标位姿

        Args:
            pose: (x, y, z, roll, pitch, yaw) 单位 cm/度

        Returns:
            是否成功到达
        """
        # TODO: 实现机械臂运动
        # 1. 调用逆运动学求解关节角度
        # 2. 控制机械臂运动
        # 3. 等待稳定
        logger.debug(f"[EnvWrapper] 移动到位姿: {pose} (TODO: 接入运动学)")
        time.sleep(0.05)
        return True

    def observe(self, viewpoint) -> "Observation":
        """采集当前视角的 RGB + 深度图

        Args:
            viewpoint: Viewpoint 对象（来自 active_planner）

        Returns:
            Observation 对象（含图像路径和元数据）
        """
        from .active_planner import Observation

        self._step += 1
        image_path = os.path.join(
            self.config.output_dir,
            f"step_{self._step:03d}_{viewpoint.name}.png",
        )

        # TODO: 实际渲染图像
        # rgb = self._env.render(camera="wrist_cam")
        # depth = self._env.render(camera="wrist_cam", mode="depth")
        # cv2.imwrite(image_path, rgb)
        logger.debug(f"[EnvWrapper] 模拟采集图像: {image_path} (TODO: 实际渲染)")

        return Observation(
            viewpoint=viewpoint,
            image_path=image_path,
        )

    def close(self) -> None:
        """清理资源"""
        if self._env is not None:
            # self._env.close()
            self._env = None
        logger.info("[EnvWrapper] 环境关闭")


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("[EnvWrapper] 模块加载测试")
    env = EnvWrapper()
    env.reset()
    env.close()
    print("骨架可正常调用，等待 Day 2-3 接入 RoboCasa")