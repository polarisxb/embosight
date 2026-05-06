"""创新点②: 零样本主动视角规划器（LLM-NBV）

将经典 Next-Best-View 问题转化为 LLM 选择问题。
设计离散视角库 + LLM 决策的新范式。

核心设计:
    1. 离散视角库（12 个标准视角）回避连续动作空间
    2. LLM 任务驱动决策（而非几何驱动）
    3. LLM 自评估早停机制

使用示例:
    >>> from src.active_planner import ActivePlanner, ViewpointLibrary
    >>> from src.llm_backend import LLMBackend
    >>> llm = LLMBackend()
    >>> vp_lib = ViewpointLibrary("configs/viewpoints.yaml")
    >>> planner = ActivePlanner(llm_client=llm, viewpoint_lib=vp_lib)
    >>> observations = planner.plan(subtasks, env)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Viewpoint:
    """单个视角

    Attributes:
        name: 视角名称（例 "top_view"）
        position: 末端位置 (x, y, z) 单位 cm
        orientation: 末端朝向 (roll, pitch, yaw) 单位度
        purpose: 该视角的用途说明
    """

    name: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float]
    purpose: str = ""

    def to_pose(self) -> tuple[float, float, float, float, float, float]:
        """转换为 6D 位姿"""
        return (*self.position, *self.orientation)

    def __repr__(self) -> str:
        return f"Viewpoint(name='{self.name}', purpose='{self.purpose}')"


@dataclass
class Observation:
    """单次视角下的观察结果

    Attributes:
        viewpoint: 当前视角
        image_path: 渲染图像保存路径
        depth_map_path: 深度图保存路径（可选）
        description: VLM 描述结果
        timestamp: 观察时间戳
    """

    viewpoint: Viewpoint
    image_path: str
    depth_map_path: Optional[str] = None
    description: str = ""
    timestamp: float = field(default_factory=time.time)


# ============================================================
# 离散视角库
# ============================================================

class ViewpointLibrary:
    """离散视角库

    管理 12 个标准视角，从 YAML 配置加载。
    """

    def __init__(self, config_path: str = "configs/viewpoints.yaml") -> None:
        self.config_path = Path(config_path)
        self.viewpoints: list[Viewpoint] = []
        self._load()

    def _load(self) -> None:
        """从 YAML 加载视角库"""
        if not self.config_path.exists():
            logger.warning(f"视角配置不存在: {self.config_path}，使用内置默认")
            self._builtin_viewpoints()
            return

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for vp_data in data.get("viewpoints", []):
            self.viewpoints.append(
                Viewpoint(
                    name=vp_data["name"],
                    position=tuple(vp_data["position"]),
                    orientation=tuple(vp_data["orientation"]),
                    purpose=vp_data.get("purpose", ""),
                )
            )
        logger.info(f"加载视角库: {len(self.viewpoints)} 个视角")

    def _builtin_viewpoints(self) -> None:
        """内置最小视角集（保证系统能跑，对应 RoboCasa 实际摄像头名）"""
        self.viewpoints = [
            Viewpoint("robot0_agentview_center", (0, 0, 60), (0, -45, 0), "全景中央视角，用于场景概览"),
            Viewpoint("robot0_agentview_left", (-60, 0, 60), (0, -45, 45), "左侧全景视角，用于左半区观察"),
            Viewpoint("robot0_agentview_right", (60, 0, 60), (0, -45, -45), "右侧全景视角，用于右半区观察"),
            Viewpoint("robot0_frontview", (0, -60, 30), (0, -30, 0), "正面视图，用于近距识别"),
            Viewpoint("robot0_robotview", (0, 30, 60), (0, -45, 180), "机器人视角，用于操作区域观察"),
            Viewpoint("robot0_eye_in_hand", (0, 0, 30), (0, -90, 0), "机械臂末端视角，用于物体特写"),
        ]

    def __len__(self) -> int:
        return len(self.viewpoints)

    def __getitem__(self, idx: int) -> Viewpoint:
        return self.viewpoints[idx]

    def list_for_prompt(self) -> str:
        """生成用于 LLM Prompt 的视角列表字符串"""
        lines = []
        for i, vp in enumerate(self.viewpoints):
            lines.append(f"  {i}: {vp.name} - {vp.purpose}")
        return "\n".join(lines)


# ============================================================
# 核心类: ActivePlanner
# ============================================================

class ActivePlanner:
    """LLM-NBV 主动视角规划器

    核心循环:
        1. 初始全景视角观察
        2. LLM 选择下一个最优视角
        3. 机械臂执行 + VLM 描述
        4. LLM 自评估是否信息足够
        5. 重复 2-4 直到信息足够或达到上限
    """

    def __init__(
        self,
        llm_client,
        viewpoint_lib: ViewpointLibrary,
        max_viewpoints: int = 6,
        coverage_threshold: float = 0.85,
        prompt_path: str = "prompts/active_planner.txt",
    ) -> None:
        """
        Args:
            llm_client: LLM 客户端
            viewpoint_lib: 离散视角库
            max_viewpoints: 最大视角数（防死循环）
            coverage_threshold: 任务覆盖率阈值
            prompt_path: 系统 Prompt 模板路径
        """
        self.llm = llm_client
        self.vp_lib = viewpoint_lib
        self.max_vp = max_viewpoints
        self.coverage_threshold = coverage_threshold
        self.system_prompt = self._load_prompt(prompt_path)

    def _load_prompt(self, prompt_path: str) -> str:
        path = Path(prompt_path)
        if not path.exists():
            logger.warning(f"Prompt 文件不存在: {path}")
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # ==========================================================
    # 主入口
    # ==========================================================

    def plan(self, subtasks: list, env) -> list[Observation]:
        """主入口: 执行完整主动视角规划循环

        Args:
            subtasks: 子任务列表（来自 TaskDecomposer）
            env: 仿真环境对象（来自 src/env_wrapper.py）

        Returns:
            观察列表
        """
        observations: list[Observation] = []
        used_indices: set[int] = set()

        init_idx = 0
        init_vp = self.vp_lib[init_idx]
        env.move_arm_to(init_vp.to_pose())
        init_obs = env.observe(init_vp)
        observations.append(init_obs)
        used_indices.add(init_idx)
        logger.info(f"初始视角: {init_vp.name}")

        while len(observations) < self.max_vp:
            if self._is_sufficient(subtasks, observations):
                logger.info(f"早停: 已收集 {len(observations)} 个视角")
                break

            next_idx = self.select_next_viewpoint(subtasks, observations, used_indices)
            if next_idx < 0 or next_idx >= len(self.vp_lib):
                logger.info(f"LLM 输出 -1（早停信号）")
                break

            next_vp = self.vp_lib[next_idx]
            env.move_arm_to(next_vp.to_pose())
            new_obs = env.observe(next_vp)
            observations.append(new_obs)
            used_indices.add(next_idx)
            logger.info(f"视角 {len(observations)}: {next_vp.name}")

        return observations

    # ==========================================================
    # 视角选择
    # ==========================================================

    def select_next_viewpoint(
        self,
        subtasks: list,
        observations: list[Observation],
        used_indices: set[int],
    ) -> int:
        """让 LLM 选择下一个最优视角

        Args:
            subtasks: 子任务列表
            observations: 已有观察
            used_indices: 已使用的视角索引（避免重复）

        Returns:
            视角库中的索引；-1 表示早停
        """
        prompt = self._build_nbv_prompt(subtasks, observations, used_indices)

        try:
            response = self.llm.generate(
                user_message=prompt,
                system=self.system_prompt,
                json_mode=True,
            )
            data = json.loads(response)
            idx = int(data.get("viewpoint_idx", -1))
            reason = data.get("reason", "")
            logger.debug(f"NBV 决策: idx={idx}, reason={reason}")
            return idx
        except Exception as e:
            logger.warning(f"NBV 决策失败: {e}, fallback 到第一个未使用视角")
            for i in range(len(self.vp_lib)):
                if i not in used_indices:
                    return i
            return -1

    def _build_nbv_prompt(
        self,
        subtasks: list,
        observations: list[Observation],
        used_indices: set[int],
    ) -> str:
        """构建 NBV 决策 Prompt"""
        lines = ["## 未完成子任务"]
        for i, t in enumerate(subtasks, 1):
            if not t.coverage_status:
                lines.append(
                    f"  {i}. [{t.type.value}] {t.target} (priority={t.priority}, dim={t.blind_dimension.value})"
                )

        lines.append("\n## 当前已有观察")
        for i, obs in enumerate(observations, 1):
            desc = obs.description[:100] if obs.description else "（暂无描述）"
            lines.append(f"  视角 {i} [{obs.viewpoint.name}]: {desc}")

        lines.append("\n## 离散视角库（已用视角已标注）")
        for i, vp in enumerate(self.vp_lib.viewpoints):
            mark = " [已用]" if i in used_indices else ""
            lines.append(f"  {i}: {vp.name} - {vp.purpose}{mark}")

        lines.append(
            "\n请选择下一个最有助于完成所有未完成子任务的视角索引。\n"
            "若当前观察已经足够回答所有子任务，输出 viewpoint_idx = -1。"
        )

        return "\n".join(lines)

    # ==========================================================
    # 早停判断
    # ==========================================================

    def _is_sufficient(
        self,
        subtasks: list,
        observations: list[Observation],
    ) -> bool:
        """LLM 自评估: 当前观察是否足以回答查询？"""
        if not observations:
            return False

        prompt = self._build_sufficiency_prompt(subtasks, observations)
        try:
            response = self.llm.generate(prompt)
            return any(kw in response.lower() for kw in ["yes", "sufficient", "足够", "已够"])
        except Exception as e:
            logger.warning(f"早停判断失败: {e}")
            return False

    def _build_sufficiency_prompt(
        self,
        subtasks: list,
        observations: list[Observation],
    ) -> str:
        """构建早停判断 Prompt"""
        sub_lines = "\n".join(
            [f"  - [{t.type.value}] {t.target}" for t in subtasks]
        )
        obs_lines = "\n".join(
            [f"  视角{i+1}: {o.description[:80]}..." for i, o in enumerate(observations)]
        )
        return (
            f"## 子任务\n{sub_lines}\n\n"
            f"## 已有观察\n{obs_lines}\n\n"
            f"基于以上观察，所有子任务的信息是否已经足够？请回答 'yes' 或 'no'。"
        )


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("[ActivePlanner] 模块加载测试")
    vp_lib = ViewpointLibrary()
    print(f"  视角库大小: {len(vp_lib)}")
    print(f"  视角列表预览:\n{vp_lib.list_for_prompt()}")