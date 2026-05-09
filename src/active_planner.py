"""主动视角选择 v1: ViewpointLibrary + ActiveViewpointSelector。

老 ActivePlanner / Observation / plan / plan_with_grounding 已删除 (Phase 15)。
v1 用 EmboSightAgent.decide_next 直接调度 ActiveViewpointSelector。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import yaml

from src.world_belief import WorldBelief

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Viewpoint:
    """单个视角 (位置 + 朝向 + 用途)。"""
    name: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float]
    purpose: str = ""

    def to_pose(self) -> tuple[float, float, float, float, float, float]:
        return (*self.position, *self.orientation)

    def __repr__(self) -> str:
        return f"Viewpoint(name='{self.name}', purpose='{self.purpose}')"


# ============================================================
# 离散视角库
# ============================================================

class ViewpointLibrary:
    """从 YAML 加载视角库; 文件不存在时使用内置默认。"""

    def __init__(self, config_path: str = "configs/viewpoints.yaml") -> None:
        self.config_path = Path(config_path)
        self.viewpoints: list[Viewpoint] = []
        self._load()

    def _load(self) -> None:
        if not self.config_path.exists():
            logger.warning(f"视角配置不存在: {self.config_path}, 使用内置默认")
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
        """内置最小视角集 (对应 RoboCasa 默认 6 摄像头)。"""
        self.viewpoints = [
            Viewpoint("robot0_agentview_center", (0, 0, 60), (0, -45, 0),
                      "全景中央视角, 用于场景概览"),
            Viewpoint("robot0_agentview_left", (-60, 0, 60), (0, -45, 45),
                      "左侧全景视角, 用于左半区观察"),
            Viewpoint("robot0_agentview_right", (60, 0, 60), (0, -45, -45),
                      "右侧全景视角, 用于右半区观察"),
            Viewpoint("robot0_frontview", (0, -60, 30), (0, -30, 0),
                      "正面视图, 用于近距识别"),
            Viewpoint("robot0_robotview", (0, 30, 60), (0, -45, 180),
                      "机器人视角, 用于操作区域观察"),
            Viewpoint("robot0_eye_in_hand", (0, 0, 30), (0, -90, 0),
                      "机械臂末端视角, 用于物体特写"),
        ]

    def __len__(self) -> int:
        return len(self.viewpoints)

    def __getitem__(self, idx: int) -> Viewpoint:
        return self.viewpoints[idx]

    def __iter__(self):
        return iter(self.viewpoints)

    def list_for_prompt(self) -> str:
        return "\n".join(
            f"  {i}: {vp.name} - {vp.purpose}"
            for i, vp in enumerate(self.viewpoints)
        )


# ============================================================
# v1: ActiveViewpointSelector (LLM-NBV with 4 preference)
# ============================================================

_NBV_PROMPT_PATH = "prompts/agent/nbv_select.txt"


class ActiveViewpointSelector:
    """LLM 选下一视角。4 种 preference 注入 prompt; 越界/重复 → None。"""

    def __init__(self, llm, viewpoint_lib, prompt_path: str = _NBV_PROMPT_PATH):
        self.llm = llm
        self.vp_lib = viewpoint_lib
        p = Path(prompt_path)
        self._template = p.read_text(encoding="utf-8") if p.exists() else None

    def select(
        self,
        belief: WorldBelief,
        exclude: set[str],
        preference: Literal[
            "search_target", "disambiguate_label",
            "parallax_position", "grasp_pose",
        ] = "search_target",
    ) -> Optional[Viewpoint]:
        candidates = [
            (i, vp) for i, vp in enumerate(self.vp_lib)
            if vp.name not in exclude
        ]
        if not candidates:
            return None

        prompt = self._build_prompt(belief, exclude, preference)
        try:
            raw = self.llm.generate(prompt, system="")
        except Exception as e:
            logger.warning(f"[viewpoint_selector] LLM failed: {e}")
            return candidates[0][1]

        m = re.search(r"-?\d+", raw)
        if not m:
            return None
        idx = int(m.group())
        if idx == -1:
            return None
        if idx < 0 or idx >= len(self.vp_lib):
            return None
        vp = self.vp_lib[idx]
        if vp.name in exclude:
            return None
        return vp

    def _build_prompt(
        self, belief: WorldBelief, exclude: set[str], preference: str,
    ) -> str:
        if self._template is None:
            return f"Pick a viewpoint index for {preference}, skip {exclude}."
        primary = belief.decomposed.primary_target if belief.decomposed else "?"
        hyp_lines = [
            f"  - {h.label} (entropy={h.label_entropy:.2f}, "
            f"pos_std={h.position_std_m:.2f}m, views={h.observed_in_views})"
            for h in belief.hypotheses
        ] or ["  (无)"]
        vp_lines = [f"  {i}: {vp.name}" for i, vp in enumerate(self.vp_lib)]
        return (
            self._template
            .replace("{primary_target}", primary)
            .replace("{n_hyp}", str(len(belief.hypotheses)))
            .replace("{hyp_list}", "\n".join(hyp_lines))
            .replace("{used_views}", ", ".join(sorted(exclude)) or "(无)")
            .replace("{vp_list}", "\n".join(vp_lines))
            .replace("{preference}", preference)
        )
