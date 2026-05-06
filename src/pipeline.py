"""EmboSight 主流程

6 步 Pipeline: 任务分解 → 主动视角 → 场景描述 → 聚合 → 行动决策 → 行动执行

使用示例:
    >>> from src.pipeline import EmboSightPipeline
    >>> pipeline = EmboSightPipeline("configs/default.yaml")
    >>> result = pipeline.run("我的药瓶在哪？", env)
    >>> print(result["speech"])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from .action_decider import ActionDecider
from .action_executor import ActionExecutor
from .active_planner import ActivePlanner, Observation, ViewpointLibrary
from .llm_backend import LLMBackend
from .scene_describer import SceneDescriber, StructuredDescription
from .task_decomposer import Subtask, TaskDecomposer
from .vlm_backend import VLMBackend

logger = logging.getLogger(__name__)


class EmboSightPipeline:
    """EmboSight 端到端 pipeline"""

    def __init__(self, config_path: str = "configs/default.yaml") -> None:
        """从 YAML 配置初始化 pipeline"""
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        logger.info(f"加载配置: {cfg_path}")

        self.llm = LLMBackend(**self.config.get("llm", {}))
        self.vlm = VLMBackend(**self.config.get("vlm", {}))

        self.task_decomposer = TaskDecomposer(
            self.llm,
            prompt_path=self.config.get("prompts", {}).get(
                "task_decompose", "prompts/task_decompose.txt"
            ),
        )
        self.viewpoint_lib = ViewpointLibrary(
            self.config.get("viewpoints_path", "configs/viewpoints.yaml")
        )
        self.active_planner = ActivePlanner(
            llm_client=self.llm,
            viewpoint_lib=self.viewpoint_lib,
            **self.config.get("active_planner", {}),
        )
        self.scene_describer = SceneDescriber(
            self.vlm,
            prompt_path=self.config.get("prompts", {}).get(
                "scene_describer", "prompts/scene_describer.txt"
            ),
        )
        self.action_decider = ActionDecider(
            self.llm,
            prompt_path=self.config.get("prompts", {}).get(
                "action_decider", "prompts/action_decider.txt"
            ),
        )
        self.action_executor = ActionExecutor(
            scene_describer=self.scene_describer,
        )

    def run(self, query: str, env) -> dict[str, Any]:
        """主入口: 执行完整 pipeline

        Args:
            query: 视障者自然语言查询
            env: 仿真环境对象（来自 src/env_wrapper.py）

        Returns:
            完整结果字典，包括:
                query: 原查询
                subtasks: 分解出的子任务
                observations: 多视角观察
                description: 聚合后的结构化描述
                action_plan: 行动决策 (type/target/reason/constraints)
                action_result: 行动执行结果 (success/message/waypoints) 或 None
                speech: TTS 语音文本
        """
        logger.info("=" * 60)
        logger.info(f"Pipeline 启动 | 查询: {query}")
        logger.info("=" * 60)

        subtasks = self.task_decomposer.decompose(query)
        logger.info(f"[Step 1] 分解出 {len(subtasks)} 个子任务")
        for t in subtasks:
            logger.info(f"  - {t}")

        observations = self.active_planner.plan(subtasks, env)
        logger.info(f"[Step 2] 采集 {len(observations)} 个视角")

        descriptions: list[StructuredDescription] = []
        for i, obs in enumerate(observations):
            desc = self.scene_describer.describe(
                image_path=obs.image_path,
                viewpoint=obs.viewpoint,
                subtasks=subtasks,
            )
            descriptions.append(desc)
            obs.description = desc.to_speech()
            logger.info(f"[Step 3.{i+1}] 视角 {obs.viewpoint.name} 描述完成")

        final_desc = self.scene_describer.aggregate(descriptions)
        speech = final_desc.to_speech()
        logger.info("[Step 4] 聚合完成")

        # Step 5: 行动决策
        logger.info("[Step 5] 行动决策")
        action_plan = self.action_decider.decide(query, final_desc)
        logger.info(
            f"  → {action_plan.action_type} "
            f"target='{action_plan.target_object}'"
        )

        # Step 6: 行动执行
        action_result = None
        if action_plan.needs_execution:
            logger.info("[Step 6] 行动执行")
            action_result = self.action_executor.execute(action_plan, env)
            logger.info(
                f"  → success={action_result.success}, "
                f"msg={action_result.message}"
            )
            speech += f"\n[行动结果] {action_result.message}"
        else:
            logger.info("[Step 6] 跳过 (无需物理动作)")

        logger.info(f"最终输出: {speech}")

        return {
            "query": query,
            "subtasks": [s.to_dict() for s in subtasks],
            "observations": [
                {
                    "viewpoint": {
                        "name": o.viewpoint.name,
                        "position": list(o.viewpoint.position),
                        "orientation": list(o.viewpoint.orientation),
                    },
                    "image_path": o.image_path,
                    "description": o.description,
                }
                for o in observations
            ],
            "description": final_desc.to_dict(),
            "action_plan": {
                "action_type": action_plan.action_type,
                "target_object": action_plan.target_object,
                "reason": action_plan.reason,
                "safety_constraints": action_plan.safety_constraints,
            },
            "action_result": (
                {
                    "success": action_result.success,
                    "executed": action_result.executed,
                    "message": action_result.message,
                    "verification_match": action_result.verification_match,
                    "waypoints": action_result.waypoints,
                }
                if action_result
                else None
            ),
            "speech": speech,
        }


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("[EmboSightPipeline] 模块加载测试")
    print("注: 完整运行需要配置 .env + 启动 GPU + 仿真环境")