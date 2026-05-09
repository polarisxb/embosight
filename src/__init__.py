"""EmboSight: 零样本视障具身辅助智能体

A Zero-Shot Embodied Visual Assistant for the Visually Impaired
via Active Perception and Multimodal Large Models.

主要模块:
    - task_decomposer:  创新点① 零样本视障任务分解
    - active_planner:   创新点② 零样本主动视角规划
    - scene_describer:  创新点③ 零样本视障友好描述
    - pipeline:         端到端主流程
    - llm_backend:      LLM 后端封装（DeepSeek API）
    - vlm_backend:      VLM 后端封装（Qwen2.5-VL 本地）
    - env_wrapper:      仿真环境封装（RoboCasa）
    - eval:             零样本评估脚本
    - utils:            工具函数
"""

__version__ = "0.1.0"
__author__ = "EmboSight Author"
__license__ = "MIT"


# ============================================================
# v1 公开 API (设计稿 §15 Appendix B)
# ============================================================
# 主入口
from src.agent import EmboSightAgent

# 数据结构 (供脚本/外部测试 import 使用)
from src.world_belief import (
    Action,
    BeliefSnapshot,
    Constraint,
    DecomposedTask,
    EpisodeResult,
    Evidence,
    GraspAttempt,
    GraspCandidate,
    Hypothesis,
    Pose,
    WorldBelief,
)

# 辅助
from src.episode_logger import EpisodeLogger
from src.user_channel import (
    CLIUserChannel,
    FakeUserChannel,
    UserChannel,
    VoiceUserChannel,
)
from src.vlm_cache import VLMCache

__all__ = [
    # 主入口
    "EmboSightAgent",
    # 数据结构
    "Action", "BeliefSnapshot", "Constraint", "DecomposedTask",
    "EpisodeResult", "Evidence", "GraspAttempt", "GraspCandidate",
    "Hypothesis", "Pose", "WorldBelief",
    # 辅助
    "EpisodeLogger",
    "CLIUserChannel", "FakeUserChannel", "UserChannel", "VoiceUserChannel",
    "VLMCache",
]