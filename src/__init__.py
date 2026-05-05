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