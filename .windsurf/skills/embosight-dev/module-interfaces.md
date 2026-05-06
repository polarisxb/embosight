# EmboSight 模块接口速查

## TaskDecomposer

```python
from src.task_decomposer import TaskDecomposer, Subtask, BlindDimension

decomposer = TaskDecomposer(llm_client, prompt_path="prompts/task_decompose.txt")
subtasks: list[Subtask] = decomposer.decompose("我的药瓶在哪里？")

# Subtask 字段
subtask.type           # SubtaskType enum: identify/locate/describe/alert/guide
subtask.target         # str: 具体目标
subtask.priority       # int: 1-5
subtask.blind_dimension # BlindDimension enum: position/distance/tactile/safety/action
subtask.output_format  # str: 输出格式要求
subtask.coverage_status # bool: 是否已被视角覆盖
```

## ActivePlanner

```python
from src.active_planner import ActivePlanner, ViewpointLibrary, Viewpoint, Observation

vp_lib = ViewpointLibrary("configs/viewpoints.yaml")
planner = ActivePlanner(llm_client, viewpoint_lib=vp_lib, max_viewpoints=6, coverage_threshold=0.85)
observations: list[Observation] = planner.plan(subtasks, env)

# Viewpoint 字段
vp.name         # str: 摄像头名 (robot0_agentview_center 等)
vp.position     # tuple[float, float, float]: (x, y, z) cm
vp.orientation  # tuple[float, float, float]: (roll, pitch, yaw) 度
vp.purpose      # str: 用途说明
vp.to_pose()    # -> 6D tuple

# Observation 字段
obs.viewpoint   # Viewpoint
obs.image_path  # str: 图像文件路径
obs.description # str: VLM 描述文本
```

## SceneDescriber

```python
from src.scene_describer import SceneDescriber, StructuredDescription

describer = SceneDescriber(vlm_client, prompt_path="prompts/scene_describer.txt")
desc: StructuredDescription = describer.describe(image_path, viewpoint=vp, subtasks=subtasks)
final: StructuredDescription = describer.aggregate(descriptions_list)
speech: str = desc.to_speech()

# StructuredDescription 字段
desc.objects           # list[str]: 物体名列表
desc.positions         # list[Position]: 方位距离
desc.tactile           # list[str]: 触觉特征
desc.safety_alerts     # list[str]: 安全提示 (含严重等级)
desc.actionable_advice # list[str]: 行动建议
desc.to_dict()         # -> dict
desc.to_speech()       # -> str (语音播报文本)
```

## EnvWrapper

```python
from src.env_wrapper import EnvWrapper, EnvConfig

config = EnvConfig(env_name="PickPlaceCounterToCabinet", robots="PandaMobile")
env = EnvWrapper(config)
obs_dict = env.reset()
observation = env.observe(viewpoint)
success = env.move_arm_to(pose_6d)  # (x,y,z,roll,pitch,yaw)
env.close()
```

## Pipeline

```python
from src.pipeline import EmboSightPipeline

pipeline = EmboSightPipeline("configs/default.yaml")
result: dict = pipeline.run("我的药瓶在哪里？", env)

# result 字段
result["query"]        # str: 原始查询
result["subtasks"]     # list[dict]: 子任务列表
result["observations"] # list[dict]: 视角观察
result["description"]  # dict: 聚合描述
result["speech"]       # str: 语音播报文本
```

## LLMBackend / VLMBackend

```python
from src.llm_backend import LLMBackend
from src.vlm_backend import VLMBackend

llm = LLMBackend()  # 从 .env 读 DEEPSEEK_API_KEY
response: str = llm.generate(user_message, system="...", json_mode=False)

vlm = VLMBackend(model_id="./checkpoints/Qwen2.5-VL-7B-Instruct")
description: str = vlm.describe("image.png", prompt="请描述图像")
```
