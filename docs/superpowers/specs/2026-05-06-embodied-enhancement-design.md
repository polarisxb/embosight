# EmboSight 具身增强设计文档

> 日期: 2026-05-06
> 状态: 待审核

## 目标

将 EmboSight 从"被动观察 + API 调用"升级为**完整具身闭环系统**，新增三个具身创新点。

## 现状

- Pipeline 4 步：分解 → 选视角 → VLM 描述 → 聚合输出
- `move_arm_to()` 是 no-op，手臂不动
- `observe()` 从 reset 时的静态图像中取，视角不随手臂变化
- 无任何物理交互能力

## 创新点总览

| 编号 | 创新点 | 核心思想 |
|---|---|---|
| ④ | 视觉触觉属性推理 + 主动近距离观察 | VLM 世界知识推断触觉属性 + 置信度估计 |
| ⑤ | 安全感知运动规划 | VLM 安全分析 → 约束区域 → 避障路径 |
| ⑥ | 抓取验证闭环 | 抓取后 VLM 二次描述 → 一致性校验 → 用户确认 |

## 架构设计

### Pipeline 升级为 6 步闭环

```
Step 1: TaskDecomposer      → 子任务列表 (五维度)
Step 2: ActivePlanner       → 选视角 + 真实移动手臂 + 实时渲染 [创新④]
Step 3: SceneDescriber      → 每视角五维度描述
Step 4: Aggregate           → 聚合描述 + 语音文本
Step 5: ActionDecider (新)  → LLM 判断是否需要物理动作
Step 6: ActionExecutor (新) → 安全规划 + 抓取 + 验证 [创新⑤⑥]
```

### 数据流

```
用户查询 "帮我拿药瓶"
    │
    ▼
[Step 1] TaskDecomposer
    │  → subtasks: [{safety, 药瓶周围危险}, {position, 药瓶}, {distance, 药瓶},
    │               {tactile, 药瓶触觉}, {action, 安全取药瓶}]
    ▼
[Step 2] ActivePlanner + 真实手臂控制
    │  → 全景 agentview_center (远景, position+safety)
    │  → LLM: "触觉维度未覆盖，需要近距离观察"
    │  → move_arm_to(药瓶上方) + eye_in_hand 特写 [创新④]
    │  → 覆盖率 100%, 早停
    ▼
[Step 3-4] SceneDescriber + Aggregate
    │  → 远景: 位置+安全
    │  → 近景: 触觉+材质+表面纹理 (eye-in-hand 补充)
    │  → 聚合五维度描述
    ▼
[Step 5] ActionDecider
    │  → LLM: "帮我拿" → action_type=grasp, target="药瓶"
    │  → "药瓶在哪" → action_type=none
    ▼
[Step 6] ActionExecutor [创新⑤⑥]
    │  → 从 safety_alerts 提取 no-go zones (热锅区域)
    │  → 安全路径规划: 从左侧绕过热锅 [创新⑤]
    │  → env.get_object_pos("药瓶") → GT 坐标
    │  → 开夹爪 → 预抓取 → 下降 → 关夹爪 → 提升
    │  → eye-in-hand 拍确认照 → VLM 二次描述 [创新⑥]
    │  → 一致性校验: "磨砂塑料圆柱药瓶" == 目标? ✓
    ▼
语音输出: "注意右侧有热锅。我从左侧拿到了磨砂塑料圆柱药瓶，
          在您左前方25cm处，请伸出左手接取。"
```

## 模块设计

### 1. env_wrapper.py — 真实手臂控制

#### move_arm_to(target_pos, max_steps, threshold)
- 获取 `obs['robot0_eef_pos']` 当前末端位置
- 计算 delta = target - current
- 归一化 + 限幅 (max 5cm/step)
- 构建 action 向量: `action[0:3] = pos_delta`, 其余为 0
- `env.step(action)` 循环直到收敛或超时
- 每次 step 后更新 `_latest_obs`

#### observe(viewpoint) — 改为实时渲染
- 不再从 reset 时的静态 obs 取图像
- 如果 viewpoint 是 eye_in_hand: 先 move_arm_to 目标位置, 然后取 `_latest_obs` 中的最新图像
- 其他固定摄像头: 做一次 zero-action step 刷新图像

#### grasp_at(target_pos, pre_grasp_height=0.10)
1. 开夹爪 (gripper_action = -1, 执行 N 步)
2. move_arm_to(target + [0,0,pre_grasp_height]) — 预抓取位置
3. move_arm_to(target) — 下降到抓取位置
4. 关夹爪 (gripper_action = +1, 执行 N 步)
5. move_arm_to(target + [0,0,pre_grasp_height]) — 提升
6. 返回成功/失败

#### get_object_pos(object_name) -> np.ndarray
- 从 `env.sim.data` 中查询物体的世界坐标
- 遍历 `env.sim.model.body_names` 匹配物体名
- 返回 (x, y, z)

### 2. action_decider.py — 新模块

```python
@dataclass
class ActionPlan:
    action_type: str       # "grasp" | "point" | "none"
    target_object: str     # "药瓶"
    reason: str            # "用户请求'帮我拿'"
    safety_constraints: list[str]  # ["避开右侧热锅"]
```

- 输入: query + StructuredDescription
- LLM 判断意图: 含"拿/取/递/给我"→ grasp; 含"在哪/有什么"→ none
- 从 safety_alerts 提取安全约束
- 输出 ActionPlan

### 3. action_executor.py — 新模块

```python
class ActionExecutor:
    def execute(self, plan: ActionPlan, env, vlm, describer) -> ActionResult:
        # 1. 获取目标 GT 坐标
        target_pos = env.get_object_pos(plan.target_object)
        
        # 2. 安全路径规划 [创新⑤]
        safe_waypoints = self._plan_safe_path(
            start=current_ee_pos,
            goal=target_pos,
            no_go_zones=self._extract_no_go_zones(plan.safety_constraints, env)
        )
        
        # 3. 沿安全路径移动
        for wp in safe_waypoints:
            env.move_arm_to(wp)
        
        # 4. 执行抓取
        env.grasp_at(target_pos)
        
        # 5. 验证闭环 [创新⑥]
        verify_obs = env.observe(eye_in_hand_viewpoint)
        verify_desc = describer.describe(verify_obs.image_path)
        match = self._verify_consistency(plan.target_object, verify_desc)
        
        return ActionResult(success=match, description=verify_desc)
```

#### 安全路径规划 (创新⑤)
- `_extract_no_go_zones()`: 从 safety_alerts 解析危险物体 → 查询 GT 坐标 → 生成半径 r 的球形禁区
- `_plan_safe_path()`: 起点 → (中间避障 waypoint) → 预抓取点 → 目标点
- 简单实现: 检测直线路径是否穿过禁区，若是则添加绕行 waypoint

#### 抓取验证 (创新⑥)
- `_verify_consistency()`: 比较 VLM 对手中物体的描述与之前目标描述的关键词匹配度
- 匹配维度: 材质、形状、颜色
- 阈值: ≥ 0.6 认为一致

### 4. prompts/action_decider.txt — 新 prompt

输入: 用户查询 + 场景描述
输出 JSON:
```json
{
  "action_type": "grasp",
  "target_object": "药瓶",
  "reason": "用户说'帮我拿'，明确需要物理操作",
  "safety_notes": ["右侧45cm处有热锅，需从左侧接近"]
}
```

### 5. pipeline.py — 扩展

```python
def run(self, query, env):
    # Step 1-4: 现有流程不变
    subtasks = self.task_decomposer.decompose(query)
    observations = self.active_planner.plan(subtasks, env)  # 真实移动
    descriptions = [self.scene_describer.describe(...) for obs in observations]
    final_desc = self.scene_describer.aggregate(descriptions)
    
    # Step 5: 行动决策 (新增)
    action_plan = self.action_decider.decide(query, final_desc)
    
    # Step 6: 行动执行 (新增)
    action_result = None
    if action_plan.action_type != "none":
        action_result = self.action_executor.execute(
            action_plan, env, self.vlm, self.scene_describer
        )
    
    # 生成最终语音 (含行动结果)
    speech = self._build_final_speech(final_desc, action_plan, action_result)
    return {...}
```

## 文件变动清单

| 操作 | 文件 | 说明 |
|---|---|---|
| 改 | `src/env_wrapper.py` | 实现 move_arm_to, grasp_at, get_object_pos, 改 observe |
| 新 | `src/action_decider.py` | LLM 行动决策模块 |
| 新 | `src/action_executor.py` | 安全规划 + 抓取 + 验证闭环 |
| 新 | `prompts/action_decider.txt` | 行动决策 prompt |
| 改 | `src/pipeline.py` | 新增 Step 5-6 |
| 改 | `configs/default.yaml` | 新增 action_decider/executor 配置 |
| 新 | `scripts/test_embodied.py` | 具身功能集成测试 |

## 测试策略

1. **单元测试**: move_arm_to 收敛性, grasp_at 流程, ActionDecider 意图判断
2. **集成测试**: Pipeline 6 步全流程 (test_embodied.py)
3. **对比实验**: 有/无触觉探索的描述完整度, 有/无安全规划的路径安全性

## 论文定位

**方法**: 零样本具身智能 — 基础模型 (LLM+VLM) 替代所有需要训练的模块
**应用**: 视障辅助 — 五维度感知 + 安全操作 + 语音反馈

### 零样本体现

传统 pipeline 每个环节需要训练专门模型（NLU 分类器、RL 策略、目标检测器、抓取策略、安全分类器）。
EmboSight 全部用 LLM/VLM zero-shot 替代：

| 环节 | 传统方法 (需训练) | EmboSight (zero-shot) |
|---|---|---|
| 任务理解 | NLU 分类器 | LLM + prompt + 模板检索 |
| 视角选择 | RL 策略网络 | LLM 信息增益推理 |
| 场景描述 | captioning 模型 | VLM + 五维度 prompt |
| 物体识别 | 目标检测器 | VLM 语义理解 |
| 安全路径 | 训练 safety classifier | VLM 分析 → LLM 约束 |
| 抓取验证 | 力觉/视觉检测器 | VLM 语义一致性 |

唯一用到 GT: 物体 3D 坐标 (单目视觉估深度不可靠, 未来可用深度相机替代)

### 创新④学术依据

- **NeRF2Physics (CVPR 2024)**: 证明 LLM 可从视觉外观 zero-shot 推断物理属性 (质量/摩擦力/硬度)
- **Tactile-VLA (ICLR 2026, 清华)**: 证明 VLM 先验知识包含物理交互的语义理解
- **Cosmos-Reason1 (NVIDIA 2025)**: LLM/VLM 可推理 Physical Common Sense

我们的独特性: 不估计精确物理量，而是输出**面向视障者的自然语言触觉描述 + 置信度标签**

置信度机制:
- high: 基于可见视觉特征 (材质纹理、形状) → "不锈钢材质，光滑硬质"
- medium: 基于尺寸+材质推断 (重量估计) → "约200g"
- low: 基于场景上下文推断 (温度等) → "可能是热的，建议先不要直接触碰"

语音播报中明确区分确定/不确定信息，对视障者更负责任

### 六大创新点归属

- 创新①②③: 零样本感知 (Day 2 已实现)
- 创新④⑤⑥: 零样本具身操作 (本次实现)

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| OSC 控制不收敛 | 设 max_steps=200 + 超时兜底 |
| get_object_pos 找不到物体 | 遍历 sim 所有 body, 模糊匹配名称 |
| 安全区域划定不准 | 使用保守半径 (15cm), 宁可绕远 |
| VLM 验证一致性低 | 降低匹配阈值 + 允许重试 1 次 |
