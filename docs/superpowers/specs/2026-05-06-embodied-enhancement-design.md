# EmboSight 具身增强设计文档

> 日期: 2026-05-06
> 状态: 待审核

## 目标

将 EmboSight 从"被动观察 + API 调用"升级为**training-free foundation model 驱动的视障具身辅助闭环系统**，新增三个具身创新点。

## 现状

- Pipeline 4 步：分解 → 选视角 → VLM 描述 → 聚合输出
- `move_arm_to()` 是 no-op，手臂不动
- `observe()` 从 reset 时的静态图像中取，视角不随手臂变化
- 无任何物理交互能力

## 创新点总览

| 编号 | 创新点 | 核心思想 |
|---|---|---|
| ④ | 视觉触觉属性推理 + 主动近距离观察 | VLM 世界知识推断可见/可推断触觉属性 + 置信度估计 |
| ⑤ | VLM 驱动的安全约束自动发现与风险感知运动 | VLM 安全分析 → 风险区域 → 路径避让偏置 |
| ⑥ | 语义一致性验证闭环 | 抓取后 VLM 二次描述 → 目标语义匹配 → 用户确认 |

## 架构设计

### Pipeline 升级为 6 步闭环

```
Step 1: TaskDecomposer      → 子任务列表 (五维度)
Step 2: ActivePlanner       → 选视角 + 真实移动手臂 + 实时渲染 [创新④]
Step 3: SceneDescriber      → 每视角五维度描述
Step 4: Aggregate           → 聚合描述 + 语音文本
Step 5: ActionDecider (新)  → LLM 判断是否需要物理动作
Step 6: ActionExecutor (新) → 风险感知运动 + 抓取 + 语义验证 [创新⑤⑥]
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
    │  → object grounding 得到目标候选位置
    │  → move_arm_to(目标上方近距离观察点) + eye_in_hand 特写 [创新④]
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
    │  → 风险感知路径: 从左侧绕过热锅 [创新⑤]
    │  → env.ground_object("药瓶") → 仿真对象 + GT 坐标
    │  → 开夹爪 → 预抓取 → 下降 → 关夹爪 → 提升
    │  → eye-in-hand 拍确认照 → VLM 二次描述 [创新⑥]
    │  → 一致性校验: "磨砂塑料圆柱药瓶" == 目标? ✓
    ▼
语音输出: "注意右侧有热锅。我从左侧拿到了磨砂塑料圆柱药瓶，
          在您左前方25cm处，请伸出左手接取。"
```

## 模块设计

### 全局工程约束

#### 单位约定
- RoboCasa / robosuite 内部位置单位统一为 **meter (m)**
- `configs/viewpoints.yaml` 当前 `position` 是语义参考值，单位为 **cm**，用于描述视角含义，不直接传给控制器
- 新增控制接口只接受 meter 坐标：`target_pos_m: tuple[float, float, float]`
- 若后续需要复用 viewpoint 的语义位置，必须显式调用 `cm_to_m()` 转换，禁止隐式混用 cm/m

#### Object grounding
用户自然语言目标（如"药瓶"）不能直接等同于仿真 body name。必须新增 grounding 层：

```python
@dataclass
class ObjectGrounding:
    user_target: str          # "药瓶"
    canonical_name: str       # "medicine_bottle"
    sim_body_name: str        # RoboCasa 内部 body name
    position_m: tuple[float, float, float]
    confidence: float
    source: str               # "alias_map" | "fuzzy_match" | "llm_match"
```

grounding 顺序:
1. `object_alias_map` 精确匹配中文/英文别名
2. RoboCasa body name 模糊匹配
3. LLM 在候选 body list 中选择最可能目标
4. 仍失败则返回 `None`，ActionExecutor 不执行抓取，只播报无法定位

### 1. env_wrapper.py — 真实手臂控制

#### move_arm_to(target_pos_m, max_steps, threshold_m)
- 获取 `obs['robot0_eef_pos']` 当前末端位置
- 计算 delta = target - current
- 归一化 + 限幅 (max 0.05m/step)
- 构建 action 向量: `action[0:3] = pos_delta`, 其余为 0
- `env.step(action)` 循环直到收敛或超时
- 每次 step 后更新 `_latest_obs`

#### observe(viewpoint) — 改为实时渲染
- 不再从 reset 时的静态 obs 取图像
- 如果 viewpoint 是 eye_in_hand: 只负责读取当前末端相机图像；目标近景位置由 ActivePlanner 或 ActionExecutor 在调用 observe 前显式移动
- 其他固定摄像头: 做一次 zero-action step 刷新图像

#### grasp_at(target_pos_m, pre_grasp_height_m=0.10)
1. 开夹爪 (gripper_action = -1, 执行 N 步)
2. `pre_grasp_pos_m = target_pos_m + z_offset(pre_grasp_height_m)` — 计算预抓取位置
3. move_arm_to(pre_grasp_pos_m) — 移动到预抓取位置
4. move_arm_to(target_pos_m) — 下降到抓取位置
5. 关夹爪 (gripper_action = +1, 执行 N 步)
6. move_arm_to(pre_grasp_pos_m) — 提升
7. 返回成功/失败

#### ground_object(user_target) -> ObjectGrounding | None
- 输入用户目标名（中文或英文）
- 用 alias map + body name 模糊匹配 + LLM 候选选择定位仿真物体
- 返回 meter 单位世界坐标
- 不再暴露裸 `get_object_pos("药瓶")` 作为主接口，避免中文目标名直接匹配仿真内部名失败

### 2. action_decider.py — 新模块

```python
@dataclass
class ActionPlan:
    action_type: str                 # "grasp" | "point" | "none"
    target_object: str               # "药瓶"
    reason: str                      # "用户请求'帮我拿'"
    safety_constraints: list[str]    # ["避开右侧热锅"]
    require_confirmation: bool = True
```

- 输入: query + StructuredDescription
- LLM 判断意图: 含"拿/取/递/给我"→ grasp; 含"在哪/有什么"→ none
- 从 safety_alerts 提取安全约束
- 输出 ActionPlan

### 3. action_executor.py — 新模块

```python
@dataclass
class NoGoZone:
    name: str
    center_m: tuple[float, float, float]
    radius_m: float
    risk_level: str       # "high" | "medium" | "low"
    reason: str


@dataclass
class ActionResult:
    success: bool
    executed: bool
    grounding: ObjectGrounding | None
    verification_match: bool
    message: str


class ActionExecutor:
    def execute(self, plan: ActionPlan, env, vlm, describer) -> ActionResult:
        # 1. 将用户目标 grounding 到仿真对象
        grounding = env.ground_object(plan.target_object)
        if grounding is None:
            return ActionResult(
                success=False,
                executed=False,
                grounding=None,
                verification_match=False,
                message=f"无法定位目标物体: {plan.target_object}",
            )
        target_pos_m = grounding.position_m
        
        # 2. 风险感知路径规划 [创新⑤]
        current_ee_pos_m = env.get_eef_pos()
        pre_grasp_pos_m = self._pre_grasp_pos(target_pos_m)
        safe_waypoints = self._plan_safe_path(
            start_m=current_ee_pos_m,
            goal_m=pre_grasp_pos_m,
            no_go_zones=self._extract_no_go_zones(plan.safety_constraints, env)
        )
        
        # 3. 沿风险感知 waypoint 移动
        for wp in safe_waypoints:
            env.move_arm_to(wp)
        
        # 4. 执行抓取
        env.grasp_at(target_pos_m)
        
        # 5. 验证闭环 [创新⑥]
        verify_obs = env.observe(env.eye_in_hand_viewpoint())
        verify_desc = describer.describe(verify_obs.image_path)
        match = self._verify_consistency(plan.target_object, verify_desc)
        
        return ActionResult(
            success=match,
            executed=True,
            grounding=grounding,
            verification_match=match,
            message=verify_desc.to_speech(),
        )
```

#### 风险感知运动 (创新⑤)
- `_extract_no_go_zones()`: 从 safety_alerts 解析危险物体 → 查询 GT 坐标 → 生成半径 r 的球形禁区
- `_plan_safe_path()`: 起点 → (中间避障 waypoint) → 预抓取点
- 简单实现: 检测直线路径是否穿过禁区，若是则添加绕行 waypoint
- 不声称形式化安全保证；只作为 VLM 风险感知的运动偏置

#### 抓取验证 (创新⑥)
- `_verify_consistency()`: 比较 VLM 对手中物体的描述与之前目标描述的关键词匹配度
- 匹配维度: 材质、形状、颜色
- 阈值: ≥ 0.6 认为一致
- 目标是回答"抓到的是不是用户要的"，这是视障用户无法目视确认时的特有需求；不是单纯判断 gripper 是否闭合或物体是否被抬起

### 4. prompts/action_decider.txt — 新 prompt

输入: 用户查询 + 场景描述
输出 JSON:
```json
{
  "action_type": "grasp",
  "target_object": "药瓶",
  "reason": "用户说'帮我拿'，明确需要物理操作",
  "safety_constraints": ["右侧45cm处有热锅，需从左侧接近"],
  "require_confirmation": true
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
| 改 | `src/env_wrapper.py` | 实现 move_arm_to, grasp_at, ground_object, get_eef_pos, eye_in_hand_viewpoint, 改 observe |
| 新 | `src/action_decider.py` | LLM 行动决策模块 |
| 新 | `src/action_executor.py` | 风险感知运动 + 抓取 + 语义验证闭环 |
| 新 | `prompts/action_decider.txt` | 行动决策 prompt |
| 新 | `configs/object_aliases.yaml` | 中文目标名到 RoboCasa 对象名的别名映射 |
| 改 | `src/pipeline.py` | 新增 Step 5-6 |
| 改 | `configs/default.yaml` | 新增 action_decider/executor 配置 |
| 新 | `scripts/test_embodied.py` | 具身功能集成测试 |

## 测试策略

1. **单元测试**: move_arm_to 收敛性, grasp_at 流程, ActionDecider 意图判断
2. **集成测试**: Pipeline 6 步全流程 (test_embodied.py)
3. **对比实验**: 有/无触觉探索的描述完整度, 有/无风险感知运动的路径安全性

## 论文定位

**方法**: Training-free zero-shot transfer — 不在目标任务上做数据标注、梯度更新或微调，依赖基础模型 (LLM+VLM) 的世界知识与结构化 prompt
**应用**: 视障辅助 — 五维度感知 + 风险感知操作 + 语音反馈

### Zero-shot transfer 的严格定义

本文不声称经典 Zero-Shot Learning 中的"未见类别识别"，也不声称模型没有预训练数据。本文的 zero-shot 指：

1. **No task-specific training data**: 不收集视障厨房辅助专用标注数据
2. **No fine-tuning**: 不对 DeepSeek/Qwen2.5-VL 做梯度更新
3. **No human demonstrations**: 不用任务演示训练抓取/规划策略
4. **Prompt-guided transfer**: 通过结构化 prompt、模板和约束把基础模型迁移到视障辅助任务

传统 pipeline 每个环节需要训练专门模型（NLU 分类器、RL 策略、目标检测器、抓取策略、安全分类器）。EmboSight 用 foundation model reasoning + 工程约束替代这些 task-specific training 模块：

| 环节 | 传统方法 (需训练) | EmboSight (training-free transfer) |
|---|---|---|
| 任务理解 | NLU 分类器 | LLM + prompt + 模板检索 |
| 视角选择 | RL 策略网络 | LLM 信息增益推理 |
| 场景描述 | captioning 模型 | VLM + 五维度 prompt |
| 物体识别 | 目标检测器 | VLM 语义理解 |
| 风险感知运动 | 训练 safety classifier / 人工规则 | VLM 分析 → 风险约束 |
| 抓取验证 | 力觉/视觉检测器 | VLM 语义一致性 |

明确限制: 当前执行层使用 RoboCasa GT 物体 3D 坐标做抓取 grounding。这不影响高层感知/决策的 training-free 定位，但应在论文中作为限制说明；未来可替换为 RGB-D 或 3D perception。

### 创新④学术依据 — 视觉触觉属性推理

**对标论文**:
- **NeRF2Physics (CVPR 2024)**: LLM 从视觉外观 zero-shot 推断物理属性 (质量/摩擦力/硬度)
- **Tactile-VLA (ICLR 2026, 清华)**: VLM 先验知识包含物理交互语义理解
- **Cosmos-Reason1 (NVIDIA 2025)**: LLM/VLM 可推理 Physical Common Sense

**我们的区别**: 不估计精确物理量，输出面向视障者的自然语言触觉描述 + 置信度标签
- high: 基于可见视觉特征 → "不锈钢材质，光滑硬质"
- medium: 基于尺寸+材质推断 → "约200g"
- low: 基于场景上下文推断 → "可能是热的，建议先不要直接触碰"

边界声明:
- 可较可靠推断: 材质、纹理、形状、粗略握感、可见危险线索
- 只能低置信度推断: 温度、重量、湿滑程度、柔软度
- 不做保证: 实时温度、真实重量、内部液体、不可见污渍/破损
- 对低置信度属性必须使用提醒式播报，不使用确定性语气

### 创新⑤学术依据 — VLM 安全约束自动发现与风险感知运动

**对标论文**:
- **Safety Chip (ICRA 2024, Brown)**: NL → LTL 公式 → 动作剪枝 (100% safety rate)，但需人工预定义安全规则
- **SAFER (arXiv 2025)**: 多 LLM 框架，专门的 Safety Agent 做安全评估，但需要独立 Agent
- **VoxPoser (CoRL 2023, Stanford)**: LLM → 3D affordance + avoidance map，但无视障场景适配
- **Code-as-Monitor (CVPR 2025)**: LLM 生成约束代码做主动+被动失败检测

**我们的区别**: 无需人工预定义规则、无需额外 Safety Agent。VLM 看图自动发现危险源，同时生成机器人运动避让偏置 + 视障者语音安全播报 ("我从左侧绕过热锅来拿药瓶")。本文不声称达到 Safety Chip/LTL 或 CBF 级别的形式化安全保证。

### 创新⑥学术依据 — 语义一致性验证闭环

**对标论文**:
- **AHA (ICLR 2025, NVIDIA)**: 专门训练 VLM 做操作失败检测 (比 GPT-4 高 21.4%)，但需微调，非零样本
- **NovaPlan (arXiv 2025)**: VLM 闭环监控 sub-goal 完成度，零样本，但只验证"做没做到"
- **Code-as-Monitor (CVPR 2025)**: LLM 生成检查代码，reactive + proactive，但验证的是约束违反

**我们的区别**: 验证的不是"操作是否成功"而是"抓到的是不是用户要的"——语义层面的一致性校验，VLM 对手中物体做二次描述 → 与目标描述匹配 → 转化为视障者确认信息。这个需求来自视障场景：用户无法目视确认机器人拿到的物体，必须由系统给出可理解、可追责的确认描述。

### 六大创新点归属

| 编号 | 创新点 | 对标论文 | 我们的差异化 |
|---|---|---|---|
| ① | IDF 加权模板检索 + 五维度分解 | — | 视障专属维度保证 |
| ② | LLM-NBV 信息增益规划 | NBV-Net (需训练) | Training-free transfer + 覆盖率早停 |
| ③ | VLM 安全分级描述 | — | 五维度 + 严重度分级 |
| ④ | 视觉触觉属性推理 | NeRF2Physics, Tactile-VLA | 置信度标签 + 主动近距离观察 |
| ⑤ | VLM 安全约束自动发现与风险感知运动 | Safety Chip, SAFER, VoxPoser | 无需预定义规则 + 视障播报，不声称形式化安全保证 |
| ⑥ | 语义一致性验证闭环 | AHA, NovaPlan | 语义匹配而非动作验证 |

- 创新①②③: Training-free 感知 (Day 2 已实现)
- 创新④⑤⑥: Training-free 具身操作 (本次实现)

## 实验设计

### 指标定义

- **DCR (Dimension Coverage Rate)**: 输出中有效覆盖的视障维度数 / 5。有效覆盖要求该维度字段非空且与场景相关。
- **Grounded DCR (G-DCR)**: 人工检查每个维度是否被图像/仿真状态支持，过滤只填字段但无依据的情况。
- **Hallucination Rate (HR)**: 描述中无图像/仿真依据的事实数 / 总事实数。
- **Info Gain / View**: 当前视角相对已有观察新增的有效事实数，包括新增物体、位置、距离、触觉属性、安全风险和行动建议。
- **TAR (Tactile Attribute Recall)**: 命中的触觉属性数 / GT 触觉属性数。
- **HZIR (Hazard Zone Intrusion Rate)**: 轨迹点落入危险区域的点数 / 总轨迹点数。
- **Semantic Accuracy (SA)**: 抓取后验证为目标物体且人工/GT 确认为目标的比例。

### Exp 1: 五维度覆盖消融

**目的**: 验证五维度强制覆盖对描述完整性的影响

| 配置 | 说明 |
|---|---|
| Full (Ours-Decomposer) | 只启用五维度强制覆盖 + 模板检索 |
| w/o dim-force | 去掉维度强制覆盖，让 LLM 自由分解 |
| w/o templates | 去掉模板检索，纯 LLM 分解 |

**度量指标**:
- **DCR / G-DCR**: 避免只靠 prompt 填满字段刷分
- **Useful Detail Score (UDS)**: 人工评分 1-5，评估对视障者是否有帮助
- **Hallucination Rate (HR)**: 越低越好
- **数据**: 30 条视障查询 × 5 个场景 = 150 个测试样例

### Exp 2: 主动视角规划消融

**目的**: 验证 LLM-NBV 对信息增益的影响

| 配置 | 说明 |
|---|---|
| LLM-NBV (Ours) | LLM 推理信息增益 + 覆盖率早停 |
| All-Scan | 遍历所有 6 个视角 |
| Random | 随机选视角 |
| Fixed-3 | 固定 3 个视角 (center + left + right) |

**度量指标**:
- **G-DCR**: grounded 维度覆盖率
- **#Views**: 使用的视角数 (效率)
- **Info Gain / View**: 每个视角新增有效事实数
- **数据**: 同 Exp 1

### Exp 3: 触觉属性推理消融 [创新④]

**目的**: 验证 eye-in-hand 近距离观察对触觉描述的增益

| 配置 | 说明 |
|---|---|
| Full (Ours) | 远景 + 近景 + 置信度标签 |
| w/o close-up | 只用远景，不移动手臂近距离观察 |
| w/o confidence | 有近景但不标注置信度 |

**度量指标**:
- **TAR**: 触觉属性命中率
- **Confidence Calibration**: 高置信度属性的准确率应 > 中 > 低
- **HR-tactile**: 触觉相关幻觉率
- **GT 标注 schema**: material, texture, hardness, graspability, risk_temperature
- **数据**: 50 个厨房常见物品，每个物品人工标注上述 5 类触觉属性

### Exp 4: 安全约束自动发现消融 [创新⑤]

**目的**: 验证 VLM 驱动的安全约束发现和风险感知运动

| 配置 | 说明 |
|---|---|
| Full (Ours) | VLM 安全分析 → no-go zones → 风险感知绕行 |
| w/o safety | 直线路径，无安全约束 |
| Manual-rules | 人工标注危险源生成 no-go zones (规则上限，不对标形式化 Safety Chip) |

**度量指标**:
- **HZIR**: 手臂轨迹点进入危险区域的比率
- **Safety Alert Recall**: 场景中实际危险源的检出率
- **Path Length Ratio**: 风险感知路径 / 直线路径长度比 (效率代价)
- **GT 标注 schema**: hazard_object, position_m, risk_level, radius_m, reason
- **数据**: 10 个不同厨房场景，每个 2-4 个危险源

### Exp 5: 抓取验证闭环消融 [创新⑥]

**目的**: 验证语义一致性校验对抓取可靠性的影响

| 配置 | 说明 |
|---|---|
| Full (Ours) | 抓取 + VLM 二次描述 + 语义匹配 |
| w/o verify | 抓了就算完，不验证 |
| Gripper-state-only | 只用夹爪状态/物体抬升判断是否抓到，不验证语义 |

**度量指标**:
- **SA**: 抓到的是不是用户要的物体
- **False Positive Rate (FPR)**: 抓错了但没发现的比率
- **Recovery Rate**: 发现错误后重试成功的比率
- **数据**: 每场景 3 个相似物体 (如: 药瓶/水瓶/调味瓶)，20 个场景

### Exp 6: 端到端对比

**目的**: 与 baseline 方法对比整体性能

| 方法 | 说明 |
|---|---|
| **EmboSight (Ours)** | 完整 6 步 pipeline |
| VLM-only | 单张图 + VLM 描述，无主动探索/抓取 |
| LLM-action-only | LLM 选高层动作 + 固定描述模板，不做五维度覆盖/主动探索/验证 |
| Random-explore | 随机视角 + 直线路径 + 无验证 |

**度量指标**:
- **Task Success Rate**: 查询意图完成率
- **G-DCR**: grounded 五维度覆盖率
- **Safety Score**: 安全指标综合分
- **User Satisfaction** (可选): 模拟用户评分
- **数据**: 50 条多样化查询 × 10 个场景

## 实施阶段

| 阶段 | 目标 | 完成标准 |
|---|---|---|
| Phase 1 | 真实手臂移动 + observe 实时刷新 | `move_arm_to()` 能让 eef 收敛到 meter 坐标；连续观察图像随 step 更新 |
| Phase 2 | Object grounding + grasp_at | 中文目标能映射到 RoboCasa 对象；预抓取/下降/闭合/提升流程可执行 |
| Phase 3 | ActionDecider + ActionExecutor | 查询触发动作决策；无法 grounding 时安全退出；成功时输出 ActionResult |
| Phase 4 | 三个具身创新实验脚本 | 触觉近景、安全约束、语义验证三个消融可复现 |
| Phase 5 | 端到端评估 | 完整 pipeline 输出实验指标表 |

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| OSC 控制不收敛 | 设 max_steps=200 + 超时兜底 |
| ground_object 找不到物体 | alias map + body list 模糊匹配 + LLM 候选选择；失败时不执行抓取 |
| 安全区域划定不准 | 使用保守半径 (0.15m), 宁可绕远 |
| VLM 验证一致性低 | 降低匹配阈值 + 允许重试 1 次 |
| 实验规模不够 | 先小规模验证 (10场景)，通了再扩到 50 |
