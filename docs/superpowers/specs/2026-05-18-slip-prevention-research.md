# 圆形/低摩擦物体抓取打滑：问题分析与文献调研

**Created**: 2026-05-18  
**Owner**: Cascade + user  
**Status**: Research / Pre-design  
**Context**: GPU 正在跑 commit `dacf9e7`（lift Phase 1 移除）。本文整理调研结果，待 GPU 结果回来后据此设计下一步方案。

---

## 1. 核心问题陈述

### 1.1 现象

`PickPlaceCounterToCabinet` 场景下，`obj_main=lemon` 多次 episode 失败模式为 `slipped_lift`：
- `close_gripper` 接触检测成功（`force_loop` 报告 grasp_confirmed）
- `micro_lift` 测试通过（物体随夹爪上升 1cm）
- **正式 `lift(height_m=0.10)` 阶段，柠檬从指间滑出，Δz_obj < 5mm**

### 1.2 已修复的层次（不充分但必要）

| Commit | 内容 | 解决的子问题 |
|---|---|---|
| `4121f37` | navigate_base safe min offset 0.55m | 防止 base 太近导致 arm OSC 锁死 |
| `e6458de` | move_arm_to 先位置后姿态 | 防止 ori 项干扰位置收敛 |
| `de4e53d` | torso adaptive workspace + nudge_base | 解决 z-stall 导致的 descend 失败 |
| `718dbaf` 系列 | diagnostic pre-grasp handoff | 替代全局 Euclidean 阈值 |
| `fbda95c` | 抑制 torso 介入时的 IK-regression false abort | 防止 lift 被自己打断 |
| `dacf9e7` | 移除 lift Phase 1（4×5mm 微步） | 防止 OSC 长时间低速振荡 |

**这些修复的共同特点**：解决"控制不该自残"的问题，让 lift **能跑完**。但**没有**回答"夹得够不够紧"。

### 1.3 物理根因（dacf9e7 即便完全修好仍未解决）

**几何根因（不可消除）：**
- 球面 + 平行直夹爪 → 接触点在赤道两侧切向，垂直方向支撑全靠摩擦
- `F_required = m·g / μ`（DeliGrasp 静态平衡公式）
- 柠檬 m≈100g, μ≈0.3 → 需要约 3.3 N 法向力。低于此即必滑

**控制根因（这是 robosuite 仿真的关键限制）：**
- robosuite SimpleGripController 是 **position controller**（命令开度）, 不是 force controller
- 我们没有 `target_force` 这个 API
- `_close_gripper_force_loop` 当前实现：检测接触后再多关 N 步，本质等价于"继续压缩弹性接触"间接获得更大法向力。但 N 是**对所有物体写死的常数**

### 1.4 决策根因（信息没用上）

`grasp_planner.select_strategy()` 已经在 prompt 里收到 label 和 visible_features，但当前只输出 `strategy ∈ {top_down, gentle_side, ...}` 和 `approach_axis`。**LLM 知道柠檬是滑的、橙子比柠檬大、面包很轻，但这些信息从未影响夹爪闭合参数。**

**核心矛盾**：我们已经有 LLM 的常识先验、有 dual-store memory 系统、有 force_loop squeeze 机制，但三者**没有打通**。

---

## 2. 文献调研

### 2.1 调研范围

关键词：robotic grasping, round/slippery objects, parallel gripper, slip detection/prevention, LLM/VLM physical property estimation, adaptive grasp force, simulation。优先 2024-2025 工作。

### 2.2 四条主流路线

| 路线 | 代表文献 | 是否需要触觉传感器 | 与本项目契合度 |
|---|---|---|---|
| **A. LLM 推理物理属性 → 解析公式算力** | DeliGrasp (Xie+, CoRL 2024) | 需要力控夹爪，不需触觉 | ★★★★★ |
| **B. VLM + RAG 检索经验决定力** | Exp-Force (arXiv 2603.08668, 2025) | 需要力控夹爪，不需触觉 | ★★★★☆ |
| **C. 轨迹调制（不改力）** | Nazari+, Nature Machine Intelligence 2025 | 需触觉做 forward model 训练 | ★★★☆☆ |
| **D. 触觉滑移检测+反馈** | FORTE (arXiv 2506.18960), Waltersson+ (arXiv 2410.19660) | 必需触觉 | ★☆☆☆☆ |

### 2.3 路线 A：DeliGrasp 详解

**Xie et al., "DeliGrasp: Inferring Object Properties with LLMs for Adaptive Grasp Policies", CoRL 2024**

**Pipeline:**
1. 给 LLM 一个 object description + grasp verb
2. LLM 输出 `m`（质量）、`μ`（摩擦系数）、`k`（弹性常数）
3. 静态平衡公式 `F_min = m·g / μ` 计算最小握力
4. force-controlled gripper 闭环到 F_min（接触后逐步增力）
5. 检测到 slip（编码器位置回跳）则把 F_min 拉高再重抓

**实验：** 12 类易损物体（覆盆子、面包、葡萄、纸飞机等），DeliGrasp 在脆弱物体上的 success / non-damaging rate **远高于** force-limited / 经典 adaptive grasping baseline。

**Prompt 设计（appendix A.3-A.5 完整公开）:**
- "Thinker" prompt：给 LLM 推理 m、μ、k 的物理过程
- "Coder" prompt：把推理结果转成 Python 控制策略
- CoT prompting 明显比 zero-shot 准

**对本项目的启示：**
- 你（user）的直觉路线**完全正确**，且已被 CoRL 2024 验证有效
- prompt 模板可直接复用
- **限制**：DeliGrasp 假设 force-controlled gripper（DC servo + 电流控制）。我们 robosuite SimpleGripController 是位置控制，**不能直接用 F_min**

**适配方案（DeliGrasp 简化版）：**
- LLM 仍输出 `m_g`、`slip_risk`、`fragility`
- 不算 F_min，而是把这些映射到 **`squeeze_extra_steps`** 和 **`finger_width_margin`** 两个参数
- `squeeze_extra_steps` 越大，position gripper 关得越深，间接法向力越大
- `finger_width_margin = w_object - target_width`：负值表示"故意夹得比物体窄"，靠夹爪指节弹性产生法向力
- 公式（建议初版）：

  ```
  squeeze_extra_steps = clip(round(m_g/10) + 10·slip_risk, 0, 30)
  finger_width_margin = clip(0.002·m_g/100 + 0.004·slip_risk, 0.001, 0.012)
  ```
  其中 `slip_risk ∈ {0:low, 1:med, 2:high}`

### 2.4 路线 B：Exp-Force 详解

**arXiv 2603.08668, "Exp-Force: Experience-Conditioned Pre-Grasp Force Selection with Vision-Language Models", 2025**

**Pipeline:**
1. 维护 experience pool（每条记录：图像 + 物体描述 + 该物体最佳力）
2. 新物体来了，VLM 生成对该物体的 task description
3. **检索**：从 experience pool 找语义/视觉最相似的 top-k 经验
4. 把 top-k 案例塞进 predictor VLM 的 prompt → in-context inference 输出力
5. **不**用解析公式，**不**用人工 heuristic

**优势：** 完全 end-to-end、不依赖 force model、跨夹爪 embodiment 通用

**对本项目的启示：**
- 我们已经有 `memory/grasp_experience.yaml`（commit `098263a`）双层记忆
- **Exp-Force 等于把 memory 系统从"策略选择"扩展到"力/开度参数选择"**
- 实施成本：扩 schema + 改 retrieval prompt
- 论文价值：Exp-Force 是 2025 最新工作，对标性强（可作为 EmboSight 的"具身感知 + 记忆 + LLM"闭环的实证）

**实施方案：**

扩展 `memory/grasp_experience.yaml`:
```yaml
lemon:
  best_strategy: top_down
  best_squeeze_extra_steps: 18
  best_finger_width_margin: 0.006
  successful_grasps:
    - {squeeze: 18, margin: 0.006, lift_height: 0.10, outcome: success}
  failed_grasps:
    - {squeeze: 8,  margin: 0.002, outcome: slipped_lift}
    - {squeeze: 30, margin: 0.012, outcome: crushed}
```

下次遇到柠檬/橙子/酸橙（语义相似），retrieve 这些经验喂给 LLM 做决策。

### 2.5 路线 C：Bioinspired Trajectory Modulation

**Nazari et al., "Bioinspired trajectory modulation for effective slip control in robot manipulation", Nature Machine Intelligence 2025**

**核心论点：** "传统认为防滑只能改握力——错。改**轨迹**也能防滑，且对力受限场景反而更有效。"

**Pipeline:**
1. 离线训练 action-conditioned forward model：给定当前抓取状态 + 候选轨迹片段 → 预测下一步是否滑
2. 在线 MPC：每步选择"预测不会滑"的轨迹增量
3. 在 reactive 模式下也能减少滑事件，predictive 模式下进一步降低

**对本项目的启示：**
- 这恰好解释了我们移除 Phase 1（小幅多步）的逻辑边界——"温柔"应该体现在**加速度有限**而不是**位移分块**
- 实操简化版：lift 时给 OSC 一个**最大速度/加速度上限**，平滑加速到目标
- 这条路线**不依赖 LLM**，是纯控制层改进
- **限制**：原文 forward model 用真触觉数据训练，我们不能照搬。但可以把"forward model"退化为"已知公式"——例如限制 `a_lift < μ·g`（即不让 lift 加速度超过摩擦能提供的最大加速度）

### 2.6 路线 D：触觉滑移检测（暂不适用）

- **FORTE (arXiv 2506.18960)**: 在 fin-ray 软指里嵌气道做内置压力传感，91.9% 易损物抓取成功率
- **Waltersson+ (arXiv 2410.19660)**: 力 + 滑速双通道触觉的并联夹爪 + 4 种滑控控制器

**为何不适用：** robosuite 默认环境无触觉传感器；我们的项目定位是 vision-language（视障辅助），加触觉硬件假设违背 setting。**记录在案，论文 future work 提及即可。**

---

## 3. 推荐方案三步走

### Step 1 — DeliGrasp 简化版（最小改动）

**改动范围：**

| 文件 | 改动 |
|---|---|
| `prompts/grasp/select_strategy.txt` | 新增字段输出：`mass_g`, `slip_risk`, `squeeze_extra_steps`, `finger_width_margin` |
| `src/world_belief.py::GraspStrategy` | dataclass 加上述字段 |
| `src/grasp_planner.py::select_strategy` | 解析新字段 |
| `src/env_wrapper.py::close_gripper` / `_close_gripper_force_loop` | 接受 `squeeze_extra_steps`, `finger_width_margin` 参数 |
| `src/action_executor.py::act` | 把 strategy 的力参数传给 close_gripper |

**测试设计：**
- `tests/test_grasp_strategy.py`：验证 LLM 输出 squeeze_extra_steps ∈ [0, 30]，圆形物体 slip_risk=high
- `tests/test_env_wrapper_grasp.py::test_close_gripper_uses_squeeze_extra_steps`：mock LLM 输出，验证 `_close_gripper_force_loop` 被调用 `default_steps + squeeze_extra_steps` 次
- 回归测试：现有 436 个测试不能破

**论文对标：** DeliGrasp 的"LLM 推理物理属性"思路，但因 SimpleGripController 限制，做了 force→position 的工程映射。

**预期收益：** 柠檬抓取从依赖固定 squeeze=10 改为按物体属性自适应；圆形/重物体获得 squeeze≥20，提升摩擦支持力。

### Step 2 — Exp-Force 集成（用上 memory）

**前提：** Step 1 已完成。

**改动范围：**

| 文件 | 改动 |
|---|---|
| `memory/grasp_experience.yaml` schema | 增加 `best_squeeze_extra_steps`, `best_finger_width_margin`, `successful_grasps[]`, `failed_grasps[]` |
| `src/memory_manager.py` | 写入路径增加力参数；retrieval 增加"返回相似物体的最佳力参数列表" |
| `src/grasp_planner.py::select_strategy` | prompt 注入 `past_experience` 时包含力参数 |
| `prompts/grasp/select_strategy.txt` | 新增 placeholder 让 LLM 利用过去经验 |
| `src/agent.py` | grasp 完成时记录 squeeze 实际用值 + 结果 |

**测试设计：**
- `tests/test_memory_integration.py`：验证 squeeze 参数能写入并读出
- `tests/test_memory_manager.py::test_retrieve_force_params_for_similar_object`：lemon 经验存在时，lime 查询返回 lemon 经验
- `tests/test_grasp_strategy.py::test_strategy_uses_past_squeeze_advice`：mock past_experience 包含 squeeze=18 成功，验证 LLM 输出倾向于该值

**论文对标：** Exp-Force 的"experience-conditioned VLM"。EmboSight 的优势：dual-store memory 已有，long-term memory 跨 episode 持久。

### Step 3 — 轨迹层防滑（可选，中期）

**改动范围：**

| 文件 | 改动 |
|---|---|
| `src/env_wrapper.py::move_arm_to` | 新增 `max_velocity_mps`, `max_accel_mps2` 参数；OSC delta 命令做速度/加速度限幅 |
| `src/env_wrapper.py::lift` | 调用 `move_arm_to` 时显式传 `max_accel = μ·g`（按当前物体的 LLM-推理 μ）|

**测试设计：**
- `tests/test_env_wrapper_orientation.py`：lift 加速度不超过限幅
- 回归：micro_lift 不受影响

**论文对标：** Nazari+ 的轨迹调制思想（简化版，不做 forward model 训练）。

---

## 4. 等 GPU 结果的判断准则

`dacf9e7` 跑完后看：

| GPU 结果 | 判断 | 推荐路径 |
|---|---|---|
| 柠檬抓取成功率 ≥80% | 控制层修复已足够，但只解决了"机械层"问题 | 仍建议做 Step 1（提升泛化到其他圆形物体） |
| 仍有 ≥30% slipped_lift | 确认根因是"夹爪开度对所有物体相同"，需要属性自适应 | **直接 Step 1 + Step 2** |
| 出现新失败模式（crushed/dropped） | 说明 lift 加速度过大 | Step 1 + Step 3 |

---

## 5. 论文写作角度

EmboSight 已有六大创新点（见 memory `95444708`），slip prevention 这块可以加为：

> **创新点 4 扩展（视觉触觉属性推理）→ "LLM-推理物体物理属性 + 经验检索 + 自适应夹持参数闭环"**
> 
> 对标：DeliGrasp (CoRL 2024) + Exp-Force (2025)
> 
> EmboSight 优势：
> 1. 嵌入到完整的 perception → planning → action 闭环里，不只是 grasp 模块独立
> 2. 复用已有 dual-store memory，跨 episode 学习
> 3. 在 robosuite 仿真里 zero-shot transfer（训练-free）
> 4. 视障辅助场景：力参数同时影响"是否安全告知用户"（重物 → 提醒小心）

---

## 6. 引用清单

1. **Xie et al., 2024.** "DeliGrasp: Inferring Object Properties with LLMs for Adaptive Grasp Policies." CoRL 2024. arXiv:2403.07832.
2. **Anonymous, 2025.** "Exp-Force: Experience-Conditioned Pre-Grasp Force Selection with Vision-Language Models." arXiv:2603.08668.
3. **Nazari, Mandil, Santello, Park, Ghalamzan-E., 2025.** "Bioinspired trajectory modulation for effective slip control in robot manipulation." *Nature Machine Intelligence* (2025). DOI:10.1038/s42256-025-01062-2.
4. **Anonymous, 2025.** "FORTE: Tactile Force and Slip Sensing on Compliant Fingers for Delicate Manipulation." arXiv:2506.18960.
5. **Waltersson & Karayiannidis, 2024.** "Perception, Control and Hardware for In-Hand Slip-Aware Object Manipulation with Parallel Grippers." arXiv:2410.19660.
6. **Zhai et al., 2024.** "NeRF2Physics: Physical Property Understanding from Language-Embedded Feature Fields." CVPR 2024.
7. **Anonymous, 2025.** "PhysQuantAgent: An Inference Pipeline of Mass Estimation for Vision-Language Models." arXiv:2603.16958.
