# 薄长物体抓取问题深度调研与方案设计

**Created**: 2026-05-16  
**Owner**: Cascade + user  
**Status**: Analysis / Pre-implementation

---

## 1. 问题陈述

### 1.1 现象

在 `random_seed_3` 场景（`obj_main=wooden_spoon`）中，系统连续三轮尝试抓取均失败：

```text
Round 1: top_down      → [act] object NOT lifted: Δz=0.000  (slipped)
Round 2: handle_grasp  → [close_gripper] no grasp after 30 steps
Round 3: refuse        → "无法安全抓取"
```

整轮 episode 耗时 **537 秒**，最终标记为 FAIL。视频显示：机器人手臂下降到木勺上方，闭合夹爪，向上抬起——**但木勺纹丝不动**。

### 1.2 此前的修复链

1. **commit 7e55520**: 删除 z-stall recovery 的 +5mm 偏移（往上抬反而远离物体）
2. **commit ed3725e**: 引入 workspace recovery（z-stall 时移动 base 靠近物体）
3. **commit 3ebef47**: close_gripper 接触后继续 squeeze 10 步
4. **commit 0f400c5**: lift 拆成 4×5mm 微步慢起
5. **commit c64f84d**: descend 接触检测要求连续 2 帧 + 已下降 ≥3cm（防瞬时擦碰）

以上每一项都是**对的**，但对于这个 seed 的木勺**都没解决**。说明问题不在控制环节，而在**几何 + 运动学层面**。

### 1.3 几何诊断

从日志反推关键尺度：

| 量 | 值 | 来源 |
|---|---|---|
| 木勺质心 z | 0.928 m | `obj_z_before` |
| 台面 z | ≈0.915 m | 推断（木勺底部贴台面） |
| 木勺高度 | ≈2 cm | 典型尺寸（直接观察） |
| 木勺顶面 z | ≈0.938 m | 0.928 + 0.010 |
| **手臂最低 EEF z** | **0.942 m** | descend stall 位置（重复多次） |
| 工作空间 gap | 1.4 cm | 0.942 - 0.928 |

robosuite Panda 的 `eef_site` 定义为**两指中点**。因此：

- 双指闭合时指尖在 z=0.942（手臂极限）
- 木勺顶面在 z=0.938
- **指尖在木勺顶面上方 4mm**

夹爪从开口位（指距 ≈8cm）水平闭合，闭合扫过的平面在 z=0.942。木勺整体在 z=0.918~0.938 范围。**夹爪闭合过程中不会穿过木勺的几何体**——除非台面震动让木勺微跳，否则物理上无法接触。

### 1.4 工作空间约束分析

每次 `descend` 的子步 `move_arm_to` 都报告：

```text
[move_arm_to] max_steps reached, dist=0.0100m
```

含义：子步目标在 EEF 下方 1cm，但跑了 200 个 sim step **完全没移动**。这不是控制收敛慢，是 **IK 在 PandaOmron 当前底盘位置下无解**——手臂已伸直到极限。

底盘靠近（commit ed3725e）后第二次 descend 起点是 z=0.944（**反而更高**了 2mm），后续 stall 在 0.958。底盘前移并未扩大 z 下界，反而因构型变化让手臂下伸能力**变差**。

**结论**：这个 seed 的木勺在 PandaOmron 当前底盘 reachability map 内**不可 top_down 抓取**。

---

## 2. 文献调研

### 2.1 薄长物体抓取的主流范式

| 范式 | 代表文献 | 适用条件 | 核心思想 |
|---|---|---|---|
| **Side-pinch / Handle grasp** | RT-1 (Brohan 2023), BC-Z (Jang 2022), VIMA (Jiang 2023) | 有侧面空间、物体形状已知 | 末端旋转使指轴垂直物体长轴，从侧面夹 |
| **Scoop-under** | Mason (1986), Dogar & Srinivasa (2010) "Push-Grasping" | 桌面薄物，物体可推动 | 一指先插入物-桌缝隙，另一指从上盖下 |
| **Push-to-edge** | Mason (1986), MIT Push (2018) | 物体在桌面但不在桌沿 | 先把物体推到桌沿，再从下方抓 |
| **Tilt-and-grasp** | Hang et al. (2019) "Pre-grasp Manipulation" | 物体可被一指顶起 | 一指撬起物体形成抓取空间 |
| **Antipodal grasp synthesis** | GraspNet-1B (Fang 2020), ContactGraspNet (Sundermeyer 2021) | 有 3D 模型 / 点云 | 计算几何最优夹持对 |
| **Suction** | Amazon Picking Challenge 系列 | 物体表面平整 | 吸附而非夹持 |
| **Soft/compliant fingers** | RBO Hand (Deimel 2016), Festo Bionic | 硬件可选 | 软指被动顺应物体形状 |

### 2.2 关键洞察

> **几何决定一切**：当 EEF site 物理上无法降到物体几何范围内时，**所有的控制层修复（squeeze、gentle lift、contact streak）都救不了**。

文献中针对此问题的方案可归为三类：

1. **改变末端姿态**（side approach）— 让接近向量平行于地面
2. **改变物体姿态**（push, tilt）— 让物体的可抓部分进入工作空间
3. **改变抓取硬件**（soft / suction）— 工程边界外，本项目不考虑

### 2.3 与 EmboSight 项目定位的关联

项目六大创新点中：
- **2 LLM-NBV** 与 **4 视觉触觉属性推理** 已部分涉及"对薄物体应更主动近距观察"
- **5 VLM 安全约束** 与 **6 语义一致性验证** 提供了 refuse 的合理依据

但**抓取执行层（GraspPlanner + ActionExecutor + EnvWrapper）目前只支持 top-down**。这与 5 策略（top_down / gentle_side / handle_grasp / scoop_under / refuse）在 prompt 中的描述**不一致**——是一个隐性的实现-设计漂移。

---

## 3. 当前架构下的根本矛盾

### 3.1 接口承诺 vs 实际行为

```python
# src/grasp_planner.py — 设计意图
_STRATEGY_PARAMS = {
    "top_down":     {"approach_dir": [0, 0, -1.0], ...},  # 从上往下
    "gentle_side":  {"approach_dir": [1, 0,  0.0], ...},  # 从+x侧面
    "handle_grasp": {"approach_dir": [1, 0,  0.0], ...},  # 从+x侧面
    "scoop_under":  {"approach_dir": [0, 0, -1.0], ...},  # ???
}
```

```python
# src/env_wrapper.py — 实际执行
def descend(self, point_3d, ...):
    """从当前 pre-grasp 位下降到 point_3d。"""
    # 永远沿 z 轴下降, 完全忽略 approach_dir
```

**5 个策略在执行层只有 1 个实现**。LLM 输出 `handle_grasp` 时执行的依然是 top_down，区别只是 `finger_width` 参数不同。

### 3.2 末端朝向控制缺失

robosuite PandaOmron 使用 OSC_POSE 控制器，action 布局 `right[0:6] = (Δx,Δy,Δz, Δrx,Δry,Δrz)`。我们的 `move_arm_to`：

```python
action[0:3] = dir_base * step_size  # 位置增量
# action[3:6] 始终为 0  ← 不改变末端朝向
```

结果：末端朝向**永远保持初始值**（默认 z 轴向下）。即使 `approach_dir=[1,0,0]`，手臂只是把 EEF site 平移到侧面，**夹爪开口方向依然是水平面内**，闭合时仍然是上下挤压。对侧面圆柱物体（spoon handle）来说，这跟 top_down 几乎没区别。

### 3.3 几何不可达的根因

PandaOmron 是 7-DOF Panda + 全向底盘。在当前桌面任务设置下：
- 底盘距桌沿固定距离（避免碰撞）
- 桌面高度 ≈0.915 m
- Panda 工作空间最低 EEF 位置依赖于 (base_xyz, target_xy)

对于桌面深处的物体（y 远离底盘），Panda 伸直后 EEF 最低能到 z≈0.94，比桌面高约 2.5cm。这正是日志观察到的 stall 位置。

**结构性约束**：除非底盘能贴桌沿或者机械臂能下倾，否则桌面远处的薄物体（高度 <2cm）几何上无法 top_down 抓取。

---

## 4. 方案设计：四条可行路径

### 4.1 路径 A：实现真正的侧抓（推荐主路径）

**思路**：在 OSC_POSE 控制下加入末端朝向控制，让 `approach_dir` 不只是位置规划，也驱动末端旋转。

**关键改动**：

1. **`env_wrapper.move_arm_to` 增加 `approach_dir` 参数**
   ```python
   def move_arm_to(self, target_pos, approach_dir=None, ...):
       # 若 approach_dir 提供, 计算目标末端朝向
       # action[3:6] = 朝向误差的 axis-angle 表示
   ```

2. **`descend` 拆分为 `approach`**
   ```python
   def approach(self, point_3d, approach_dir, ...):
       """沿 approach_dir 接近 point_3d, 支持任意方向。
       - top_down: approach_dir=[0,0,-1], 行为=旧 descend
       - side:     approach_dir=[1,0,0], 沿 +x 接近, 末端朝向 -x
       """
   ```

3. **`move_to_pre_grasp` 按 `approach_dir` 算 pre-grasp 位置**
   ```python
   pre_pos = target_pos - approach_dir * pre_grasp_offset
   # top_down: 物体上方 5cm
   # side:     物体 -x 方向 10cm
   ```

4. **抓后回退也沿 -approach_dir**
   ```python
   retreat_pos = current + (-approach_dir) * retreat_distance
   ```

**优点**：
- 5 个策略真正可用，1 个改动解锁 4 倍能力
- 论文上有 implementable contribution
- 与 LLM 策略选择形成闭环（策略真的有差异）

**缺点**：
- 工程量 1-2 天（OSC 朝向控制不平凡，需测试稳定性）
- 朝向运动可能引入新的 IK 不可达情况
- 需要扩展单测（mock OSC 朝向控制）

**风险点**：
- robosuite OSC_POSE 的朝向 action 是 axis-angle 还是 quaternion 需查文档
- 底盘 + 末端协同旋转可能震荡

**预期收益**：薄长物体抓取成功率 0% → 40~60%

---

### 4.2 路径 B：薄物体专用 depth_margin

**思路**：检测到"薄/平/扁"物体时，把 `descend` 的目标 z 从"质心下方"改成"物体顶上方"，避免追求物理上不可达的低点。

**关键改动**：

1. **`Hypothesis.visible_features` 中识别薄物体关键词**
   ```python
   THIN_KEYWORDS = {"thin", "flat", "thin object", "扁", "薄", "平"}
   is_thin = any(k in f.lower() for f in hyp.visible_features for k in THIN_KEYWORDS)
   ```

2. **`GraspPlanner._STRATEGY_PARAMS` 动态调整 margin**
   ```python
   if is_thin and strategy == "top_down":
       params["depth_margin"] = -0.005  # 下降到物体顶+5mm 就停
   ```

3. **配合 close_enough 阈值放宽**
   ```python
   close_enough_threshold = 0.03 if is_thin else 0.01
   ```

**优点**：
- 半天工作量
- 不引入新接口，回退安全
- 对其他物体零影响

**缺点**：
- **不解决根本问题**——夹爪还是在物体上方闭合，靠运气
- 对真正几何不可达的物体仍然失败
- 难写测试（依赖具体几何）

**预期收益**：边界场景成功率 +10~20%

---

### 4.3 路径 C：能力边界承认 + 评测补全

**思路**：承认薄物体是当前架构边界，强化 refuse 逻辑和 LLM prompt，跑 50 seeds 评测看总体表现。

**关键改动**：

1. **`prompts/grasp/select_strategy.txt`** 增加规则：
   ```
   若 visible_features 含 "thin", "flat", 或 object width<3cm:
       优先选 scoop_under 或 gentle_side (从侧面)
       连续 2 次失败则 refuse
   ```

2. **`agent.py`** 加 refuse threshold：
   ```python
   if consecutive_grasp_failures >= 2:
       force_refuse = True
   ```

3. **跑 `eval/run_long_generalization.py` 50 seeds**，统计：
   - 总体成功率
   - 薄物体场景占比
   - 不同策略选择分布

**优点**：
- 半天到一天
- 直接给论文实验数据
- 系统行为更可预测

**缺点**：
- 不提升能力，只是更"诚实地"失败
- 论文上可能被审稿人质疑"你为什么不解决"

**预期收益**：评测可重复性 +; 平均成功率不变

---

### 4.4 路径 D：Push-to-edge / 推动策略

**思路**：薄物体无法直接抓时，先用单指或夹爪侧面把物体推到桌沿，再从下方抓。

**关键改动**：

1. 新增 `EnvWrapper.push_object(target_xy, direction, distance)`
2. `GraspStrategy` 增加 `push_then_grasp` 策略
3. LLM 在 thin + table_position=center 时考虑推到 edge

**优点**：
- 文献支持充分（Mason, Dogar）
- 视觉效果明显，演示效果好
- 适合论文做 contribution

**缺点**：
- 工程量 2-3 天
- 需要 edge 位置感知
- 推的过程可能把物体推下桌（碰撞控制复杂）

**预期收益**：薄物体成功率 +30~50%（如果实现得好）

---

## 5. 成本-收益矩阵

| 路径 | 实现成本 | 薄物体收益 | 普通物体影响 | 论文价值 | 风险 |
|---|---|---|---|---|---|
| A 真侧抓 | ★★★ (1-2d) | ★★★★ | 中性 | ★★★★ | 朝向控制不稳 |
| B depth_margin | ★ (0.5d) | ★★ | 零 | ★ | 治标 |
| C 承认边界 | ★ (0.5d) | 零 | 零 | ★★ | 审稿质疑 |
| D Push-to-edge | ★★★★ (2-3d) | ★★★★ | 风险中 | ★★★★ | 几何复杂 |

---

## 6. 推荐路线

### 6.1 短期（本周内）

**B + C 组合**：
- B 给薄物体一个"假装能抓"的最后机会（约 0.5 天）
- C 跑 50 seeds 评测，**得到真实成功率基线**（约 1 天，主要是等 sim 跑）

这一步的目标是 **量化问题**：薄物体占评测多少比例？除薄物体外的成功率是多少？只有有了数据才能决定后续投入。

### 6.2 中期（评测之后）

根据评测结果择路：
- 若薄物体 < 15% 且总成功率 > 60%：路径 C 收尾，写论文
- 若薄物体 ≥ 15% 或失败集中：投入 **路径 A 真正侧抓**（最高回报）

### 6.3 论文叙事

无论哪条路径，本调研本身就是论文 **Discussion / Limitation** 章节的素材：

> "Our system's grasp execution is constrained by the workspace of PandaOmron 
> under the fixed base placement. For thin objects (<2cm height) located at 
> the far end of the counter, top-down grasping is geometrically infeasible. 
> We identified this through systematic failure-mode analysis (Section X), 
> and provide a roadmap for orientation-aware grasping in future work."

这恰好对应六大创新点中的 **"6 语义一致性验证闭环"**——系统能**正确识别并承认自己的能力边界**，而不是谎报成功。这是 training-free 系统的优势之一。

---

## 7. 立即可做的下一步

1. **不改代码**：本调研文档 commit + push（本步骤）
2. **路径 B**：薄物体 depth_margin 自适应（写在下一个 spec 中）
3. **路径 C**：跑 50 seeds 评测，输出 `logs/long_generalization/<run-id>/summary.json` 作为基线
4. **决策点**：评测结果出来后再决定是否上路径 A

---

## 8. 附录：相关文献条目

- Mason, M. T. (1986). Mechanics and planning of manipulator pushing operations. *IJRR*.
- Dogar, M., & Srinivasa, S. (2010). Push-grasping with dexterous hands: Mechanics and a method. *IROS*.
- Hang, K., et al. (2019). Pre-grasp sliding manipulation of thin objects. *ICRA*.
- Fang, H. S., et al. (2020). GraspNet-1Billion: A large-scale benchmark for general object grasping. *CVPR*.
- Sundermeyer, M., et al. (2021). Contact-GraspNet: Efficient 6-DoF grasp generation. *ICRA*.
- Brohan, A., et al. (2023). RT-1: Robotics Transformer for real-world control at scale. *RSS*.
- Jang, E., et al. (2022). BC-Z: Zero-shot task generalization with robotic imitation learning. *CoRL*.

