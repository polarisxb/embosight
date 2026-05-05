# EmboSight 查新报告（盖章版）

> **用途**：CRAIC 必交材料之一，需 **学院盖章 + 申请人签字**
> **格式**：参照 CRAIC PDF P26-30 标准格式
> **提交格式**：彩色扫描盖章页 → PDF
> **状态**：模板版（学院实际盖章流程因校而异，最终格式以学院要求为准）

---

## ⚠ 操作流程提醒

```
─────────────────────────────────────────────────────
Step 1（提前 5-7 工作日）：联系学院
  → 找学院教务办 / 科研办 / 学工办
  → 询问"参加 CRAIC 大赛要查新报告盖章，流程是？"
  → 各校流程：
     • 简单：学院直接盖章
     • 中等：先去图书馆做查新，再学院盖章
     • 复杂：去校级科研处或第三方查新机构

Step 2：自行查新（用本文档下方"检索词"）
  → CNKI / Web of Science / arXiv 检索
  → 截图保存检索结果（关键词 + 检索数量）
  → 下载主要参考文献摘要

Step 3：填写本模板（替换占位符 → 实际信息）

Step 4：打印 + 个人签字 → 学院盖章 → 彩色扫描

⚠ 千万别在最后一刻才找学院盖章
─────────────────────────────────────────────────────
```

---

## 标准格式正文（按 CRAIC 模板填写）

```
═════════════════════════════════════════════════════
中国机器人及人工智能创新大赛
查新报告
═════════════════════════════════════════════════════
```

### 一、查新项目基本情况

| 项目 | 内容 |
|------|------|
| 项目名称（中文） | EmboSight——零样本视障具身辅助智能体 |
| 项目副题 | 基于主动感知与多模态大模型的助盲机器人研究 |
| 项目名称（英文） | EmboSight: A Zero-Shot Embodied Visual Assistant for the Visually Impaired via Active Perception and Multimodal Large Models |
| 项目申请者 | [姓名] |
| 申请者单位 | [学校全称][学院全称] |
| 联系电话 | [手机] |
| 电子邮箱 | [邮箱] |
| 项目类型 | 大学生科技创新作品 |

### 二、查新点（自我声明的创新点）

**创新点 1：零样本视障专属任务分解机制**

设计了面向视障辅助场景的任务分解模板库（10 大类，覆盖 50+ 子任务原型），并结合 Few-shot Prompt 框架，使大语言模型能够在零样本场景下正确理解视障者多样化指令，并强制输出包含五个视障关键维度（方位、距离、触觉、安全、行动）的结构化子任务序列。

**创新点 2：基于大语言模型的主动视角规划方法（LLM-NBV）**

将经典 Next-Best-View 问题转化为大语言模型的离散选择问题，设计离散视角库（12 个标准位姿）作为动作空间，提出任务驱动而非几何驱动的视角选择策略，结合 LLM 自评估早停机制，实现视障辅助场景下的零样本主动感知。

**创新点 3：视障友好的多模态场景描述生成方法**

首次系统化将无障碍设计学（Accessibility Design）方法论引入视觉-语言模型输出改造，设计五维度（物体识别、方位距离、触觉特征、安全提示、行动建议）强制输出 Prompt 模板，结合几何后处理实现厘米级距离精度，并建立量化的视障友好度评估指标体系。

### 三、查新范围

- ☑ 中国知网（CNKI）
- ☑ 万方数据知识服务平台
- ☑ 维普资讯（VIP）
- ☑ Web of Science Core Collection
- ☑ IEEE Xplore Digital Library
- ☑ ACM Digital Library
- ☑ Springer Link
- ☑ ScienceDirect
- ☑ arXiv 预印本服务器
- ☑ Google Scholar
- ☑ 国家知识产权局专利检索（CNIPA）

### 四、检索词

**中文检索词组合**（建议交叉检索）：

| 序号 | 检索词组 |
|------|---------|
| C1 | 具身智能 + 视障辅助 |
| C2 | 零样本任务泛化 + 助盲 |
| C3 | 主动感知 + 视觉障碍 |
| C4 | 视觉-语言-动作模型 + 无障碍 |
| C5 | 视障 + 大模型 + 机器人 |
| C6 | 助盲机器人 + 仿真 |
| C7 | 视觉-语言模型 + 视障描述 |
| C8 | 无障碍设计 + 人工智能 + 机器人 |

**英文检索词组合**：

| 序号 | 检索词组 |
|------|---------|
| E1 | "Embodied AI" + "visual impairment" + "assistance" |
| E2 | "Zero-shot" + "blind" + "robot" |
| E3 | "Active perception" + "VLM" + "accessibility" |
| E4 | "Vision-Language-Action" + "blind users" |
| E5 | "Next-Best-View" + "language-driven" |
| E6 | "VLA" + "visually impaired" |
| E7 | "LLM-driven NBV" |
| E8 | "Multimodal" + "accessibility design" + "robot" |

### 五、检索时间范围

2020 年 1 月 1 日 — 2026 年 [当前月]（覆盖近 6 年研究）

### 六、主要参考文献（已选 30 篇代表性文献）

#### 6.1 视觉-语言-动作（VLA）模型方向

[1] KIM M J, PERTSCH K, KARAMCHETI S, et al. OpenVLA: An Open-Source Vision-Language-Action Model[C]//Proceedings of Robotics: Science and Systems (RSS), 2024.

[2] BLACK K, BROHAN A, FU C Y, et al. π0: A Vision-Language-Action Flow Model for General Robot Control[J/OL]. arXiv preprint, arXiv:2410.24164, 2024.

[3] OCTO MODEL TEAM. Octo: An Open-Source Generalist Robot Policy[J/OL]. arXiv preprint, arXiv:2405.12213, 2024.

[4] BROHAN A, BROWN N, CARBAJAL J, et al. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control[J/OL]. arXiv preprint, arXiv:2307.15818, 2023.

[5] DRIESS D, XIA F, SAJJADI M S M, et al. PaLM-E: An Embodied Multimodal Language Model[J/OL]. arXiv preprint, arXiv:2303.03378, 2023.

#### 6.2 视觉-语言模型方向

[6] BAI J, BAI S, YANG S, et al. Qwen-VL: A Versatile Vision-Language Model[J/OL]. arXiv preprint, arXiv:2308.12966, 2023.

[7] OPENAI. GPT-4V(ision) System Card[R/OL]. (2023). https://openai.com/research/gpt-4v-system-card.

[8] LIU H, LI C, WU Q, et al. Visual Instruction Tuning[C]//NeurIPS, 2023.

[9] WANG W, LV Q, YU W, et al. CogVLM: Visual Expert for Pretrained Language Models[J/OL]. arXiv preprint, arXiv:2311.03079, 2024.

#### 6.3 仿真环境与基准

[10] NASIRIANY S, MAHESHWARI A, XU Z, et al. RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots[C]//RSS, 2024.

[11] LIU B, ZHU Y, GAO C, et al. LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning[C]//NeurIPS Datasets and Benchmarks Track, 2023.

[12] PUIG X, UNDERSANDER E, SZOT A, et al. Habitat 3.0: A Co-habitat for Humans, Avatars, and Robots[C]//ICLR, 2024.

[13] TAO S, XIANG F, SHUKLA A, et al. ManiSkill3: GPU Parallelized Robotics Simulation[J/OL]. arXiv preprint, arXiv:2410.00425, 2024.

[14] XIANG F, QIN Y, MO K, et al. SAPIEN: A SimulAted PartNet Environment[C]//CVPR, 2020.

#### 6.4 主动感知与具身问答

[15] BAJCSY R. Active Perception[J]. Proceedings of the IEEE, 1988, 76(8): 966-1005.

[16] CONNOLLY C. The Determination of Next Best Views[C]//IEEE ICRA, 1985, 2: 432-435.

[17] CHAPLOT D S, GANDHI D P, GUPTA S, et al. Learning to Explore Using Active Neural SLAM[C]//ICLR, 2020.

[18] DAS A, DATTA S, GKIOXARI G, et al. Embodied Question Answering[C]//CVPR, 2018.

[19] KRAININ M, CURLESS B, FOX D. Autonomous Generation of Complete 3D Object Models Using Next Best View Manipulation Planning[C]//IEEE ICRA, 2011.

#### 6.5 视障辅助应用

[20] BIGHAM J P, JAYANT C, JI H, et al. VizWiz: Nearly Real-time Answers to Visual Questions[C]//ACM UIST, 2010.

[21] GURARI D, LI Q, STANGL A J, et al. VizWiz Grand Challenge: Answering Visual Questions from Blind People[C]//CVPR, 2018.

[22] BE MY EYES & OPENAI. Be My AI: GPT-4 Vision-Powered Visual Assistance[EB/OL]. (2023). https://www.bemyeyes.com/blog/announcing-be-my-ai.

[23] MICROSOFT. Seeing AI: A Free App That Narrates the World[EB/OL]. https://www.seeingai.com/.

#### 6.6 政策与统计

[24] 中国残疾人联合会. 2023 年残疾人事业发展统计公报[R]. 北京, 2024.

[25] 中华人民共和国民政部. 2024 年民政事业发展统计公报[R]. 北京, 2025.

[26] 全国人民代表大会常务委员会. 中华人民共和国无障碍环境建设法[Z]. 2023-06-28.

[27] 国务院. "十四五"残疾人保障和发展规划[Z]. 国发〔2021〕10 号.

[28] WORLD HEALTH ORGANIZATION. World Report on Vision[R/OL]. Geneva: WHO, 2019.

#### 6.7 基础理论

[29] RADFORD A, KIM J W, HALLACY C, et al. Learning Transferable Visual Models From Natural Language Supervision[C]//ICML, 2021.

[30] OPEN X-EMBODIMENT COLLABORATION. Open X-Embodiment: Robotic Learning Datasets and RT-X Models[C]//IEEE ICRA, 2024.

### 七、查新结论

经过对国内外主要文献数据库的系统检索与分析，得出以下结论：

**1. 关于视觉-语言-动作（VLA）模型研究**

经检索，OpenVLA、π0、Octo、RT-2 等代表性 VLA 模型均聚焦于通用机器人操作任务（抓取、放置、推动等），训练数据来自 Open X-Embodiment 等通用机器人数据集。**未检索到专门针对视觉障碍人士辅助场景设计的 VLA 系统报道**。

**2. 关于主动感知（Active Perception）研究**

经检索，主动感知领域以 Bajcsy（1988）的开创性工作为起点，经历了从几何方法（Connolly 1985）到强化学习方法（Chaplot 2020）的发展。**未检索到将大语言模型驱动的离散视角选择应用于视障辅助场景的研究报道**。

**3. 关于视觉-语言模型（VLM）在助盲应用研究**

经检索，VLM 助盲应用主要包括 Be My AI、SeeingAI 等商业产品，以及 VizWiz 等学术数据集。这些工作均采用单图被动描述模式。**未检索到基于具身机器人的多视角主动感知 + 视障辅助方案报道**。

**4. 关于零样本任务泛化研究**

经检索，零样本任务泛化研究主要在通用任务（桌面操作、视觉问答）上评估。**未检索到视障辅助场景专属的零样本评估体系报道**。

**5. 综合结论**

综上所述，在所检索的国内外公开文献与专利数据库中，**未发现与本项目"零样本视障具身辅助智能体"主要技术方案相同的研究报道**。本项目所提出的三项核心创新（零样本视障专属任务分解机制、基于大语言模型的主动视角规划方法、视障友好的多模态场景描述生成方法）具有显著创新性，该研究方向具有重要的学术价值与社会意义。

### 八、申请者本人、所在学院签字盖章

**申请者承诺**：

（1）本报告中所列项目情况真实、准确；

（2）所附检索文献为公开发表的可获取文献；

（3）本报告中陈述的事实真实、准确；

（4）本人已按照大赛查新规范进行查新、文献分析和审核，并做出上述查新结论；

（5）所提交的所有材料均为申请者本人独立完成（含在指导教师指导下完成），无任何抄袭、剽窃等学术不端行为。

申请者（签字）：________________________________

日期：________ 年 ________ 月 ________ 日

申请者所在学院（盖章）：

```
                                  ┌──────────────────┐
                                  │                  │
                                  │   学院盖章位置   │
                                  │                  │
                                  └──────────────────┘
```

### 九、附件清单

- ☑ CNKI 检索结果截图（按检索词组分别截图）
- ☑ Web of Science 检索结果截图
- ☑ arXiv 检索结果截图
- ☑ 主要参考文献摘要复印件（30 篇）
- ☐ 学院科研处证明（如学校要求）
- ☐ 第三方查新机构出具证明（如学校要求）

### 十、备注

（如有特殊说明，写在此处。常见情况包括：本项目与某项已发表工作有部分相似但本质区别于…；本项目已申请发明专利受理号 CN…）

---

## 风险红线提醒

```
❌ 不要写"国内首创""填补空白"等绝对化表述
   → 改为"未发现与本项目主要技术方案相同的研究报道"

❌ 不要虚报检索数据库（被追问会很尴尬）
   → 真实检索过哪几个就只列哪几个

❌ 不要虚报参考文献（评委会随机抽查）
   → 上面 30 篇都是真实存在的，可放心使用
   → 但请按本人能查阅到的实际情况做删减

❌ 不要在最后一刻才找学院盖章（流程通常需 3-7 天）
   → 强烈建议 Day 1 就联系学院老师启动流程
```