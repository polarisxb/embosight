# EmboSight - 零样本视障具身辅助智能体

> *基于主动感知与多模态大模型的助盲机器人研究*

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status: WIP](https://img.shields.io/badge/Status-WIP-orange.svg)]()

> 中国 1731 万视障人士的具身智能"第二双眼"

---

## 项目简介

EmboSight 是首个面向视障人群的零样本具身智能辅助系统。通过将**主动感知**（Active Perception）、**多模态大模型**（VLM/LLM）与**零样本任务泛化**（Zero-Shot Generalization）深度融合，让机器人为视障者"看世界、听指令、帮做事"。

## 三大核心创新

| # | 创新点 | 一句话描述 |
|---|---|---|
| ① | 零样本视障任务分解 | LLM + 视障专属任务模板库，理解多样化视障需求 |
| ② | 零样本主动视角规划 ⭐ | LLM 驱动机械臂自主选择最优观察视角，填补 LLM-NBV × 视障辅助交叉空白 |
| ③ | 零样本视障友好描述 | 五维度强制输出 + 几何后处理，让通用 VLM 服务视障者 |

## 系统架构

```
视障者输入 → 任务分解 → 主动视角规划 → 机械臂执行 → 视障描述 → 视障者输出
                ↑                                               ↓
                └──────────────  闭环  ──────────────────────────┘
```

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 12.1+
- GPU: 24GB 以上（推荐 RTX 4090）
- OS: Ubuntu 22.04（AutoDL 推荐）

### 安装

```bash
git clone https://github.com/<your-username>/embodied-AI-one.git
cd embodied-AI-one

conda create -n embosight python=3.10 -y
conda activate embosight

pip install -r requirements.txt

pip install git+https://github.com/robocasa/robocasa.git

cp .env.example .env
# 编辑 .env 写入 DEEPSEEK_API_KEY
```

### 运行 Demo

```bash
python scripts/run_demo.py --query "我的药瓶在哪？"

python scripts/run_demo.py --baseline --query "桌上有什么"

python -m src.eval --config configs/default.yaml
```

## 项目结构

```
embodied-AI-one/
├── README.md                  # 你正在看的文件
├── requirements.txt           # Python 依赖
├── .gitignore
├── .env.example               # API Key 占位
├── LICENSE
├── docs/                      # 项目文档
│   ├── proposal.md            # 战略方案
│   ├── 01_report_draft.md     # 项目研究报告（CRAIC 提交版）
│   ├── 02_project_brief.md    # 项目简介 400 字
│   ├── 03_novelty_search.md   # 查新报告
│   └── 04_registration_guide.md
├── src/                       # 核心代码
│   ├── pipeline.py            # 主流程
│   ├── task_decomposer.py     # 创新①
│   ├── active_planner.py      # 创新②
│   ├── scene_describer.py     # 创新③
│   ├── llm_backend.py         # DeepSeek API
│   ├── vlm_backend.py         # Qwen2.5-VL
│   ├── env_wrapper.py         # RoboCasa 封装
│   ├── eval.py                # 评估脚本
│   └── utils.py
├── configs/
│   ├── default.yaml
│   └── viewpoints.yaml        # 离散视角库
├── prompts/                   # 三个核心 Prompt
│   ├── task_decompose.txt
│   ├── active_planner.txt
│   └── scene_describer.txt
├── scripts/
│   ├── setup_autodl.sh
│   └── run_demo.py
├── results/                   # 实验结果（gitignored）
├── checkpoints/               # 模型权重（gitignored）
└── data/                      # 数据（gitignored）
```

## 路线图

### 校赛（15 天）— 当前阶段
- [x] 项目方案锁定
- [x] 项目研究报告初稿
- [x] 查新报告模板
- [x] 项目骨架搭建
- [ ] AutoDL 环境部署
- [ ] RoboCasa Hello World
- [ ] 三大创新模块跑通
- [ ] 实验数据完成
- [ ] Demo 视频录制
- [ ] 答辩 PPT 准备

### 省赛（1-2 月）
- [ ] 跨任务记忆库（FAISS）
- [ ] 视障者真实访谈
- [ ] 适老化场景扩展

### 国赛（1-2 月）
- [ ] Sim2Real 真机部署
- [ ] 论文级实验
- [ ] 专利申请

## 文档

- [项目战略方案](docs/proposal.md)
- [项目研究报告](docs/01_report_draft.md)
- [项目简介](docs/02_project_brief.md)
- [查新报告](docs/03_novelty_search.md)
- [报名表填写指南](docs/04_registration_guide.md)
- [提交清单](docs/00_submission_checklist.md)

## 引用

```bibtex
@misc{embosight2026,
  title  = {EmboSight: A Zero-Shot Embodied Visual Assistant for the Visually Impaired
            via Active Perception and Multimodal Large Models},
  author = {[Author Name]},
  year   = {2026},
  note   = {Submitted to the 28th China Robot and Artificial Intelligence Competition},
}
```

## 致谢

- 仿真环境基于 [RoboCasa](https://robocasa.ai/)（Nasiriany 等, 2024）
- 视觉理解基于 [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)（阿里云）
- 大语言模型 API 由 [DeepSeek](https://www.deepseek.com/) 提供
- 项目灵感源于 [Be My Eyes](https://www.bemyeyes.com/) 等无障碍社区

## 许可证

MIT License - 详见 [LICENSE](LICENSE)