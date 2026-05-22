# EmboSight

零样本视障具身辅助智能体。项目面向视障辅助场景，将 RoboCasa 仿真、Qwen-VL 视觉语言模型、DeepSeek/OpenAI 兼容 LLM、主动视角规划和安全抓取策略组合成一个可运行的具身智能闭环。

> 当前仓库仍处于竞赛研发阶段。Windows 本地适合做代码开发和单元测试；完整仿真、VLM 推理和视频录制建议在 Linux + CUDA 机器上运行，例如 AutoDL RTX 4090 实例。

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: WIP](https://img.shields.io/badge/Status-WIP-orange.svg)]()

## 核心能力

- 视障任务分解：把自然语言请求拆成包含方位、距离、触觉、安全和行动建议的子任务。
- 主动视角规划：通过离散视角库和 LLM 选择下一步观察视角。
- 多模态感知：用 Qwen2.5-VL / Qwen3-VL 对 RoboCasa 观测图像做语义理解和定位辅助。
- 安全门控：对刀具、高温、易碎物等高风险对象进行额外确认。
- 抓取执行与恢复：在仿真环境中执行导航、接近、抓取和失败恢复策略。
- 评估与复现：提供固定 seed 场景、episode 日志、oracle 汇总和视频录制脚本。

## 目录结构

```text
embodied-AI-one/
├── configs/
│   ├── default.yaml              # 顶层 LLM/VLM/RoboCasa 配置
│   ├── agent.yaml                # agent 阈值、cache、logger、策略配置
│   ├── eval_scenarios.yaml       # 固定评估场景
│   └── viewpoints.yaml           # 离散视角库
├── docs/                         # 申报材料、研究报告、方案文档
├── eval/
│   ├── run_fixed.py              # 单个固定场景评估
│   ├── run_batch.py              # 批量评估
│   └── run_long_generalization.py
├── prompts/                      # LLM/VLM prompt 模板
├── scripts/
│   ├── run_agent.py              # 端到端 agent 入口
│   ├── verify_robocasa.py        # RoboCasa 安装与渲染验证
│   ├── test_real_llm.py          # DeepSeek/LLM 连通性测试
│   ├── test_real_vlm.py          # VLM 下载与推理测试
│   ├── record_video.py           # episode 视频录制
│   └── setup_autodl.sh           # AutoDL 辅助安装脚本
├── src/                          # 核心代码
├── tests/                        # 单元测试与回归测试
├── .env.example                  # 环境变量模板
├── requirements.txt              # Python 依赖，torch 需单独安装
└── README.md
```

## 环境要求

完整运行推荐：

- Ubuntu 22.04 或兼容 Linux 环境
- Python 3.10
- CUDA 12.1+
- NVIDIA GPU，建议 24GB 显存及以上
- Git、Conda 或 venv
- DeepSeek API Key，或其他 OpenAI 兼容 API Key

本地开发和纯单元测试：

- Python 3.10
- 不强制需要 GPU
- 可不安装 RoboCasa/Qwen 权重，但与仿真、VLM、真实 LLM 相关的集成测试会跳过或失败

## 快速部署

### 1. 获取仓库

```bash
git clone https://github.com/<your-username>/embodied-AI-one.git
cd embodied-AI-one
```

如果已经在本仓库目录中，直接进入下一步。

### 2. 创建 Python 环境

Conda：

```bash
conda create -n embosight python=3.10 -y
conda activate embosight
python -m pip install --upgrade pip setuptools wheel
```

venv：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 3. 安装 PyTorch

`requirements.txt` 不直接锁定 torch，因为 CUDA wheel 需要按机器环境选择。CUDA 12.1 环境推荐：

```bash
pip install torch==2.3.0 torchvision==0.18.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

验证：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda :", torch.cuda.is_available())
print("gpu  :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

### 4. 安装项目依赖

```bash
pip install --upgrade-strategy only-if-needed -r requirements.txt
```

### 5. 安装 RoboCasa

RoboCasa 建议源码安装，避免训练依赖把环境拉得过重：

```bash
cd ..
git clone https://github.com/robocasa/robocasa.git
cd robocasa
pip install -e . --no-deps
pip install pygame pynput hidapi lxml gymnasium
python robocasa/scripts/download_kitchen_assets.py
cd ../embodied-AI-one
```

如果你使用 AutoDL 并且仓库在 `/root/autodl-tmp/embodied-AI-one`，可以把上面的路径对应替换。

### 6. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
HF_ENDPOINT=https://hf-mirror.com
PROJECT_ROOT=.
LOG_LEVEL=INFO
```

说明：

- `scripts/run_agent.py` 会优先读取项目根目录的 `.env`。
- `.env` 已被 `.gitignore` 忽略，不要提交真实 API Key。
- 如果不用 DeepSeek，可以把 `DEEPSEEK_API_KEY` 换成 `OPENAI_API_KEY`，同时在 `configs/default.yaml` 中改 `llm.base_url` 和 `llm.model`。

### 7. 下载 VLM 权重

当前 `configs/default.yaml` 默认使用：

```yaml
vlm:
  model_id: "./checkpoints/Qwen3-VL-8B-Instruct"
```

推荐显式下载到该目录：

```bash
mkdir -p checkpoints
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir checkpoints/Qwen3-VL-8B-Instruct
```

如果你想使用 Qwen2.5-VL，可以运行仓库脚本下载：

```bash
python scripts/test_real_vlm.py --download-only
```

然后把 `configs/default.yaml` 里的 `vlm.model_id` 改成实际本地目录，例如：

```yaml
vlm:
  model_id: "./checkpoints/Qwen2.5-VL-7B-Instruct"
```

## AutoDL 部署

推荐镜像：

- Ubuntu 22.04
- Python 3.10
- CUDA 12.1
- RTX 4090 或同级 24GB 显存 GPU

如果镜像已经预装 PyTorch + CUDA，可以直接：

```bash
cd /root/autodl-tmp
git clone https://github.com/<your-username>/embodied-AI-one.git
cd embodied-AI-one
bash scripts/setup_autodl.sh
cp .env.example .env
```

然后编辑 `.env`，填入 API Key。

如果镜像没有预装合适的 PyTorch，请先执行本文“安装 PyTorch”步骤，再运行 `scripts/setup_autodl.sh` 或手动安装依赖。

## 运行验证

建议按下面顺序验证。前一层不通过时，不要直接跑后一层。

### 1. 代码级快速检查

```bash
python -m pytest tests/test_public_api.py -q
python -m pytest tests/test_task_decomposer_v1.py tests/test_safety_classifier.py -q
python -m ruff check .
```

### 2. 完整单元测试

```bash
python -m pytest -q
```

如果当前机器没有 GPU、RoboCasa 或模型权重，部分集成测试可能失败。此时先用快速检查确认纯代码路径，再去 Linux/GPU 环境跑完整验证。

### 3. RoboCasa 仿真验证

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python scripts/verify_robocasa.py
```

成功后会生成：

```text
results/verify/robocasa_test_render.png
```

### 4. LLM 连通性测试

确保 `.env` 中已经配置 `DEEPSEEK_API_KEY`：

```bash
python scripts/test_real_llm.py
```

这个脚本会测试：

- API 连通性
- JSON 输出
- 任务分解
- 主动视角规划

### 5. VLM 推理测试

先确保已有测试图片。最简单方式是先运行 RoboCasa 验证生成渲染图：

```bash
python scripts/verify_robocasa.py
```

再运行：

```bash
python scripts/test_real_vlm.py
```

### 6. 单场景端到端运行

```bash
python scripts/run_agent.py \
  --query "pick up the apple" \
  --config configs/default.yaml \
  --agent-config configs/agent.yaml \
  --user-mode fake_from_robocasa \
  --seed 42
```

常用 `--user-mode`：

- `fake_from_robocasa`：从 RoboCasa 当前场景读取真实目标并改写 query，适合自动评估。
- `fake_from_query`：保留命令中的目标对象，适合固定目标实验。
- `cli`：命令行交互模式。

### 7. 固定场景评估

```bash
python eval/run_fixed.py --scenario fixed_seed_discover_001
python eval/run_fixed.py --scenario fixed_lemon_001 --memory-dir memory/eval_fixed_lemon_001
```

可用场景定义在 `configs/eval_scenarios.yaml`。运行后会在 `logs/episodes/` 写入 episode JSON，并打印 oracle summary。

### 8. 录制成功演示视频

竞赛材料建议优先使用成功优先包装脚本。它会先尝试 `fixed_lemon_001`，不成功再依次尝试其他固定厨房场景；只有 episode 成功并且录到足够帧数时，才会输出最终 MP4。

```bash
python scripts/record_success_video.py \
  --output results/videos/embosight_kitchen_success.mp4 \
  --multi \
  --min-frames 8
```

如果想录单路高清正面视角：

```bash
python scripts/record_success_video.py \
  --output results/videos/embosight_kitchen_success_hd.mp4 \
  --hd \
  --min-frames 8
```

输出成功时会同时生成：

```text
results/videos/embosight_kitchen_success.mp4
results/videos/embosight_kitchen_success.json
```

JSON 摘要里会记录最终采用的场景、每次尝试的退出码和中间视频路径。

单场景录制仍可直接使用 `record_video.py`。加上 `--require-success` 后，失败 episode 不会保存为最终视频：

```bash
python scripts/record_video.py \
  --scenario fixed_seed_discover_001 \
  --output results/videos/fixed_seed_discover_001.mp4 \
  --hd \
  --require-success \
  --min-frames 8
```

多视角拼接：

```bash
python scripts/record_video.py \
  --scenario fixed_lemon_001 \
  --output results/videos/fixed_lemon_001_multi.mp4 \
  --multi \
  --require-success \
  --min-frames 8
```

输出目录：

```text
results/videos/
```

## 常用开发命令

```bash
# 查看固定场景配置
python - <<'PY'
import yaml
data = yaml.safe_load(open("configs/eval_scenarios.yaml", encoding="utf-8"))
print([s["id"] for s in data["scenarios"]])
PY

# 只跑抓取相关测试
python -m pytest tests/test_grasp_policy.py tests/test_grasp_planner.py tests/test_grasp_execution.py -q

# 只跑 agent 决策相关测试
python -m pytest tests/test_agent_run.py tests/test_agent_decide_next.py tests/test_action_executor_v1.py -q

# 清理本地记忆缓存
python scripts/clean_memory.py
```

## 输出文件

运行过程中常见输出：

```text
logs/episodes/                 # episode JSON、belief trace
results/verify/                # RoboCasa 验证渲染图
results/videos/                # 演示视频
results/observations/          # 观测图像
memory/                        # 可选长期记忆目录
checkpoints/                   # 本地模型权重
```

这些目录多数是运行产物，不应把大文件直接提交到 Git。

## 常见问题

### CUDA 不可用

先确认驱动和 torch wheel 匹配：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

如果 `torch.cuda.is_available()` 是 `False`，通常是安装了 CPU 版 torch，重新安装 CUDA wheel。

### MuJoCo 或 RoboCasa 渲染失败

Linux 服务器上优先使用 EGL：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python scripts/verify_robocasa.py
```

如果提示缺少 kitchen assets，进入 RoboCasa 源码目录执行：

```bash
python robocasa/scripts/download_kitchen_assets.py
```

### LLM 测试失败

检查：

- `.env` 是否存在于项目根目录。
- `DEEPSEEK_API_KEY` 是否有效。
- 服务器是否能访问 `https://api.deepseek.com/v1`。
- `configs/default.yaml` 中 `llm.base_url` 和 `llm.model` 是否与 API 服务匹配。

### VLM 权重找不到

检查 `configs/default.yaml` 的 `vlm.model_id` 是否指向真实目录：

```bash
ls checkpoints
```

如果模型目录名和配置不一致，改配置或重新下载。

### Windows 本地怎么跑

Windows 本地建议只跑纯 Python 单元测试：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python -m pytest tests/test_public_api.py -q
```

RoboCasa、MuJoCo EGL、CUDA VLM 推理建议放到 Linux/GPU 环境运行。

## 申报材料

竞赛材料主要放在 `docs/`：

- `docs/00_submission_checklist.md`：提交材料清单
- `docs/01_report_draft.md`：项目研究报告草稿
- `docs/02_project_brief.md`：400 字项目简介
- `docs/03_novelty_search.md`：查新报告模板
- `docs/04_registration_guide.md`：报名表填写指南
- `docs/CRAIC2026_项目研究报告.md`：CRAIC 研究报告版本
- `docs/CRAIC2026_查新报告.md`：CRAIC 查新报告版本

## 引用

```bibtex
@misc{embosight2026,
  title  = {EmboSight: A Zero-Shot Embodied Visual Assistant for the Visually Impaired via Active Perception and Multimodal Large Models},
  author = {[Author Name]},
  year   = {2026},
  note   = {Submitted to the China Robot and Artificial Intelligence Competition},
}
```

## License

MIT License. See [LICENSE](LICENSE).
