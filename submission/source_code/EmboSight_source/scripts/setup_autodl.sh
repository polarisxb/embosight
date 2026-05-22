#!/usr/bin/env bash
# ============================================================
# EmboSight - AutoDL 一键环境配置脚本
# ============================================================
# 用法:
#   bash scripts/setup_autodl.sh
#
# 前置条件:
#   - AutoDL RTX 4090 实例（24GB 显存）
#   - PyTorch 2.3.0 + CUDA 12.1 镜像
#   - Python 3.10
# ============================================================

set -e
set -o pipefail

echo "=== [1/7] 系统信息 ==="
nvidia-smi
python --version
pip --version

echo ""
echo "=== [2/7] 设置 HuggingFace 镜像（中国大陆加速）==="
export HF_ENDPOINT=https://hf-mirror.com
echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc

echo ""
echo "=== [3/7] 升级 pip ==="
pip install --upgrade pip setuptools wheel

echo ""
echo "=== [4/7] 安装核心依赖 ==="
pip install -r requirements.txt

echo ""
echo "=== [5/7] 安装 RoboCasa（从源码）==="
if ! python -c "import robocasa" 2>/dev/null; then
    pip install git+https://github.com/robocasa/robocasa.git
else
    echo "  RoboCasa 已安装，跳过"
fi

echo ""
echo "=== [6/7] 预下载 Qwen2.5-VL 模型权重 ==="
mkdir -p ./checkpoints
python -c "
from huggingface_hub import snapshot_download
print('正在下载 Qwen2.5-VL-7B-Instruct...')
snapshot_download(
    repo_id='Qwen/Qwen2.5-VL-7B-Instruct',
    cache_dir='./checkpoints',
    resume_download=True,
)
print('Qwen2.5-VL 下载完成')
"

echo ""
echo "=== [7/7] 验证安装 ==="
python -c "
import torch
print(f'PyTorch:    {torch.__version__}')
print(f'CUDA OK:    {torch.cuda.is_available()}')
print(f'GPU:        {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')
print(f'显存:       {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
python -c "import mujoco; print(f'mujoco:      {mujoco.__version__}')"

echo ""
echo "==================================================="
echo "环境配置完成！"
echo ""
echo "下一步:"
echo "  1. 编辑 .env 写入 DEEPSEEK_API_KEY"
echo "  2. 运行 demo:"
echo "       python scripts/run_demo.py --query \"我的药瓶在哪？\""
echo "==================================================="