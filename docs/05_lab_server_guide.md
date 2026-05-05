# EmboSight 部署指南（实验室服务器版 - Rocky Linux）

> **目标**：在学校/实验室服务器上部署 EmboSight + RoboCasa
> **总耗时**：约 1-1.5 小时（首次）
> **场景**：Rocky Linux 8.x / RHEL 8.x / CentOS 8.x，无 sudo 权限

---

## 服务器环境约束

```
✅ 已确认条件:
   - Rocky Linux 8.8 (RHEL 系)
   - GPU: A10 24GB × 4
   - RAM: 502 GB, 32 CPU 核
   - 共享存储: /share (50 TB)
   - 用户家目录: /share/home/<user>
   - Anaconda 已装在 /share/apps/anaconda3
   - libEGL/libGL/libcuda 系统库齐全（无需 sudo 安装）

⚠️ 限制:
   - 无 sudo 权限
   - 共享 anaconda（不能修改系统 env）
   - 网络走代理（curl 能通 GitHub/HF）
```

---

## 部署 4 大步骤

```
Step 1: 创建用户级 conda 环境           5 分钟
Step 2: 同步代码 + 安装 Python 依赖     20 分钟
Step 3: 安装 RoboCasa + 下载 assets     30 分钟
Step 4: 配置渲染 + 验证                 10 分钟
合计                                    ~1 小时
```

---

## Step 1：创建用户级 conda 环境

```bash
cd ~/embodied
mkdir -p ~/embodied/embodied-AI-one
cd ~/embodied

conda create -n embosight python=3.10 -y
conda activate embosight

python --version
which python
```

期望输出：

```
Python 3.10.x
/share/home/<user>/.conda/envs/embosight/bin/python
```

**所有后续命令都在 `embosight` 环境下执行**，每次新开 terminal 后都要：

```bash
conda activate embosight
```

---

## Step 2：同步代码 + 安装依赖

### 2.1 配置 pip 国内镜像（一次配置）

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.timeout 120
```

### 2.2 配置 HuggingFace 镜像（一次配置）

```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
echo $HF_ENDPOINT
```

### 2.3 同步代码

**方式 A：从 GitHub clone（推荐）**

先在你本地 Windows 把代码推到 GitHub：
```bash
cd c:\all_project\embodied-AI-one
git init && git add -A && git commit --trailer "Co-authored-by: Cursor <cursoragent@cursor.com>" -m "initial scaffold"
git remote add origin https://github.com/<your-user>/embodied-AI-one.git
git branch -M main
git push -u origin main
```

然后在服务器：
```bash
cd ~/embodied
git clone https://github.com/<your-user>/embodied-AI-one.git
cd embodied-AI-one
ls -la
```

**方式 B：scp 上传（不用 git）**

在本地 Windows PowerShell：
```powershell
cd c:\all_project
scp -r embodied-AI-one <user>@<server-ip>:~/embodied/
```

### 2.4 安装 PyTorch（针对 CUDA 12.4 驱动）

```bash
pip install torch==2.3.0 torchvision==0.18.0 \
  --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

期望输出：

```
2.3.0+cu121 True 4
```

> 💡 **为什么用 cu121 不用 cu124**：CUDA 向后兼容，PyTorch 官方 wheel 编译针对 cu121，能在 12.4 驱动上稳定运行。

### 2.5 安装项目依赖

```bash
cd ~/embodied/embodied-AI-one
pip install -r requirements.txt
```

预计 5-10 分钟。

> 如果 `flash-attn` 安装失败：跳过它（注释掉 requirements.txt 里那行），不影响主流程。

### 2.6 验证基础 import

```bash
python -c "
import torch, transformers, mujoco, robosuite
print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
print('GPU 数量:', torch.cuda.device_count())
print('GPU 型号:', torch.cuda.get_device_name(0))
print('transformers:', transformers.__version__)
print('mujoco:', mujoco.__version__)
print('robosuite:', robosuite.__version__)
"
```

---

## Step 3：安装 RoboCasa（最耗时）

### 3.1 从源码安装

```bash
cd ~/embodied
git clone https://github.com/robocasa/robocasa.git
cd robocasa
pip install -e .
```

### 3.2 下载 kitchen assets（约 4-6 GB，10-30 分钟）

```bash
cd ~/embodied/robocasa
python robocasa/scripts/download_kitchen_assets.py
```

> ⚠️ 走代理网络可能慢，耐心等。如果中断，重新运行（脚本会续传）。

### 3.3 setup macros

```bash
python robocasa/scripts/setup_macros.py
```

---

## Step 4：配置渲染 + 验证

### 4.1 设置 MuJoCo 渲染后端

```bash
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
echo 'export PYOPENGL_PLATFORM=egl' >> ~/.bashrc
source ~/.bashrc

echo "MUJOCO_GL=$MUJOCO_GL"
echo "PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM"
```

### 4.2 设置 GPU 选择（4 张 A10 中选 1 张）

```bash
echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc
source ~/.bashrc
nvidia-smi -i 0
```

> 💡 后续如果想切换到 GPU 1/2/3，直接在终端执行 `export CUDA_VISIBLE_DEVICES=1`。
> 💡 如果想多卡并行实验，把 4 张 GPU 分给不同的实验脚本。

### 4.3 写 .env 配置 DeepSeek API Key

```bash
cd ~/embodied/embodied-AI-one
cp .env.example .env
nano .env
```

把 `your_deepseek_api_key_here` 改成你在 https://platform.deepseek.com/ 申请的真实 Key。

### 4.4 跑验证脚本

```bash
cd ~/embodied/embodied-AI-one
python scripts/verify_robocasa.py
```

期望全部 [OK]，并在 `./results/verify/robocasa_test_render.png` 生成一张厨房图。

---

## 常见问题

### Q1: `conda activate embosight: command not found`
原因：conda init 没运行
```bash
conda init bash
source ~/.bashrc
```

### Q2: `RuntimeError: Found no NVIDIA driver`
原因：`CUDA_VISIBLE_DEVICES` 设置错误
```bash
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
```

### Q3: `mujoco.FatalError: gladLoadGL error`
原因：`MUJOCO_GL` 未设
```bash
export MUJOCO_GL=egl
echo $MUJOCO_GL
```

### Q4: `PermissionError: /share/apps/anaconda3/...`
原因：试图写入共享 anaconda
```bash
conda activate embosight
which python   # 应该是 /share/home/.../.conda/envs/embosight/bin/python
```

### Q5: HuggingFace 下载超时
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./checkpoints/Qwen2.5-VL-7B-Instruct
```

### Q6: RoboCasa download_kitchen_assets 卡住
原因：网络代理不稳定
```bash
unset http_proxy https_proxy
python robocasa/scripts/download_kitchen_assets.py
```

### Q7: pip install 包冲突
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-deps
pip install <冲突的包> --upgrade
```

---

## 多卡使用建议

A10 × 4 张，可以这样分配：

```
GPU 0: 主开发 / 跑 demo
GPU 1: Baseline 实验
GPU 2: 消融实验
GPU 3: 备用 / VLM 多副本

切换方法:
  CUDA_VISIBLE_DEVICES=1 python scripts/run_demo.py --query "..."
  CUDA_VISIBLE_DEVICES=2 python -m src.eval --config configs/baseline.yaml
```

---

## 服务器维护建议

### 关闭未用进程（节省 GPU）

```bash
# 查看所有 Python 进程
ps -ef | grep python | grep $USER

# 查看 GPU 占用
nvidia-smi

# 杀掉特定 PID
kill -9 <PID>
```

### 看磁盘占用

```bash
du -sh ~/embodied/*
```

---

## Day 1 自查清单

```
环境配置
- [ ] conda env "embosight" 创建并激活
- [ ] HF_ENDPOINT 设置为 https://hf-mirror.com
- [ ] PyPI 镜像设为清华源
- [ ] MUJOCO_GL=egl 设置完成
- [ ] CUDA_VISIBLE_DEVICES=0 设置完成

代码与依赖
- [ ] 代码同步到 ~/embodied/embodied-AI-one
- [ ] PyTorch 2.3.0 + CUDA 12.1 可用
- [ ] requirements.txt 全部安装
- [ ] robocasa 从源码安装完成
- [ ] kitchen_assets 下载完成
- [ ] DeepSeek API Key 已配置

验证
- [ ] python scripts/verify_robocasa.py 全部通过
- [ ] ./results/verify/robocasa_test_render.png 渲染成功
```