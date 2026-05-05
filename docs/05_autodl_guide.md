# EmboSight 仿真环境部署指南（AutoDL + RoboCasa）

> **目标**：在 AutoDL 上从零搭建 RoboCasa 仿真环境
> **总耗时**：约 1.5-2 小时（首次）
> **难度**：⭐⭐⭐（按本指南一步步走，不会踩坑）

---

## Day 1 时间预算

| 阶段 | 预计耗时 | 关键产出 |
|---|---|---|
| 1. 账号 + 租卡 | 20 min | 4090 实例运行中 |
| 2. 代码部署 | 15 min | 项目同步到云端 |
| 3. 依赖安装 | 30 min | requirements 全装完 |
| 4. RoboCasa 资源 | 30 min | 厨房 assets 就位 |
| 5. 头显渲染配置 | 5 min | EGL 环境变量 |
| 6. Hello World | 15 min | 渲染第一张厨房图 |
| **合计** | **~2 hr** | **RoboCasa 跑通** |

---

## Step 1：AutoDL 账号准备

### 1.1 注册 + 实名

访问 **https://www.autodl.com/** → 手机号注册 → **实名认证**（必须，否则不能租卡）。

### 1.2 充值

建议**首次充值 100 元**（够 15 天校赛阶段用）。

> 💡 学生身份认证可享 8 折优惠，注册时记得勾选"我是学生"。

---

## Step 2：创建 GPU 实例

### 2.1 算力市场选卡

控制台 → **算力市场**，筛选：

| 配置项 | 选择 |
|---|---|
| GPU 型号 | **RTX 4090 24GB** |
| GPU 数量 | 1 |
| 付费类型 | **按量计费**（约 ¥2.5-3/小时） |
| 地区 | 任选（北京 / 上海 / 内蒙等都行） |

> ⚠️ 4090 缺货时退而求其次：RTX 3090 24GB 或 A5000 24GB 都行，避开 16GB 卡（VLM 跑不动）。

### 2.2 选择镜像

镜像市场 → 搜索"PyTorch" → 选：

```
框架: PyTorch 2.3.0
Python: 3.10 (ubuntu22.04)
CUDA: 12.1
镜像名: PyTorch 2.3.0 / Python 3.10 / Ubuntu22.04 / CUDA 12.1
```

### 2.3 数据盘扩展

- 系统盘：保持默认 30GB
- 数据盘：**扩展到 50GB**（用于 Qwen2.5-VL 权重 16GB + RoboCasa 资源 6GB + 实验数据）

### 2.4 立即创建

点击"立即创建" → 等约 1-2 分钟开机。开机后实例状态变为"运行中"。

---

## Step 3：连接实例

### 3.1 推荐方式：JupyterLab 网页

控制台 → 实例列表 → 点击"**JupyterLab**"按钮 → 浏览器打开。

里面有：
- **文件管理器**（左侧，可拖拽上传文件）
- **Terminal**（右上"+"号 → Other → Terminal）
- **Notebook**（用于交互调试）

### 3.2 备用：SSH 连接（高级用户）

```bash
# 控制台"实例"页面会显示 SSH 命令，类似：
ssh -p 12345 root@region-X.autodl.cloud
```

### 3.3 用 Cursor 远程连接（推荐高级配置）

```
Cursor → 左下角 ><(打开远程窗口) → 连接到主机
→ SSH 配置文件添加 AutoDL ssh 信息
→ 重新打开远程文件夹 /root/autodl-tmp
```

这样你可以**在本地 Cursor 里直接编辑云端代码**，省去同步麻烦。

---

## Step 4：代码部署

打开 JupyterLab 的 Terminal，执行：

```bash
# 进入持久化数据盘（关机不会丢失）
cd /root/autodl-tmp

# 启用 AutoDL 学术加速（GitHub 提速 5-10 倍）
source /etc/network_turbo
```

### 4.1 方式 A：Git 同步（推荐）

**首先在本地把代码推到 GitHub**：

```bash
# 在你本地 Windows 的 Cursor 终端
cd c:\all_project\embodied-AI-one
git init
git add -A
git commit --trailer "Co-authored-by: Cursor <cursoragent@cursor.com>" -m "initial scaffold: EmboSight v0.1"

# 到 github.com 创建空仓库 (不要勾选 README)
# 命名为 embodied-AI-one 或 EmboSight

git remote add origin https://github.com/<你的用户名>/embodied-AI-one.git
git branch -M main
git push -u origin main
```

**然后在 AutoDL 终端 clone**：

```bash
cd /root/autodl-tmp
git clone https://github.com/<你的用户名>/embodied-AI-one.git
cd embodied-AI-one
ls -la
```

### 4.2 方式 B：拖拽上传（不熟悉 Git 的）

- 在 JupyterLab 文件浏览器中打开 `/root/autodl-tmp`
- 把本地的整个 `embodied-AI-one` 文件夹拖进去
- 上传约 5-10 分钟（看网速）

> ⚠️ 不要上传 PDF（CRAIC2026.pdf 11MB 的那个），AutoDL 上不需要。

---

## Step 5：基础环境检查

```bash
nvidia-smi                                      # 应该看到 RTX 4090
python --version                                # 应该是 3.10.x
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望: 2.3.0 True
```

---

## Step 6：配置加速源

```bash
# 1. 设置 PyPI 国内镜像（清华源）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 设置 HuggingFace 镜像（重要！否则下载 Qwen2.5-VL 会超时）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc

# 3. 验证
echo $HF_ENDPOINT
# 期望输出: https://hf-mirror.com
```

---

## Step 7：安装项目依赖

```bash
cd /root/autodl-tmp/embodied-AI-one
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

预计耗时 5-10 分钟。

> ⚠️ **如果某个包安装失败**，单独装：
> ```bash
> pip install <包名> -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

---

## Step 8：安装 RoboCasa（关键 30 分钟）

### 8.1 验证 robosuite

robosuite 应已被 requirements.txt 装好：

```bash
python -c "import robosuite; print(robosuite.__version__)"
# 期望: 1.5.x
```

### 8.2 从源码安装 robocasa

```bash
cd /root/autodl-tmp
git clone https://github.com/robocasa/robocasa.git
cd robocasa
pip install -e .
```

### 8.3 下载 kitchen assets（最耗时步骤）

RoboCasa 需要下载 mesh + texture + 物理参数，约 **4-6 GB**：

```bash
cd /root/autodl-tmp/robocasa
python robocasa/scripts/download_kitchen_assets.py
```

下载约 10-30 分钟。

> ⚠️ **慢/卡住怎么办？**
> ```bash
> # 重启网络加速
> source /etc/network_turbo
> # 重新运行下载
> python robocasa/scripts/download_kitchen_assets.py
> ```

### 8.4 setup macros（关键）

```bash
python robocasa/scripts/setup_macros.py
```

---

## Step 9：配置头显渲染（关键 5 分钟）

云端没有显示器，必须用 GPU 加速的离屏渲染（EGL）：

```bash
# 1. 安装系统依赖
apt-get update
apt-get install -y libgl1-mesa-glx libegl1 libgles2 libglfw3 libosmesa6

# 2. 设置环境变量（关键）
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
echo 'export PYOPENGL_PLATFORM=egl' >> ~/.bashrc
source ~/.bashrc

# 3. 验证
echo $MUJOCO_GL
# 期望: egl
```

---

## Step 10：Hello World 验证

```bash
cd /root/autodl-tmp/embodied-AI-one
python scripts/verify_robocasa.py
```

期望输出：

```
============================================================
EmboSight 仿真环境验证
============================================================

Step 1: Python 环境
[OK] Python 3.10.x

Step 2: PyTorch
[OK] PyTorch 2.3.0  CUDA True  GPU RTX 4090  24.0 GB

Step 3: MuJoCo
[OK] MuJoCo 3.x  MUJOCO_GL=egl

Step 4: robosuite
[OK] robosuite 1.5.x

Step 5: robocasa
[OK] robocasa

Step 6: Kitchen Assets
[OK] 找到 NNN 个 STL 文件

Step 7: 渲染测试
环境创建成功
渲染图像 shape: (256, 256, 3)
[OK] 渲染图像保存到 ./results/verify/robocasa_test_render.png
```

打开生成的 png 看看，应该能看到一张厨房场景的相机视角图。

---

## 常见问题排查

### Q1: `RuntimeError: GLFW... cannot open display`
原因：MUJOCO_GL 没设成 egl
```bash
export MUJOCO_GL=egl
```

### Q2: `nvidia-smi` 看到 GPU 但 PyTorch CUDA = False
原因：PyTorch 版本和 CUDA 不匹配
```bash
pip install --force-reinstall torch torchvision \
  --index-url https://download.pytorch.org/whl/cu121
```

### Q3: `mujoco.FatalError: gladLoadGL error`
原因：缺 OpenGL 库
```bash
apt-get install -y libgl1-mesa-glx libegl1 libgles2 libglfw3 libosmesa6
```

### Q4: HuggingFace 下载超时
原因：未设置镜像
```bash
env | grep HF
# 应该看到 HF_ENDPOINT=https://hf-mirror.com
# 如果没有，重新执行 source ~/.bashrc
```

### Q5: `pip install` 卡住
原因：默认源慢
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q6: RoboCasa assets 下载失败
原因：网络不稳定
```bash
# 解除可能的代理
unset http_proxy https_proxy

# 重新启用 AutoDL 学术加速
source /etc/network_turbo

# 重试
python robocasa/scripts/download_kitchen_assets.py
```

### Q7: 内存/显存不够
```bash
# 查看占用
nvidia-smi
free -h

# 杀进程
kill -9 <PID>
```

### Q8: AutoDL 实例突然关机
原因：余额不足
- 控制台"账户" → 充值

### Q9: `ImportError: libpython3.10.so.1.0: cannot open shared object`
原因：conda env 的 LD_LIBRARY_PATH 没设
```bash
echo 'export LD_LIBRARY_PATH=/root/miniconda3/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Q10: RoboCasa 报 `xml not found`
原因：setup_macros.py 没运行
```bash
cd /root/autodl-tmp/robocasa
python robocasa/scripts/setup_macros.py
```

---

## Day 1 自查清单

```
账号准备
- [ ] AutoDL 账号实名认证完成
- [ ] 充值 ≥ 50 元

实例创建
- [ ] 4090 24GB 实例创建并开机
- [ ] PyTorch 2.3.0 + CUDA 12.1 镜像

环境配置
- [ ] 学术加速已启用（source /etc/network_turbo）
- [ ] HF_ENDPOINT=https://hf-mirror.com
- [ ] PyPI 镜像设为清华源
- [ ] MUJOCO_GL=egl

代码与依赖
- [ ] 代码同步到 /root/autodl-tmp/embodied-AI-one
- [ ] requirements.txt 全部安装
- [ ] robocasa 从源码安装完成
- [ ] kitchen_assets 下载完成

验证
- [ ] python scripts/verify_robocasa.py 全部通过
- [ ] ./results/verify/robocasa_test_render.png 渲染成功
```

---

## 关机前操作（省钱必读）

每次结束工作前：

```bash
# 1. 提交代码到 git
cd /root/autodl-tmp/embodied-AI-one
git add -A
git commit --trailer "Co-authored-by: Cursor <cursoragent@cursor.com>" -m "Day X: 完成 XXX"
git push

# 2. 控制台 → 实例 → 关机（按量计费, 关机不计费）
```

> ⚠️ **持久化存储**：`/root/autodl-tmp` 在关机时**不会删除**，可以放心存模型权重和数据。
> ⚠️ **临时存储**：`/root/`（系统盘）在关机后会重置，**不要把代码放这里**！

---

## 下一步（仿真环境跑通后）

1. **Day 2**：完整实现 `src/env_wrapper.py`（接入 RoboCasa）
2. **Day 3**：测试 `pipeline.py` 端到端（手动构造一个 query 跑通）
3. **Day 4-5**：完整实现三大创新模块的细节
4. **Day 6**：跑 baseline 实验
5. **Day 7-12**：跑完整实验 + 录视频
6. **Day 13-14**：写报告 + 准备 PPT
7. **Day 15**：提交校赛