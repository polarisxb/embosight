"""RoboCasa 仿真环境验证脚本

校赛 Day 1 必跑。

用法:
    python scripts/verify_robocasa.py

输出:
    控制台逐项报告
    渲染图像保存到 ./results/verify/robocasa_test_render.png
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# 必须在 import mujoco 之前设置渲染后端
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# 终端 ANSI 颜色
OK = "\033[92m[OK]\033[0m"
ERR = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def check_python() -> bool:
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info >= (3, 9):
        print(f"{OK} Python 版本符合（≥ 3.9）")
        return True
    print(f"{ERR} Python 版本过低，需要 3.9+")
    return False


def check_torch() -> bool:
    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        print(f"CUDA available: {cuda_ok}")

        if cuda_ok:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"显存: {mem_gb:.1f} GB")
            print(f"{OK} PyTorch + CUDA")
            return True

        print(f"{WARN} PyTorch 已装但 CUDA 不可用")
        return False
    except ImportError as e:
        print(f"{ERR} PyTorch 未安装: {e}")
        return False


def check_mujoco() -> bool:
    try:
        import mujoco

        print(f"MuJoCo: {mujoco.__version__}")
        gl_backend = os.environ.get("MUJOCO_GL", "未设置")
        print(f"MUJOCO_GL: {gl_backend}")

        if gl_backend != "egl":
            print(f"{WARN} 建议设置 MUJOCO_GL=egl 以启用 GPU 加速渲染")

        print(f"{OK} MuJoCo")
        return True
    except ImportError as e:
        print(f"{ERR} MuJoCo 未安装: {e}")
        print("  安装: pip install mujoco")
        return False


def check_robosuite() -> bool:
    try:
        import robosuite

        print(f"robosuite: {robosuite.__version__}")
        print(f"{OK} robosuite")
        return True
    except ImportError as e:
        print(f"{ERR} robosuite 未安装: {e}")
        print("  安装: pip install robosuite")
        return False


def check_robocasa() -> bool:
    try:
        import robocasa

        rc_path = Path(robocasa.__file__).parent
        print(f"robocasa: 已安装")
        print(f"位置: {rc_path}")
        print(f"{OK} robocasa")
        return True
    except ImportError as e:
        print(f"{ERR} robocasa 未安装: {e}")
        print("  安装步骤:")
        print("    cd /root/autodl-tmp")
        print("    git clone https://github.com/robocasa/robocasa.git")
        print("    cd robocasa && pip install -e .")
        return False


def check_assets() -> bool:
    try:
        import robocasa

        rc_path = Path(robocasa.__file__).parent
        # 检查关键 assets 目录
        candidate_paths = [
            rc_path / "models" / "assets" / "fixtures",
            rc_path / "models" / "assets" / "kitchen_components",
            rc_path / "assets",
        ]

        for p in candidate_paths:
            if p.exists():
                stl_count = len(list(p.rglob("*.stl"))) + len(list(p.rglob("*.obj")))
                if stl_count > 0:
                    print(f"  找到资源目录: {p.relative_to(rc_path)}")
                    print(f"  Mesh 文件数: {stl_count}")
                    print(f"{OK} Kitchen Assets")
                    return True

        print(f"{WARN} 未找到 kitchen assets，请下载:")
        print("    cd /root/autodl-tmp/robocasa")
        print("    python robocasa/scripts/download_kitchen_assets.py")
        return False
    except Exception as e:
        print(f"{WARN} 资源检查失败: {e}")
        return False


def test_render() -> bool:
    """尝试创建 RoboCasa 厨房环境并渲染一张图"""
    try:
        import imageio.v2 as imageio
        import robosuite as suite

        print("正在创建厨房环境（首次约 30-60 秒）...")
        env = suite.make(
            env_name="PnPCounterToCab",
            robots="PandaMobile",
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=["robot0_eye_in_hand"],
            camera_heights=256,
            camera_widths=256,
            control_freq=20,
        )
        obs = env.reset()
        print(f"  环境创建成功")
        print(f"  观察键预览: {list(obs.keys())[:5]}...")

        img_key = "robot0_eye_in_hand_image"
        img = obs.get(img_key)
        if img is None:
            print(f"{ERR} 未找到图像键 {img_key}")
            return False

        print(f"  渲染图像 shape: {img.shape}, dtype: {img.dtype}")

        output_dir = REPO_ROOT / "results" / "verify"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "robocasa_test_render.png"

        imageio.imwrite(str(output_path), img)
        print(f"{OK} 渲染图像保存到 {output_path}")
        env.close()
        return True

    except Exception as e:
        print(f"{ERR} 渲染失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def test_robocasa_kitchen() -> bool:
    """尝试创建 RoboCasa 自带的 Kitchen 环境（如果可用）"""
    try:
        import robocasa
        import robosuite as suite

        # RoboCasa 注册了一些 Kitchen 任务
        print("正在尝试创建 RoboCasa Kitchen 任务...")
        try:
            env = suite.make(
                env_name="PnPCounterToCab",
                robots="PandaMobile",
                has_renderer=False,
                has_offscreen_renderer=True,
                use_camera_obs=True,
                camera_names=["robot0_eye_in_hand"],
                camera_heights=256,
                camera_widths=256,
                control_freq=20,
                layout_ids=0,
                style_ids=0,
            )
            obs = env.reset()
            print(f"  Kitchen 任务创建成功")
            env.close()
            print(f"{OK} RoboCasa Kitchen")
            return True
        except Exception as e:
            print(f"{WARN} Kitchen 任务创建失败（可能 assets 未下载完整）: {e}")
            return False

    except Exception as e:
        print(f"{WARN} RoboCasa Kitchen 测试失败: {e}")
        return False


def main() -> int:
    section("EmboSight 仿真环境验证")

    results: dict[str, bool] = {}

    section("Step 1: Python 环境")
    results["python"] = check_python()

    section("Step 2: PyTorch + CUDA")
    results["torch"] = check_torch()

    section("Step 3: MuJoCo")
    results["mujoco"] = check_mujoco()

    section("Step 4: robosuite")
    results["robosuite"] = check_robosuite()

    section("Step 5: robocasa")
    results["robocasa"] = check_robocasa()

    if results.get("robocasa", False):
        section("Step 6: Kitchen Assets")
        results["assets"] = check_assets()

    if all(results.get(k, False) for k in ["robosuite", "mujoco"]):
        section("Step 7: 基础渲染测试")
        results["render"] = test_render()

    if all(results.get(k, False) for k in ["robocasa", "robosuite"]):
        section("Step 8: RoboCasa Kitchen 任务测试")
        results["robocasa_kitchen"] = test_robocasa_kitchen()

    section("总结")
    for name, ok in results.items():
        status = OK if ok else ERR
        print(f"  {status} {name}")

    if all(results.values()):
        print(f"\n{OK} 所有验证通过！可以开始开发 EmboSight 了。")
        print("  下一步: 编写 src/env_wrapper.py，接入 RoboCasa")
        return 0

    failed = [k for k, v in results.items() if not v]
    print(f"\n{WARN} 失败项: {failed}")
    print("  请按上面的提示排查，或回 docs/05_autodl_guide.md 看常见问题")
    return 1


if __name__ == "__main__":
    sys.exit(main())