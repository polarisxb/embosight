#!/usr/bin/env python3
"""Phase 3 验证: Qwen2.5-VL 本地 VLM 集成测试

在服务器上运行:
    # Step 1: 下载模型（首次约 15-30 分钟）
    python scripts/test_real_vlm.py --download-only

    # Step 2: 完整测试（需要先跑过 Phase 1 生成观察图像）
    python scripts/test_real_vlm.py

前置条件:
    1. GPU 显存 >= 16GB（推荐 24GB）
    2. pip install torch transformers qwen-vl-utils accelerate
    3. Phase 1 已跑过（results/observations/ 下有图像）
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_real_vlm")

SEPARATOR = "=" * 60
REPO_ROOT = Path(__file__).resolve().parent.parent


def check_gpu():
    """检查 GPU 状态"""
    print(f"\n[0] GPU 检查")
    try:
        import torch
        if not torch.cuda.is_available():
            print(f"  [WARN] CUDA 不可用，将使用 CPU（非常慢）")
            return "cpu"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {gpu_name}")
        print(f"  显存: {gpu_mem:.1f} GB")
        if gpu_mem < 16:
            print(f"  [WARN] 显存 < 16GB，可能 OOM，建议用 float16")
        print(f"  [OK] GPU 可用")
        return "cuda"
    except Exception as e:
        print(f"  [WARN] GPU 检查失败: {e}，使用 CPU")
        return "cpu"


def download_model():
    """仅下载模型，不执行推理"""
    print(f"\n[1] 下载 Qwen2.5-VL-7B 模型")
    print(f"  缓存目录: ./checkpoints")
    print(f"  模型大小约 15GB，首次下载需要 15-30 分钟...")

    try:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        import torch

        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        cache_dir = "./checkpoints"

        print(f"  正在下载 Processor...")
        processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=cache_dir,
        )
        print(f"  [OK] Processor 下载完成")

        print(f"  正在下载模型权重（约 15GB）...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # 下载时不占 GPU
            cache_dir=cache_dir,
        )
        print(f"  [OK] 模型下载完成")
        del model, processor

        return True
    except Exception as e:
        print(f"  [FAIL] 下载失败: {e}")
        print(f"  提示:")
        print(f"    1. pip install torch transformers accelerate")
        print(f"    2. 如果网络超时，检查 .env 中 HF_ENDPOINT=https://hf-mirror.com")
        print(f"    3. 或手动: huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./checkpoints/Qwen2.5-VL-7B-Instruct")
        return False


def find_test_image() -> str:
    """找一张测试图像"""
    obs_dir = REPO_ROOT / "results" / "observations"
    if obs_dir.exists():
        pngs = list(obs_dir.glob("*.png"))
        if pngs:
            return str(pngs[0])

    verify_dir = REPO_ROOT / "results" / "verify"
    if verify_dir.exists():
        pngs = list(verify_dir.glob("*.png"))
        if pngs:
            return str(pngs[0])

    return ""


def test_vlm_basic(device: str):
    """测试 2: VLM 基础推理"""
    print(f"\n[2] VLM 基础推理测试")
    from src.vlm_backend import VLMBackend

    image_path = find_test_image()
    if not image_path:
        print(f"  [SKIP] 未找到测试图像")
        print(f"  请先运行 Phase 1 生成图像: python scripts/run_demo.py")
        print(f"  或运行: python scripts/verify_robocasa.py")
        return None

    print(f"  测试图像: {image_path}")

    vlm = VLMBackend(device=device)
    response = vlm.describe(image_path, prompt="请简要描述图像中有哪些物体")
    print(f"  VLM 响应 ({len(response)} chars):")
    print(f"    {response[:300]}")
    print(f"  [OK] VLM 基础推理正常")
    return vlm


def test_scene_describer(vlm, device: str):
    """测试 3: SceneDescriber 真实 VLM 集成"""
    print(f"\n[3] SceneDescriber 真实 VLM 测试")
    from src.scene_describer import SceneDescriber

    image_path = find_test_image()
    if not image_path:
        print(f"  [SKIP] 未找到测试图像")
        return False

    if vlm is None:
        from src.vlm_backend import VLMBackend
        vlm = VLMBackend(device=device)

    describer = SceneDescriber(vlm_client=vlm)
    desc = describer.describe(image_path)

    print(f"  物体: {desc.objects}")
    print(f"  位置: {len(desc.positions)} 个")
    for p in desc.positions:
        print(f"    {p.obj}: {p.direction} {p.distance_cm}cm")
    print(f"  触觉: {desc.tactile[:3]}")
    print(f"  安全: {desc.safety_alerts}")
    print(f"  建议: {desc.actionable_advice[:2]}")

    speech = desc.to_speech()
    print(f"\n  语音文本 ({len(speech)} chars):")
    print(f"    {speech[:300]}")

    # 保存结果
    output_path = REPO_ROOT / "results" / "vlm_test_result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "image": image_path,
        "objects": desc.objects,
        "positions": [
            {"obj": p.obj, "direction": p.direction,
             "distance_cm": p.distance_cm, "confidence": p.confidence}
            for p in desc.positions
        ],
        "tactile": desc.tactile,
        "safety_alerts": desc.safety_alerts,
        "actionable_advice": desc.actionable_advice,
        "speech_text": speech,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  [OK] 结果保存到: {output_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: VLM 集成测试")
    parser.add_argument("--download-only", action="store_true",
                        help="仅下载模型，不执行推理测试")
    args = parser.parse_args()

    print(SEPARATOR)
    print("EmboSight Phase 3: Qwen2.5-VL 集成测试")
    print(SEPARATOR)

    device = check_gpu()

    if args.download_only:
        ok = download_model()
        if ok:
            print(f"\n[OK] 模型下载完成！接下来运行: python scripts/test_real_vlm.py")
        sys.exit(0 if ok else 1)

    results = {}

    # 测试 VLM 推理
    try:
        vlm = test_vlm_basic(device)
        results["vlm_basic"] = vlm is not None
    except Exception as e:
        print(f"  [FAIL] VLM 推理失败: {e}")
        import traceback
        traceback.print_exc()
        results["vlm_basic"] = False
        vlm = None

    # 测试 SceneDescriber
    try:
        results["scene_describer"] = test_scene_describer(vlm, device)
    except Exception as e:
        print(f"  [FAIL] SceneDescriber 失败: {e}")
        import traceback
        traceback.print_exc()
        results["scene_describer"] = False

    print(f"\n{SEPARATOR}")
    print("总结")
    print(SEPARATOR)
    for name, ok in results.items():
        status = "[OK]" if ok else ("[SKIP]" if ok is None else "[FAIL]")
        print(f"  {status} {name}")

    if all(v for v in results.values() if v is not None):
        print(f"\n[OK] Phase 3 通过！VLM 集成就绪。")
        print(f"下一步: python scripts/run_demo.py  (端到端全流程)")
    else:
        print(f"\n[WARN] 部分测试未通过，请检查错误信息。")
