"""探测 Qwen2.5-VL 的 bbox / grounding 输出能力.

目的: SafeQuery-VLM 架构决定走 bbox 路径还是几何文本 fallback.
Qwen2.5-VL 官方支持 <box> token grounding, 但 7B 模型精度未知.

运行: MUJOCO_GL=egl python scripts/probe_vlm_bbox.py
输出: 对默认场景 agentview 图跑 3 种 prompt, 打印 VLM 返回内容 + 解析后 bbox.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env_wrapper import EnvWrapper, EnvConfig
from src.vlm_backend import VLMBackend


BBOX_PROMPTS = {
    "prompt_A_qwen_native": (
        "Please locate the peeler in this image. "
        "Output the bounding box in the format: "
        "<|box_start|>(x1,y1),(x2,y2)<|box_end|>"
    ),
    "prompt_B_natural": (
        "Look at this image. Find the peeler and give me its position "
        "as bounding box coordinates [x1, y1, x2, y2] where values are "
        "integers in pixels (image is 256x256). "
        "Reply ONLY with the JSON: {\"bbox\": [x1,y1,x2,y2]} or {\"bbox\": null}"
    ),
    "prompt_C_multi_object": (
        "List all task-relevant objects in this image. For each, give:\n"
        "- name (e.g. peeler, condiment_bottle, reamer)\n"
        "- bbox: [x1, y1, x2, y2] in pixels (0-255 range)\n"
        "- confidence: 0.0 to 1.0\n"
        "Reply in JSON: {\"objects\": [{...}, ...]}"
    ),
}


def main():
    env = EnvWrapper(EnvConfig())
    env.reset()

    # 采集默认场景的 agentview_center 图
    from src.active_planner import Viewpoint
    vp = Viewpoint(
        name="robot0_agentview_center",
        position=(0, 0, 60),
        orientation=(0, -45, 0),
        purpose="probe",
    )
    obs = env.observe(vp)
    img_path = obs.image_path
    print(f"\n=== Probe image: {img_path} ===")

    # 打印 episode 真实物体类型 (ground truth)
    type_map = env._get_obj_type_map()
    print(f"Ground truth object categories: {type_map}\n")

    vlm = VLMBackend()

    for name, prompt in BBOX_PROMPTS.items():
        print(f"\n--- {name} ---")
        print(f"Prompt: {prompt[:120]}...")
        try:
            raw = vlm.describe(img_path, prompt=prompt)
            print(f"Raw output ({len(raw)} chars):")
            print(raw[:800])
            bboxes = _parse_bboxes(raw)
            print(f"Parsed bboxes: {bboxes}")
        except Exception as e:
            print(f"ERROR: {e}")

    env.close()


def _parse_bboxes(raw: str) -> list:
    """尝试从 VLM 输出解析 bbox. 多种格式都试一遍."""
    results = []

    # Format 1: Qwen native <|box_start|>(x1,y1),(x2,y2)<|box_end|>
    m = re.findall(
        r"<\|box_start\|>\(?(\d+),\s*(\d+)\)?,\s*\(?(\d+),\s*(\d+)\)?<\|box_end\|>",
        raw,
    )
    for match in m:
        results.append({"format": "qwen_native", "bbox": [int(x) for x in match]})

    # Format 2: JSON {"bbox": [x1,y1,x2,y2]} or {"objects": [{...}]}
    try:
        text = raw
        if "```" in text:
            mm = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL)
            if mm:
                text = mm.group(1)
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            if "bbox" in data and data["bbox"]:
                results.append({"format": "json_single", "bbox": data["bbox"]})
            if "objects" in data:
                for o in data["objects"]:
                    if o.get("bbox"):
                        results.append({
                            "format": "json_multi",
                            "name": o.get("name"),
                            "bbox": o["bbox"],
                            "confidence": o.get("confidence"),
                        })
    except Exception:
        pass

    return results


if __name__ == "__main__":
    main()
