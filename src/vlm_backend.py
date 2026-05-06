"""VLM 后端封装（Qwen2.5-VL 本地部署）

加载 Qwen2.5-VL-7B 模型在 GPU 上推理。
延迟加载策略：仅在第一次调用 describe() 时加载模型，避免 import 副作用。

使用示例:
    >>> from src.vlm_backend import VLMBackend
    >>> vlm = VLMBackend()
    >>> desc = vlm.describe("kitchen.png", prompt="请描述图像内容")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


class VLMBackend:
    """Qwen2.5-VL 视觉-语言模型客户端"""

    def __init__(
        self,
        model_id: str = "./checkpoints/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 1024,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_id: HuggingFace 模型 ID
            device: cuda / cpu
            torch_dtype: bfloat16 / float16 / float32
            max_new_tokens: 最大生成 tokens
            cache_dir: 模型缓存目录
        """
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.cache_dir = cache_dir
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        """延迟加载模型（避免 import 时占用资源）"""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoProcessor
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as VLModelClass
            except ImportError:
                from transformers import Qwen2VLForConditionalGeneration as VLModelClass
        except ImportError as e:
            raise ImportError(
                f"请先安装依赖: pip install torch transformers, 错误: {e}"
            ) from e

        logger.info(f"正在加载 Qwen2.5-VL: {self.model_id}")

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.torch_dtype, torch.bfloat16)

        load_kwargs = {}
        if self.cache_dir:
            load_kwargs["cache_dir"] = self.cache_dir

        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            **load_kwargs,
        )
        device_map = "auto" if self.device.startswith("cuda") else "cpu"
        self._model = VLModelClass.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=device_map,
            **load_kwargs,
        )
        self._model.eval()

        logger.info(f"Qwen2.5-VL 加载完成 (dtype={self.torch_dtype}, device={self.device})")

    def describe(
        self,
        image: Union[str, Path],
        prompt: str = "请详细描述图像内容",
    ) -> str:
        """对单张图像生成描述

        Args:
            image: 图像路径
            prompt: 用户 prompt（含视障专属指令）

        Returns:
            VLM 输出文本

        TODO:
            完整推理逻辑参考 Qwen2.5-VL 官方示例:
            https://github.com/QwenLM/Qwen2.5-VL
        """
        self._ensure_loaded()

        try:
            import torch
            from qwen_vl_utils import process_vision_info
        except ImportError as e:
            raise ImportError(
                "请先安装 qwen-vl-utils: pip install qwen-vl-utils"
            ) from e

        image_path = str(image)
        if not Path(image_path).exists():
            raise FileNotFoundError(f"图像不存在: {image_path}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        logger.debug(f"VLM 输出长度: {len(output_text)} chars")
        return output_text


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("[VLMBackend] 模块加载测试（不实际加载模型）")
    vlm = VLMBackend()
    print(f"  model_id: {vlm.model_id}")
    print(f"  device: {vlm.device}")
    print(f"  cache_dir: {vlm.cache_dir}")
    print("  注: 首次调用 describe() 时会触发模型加载")