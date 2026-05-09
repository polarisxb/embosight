"""CLIPScorer: CLIP text-image cosine similarity for semantic object matching.

当 VLM 给出的标签 (e.g. "chocolate") 与用户目标 (e.g. "cake") 不匹配时，
用 CLIP 视觉-文本相似度判断图片 crop 是否为目标物体。

参考: OK-Robot (CoRL 2024), ConceptGraphs (ICRA 2024)

设计:
- lazy 加载 CLIP (首次调用时才初始化)
- 默认 CPU 推理 (GPU 被 VLM 占满)
- 单次推理 ~100ms (ViT-B/32, CPU)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class CLIPScorer:
    """CLIP text-image cosine similarity scorer."""

    INJECT_THRESHOLD = 0.23  # cosine sim 阈值; OK-Robot 用 0.25, 我们稍低以提高召回

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ):
        self._model_name = model_name
        self._device = device or "cpu"
        self._model = None
        self._processor = None
        self._load_failed = False

    # ──────────────────────────────────────
    # Lazy loading
    # ──────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_failed:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
            logger.info("[clip] 加载 CLIP: %s (device=%s)", self._model_name, self._device)
            self._processor = CLIPProcessor.from_pretrained(
                self._model_name, use_fast=False,
            )
            # 强制 safetensors 格式, 绕过 torch.load CVE-2025-32434 限制
            self._model = CLIPModel.from_pretrained(
                self._model_name, use_safetensors=True,
            )
            self._model = self._model.to(self._device).eval()
            logger.info("[clip] CLIP 加载完成")
        except Exception as e:
            logger.warning("[clip] CLIP 加载失败: %s", e)
            self._model = None
            self._load_failed = True

    @property
    def available(self) -> bool:
        """CLIP 是否可用 (加载成功)。"""
        if self._model is None and not self._load_failed:
            self._ensure_loaded()
        return self._model is not None

    # ──────────────────────────────────────
    # Core: score crops
    # ──────────────────────────────────────

    def score_crops(
        self,
        image_path: str,
        bboxes: list[tuple[int, int, int, int]],
        text_query: str,
    ) -> list[float]:
        """对每个 bbox crop 计算与 text_query 的 CLIP cosine similarity。

        Args:
            image_path: 原始观察图像路径
            bboxes: [(x1, y1, x2, y2), ...] 每个 hypothesis 的 bbox
            text_query: 用户目标 (e.g. "cake")

        Returns:
            cosine similarities, 与 bboxes 等长; 加载失败时全 0
        """
        if not bboxes:
            return []
        self._ensure_loaded()
        if self._model is None:
            return [0.0] * len(bboxes)

        try:
            from PIL import Image
            import torch

            img = Image.open(image_path).convert("RGB")
            crops = []
            for x1, y1, x2, y2 in bboxes:
                # 安全 clamp
                w, h = img.size
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    # 无效 bbox → 用整张图
                    crops.append(img)
                else:
                    crops.append(img.crop((x1, y1, x2, y2)))

            # Batch encode
            inputs = self._processor(
                text=[text_query],
                images=crops,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                # logits_per_text shape: (1, n_crops)
                # 转为 cosine similarity (CLIP logits = 100 * cosine_sim)
                sims = (outputs.logits_per_text[0] / 100.0).cpu().numpy().tolist()

            return sims

        except Exception as e:
            logger.warning("[clip] score_crops failed: %s", e)
            return [0.0] * len(bboxes)
