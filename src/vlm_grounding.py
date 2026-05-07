"""SafeQuery-VLM: 零样本 VLM 视觉定位模块

利用 Qwen2.5-VL 的 bbox 输出能力, 对场景进行开放式物体检测,
再通过后处理匹配用户查询目标.

核心设计:
    1. Prompt D 风格 (反幻觉): 不注入目标名, VLM 自由列出所见物体
    2. 结构化输出: label + bbox_2d + confidence + visible_features
    3. 后处理 query 匹配: 模糊文本匹配 + alias 表 + LLM 语义判定

Phase 1 探测结果 (2026-05-07):
    - Qwen2.5-VL-7B 输出 bbox 可靠 (Prompt D 无幻觉)
    - fovy=45°, fx=fy=309.02, image 256x256
    - 走 PATH A (bbox + depth 3D 投影)

使用示例:
    >>> from src.vlm_grounding import VLMGrounder
    >>> from src.vlm_backend import VLMBackend
    >>> grounder = VLMGrounder(VLMBackend())
    >>> candidates = grounder.ground("kitchen.png")
    >>> matches = grounder.match_query(candidates, "削皮器")
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class GroundedCandidate:
    """单视角 VLM 检测到的一个候选物体.

    由 VLMGrounder.ground() 产出, 每个候选代表图中一个物体.
    """
    label: str                          # VLM 给出的英文名 (e.g. "red apple", "white stick")
    confidence: float                   # VLM 自评置信度 0.0-1.0
    bbox_2d: tuple[int, int, int, int]  # (x1, y1, x2, y2) 像素坐标, 256x256
    visible_features: str = ""          # VLM 给出的视觉描述 (形状/颜色/材质)

    # 后处理填充 (match_query 阶段)
    query_match_score: float = 0.0      # 与用户 query 的匹配度 0.0-1.0
    matched_category: str = ""          # 匹配到的标准类别 (e.g. "peeler")
    match_method: str = ""              # 匹配方法 (alias/fuzzy/llm/none)

    @property
    def bbox_center(self) -> tuple[int, int]:
        """bbox 中心像素坐标."""
        x1, y1, x2, y2 = self.bbox_2d
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def bbox_area(self) -> int:
        """bbox 面积 (像素²)."""
        x1, y1, x2, y2 = self.bbox_2d
        return max(0, x2 - x1) * max(0, y2 - y1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox_2d": list(self.bbox_2d),
            "visible_features": self.visible_features,
            "query_match_score": self.query_match_score,
            "matched_category": self.matched_category,
            "match_method": self.match_method,
        }


# ============================================================
# VLM Grounding 核心类
# ============================================================

class VLMGrounder:
    """零样本 VLM 视觉定位器.

    给定一张图像, 让 VLM 列出所有可见物体及其 bbox, 然后后处理匹配用户 query.

    设计原则:
        - Prompt 不注入目标名 (防幻觉)
        - VLM 只做"看到什么" → 后处理做"是不是用户要的"
        - 支持 alias 表 + 模糊匹配 + LLM 语义判定三级匹配
    """

    def __init__(
        self,
        vlm_backend,
        prompt_path: str = "prompts/vlm_grounding.txt",
        aliases_path: str = "configs/object_aliases.yaml",
    ):
        self.vlm = vlm_backend
        self._prompt_template = self._load_prompt(prompt_path)
        self._aliases = self._load_aliases(aliases_path)
        # 构建反向索引: english_keyword → [zh_name1, zh_name2, ...]
        self._reverse_aliases = self._build_reverse_aliases()

    @staticmethod
    def _load_prompt(path: str) -> str:
        p = Path(path)
        if not p.exists():
            logger.warning(f"VLM grounding prompt not found: {p}")
            return ""
        return p.read_text(encoding="utf-8")

    @staticmethod
    def _load_aliases(path: str) -> dict[str, list[str]]:
        p = Path(path)
        if not p.exists():
            logger.warning(f"Aliases file not found: {p}")
            return {}
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        return data.get("aliases", {})

    def _build_reverse_aliases(self) -> dict[str, list[str]]:
        """english keyword → [中文名1, ...]"""
        rev: dict[str, list[str]] = {}
        for zh_name, en_list in self._aliases.items():
            for en in en_list:
                en_lower = en.lower()
                if en_lower not in rev:
                    rev[en_lower] = []
                if zh_name not in rev[en_lower]:
                    rev[en_lower].append(zh_name)
        return rev

    # ----------------------------------------------------------
    # 1. VLM Grounding: 图像 → 候选列表
    # ----------------------------------------------------------

    def ground(self, image_path: str) -> list[GroundedCandidate]:
        """对单张图像运行 VLM 开放式物体检测.

        不注入任何目标名称, VLM 自由列出所见.

        Args:
            image_path: RGB 图像路径

        Returns:
            GroundedCandidate 列表 (可能为空)
        """
        prompt = self._build_prompt()
        logger.info(f"[vlm_grounding] running on {image_path}")

        try:
            raw = self.vlm.describe(image_path, prompt=prompt)
        except Exception as e:
            logger.error(f"[vlm_grounding] VLM call failed: {e}")
            return []

        candidates = self._parse(raw)
        logger.info(f"[vlm_grounding] detected {len(candidates)} candidates")
        return candidates

    def _build_prompt(self) -> str:
        """构建 Prompt D 风格的开放式检测 prompt."""
        if self._prompt_template:
            return self._prompt_template

        # fallback: 内建 prompt (与 prompts/vlm_grounding.txt 内容一致)
        return (
            "Look at this kitchen image carefully. List ONLY the physical objects "
            "you can actually see on the countertop or table. Do NOT invent objects.\n\n"
            "For each object you see, provide:\n"
            "- name: a simple English noun describing what it looks like\n"
            "- bbox_2d: [x1, y1, x2, y2] in pixels (image is 256x256)\n"
            "- confidence: 0.0 to 1.0\n"
            "- visible_features: 1 sentence describing shape/color/material\n\n"
            "If you see NOTHING on the countertop, return {\"objects\": []}.\n"
            "Reply with ONLY a JSON object:\n"
            "{\"objects\": [{\"name\":..., \"bbox_2d\":..., \"confidence\":..., "
            "\"visible_features\":...}]}"
        )

    @staticmethod
    def _parse(raw: str) -> list[GroundedCandidate]:
        """解析 VLM JSON 输出为 GroundedCandidate 列表.

        支持格式:
            - {"objects": [{...}, ...]}
            - 顶层数组 [{...}, ...]
            - Markdown fenced JSON
        """
        # 清理 markdown fence
        text = raw.strip()
        fence_match = re.search(
            r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, re.DOTALL
        )
        if fence_match:
            text = fence_match.group(1)

        candidates: list[GroundedCandidate] = []

        try:
            # 尝试找到 JSON 起始
            arr_start = text.find("[")
            obj_start = text.find("{")

            data = None

            # 优先尝试 object 格式 {"objects": [...]}
            if obj_start >= 0:
                obj_end = text.rfind("}") + 1
                if obj_end > obj_start:
                    try:
                        data = json.loads(text[obj_start:obj_end])
                    except json.JSONDecodeError:
                        pass

            # 尝试数组格式 [{...}, ...]
            if data is None and arr_start >= 0:
                arr_end = text.rfind("]") + 1
                if arr_end > arr_start:
                    try:
                        data = json.loads(text[arr_start:arr_end])
                    except json.JSONDecodeError:
                        pass

            if data is None:
                logger.warning(f"[vlm_grounding] failed to parse JSON from VLM output: {text[:200]}")
                return []

            # 统一为列表
            obj_list: list[dict] = []
            if isinstance(data, dict) and "objects" in data:
                obj_list = data["objects"]
            elif isinstance(data, list):
                obj_list = data
            elif isinstance(data, dict):
                # 单个对象
                obj_list = [data]

            for item in obj_list:
                if not isinstance(item, dict):
                    continue
                # 读取 bbox (支持 bbox_2d 和 bbox 两种 key)
                bbox_raw = item.get("bbox_2d") or item.get("bbox")
                if not bbox_raw or not isinstance(bbox_raw, (list, tuple)):
                    continue
                if len(bbox_raw) != 4:
                    continue

                try:
                    bbox = tuple(int(v) for v in bbox_raw)
                except (ValueError, TypeError):
                    continue

                # 验证 bbox 合理性 (256x256 图像)
                x1, y1, x2, y2 = bbox
                if x2 <= x1 or y2 <= y1:
                    continue
                if x1 < 0 or y1 < 0 or x2 > 256 or y2 > 256:
                    # 容忍轻微越界, 裁剪到边界
                    bbox = (max(0, x1), max(0, y1), min(256, x2), min(256, y2))

                label = str(item.get("name") or item.get("label") or "unknown").strip()
                conf = float(item.get("confidence", 0.5))
                features = str(item.get("visible_features", ""))

                candidates.append(GroundedCandidate(
                    label=label,
                    confidence=max(0.0, min(1.0, conf)),
                    bbox_2d=bbox,
                    visible_features=features,
                ))

        except Exception as e:
            logger.error(f"[vlm_grounding] parse error: {e}")

        return candidates

    # ----------------------------------------------------------
    # 2. Query 匹配: 候选列表 + 用户查询 → 排序后候选
    # ----------------------------------------------------------

    def match_query(
        self,
        candidates: list[GroundedCandidate],
        user_query: str,
        gt_categories: Optional[dict[str, str]] = None,
    ) -> list[GroundedCandidate]:
        """将 VLM 检测的候选与用户查询匹配, 返回按匹配度降序排列的候选.

        三级匹配策略:
            1. alias 表精确匹配 (confidence 0.9)
            2. 模糊文本匹配 (confidence 0.5-0.8)
            3. ground truth cross-check (如果可用)

        Args:
            candidates: VLM 检测到的候选列表
            user_query: 用户原始查询 (e.g. "帮我拿削皮器")
            gt_categories: 环境真实物体类别 (e.g. {"obj_main": "peeler"})
                可选, 用于 cross-check VLM 标签准确性

        Returns:
            匹配后的候选列表 (按 query_match_score 降序), 原列表被 mutate
        """
        # 从用户查询提取目标关键词
        target_keywords = self._extract_target_keywords(user_query)
        logger.info(f"[vlm_grounding] query='{user_query}' → targets={target_keywords}")

        # gt 类别列表 (如果可用)
        gt_types = set()
        if gt_categories:
            gt_types = {v.lower() for v in gt_categories.values()}

        for c in candidates:
            score, method, category = self._score_candidate(
                c, target_keywords, gt_types
            )
            c.query_match_score = score
            c.match_method = method
            c.matched_category = category

        # 按 query_match_score 降序
        candidates.sort(key=lambda c: c.query_match_score, reverse=True)
        return candidates

    def _extract_target_keywords(self, query: str) -> list[str]:
        """从用户查询提取目标物体关键词.

        Returns:
            英文关键词列表 (e.g. ["peeler", "削皮器"])
        """
        keywords: list[str] = []

        # 1. 中文名 → 通过 alias 表查英文
        for zh_name, en_list in self._aliases.items():
            if zh_name in query:
                keywords.append(zh_name)
                keywords.extend(en_list)

        # 2. 英文关键词直接提取
        en_words = re.findall(r"[a-zA-Z_]+", query)
        keywords.extend(en_words)

        # 3. 去重并小写化
        seen = set()
        result = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                result.append(kw_lower)

        # 如果 alias 表没匹配到, 把原始查询的中文也放进去
        if not result:
            # 提取可能的物体名 (去掉常见动词前缀)
            cleaned = re.sub(r"^(帮我|请|帮忙|我要|给我|把|拿|取|找|递|送)", "", query)
            cleaned = cleaned.strip()
            if cleaned:
                result.append(cleaned)

        return result

    def _score_candidate(
        self,
        candidate: GroundedCandidate,
        target_keywords: list[str],
        gt_types: set[str],
    ) -> tuple[float, str, str]:
        """为单个候选评分.

        Returns:
            (score, method, matched_category)
        """
        label_lower = candidate.label.lower()
        features_lower = candidate.visible_features.lower()

        best_score = 0.0
        best_method = "none"
        best_category = ""

        for kw in target_keywords:
            kw_lower = kw.lower()

            # Level 1: 精确匹配 (label 包含关键词或关键词包含 label)
            if kw_lower in label_lower or label_lower in kw_lower:
                if 0.9 > best_score:
                    best_score = 0.9
                    best_method = "exact"
                    best_category = kw_lower

            # Level 2: alias 反向匹配 (label 的某部分是某个 alias 的英文 key)
            for word in label_lower.split():
                word_clean = re.sub(r"[^a-z_]", "", word)
                if word_clean in self._reverse_aliases:
                    # 这个 word 是某些中文名的英文 alias
                    zh_names = self._reverse_aliases[word_clean]
                    for zh in zh_names:
                        if zh.lower() in [k.lower() for k in target_keywords]:
                            if 0.85 > best_score:
                                best_score = 0.85
                                best_method = "alias_reverse"
                                best_category = word_clean

            # Level 3: features 文本匹配
            if kw_lower in features_lower:
                if 0.6 > best_score:
                    best_score = 0.6
                    best_method = "features"
                    best_category = kw_lower

            # Level 4: 语义近似 (peach ↔ apple, bottle ↔ jar)
            semantic_pairs = {
                ("apple", "peach"), ("apple", "fruit"),
                ("bottle", "jar"), ("bottle", "container"),
                ("cup", "mug"), ("cup", "glass"),
                ("bowl", "dish"),
                ("knife", "blade"), ("knife", "cutter"),
                ("pot", "pan"), ("pot", "saucepan"),
            }
            for w1, w2 in semantic_pairs:
                if (kw_lower == w1 and w2 in label_lower) or \
                   (kw_lower == w2 and w1 in label_lower):
                    if 0.7 > best_score:
                        best_score = 0.7
                        best_method = "semantic_pair"
                        best_category = kw_lower

        # Bonus: GT cross-check
        if gt_types and best_score > 0:
            # 如果 VLM label 的某部分出现在 GT 里, 提升置信度
            for gt in gt_types:
                if gt in label_lower or label_lower in gt:
                    best_score = min(1.0, best_score + 0.1)
                    best_method += "+gt"
                    break

        # Penalty: 标签是通用名 (如 "black microwave") 且与查询无关
        generic_labels = {"microwave", "robot_arm", "robot arm", "cabinet", "wall", "floor"}
        if any(g in label_lower for g in generic_labels) and best_score < 0.5:
            best_score = 0.0

        return best_score, best_method, best_category


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("[VLMGrounder] 模块加载测试")

    # 测试 parse
    raw_json = '{"objects": [{"name": "red apple", "bbox_2d": [235, 65, 256, 85], "confidence": 0.9, "visible_features": "round, red, shiny"}]}'
    candidates = VLMGrounder._parse(raw_json)
    print(f"Parsed {len(candidates)} candidates:")
    for c in candidates:
        print(f"  {c.label}: bbox={c.bbox_2d} conf={c.confidence}")

    # 测试 query 匹配 (无 VLM, 仅测匹配逻辑)
    grounder = VLMGrounder.__new__(VLMGrounder)
    grounder._aliases = {"苹果": ["apple", "fruit"], "削皮器": ["peeler"]}
    grounder._reverse_aliases = grounder._build_reverse_aliases()

    matched = grounder.match_query(candidates, "帮我拿苹果")
    print(f"\nQuery '帮我拿苹果' matches:")
    for c in matched:
        print(f"  {c.label}: score={c.query_match_score} method={c.match_method}")
