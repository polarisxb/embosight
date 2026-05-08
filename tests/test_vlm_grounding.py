"""VLMGrounder 单元测试.

测试 VLM 输出解析 + query 匹配逻辑. 不需要真 VLM (全部 mock).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.vlm_grounding import GroundedCandidate, VLMGrounder


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def grounder():
    """创建一个不需要真 VLM 的 VLMGrounder (仅测 parse + match)."""
    g = VLMGrounder.__new__(VLMGrounder)
    g.vlm = None
    g._llm = None
    g._prompt_template = ""
    g._aliases = {
        "苹果": ["apple", "fruit"],
        "削皮器": ["peeler"],
        "杯子": ["cup", "mug", "glass"],
        "药瓶": ["bottle", "medicine_bottle", "pill_bottle", "medicine"],
        "瓶子": ["bottle", "jar", "container"],
    }
    g._reverse_aliases = g._build_reverse_aliases()
    return g


@pytest.fixture
def sample_candidates():
    """模拟 VLM 检测到的候选列表 (来自真实 probe_vlm_bbox.py 输出)."""
    return [
        GroundedCandidate(
            label="red apple",
            confidence=0.9,
            bbox_2d=(235, 65, 256, 85),
            visible_features="round, red, shiny",
        ),
        GroundedCandidate(
            label="black microwave",
            confidence=0.8,
            bbox_2d=(134, 78, 192, 140),
            visible_features="rectangular, black",
        ),
    ]


# ============================================================
# GroundedCandidate Tests
# ============================================================

class TestGroundedCandidate:
    def test_bbox_center(self):
        c = GroundedCandidate("apple", 0.9, (10, 20, 50, 80), "round")
        assert c.bbox_center == (30, 50)

    def test_bbox_area(self):
        c = GroundedCandidate("apple", 0.9, (10, 20, 50, 80), "round")
        assert c.bbox_area == 40 * 60  # 2400

    def test_to_dict(self):
        c = GroundedCandidate("apple", 0.9, (10, 20, 50, 80), "round")
        d = c.to_dict()
        assert d["label"] == "apple"
        assert d["bbox_2d"] == [10, 20, 50, 80]
        assert d["confidence"] == 0.9


# ============================================================
# VLMGrounder._parse Tests
# ============================================================

class TestParse:
    def test_parse_objects_dict(self):
        """标准 {"objects": [...]} 格式."""
        raw = '{"objects": [{"name": "peeler", "bbox_2d": [108, 0, 194, 63], "confidence": 0.95, "visible_features": "white plastic tool"}]}'
        result = VLMGrounder._parse(raw)
        assert len(result) == 1
        assert result[0].label == "peeler"
        assert result[0].bbox_2d == (108, 0, 194, 63)
        assert result[0].confidence == 0.95
        assert result[0].visible_features == "white plastic tool"

    def test_parse_top_level_array(self):
        """顶层数组 [{...}] 格式 (Qwen Prompt A 风格)."""
        raw = '[{"bbox_2d": [130, 104, 195, 168], "label": "peeler"}]'
        result = VLMGrounder._parse(raw)
        assert len(result) == 1
        assert result[0].label == "peeler"
        assert result[0].bbox_2d == (130, 104, 195, 168)

    def test_parse_markdown_fenced(self):
        """Markdown 包裹的 JSON."""
        raw = '```json\n{"objects": [{"name": "cup", "bbox_2d": [10, 20, 30, 40], "confidence": 0.8}]}\n```'
        result = VLMGrounder._parse(raw)
        assert len(result) == 1
        assert result[0].label == "cup"

    def test_parse_bbox_key_alias(self):
        """支持 'bbox' 和 'bbox_2d' 两种 key."""
        raw = '{"objects": [{"name": "cup", "bbox": [10, 20, 30, 40], "confidence": 0.8}]}'
        result = VLMGrounder._parse(raw)
        assert len(result) == 1
        assert result[0].bbox_2d == (10, 20, 30, 40)

    def test_parse_empty_objects(self):
        """VLM 返回空列表 (场景中没有物体)."""
        raw = '{"objects": []}'
        result = VLMGrounder._parse(raw)
        assert len(result) == 0

    def test_parse_malformed_json_returns_empty(self):
        """畸形 JSON 不崩溃, 返回空列表."""
        raw = "I don't see any objects in this image."
        result = VLMGrounder._parse(raw)
        assert len(result) == 0

    def test_parse_invalid_bbox_skipped(self):
        """bbox 无效 (零尺寸) 的候选被跳过。

        注: [50, 50, 10, 10] 现在被解读为 xywh 格式 (Qwen3-VL 兼容),
        所以真正的非法 bbox 是零宽零高。
        """
        raw = '{"objects": [{"name": "bad", "bbox_2d": [50, 50, 0, 0], "confidence": 0.8}]}'
        result = VLMGrounder._parse(raw)
        assert len(result) == 0

    def test_parse_xywh_format(self):
        """Qwen3-VL 兼容: (x, y, w, h) 格式应被识别并转换为 (x1,y1,x2,y2)."""
        raw = '{"objects": [{"name": "obj", "bbox_2d": [50, 60, 30, 40], "confidence": 0.9}]}'
        result = VLMGrounder._parse(raw)
        assert len(result) == 1
        assert result[0].bbox_2d == (50, 60, 80, 100)

    def test_parse_normalized_1000_grid(self):
        """Qwen3-VL 兼容: 在 1000-grid normalized 空间的 bbox 应被缩放到图像像素空间."""
        # 1000-grid 中 (250, 250, 500, 500) → 256-grid 中应约为 (64, 64, 128, 128)
        raw = '{"objects": [{"name": "obj", "bbox_2d": [250, 250, 500, 500], "confidence": 0.9}]}'
        result = VLMGrounder._parse(raw, img_w=256, img_h=256)
        assert len(result) == 1
        x1, y1, x2, y2 = result[0].bbox_2d
        assert 60 <= x1 <= 68 and 60 <= y1 <= 68
        assert 124 <= x2 <= 132 and 124 <= y2 <= 132

    def test_parse_clips_bbox_to_boundary(self):
        """bbox 略微越界时裁剪到 [0,256]."""
        raw = '{"objects": [{"name": "edge", "bbox_2d": [-5, 10, 260, 200], "confidence": 0.7}]}'
        result = VLMGrounder._parse(raw)
        assert len(result) == 1
        assert result[0].bbox_2d == (0, 10, 256, 200)

    def test_parse_confidence_clamped(self):
        """confidence 超出 [0,1] 时 clamp."""
        raw = '{"objects": [{"name": "x", "bbox_2d": [1,1,10,10], "confidence": 1.5}]}'
        result = VLMGrounder._parse(raw)
        assert result[0].confidence == 1.0

    def test_parse_multi_objects(self):
        """多个物体同时解析."""
        raw = '{"objects": [{"name": "a", "bbox_2d": [1,1,10,10], "confidence": 0.9}, {"name": "b", "bbox_2d": [50,50,100,100], "confidence": 0.7}]}'
        result = VLMGrounder._parse(raw)
        assert len(result) == 2
        assert result[0].label == "a"
        assert result[1].label == "b"


# ============================================================
# Query 匹配 Tests
# ============================================================

class TestMatchQuery:
    def test_exact_match_chinese(self, grounder, sample_candidates):
        """中文查询 '苹果' 通过 alias 精确匹配 'apple'."""
        result = grounder.match_query(sample_candidates, "帮我拿苹果")
        assert result[0].label == "red apple"
        assert result[0].query_match_score >= 0.85
        assert result[0].matched_category in ("apple", "fruit", "苹果")

    def test_no_match_returns_zero_score(self, grounder, sample_candidates):
        """查询 '削皮器' 在只有 apple + microwave 的场景中, 无匹配."""
        result = grounder.match_query(sample_candidates, "帮我拿削皮器")
        # microwave 和 apple 都不该匹配 peeler
        for c in result:
            assert c.query_match_score < 0.5

    def test_microwave_filtered_as_generic(self, grounder, sample_candidates):
        """通用标签 (microwave) 不匹配任何查询, score=0."""
        result = grounder.match_query(sample_candidates, "帮我拿苹果")
        mw = [c for c in result if "microwave" in c.label.lower()]
        assert len(mw) == 1
        assert mw[0].query_match_score == 0.0

    def test_gt_crosscheck_boosts_score(self, grounder, sample_candidates):
        """GT cross-check: VLM 说 'red apple', GT 包含 'apple' → boost."""
        result = grounder.match_query(
            sample_candidates,
            "帮我拿苹果",
            gt_categories={"obj_main": "apple"},
        )
        assert result[0].label == "red apple"
        assert result[0].query_match_score >= 0.95  # 0.9 + 0.1 boost
        assert "+gt" in result[0].match_method

    def test_gt_miss_penalizes_hallucination(self, grounder):
        """GT cross-check: VLM 说 'brown cup' 但 GT={tangerine,cake} → 大幅降分."""
        candidates = [
            GroundedCandidate("brown cup", 0.8, (100,100,150,150), "brown ceramic"),
        ]
        result = grounder.match_query(
            candidates,
            "帮我拿杯子",
            gt_categories={"obj_main": "tangerine", "distr_counter_main": "cake"},
        )
        # 'cup' matches query but NOT in GT → score * 0.3
        assert result[0].query_match_score < 0.4
        assert "gt_miss" in result[0].match_method

    def test_sorted_by_score_descending(self, grounder):
        """结果按 query_match_score 降序."""
        candidates = [
            GroundedCandidate("bottle", 0.8, (10,10,50,50), "plastic"),
            GroundedCandidate("apple", 0.9, (60,60,100,100), "red"),
        ]
        result = grounder.match_query(candidates, "帮我拿苹果")
        assert result[0].label == "apple"
        assert result[0].query_match_score > result[1].query_match_score

    def test_english_query(self, grounder):
        """英文查询 'cup' 也能匹配."""
        candidates = [
            GroundedCandidate("white mug", 0.8, (10,10,50,50), "ceramic"),
        ]
        result = grounder.match_query(candidates, "cup")
        assert result[0].query_match_score >= 0.7

    def test_ambiguous_query_extracts_target(self, grounder):
        """模糊中文查询 '帮我拿那个瓶子' → 提取 '瓶子'."""
        candidates = [
            GroundedCandidate("plastic bottle", 0.8, (10,10,50,50), "cylindrical"),
        ]
        result = grounder.match_query(candidates, "帮我拿那个瓶子")
        assert result[0].query_match_score >= 0.8

    def test_extract_keywords_with_no_alias_match(self, grounder):
        """查询不在 alias 表中, fallback 到原始文本."""
        keywords = grounder._extract_target_keywords("帮我拿冰块")
        assert "冰块" in keywords or len(keywords) > 0


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_candidates_list(self, grounder):
        """空候选列表不崩溃."""
        result = grounder.match_query([], "帮我拿苹果")
        assert result == []

    def test_candidate_with_default_fields(self):
        """GroundedCandidate 默认字段值正确."""
        c = GroundedCandidate("x", 0.5, (0, 0, 10, 10))
        assert c.visible_features == ""
        assert c.query_match_score == 0.0
        assert c.matched_category == ""
        assert c.match_method == ""
