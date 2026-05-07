"""SafetyGate 单元测试.

测试安全门控逻辑: 类别风险 + 置信度阈值 + VLM 文本风险检测.
不需要 GPU 或仿真环境.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.safety_gate import SafetyGate, SafetyDecision


# ============================================================
# Mock GroundedObject
# ============================================================

class MockObj:
    """模拟 GroundedObject, 仅包含 SafetyGate 需要的字段."""

    def __init__(
        self,
        matched_category: str = "",
        label: str = "unknown",
        query_match_score: float = 0.9,
        position_confidence: float = 0.85,
        per_view_features: dict = None,
    ):
        self.matched_category = matched_category
        self.label = label
        self.query_match_score = query_match_score
        self.position_confidence = position_confidence
        self.per_view_features = per_view_features or {}
        self.safety_risk = "unknown"
        self.safety_reason = ""


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def gate():
    return SafetyGate("configs/safety_rules.yaml")


# ============================================================
# Happy Path Tests
# ============================================================

class TestHappyPath:
    def test_safe_object_high_confidence_passes(self, gate):
        """安全物体 + 高置信度 → 允许执行."""
        obj = MockObj(matched_category="apple", query_match_score=0.9, position_confidence=0.85)
        dec = gate.check(obj)
        assert dec.allow_execute is True
        assert dec.risk_level == "safe"

    def test_fragile_object_high_confidence_passes(self, gate):
        """易碎物体 + 高置信度 → 允许执行 (带警告)."""
        obj = MockObj(matched_category="cup", query_match_score=0.9, position_confidence=0.85)
        dec = gate.check(obj)
        assert dec.allow_execute is True
        assert dec.risk_level == "fragile"
        assert "轻拿轻放" in dec.reason_user

    def test_sharp_object_very_high_confidence_passes(self, gate):
        """锋利物体 + 极高置信度 (>=0.9) → 允许执行 (带警告)."""
        obj = MockObj(matched_category="peeler", query_match_score=0.95, position_confidence=0.92)
        dec = gate.check(obj)
        assert dec.allow_execute is True
        assert dec.risk_level == "sharp"
        assert "刀片" in dec.reason_user or "锋利" in dec.reason_user


# ============================================================
# Rejection Tests
# ============================================================

class TestRejection:
    def test_reject_low_confidence_safe_object(self, gate):
        """安全物体 + 低置信度 (<0.75) → 拒绝."""
        obj = MockObj(matched_category="apple", query_match_score=0.5, position_confidence=0.4)
        dec = gate.check(obj)
        assert dec.allow_execute is False
        assert "手动确认" in dec.reason_user

    def test_reject_sharp_medium_confidence(self, gate):
        """锋利物体 + 中等置信度 (<0.9) → 拒绝 (高风险阈值)."""
        obj = MockObj(matched_category="peeler", query_match_score=0.85, position_confidence=0.80)
        dec = gate.check(obj)
        assert dec.allow_execute is False
        assert dec.risk_level == "sharp"

    def test_reject_hot_medium_confidence(self, gate):
        """高温物体 + 中等置信度 → 拒绝."""
        obj = MockObj(matched_category="pot", query_match_score=0.8, position_confidence=0.8)
        dec = gate.check(obj)
        assert dec.allow_execute is False
        assert dec.risk_level == "hot"

    def test_reject_high_risk_medium_confidence(self, gate):
        """高危物体 (刀) + 中等置信度 → 拒绝."""
        obj = MockObj(matched_category="knife", query_match_score=0.85, position_confidence=0.85)
        dec = gate.check(obj)
        assert dec.allow_execute is False
        assert dec.risk_level == "high"

    def test_high_risk_passes_with_very_high_confidence(self, gate):
        """高危物体 (刀) + 极高置信度 (>=0.9) → 允许 (当前无 never_execute)."""
        obj = MockObj(matched_category="knife", query_match_score=0.95, position_confidence=0.95)
        dec = gate.check(obj)
        assert dec.allow_execute is True
        assert dec.risk_level == "high"

    def test_position_confidence_affects_decision(self, gate):
        """综合置信度取 min(match, position), position 低则拒绝."""
        obj = MockObj(matched_category="apple", query_match_score=0.9, position_confidence=0.3)
        dec = gate.check(obj)
        assert dec.allow_execute is False
        assert dec.confidence == 0.3  # min(0.9, 0.3)


# ============================================================
# Default Category Tests
# ============================================================

class TestDefaultCategory:
    def test_unknown_category_uses_default(self, gate):
        """未知类别 → fallback 到 _default 规则."""
        obj = MockObj(matched_category="weird_alien_object", query_match_score=0.9, position_confidence=0.85)
        dec = gate.check(obj)
        assert dec.risk_level == "unknown"

    def test_empty_category_uses_label(self, gate):
        """matched_category 为空 → 使用 label."""
        obj = MockObj(matched_category="", label="apple", query_match_score=0.9, position_confidence=0.85)
        dec = gate.check(obj)
        assert dec.allow_execute is True
        assert dec.risk_level == "safe"


# ============================================================
# VLM Feature Risk Detection Tests
# ============================================================

class TestFeatureRiskDetection:
    def test_sharp_feature_detected(self, gate):
        """VLM 描述含 'sharp' → 额外风险警告."""
        obj = MockObj(
            matched_category="apple",
            query_match_score=0.9,
            position_confidence=0.85,
            per_view_features={"v1": "round red object with sharp metal edge"},
        )
        dec = gate.check(obj)
        assert len(dec.extra_warnings) > 0
        assert any("sharp" in w for w in dec.extra_warnings)

    def test_glass_feature_detected(self, gate):
        """VLM 描述含 'glass' → fragile 风险."""
        obj = MockObj(
            matched_category="bottle",
            query_match_score=0.9,
            position_confidence=0.85,
            per_view_features={"v1": "transparent glass bottle"},
        )
        dec = gate.check(obj)
        assert len(dec.extra_warnings) > 0
        assert any("fragile" in w for w in dec.extra_warnings)

    def test_no_features_no_warnings(self, gate):
        """无 features 文本 → 无额外警告."""
        obj = MockObj(matched_category="apple", query_match_score=0.9, position_confidence=0.85)
        dec = gate.check(obj)
        assert dec.extra_warnings == []

    def test_safe_unknown_object_with_sharp_feature_upgrades_risk(self, gate):
        """VLM 说 'sharp blade' 但类别 safe → 风险升级到 sharp."""
        obj = MockObj(
            matched_category="reamer",  # safe category
            query_match_score=0.95,
            position_confidence=0.92,
            per_view_features={"v1": "metal tool with sharp blade"},
        )
        dec = gate.check(obj)
        assert dec.risk_level == "sharp"
        # sharp → 需要 0.9 conf, 0.92>0.9 → 仍然 pass
        assert dec.allow_execute is True


# ============================================================
# update_object_safety Tests
# ============================================================

class TestUpdateObjectSafety:
    def test_update_sets_fields(self, gate):
        obj = MockObj(matched_category="peeler")
        gate.update_object_safety(obj)
        assert obj.safety_risk == "sharp"
        assert "刀片" in obj.safety_reason or "锋利" in obj.safety_reason

    def test_update_unknown_category(self, gate):
        obj = MockObj(matched_category="alien_fruit")
        gate.update_object_safety(obj)
        assert obj.safety_risk == "unknown"

    def test_update_uses_label_when_no_category(self, gate):
        obj = MockObj(matched_category="", label="pot")
        gate.update_object_safety(obj)
        assert obj.safety_risk == "hot"


# ============================================================
# SafetyDecision Tests
# ============================================================

class TestSafetyDecision:
    def test_default_extra_warnings(self):
        dec = SafetyDecision(
            allow_execute=True, risk_level="safe",
            confidence=0.9, reason_user="ok", reason_log="ok",
        )
        assert dec.extra_warnings == []

    def test_with_extra_warnings(self):
        dec = SafetyDecision(
            allow_execute=True, risk_level="safe",
            confidence=0.9, reason_user="ok", reason_log="ok",
            extra_warnings=["warning1"],
        )
        assert dec.extra_warnings == ["warning1"]


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_gate_with_missing_rules_file(self):
        """规则文件不存在时不崩溃, 使用默认值."""
        gate = SafetyGate("nonexistent/rules.yaml")
        obj = MockObj(matched_category="apple", query_match_score=0.9, position_confidence=0.85)
        dec = gate.check(obj)
        # 默认只有 _default 规则, apple 不在里面 → unknown
        assert dec.risk_level == "unknown"

    def test_zero_position_confidence(self, gate):
        """position_confidence=0 时只用 match_score."""
        obj = MockObj(matched_category="apple", query_match_score=0.9, position_confidence=0.0)
        dec = gate.check(obj)
        # conf = match_score (position=0 → 用 match_score)
        assert dec.allow_execute is True
