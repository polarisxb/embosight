"""TaskDecomposer v1 (DecomposedTask) 单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

from tests._mocks import MockLLM


class TestDecomposeV1:
    def test_basic(self):
        from src.task_decomposer import TaskDecomposer
        llm = MockLLM(responses=[json.dumps({
            "primary_target": "apple",
            "constraints": [],
        })])
        td = TaskDecomposer(llm)
        dt = td.decompose_v1("帮我拿苹果")
        assert dt.primary_target == "apple"
        assert dt.constraints == []
        assert dt.raw_query == "帮我拿苹果"

    def test_avoid_constraint(self):
        from src.task_decomposer import TaskDecomposer
        llm = MockLLM(responses=[json.dumps({
            "primary_target": "bowl",
            "constraints": [
                {"kind": "avoid", "target_label": "knife", "reason": "用户避开"},
            ],
        })])
        td = TaskDecomposer(llm)
        dt = td.decompose_v1("拿碗, 避开刀")
        assert dt.primary_target == "bowl"
        assert len(dt.constraints) == 1
        assert dt.constraints[0].kind == "avoid"
        assert dt.constraints[0].target_label == "knife"

    def test_user_hint_constraint(self):
        from src.task_decomposer import TaskDecomposer
        llm = MockLLM(responses=[json.dumps({
            "primary_target": "bottle",
            "constraints": [
                {"kind": "user_hint", "text": "水池左边", "reason": "位置提示"},
            ],
        })])
        td = TaskDecomposer(llm)
        dt = td.decompose_v1("拿水池左边的瓶子")
        assert any(c.kind == "user_hint" and "水池" in (c.text or "")
                   for c in dt.constraints)

    def test_malformed_falls_back_to_primary_only(self):
        from src.task_decomposer import TaskDecomposer
        llm = MockLLM(responses=["not json"])
        td = TaskDecomposer(llm)
        dt = td.decompose_v1("帮我拿苹果")
        assert dt.primary_target  # 非空
        assert dt.constraints == []

    def test_unknown_constraint_kind_skipped(self):
        from src.task_decomposer import TaskDecomposer
        llm = MockLLM(responses=[json.dumps({
            "primary_target": "x",
            "constraints": [{"kind": "weird_kind", "reason": "?"}],
        })])
        td = TaskDecomposer(llm)
        dt = td.decompose_v1("x")
        assert dt.constraints == []


class TestSynonymParsing:
    def test_synonyms_parsed(self):
        from src.task_decomposer import TaskDecomposer
        llm = MockLLM(responses=[json.dumps({
            "primary_target": "tangerine",
            "primary_target_synonyms": ["mandarin", "orange", "citrus"],
            "constraints": [],
        })])
        td = TaskDecomposer(llm)
        dt = td.decompose_v1("拿橘子")
        assert dt.primary_target == "tangerine"
        assert "mandarin" in dt.primary_target_synonyms
        assert "orange" in dt.primary_target_synonyms
        assert "citrus" in dt.primary_target_synonyms

    def test_synonyms_blacklist_filtered(self):
        from src.task_decomposer import TaskDecomposer
        llm = MockLLM(responses=[json.dumps({
            "primary_target": "apple",
            "primary_target_synonyms": ["fruit", "object", "thing", "red apple"],
            "constraints": [],
        })])
        td = TaskDecomposer(llm)
        dt = td.decompose_v1("拿苹果")
        # blacklist 词被过滤
        assert "object" not in dt.primary_target_synonyms
        assert "thing" not in dt.primary_target_synonyms
        # 正常词保留
        assert "fruit" in dt.primary_target_synonyms
        assert "red apple" in dt.primary_target_synonyms

    def test_synonyms_dedup_with_primary(self):
        from src.task_decomposer import TaskDecomposer
        llm = MockLLM(responses=[json.dumps({
            "primary_target": "apple",
            "primary_target_synonyms": ["apple", "Apple", "fruit"],
            "constraints": [],
        })])
        td = TaskDecomposer(llm)
        dt = td.decompose_v1("拿苹果")
        # 与 primary 同名的同义词被去重
        assert dt.primary_target_synonyms.count("apple") == 0
        assert "fruit" in dt.primary_target_synonyms

    def test_synonyms_capped_at_5(self):
        from src.task_decomposer import TaskDecomposer
        llm = MockLLM(responses=[json.dumps({
            "primary_target": "x",
            "primary_target_synonyms": [f"syn{i}" for i in range(10)],
            "constraints": [],
        })])
        td = TaskDecomposer(llm)
        dt = td.decompose_v1("x")
        assert len(dt.primary_target_synonyms) <= 5

    def test_synonyms_missing_returns_empty(self):
        from src.task_decomposer import TaskDecomposer
        llm = MockLLM(responses=[json.dumps({
            "primary_target": "apple",
            "constraints": [],
        })])
        td = TaskDecomposer(llm)
        dt = td.decompose_v1("拿苹果")
        assert dt.primary_target_synonyms == []


class TestTargetMatchSynonyms:
    """WorldBelief.target() 使用 synonyms 匹配 hypothesis。"""

    def test_target_via_synonym_label(self):
        import numpy as np

        from src.world_belief import (
            DecomposedTask, Hypothesis, WorldBelief,
        )
        belief = WorldBelief(user_query="拿橘子")
        belief.decomposed = DecomposedTask(
            primary_target="tangerine",
            primary_target_synonyms=["mandarin", "orange"],
        )
        # hypothesis label 是 "orange" — primary 不匹配, 但 synonym 匹配
        h = Hypothesis(
            object_id="o1", label="orange",
            label_alternatives=[("orange", 0.85), ("fruit", 0.15)],
            label_entropy=0.5,
            position_3d=np.array([0.5, 0, 0.9], dtype=np.float32),
            position_std_m=0.02,
        )
        belief.hypotheses = [h]
        assert belief.target() is h

    def test_target_primary_beats_synonym(self):
        """Phase 1 找到 primary → 直接返回, synonym 不参与 (方案 B)。"""
        import numpy as np

        from src.world_belief import (
            DecomposedTask, Hypothesis, WorldBelief,
        )
        belief = WorldBelief(user_query="拿橘子")
        belief.decomposed = DecomposedTask(
            primary_target="tangerine",
            primary_target_synonyms=["orange"],
        )
        h_synonym = Hypothesis(
            object_id="o1", label="orange",
            label_alternatives=[("orange", 0.85)],
            label_entropy=0.5,
            position_3d=np.array([0.5, 0, 0.9], dtype=np.float32),
            position_std_m=0.02,
        )
        h_primary = Hypothesis(
            object_id="o2", label="tangerine",
            label_alternatives=[("tangerine", 0.85)],
            label_entropy=0.5,
            position_3d=np.array([0.6, 0, 0.9], dtype=np.float32),
            position_std_m=0.02,
        )
        belief.hypotheses = [h_synonym, h_primary]
        # Phase 1 (primary only) 找到 h_primary → 直接返回, 不受 h_synonym 干扰
        result = belief.target()
        assert result is h_primary

    def test_synonym_no_ambiguity_when_primary_matches(self):
        """有 synonym 匹配的 hypothesis 存在, 但 primary 也能找到 → 无回归。"""
        import numpy as np

        from src.world_belief import (
            DecomposedTask, Hypothesis, WorldBelief,
        )
        belief = WorldBelief(user_query="拿酸奶")
        belief.decomposed = DecomposedTask(
            primary_target="yogurt",
            primary_target_synonyms=["dairy cup", "fermented milk"],
        )
        # 场景: VLM 正确识别了 yogurt, 同时场景中有一个 cup
        h_yogurt = Hypothesis(
            object_id="o1", label="yogurt",
            label_alternatives=[("yogurt", 0.75), ("container", 0.25)],
            label_entropy=0.8,
            position_3d=np.array([0.5, 0, 0.9], dtype=np.float32),
            position_std_m=0.02,
        )
        h_cup = Hypothesis(
            object_id="o2", label="cup",
            label_alternatives=[("cup", 0.90), ("mug", 0.10)],
            label_entropy=0.3,
            position_3d=np.array([0.6, 0, 0.9], dtype=np.float32),
            position_std_m=0.02,
        )
        belief.hypotheses = [h_yogurt, h_cup]
        # Phase 1 匹配 primary "yogurt" → 只有 h_yogurt, 无歧义
        result = belief.target()
        assert result is h_yogurt
