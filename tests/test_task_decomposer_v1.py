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
