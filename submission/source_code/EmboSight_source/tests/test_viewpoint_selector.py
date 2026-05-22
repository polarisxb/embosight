"""ActiveViewpointSelector 单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests._mocks import MockLLM


class FakeViewpoint:
    def __init__(self, name):
        self.name = name


class FakeViewpointLib:
    def __init__(self, names):
        self.viewpoints = [FakeViewpoint(n) for n in names]

    def __len__(self):
        return len(self.viewpoints)

    def __getitem__(self, i):
        return self.viewpoints[i]

    def __iter__(self):
        return iter(self.viewpoints)


def _basic_belief():
    from src.world_belief import DecomposedTask, WorldBelief
    b = WorldBelief(user_query="拿苹果")
    b.decomposed = DecomposedTask(primary_target="apple")
    return b


class TestSelect:
    def test_returns_viewpoint_at_index(self):
        from src.active_planner import ActiveViewpointSelector
        llm = MockLLM(responses=["1"])
        vp_lib = FakeViewpointLib(["v0", "v1", "v2"])
        sel = ActiveViewpointSelector(llm=llm, viewpoint_lib=vp_lib)
        vp = sel.select(_basic_belief(), exclude=set(), preference="search_target")
        assert vp.name == "v1"

    def test_excludes_used(self):
        from src.active_planner import ActiveViewpointSelector
        llm = MockLLM(responses=["2"])
        vp_lib = FakeViewpointLib(["v0", "v1", "v2"])
        sel = ActiveViewpointSelector(llm=llm, viewpoint_lib=vp_lib)
        vp = sel.select(_basic_belief(), exclude={"v0", "v1"},
                        preference="search_target")
        assert vp.name == "v2"

    def test_minus_one_returns_none(self):
        from src.active_planner import ActiveViewpointSelector
        llm = MockLLM(responses=["-1"])
        vp_lib = FakeViewpointLib(["v0", "v1"])
        sel = ActiveViewpointSelector(llm=llm, viewpoint_lib=vp_lib)
        vp = sel.select(_basic_belief(), exclude=set(),
                        preference="search_target")
        assert vp is None

    def test_all_excluded_returns_none(self):
        from src.active_planner import ActiveViewpointSelector
        llm = MockLLM(responses=[])
        vp_lib = FakeViewpointLib(["v0"])
        sel = ActiveViewpointSelector(llm=llm, viewpoint_lib=vp_lib)
        vp = sel.select(_basic_belief(), exclude={"v0"}, preference="search_target")
        assert vp is None

    def test_invalid_index_returns_none(self):
        """LLM 输出 99 (越界) → None。"""
        from src.active_planner import ActiveViewpointSelector
        llm = MockLLM(responses=["99"])
        vp_lib = FakeViewpointLib(["v0", "v1"])
        sel = ActiveViewpointSelector(llm=llm, viewpoint_lib=vp_lib)
        vp = sel.select(_basic_belief(), exclude=set(), preference="search_target")
        assert vp is None
