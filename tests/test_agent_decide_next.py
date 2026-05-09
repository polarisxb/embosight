"""EmboSightAgent.decide_next 单元测试 (mock-driven, 8+ belief 状态)。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def _make_belief(target_word="apple", hyps=None, evidence=None,
                 action_history=None):
    from src.world_belief import DecomposedTask, WorldBelief
    b = WorldBelief(user_query=f"拿{target_word}")
    b.decomposed = DecomposedTask(primary_target=target_word)
    b.hypotheses = hyps or []
    b.evidence = evidence or []
    b.action_history = action_history or []
    return b


def _confident_target_hyp(label="apple"):
    from src.world_belief import GraspCandidate, Hypothesis
    c = GraspCandidate(point_3d=np.array([0.5, 0, 0.9]),
                       approach_dir=np.array([0, 0, -1]),
                       finger_width_m=0.04, score=0.9,
                       source="geometric_centroid")
    return Hypothesis(
        object_id="o0", label=label,
        label_alternatives=[(label, 0.95), ("other", 0.05)],
        label_entropy=0.10,
        position_3d=np.array([0.5, 0, 0.9]), position_std_m=0.02,
        safety_dist={"safe": 0.9, "fragile": 0.1}, safety_entropy=0.10,
        grasp_candidates=[c],
    )


class FakeVPLib:
    def __init__(self, names):
        self.viewpoints = [type("VP", (), {"name": n})() for n in names]

    def __len__(self):
        return len(self.viewpoints)

    def __getitem__(self, i):
        return self.viewpoints[i]

    def __iter__(self):
        return iter(self.viewpoints)


def _make_agent(viewpoints=None, nbv_responses=None):
    """构造一个 mock 化的 agent。"""
    from src.agent import EmboSightAgent
    vp_lib = FakeVPLib(viewpoints or ["v0", "v1", "v2"])
    return EmboSightAgent.with_test_doubles(
        vp_lib=vp_lib,
        nbv_llm=__import__("tests._mocks", fromlist=["MockLLM"]).MockLLM(
            nbv_responses or ["1", "2", "-1"],
        ),
    )


class TestDecideNext:
    def test_no_evidence_returns_observe(self):
        agent = _make_agent()
        belief = _make_belief()
        action = agent.decide_next(belief)
        assert action.kind == "observe"
        assert action.viewpoint.name == "v0"

    def test_no_target_returns_nbv_observe(self):
        from src.world_belief import Evidence, Hypothesis
        agent = _make_agent()
        belief = _make_belief(
            evidence=[Evidence(source="vlm_ground", timestamp=0, raw_payload={})],
            hyps=[
                Hypothesis(object_id="o", label="banana",
                           label_alternatives=[("banana", 0.9)],
                           label_entropy=0.1,
                           position_3d=np.zeros(3), position_std_m=0.05),
            ],
        )
        action = agent.decide_next(belief)
        assert action.kind in {"observe", "ask_user"}

    def test_no_target_no_more_views_asks_user(self):
        from src.world_belief import Action, Evidence
        agent = _make_agent(viewpoints=["v0"], nbv_responses=["-1"])
        belief = _make_belief(
            evidence=[Evidence(source="vlm_ground", timestamp=0, raw_payload={})],
            action_history=[Action(kind="observe",
                                   viewpoint=type("VP", (), {"name": "v0"})())],
        )
        action = agent.decide_next(belief)
        assert action.kind == "ask_user"

    def test_label_uncertain_zooms(self):
        from src.world_belief import Evidence, Hypothesis
        agent = _make_agent()
        h = Hypothesis(
            object_id="o0", label="apple",
            label_alternatives=[("apple", 0.35), ("kiwi", 0.35), ("pear", 0.30)],
            label_entropy=1.10,   # > 0.80 阈值, 触发 zoom 而非兜底
            position_3d=np.array([0.5, 0, 0.9]), position_std_m=0.02,
            safety_entropy=0.1,
        )
        belief = _make_belief(
            hyps=[h],
            evidence=[Evidence(source="vlm_ground", timestamp=0, raw_payload={})],
        )
        action = agent.decide_next(belief)
        assert action.kind == "re_observe"
        assert action.strategy == "zoom_in"

    def test_position_uncertain_parallax(self):
        from src.world_belief import Evidence, Hypothesis
        agent = _make_agent()
        h = Hypothesis(
            object_id="o0", label="apple",
            label_alternatives=[("apple", 0.95)],
            label_entropy=0.1,
            position_3d=np.array([0.5, 0, 0.9]), position_std_m=0.20,
            safety_entropy=0.1,
        )
        belief = _make_belief(
            hyps=[h],
            evidence=[Evidence(source="vlm_ground", timestamp=0, raw_payload={})],
        )
        action = agent.decide_next(belief)
        assert action.kind == "re_observe"
        assert action.strategy == "parallax_view"

    def test_safety_uncertain_classify(self):
        from src.world_belief import Evidence, Hypothesis
        agent = _make_agent()
        h = Hypothesis(
            object_id="o0", label="apple",
            label_alternatives=[("apple", 0.95)],
            label_entropy=0.1,
            position_3d=np.array([0.5, 0, 0.9]), position_std_m=0.02,
            safety_entropy=0.9,
        )
        belief = _make_belief(
            hyps=[h],
            evidence=[Evidence(source="vlm_ground", timestamp=0, raw_payload={})],
        )
        action = agent.decide_next(belief)
        assert action.kind == "classify_safety"

    def test_grasp_no_candidates_plans(self):
        from src.world_belief import Evidence, Hypothesis
        agent = _make_agent()
        h = Hypothesis(
            object_id="o0", label="apple",
            label_alternatives=[("apple", 0.95)],
            label_entropy=0.1,
            position_3d=np.array([0.5, 0, 0.9]), position_std_m=0.02,
            safety_entropy=0.1,
        )
        belief = _make_belief(
            hyps=[h],
            evidence=[Evidence(source="vlm_ground", timestamp=0, raw_payload={})],
        )
        action = agent.decide_next(belief)
        assert action.kind == "plan_grasp_candidates"

    def test_low_hazard_safety_uncertainty_without_candidates_plans(self):
        from src.world_belief import Evidence, Hypothesis
        agent = _make_agent()
        h = Hypothesis(
            object_id="o0", label="tupperware",
            label_alternatives=[("tupperware", 0.95)],
            label_entropy=0.1,
            position_3d=np.array([0.5, 0, 0.9]), position_std_m=0.02,
            safety_dist={
                "safe": 0.60,
                "fragile": 0.38,
                "sharp": 0.01,
                "hot": 0.01,
                "chemical": 0.0,
            },
            safety_entropy=0.52,
        )
        belief = _make_belief(
            target_word="tupperware",
            hyps=[h],
            evidence=[Evidence(source="vlm_ground", timestamp=0, raw_payload={})],
        )
        action = agent.decide_next(belief)
        assert action.kind == "plan_grasp_candidates"

    def test_all_confident_returns_grasp(self):
        from src.world_belief import Evidence
        agent = _make_agent()
        h = _confident_target_hyp()
        belief = _make_belief(
            hyps=[h],
            evidence=[Evidence(source="vlm_ground", timestamp=0, raw_payload={})],
        )
        action = agent.decide_next(belief)
        assert action.kind == "grasp"
        assert action.target_hypothesis is h

    def test_max_re_observe_asks_user(self):
        from src.world_belief import Evidence, Hypothesis
        agent = _make_agent()
        h = Hypothesis(
            object_id="o0", label="apple",
            label_alternatives=[("apple", 0.5), ("pear", 0.5)],
            label_entropy=0.69,
            position_3d=np.array([0.5, 0, 0.9]), position_std_m=0.02,
            safety_entropy=0.1,
        )
        h.times_re_observed = 3
        belief = _make_belief(
            hyps=[h],
            evidence=[Evidence(source="vlm_ground", timestamp=0, raw_payload={})],
        )
        action = agent.decide_next(belief)
        assert action.kind == "ask_user"
