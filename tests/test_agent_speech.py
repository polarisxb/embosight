import numpy as np


def test_success_speech_uses_post_lift_object_position_not_belief_estimate():
    from src.agent import EmboSightAgent
    from src.world_belief import (
        DecomposedTask,
        GraspAttempt,
        GraspCandidate,
        Hypothesis,
        WorldBelief,
    )

    candidate = GraspCandidate(
        point_3d=np.array([0.188, -2.836, 0.947], dtype=np.float32),
        approach_dir=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        finger_width_m=0.04,
        score=1.0,
    )
    attempt = GraspAttempt(
        timestamp=0.0,
        candidate=candidate,
        failure_mode="success",
        end_effector_pose_reached=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        diagnostic={
            "post_lift_obj_pos": [0.1883076578, -2.8358876705, 1.0359208121],
        },
    )
    hyp = Hypothesis(
        object_id="distr_counter_main",
        label="lemon",
        label_alternatives=[("lemon", 0.95)],
        label_entropy=0.1,
        position_3d=np.array([9.0, 9.0, 9.0], dtype=np.float32),
        position_std_m=0.02,
        grasp_candidates=[candidate],
        grasp_attempts=[attempt],
    )
    belief = WorldBelief(
        user_query="pick up the lemon",
        decomposed=DecomposedTask(primary_target="lemon"),
        hypotheses=[hyp],
    )

    speech = EmboSightAgent._build_speech(belief, success=True)

    assert "当前物体世界坐标" in speech
    assert "x=0.188m" in speech
    assert "y=-2.836m" in speech
    assert "z=1.036m" in speech
    assert "正前方" not in speech
    assert "9.00m" not in speech


def test_grasp_memory_payload_separates_selected_strategy_from_executed_shape():
    from src.agent import EmboSightAgent
    from src.world_belief import (
        GraspAttempt,
        GraspCandidate,
        GraspStrategy,
        Hypothesis,
    )

    candidate = GraspCandidate(
        point_3d=np.array([0.188, -2.836, 0.947], dtype=np.float32),
        approach_dir=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        finger_width_m=0.04,
        score=1.0,
        source="vlm_top_grasp",
    )
    attempt = GraspAttempt(
        timestamp=0.0,
        candidate=candidate,
        failure_mode="success",
        end_effector_pose_reached=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    hyp = Hypothesis(
        object_id="distr_counter_main",
        label="lemon",
        label_alternatives=[("lemon", 0.95)],
        label_entropy=0.1,
        position_3d=np.array([0.188, -2.836, 0.947], dtype=np.float32),
        position_std_m=0.02,
        grasp_strategy=GraspStrategy(strategy="gentle_side", depth_margin_m=0.01),
        grasp_candidates=[candidate],
        grasp_attempts=[attempt],
    )

    context, lesson = EmboSightAgent._grasp_memory_payload(hyp, attempt)

    assert context["selected_strategy"] == "gentle_side"
    assert context["candidate_source"] == "vlm_top_grasp"
    assert context["strategy"] == "top_down"
    assert context["executed_strategy"] == "top_down"
    assert context["depth_margin_m"] == 0.01
    assert "selected gentle_side" in lesson
    assert "executed top_down" in lesson
