"""EpisodeLogger 单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import pytest
import numpy as np


@pytest.fixture
def tmp_log_dir(tmp_path):
    return str(tmp_path / "episodes")


class TestEpisodeLogger:
    def test_start_and_end_writes_json(self, tmp_log_dir):
        from src.episode_logger import EpisodeLogger
        from src.world_belief import EpisodeResult
        lg = EpisodeLogger(log_dir=tmp_log_dir)
        lg.start_episode("拿苹果")
        result = EpisodeResult(
            success=True, target=None, speech="ok",
            belief_trace=[], action_history=[], n_steps=0, elapsed_seconds=1.0,
        )
        path = lg.end_episode(result)
        assert Path(path).exists()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["query"] == "拿苹果"
        assert data["final_result"]["success"] is True
    
    def test_log_snapshot_appends(self, tmp_log_dir):
        from src.episode_logger import EpisodeLogger
        from src.world_belief import BeliefSnapshot, EpisodeResult
        lg = EpisodeLogger(log_dir=tmp_log_dir)
        lg.start_episode("q")
        for i in range(3):
            snap = BeliefSnapshot(
                step=i, timestamp=time.time(), n_hypotheses=i,
                target_summary=None, most_uncertain_axis="label",
                overall_uncertainty=0.5, n_evidence=0, open_questions_count=0,
            )
            lg.log_snapshot(snap)
        path = lg.end_episode(EpisodeResult(
            success=True, target=None, speech="", belief_trace=[],
            action_history=[], n_steps=3, elapsed_seconds=1.0,
        ))
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert len(data["snapshots"]) == 3
        assert data["snapshots"][2]["step"] == 2
    
    def test_log_action_pair(self, tmp_log_dir):
        from src.episode_logger import EpisodeLogger
        from src.world_belief import (
            Action, BeliefSnapshot, EpisodeResult,
        )
        lg = EpisodeLogger(log_dir=tmp_log_dir)
        lg.start_episode("q")
        a = Action(kind="observe")
        snap = BeliefSnapshot(
            step=0, timestamp=time.time(), n_hypotheses=0,
            target_summary=None, most_uncertain_axis="label",
            overall_uncertainty=1.0, n_evidence=0, open_questions_count=0,
        )
        lg.log_action_start(a, snap)
        lg.log_action_end(a, snap)
        path = lg.end_episode(EpisodeResult(
            success=False, target=None, speech="", belief_trace=[],
            action_history=[a], n_steps=1, elapsed_seconds=1.0,
        ))
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert len(data["actions"]) == 1
        assert data["actions"][0]["kind"] == "observe"
    
    def test_load_round_trip(self, tmp_log_dir):
        from src.episode_logger import EpisodeLogger
        from src.world_belief import EpisodeResult
        lg = EpisodeLogger(log_dir=tmp_log_dir)
        lg.start_episode("query A")
        path = lg.end_episode(EpisodeResult(
            success=True, target=None, speech="hi", belief_trace=[],
            action_history=[], n_steps=0, elapsed_seconds=0.5,
        ))
        record = EpisodeLogger.load(path)
        assert record.query == "query A"
        assert record.final_result is not None
        assert record.final_result.success is True
    
    def test_user_qa_logged(self, tmp_log_dir):
        from src.episode_logger import EpisodeLogger
        from src.world_belief import EpisodeResult
        lg = EpisodeLogger(log_dir=tmp_log_dir)
        lg.start_episode("q")
        lg.log_user_qa("您要哪个?", "圆形的")
        path = lg.end_episode(EpisodeResult(
            success=True, target=None, speech="", belief_trace=[],
            action_history=[], n_steps=0, elapsed_seconds=0.0,
        ))
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["user_qa"] == [["您要哪个?", "圆形的"]]
    
    def test_evidence_serializes_numpy(self, tmp_log_dir):
        from src.episode_logger import EpisodeLogger
        from src.world_belief import Evidence, EpisodeResult
        lg = EpisodeLogger(log_dir=tmp_log_dir)
        lg.start_episode("q")
        ev = Evidence(
            source="vlm_ground", timestamp=1.0,
            raw_payload={"objects": [{"pos": np.array([0.1, 0.2, 0.3])}]},
        )
        lg.log_evidence(ev)
        path = lg.end_episode(EpisodeResult(
            success=True, target=None, speech="", belief_trace=[],
            action_history=[], n_steps=0, elapsed_seconds=0.0,
        ))
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # numpy 数组应被转成 list
        assert isinstance(
            data["evidence"][0]["raw_payload"]["objects"][0]["pos"], list
        )
