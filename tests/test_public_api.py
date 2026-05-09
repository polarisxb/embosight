"""验证 src 包级别 v1 公开 API 可用 (设计稿 §15 Appendix B)。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_public_api_imports():
    """所有声明的公开符号都能 from src import 到。"""
    import src
    expected = {
        "EmboSightAgent",
        "Action", "BeliefSnapshot", "Constraint", "DecomposedTask",
        "EpisodeResult", "Evidence", "GraspAttempt", "GraspCandidate",
        "Hypothesis", "Pose", "WorldBelief",
        "EpisodeLogger",
        "CLIUserChannel", "FakeUserChannel", "UserChannel", "VoiceUserChannel",
        "VLMCache",
    }
    actual = set(src.__all__)
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"src.__all__ 缺: {missing}"
    assert not extra, f"src.__all__ 多: {extra}"
    for name in expected:
        assert hasattr(src, name), f"src.{name} 未导出"
