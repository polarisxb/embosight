"""UserChannel 单元测试 (FakeUser/CLI/Voice)。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from tests._mocks import MockLLM


class TestFakeUserChannel:
    def test_from_query_extracts_intent(self):
        from src.user_channel import FakeUserChannel
        llm = MockLLM(responses=["削皮器", "圆形的"])
        ch = FakeUserChannel.from_query(llm, "帮我拿那个削皮器")
        assert ch.intent == "削皮器"
        ans = ch.ask("您要哪个?")
        assert ans == "圆形的"
        # history 累积
        assert len(ch.history) == 1
    
    def test_from_explicit_passes_through(self):
        from src.user_channel import FakeUserChannel
        llm = MockLLM(responses=["a"])
        ch = FakeUserChannel.from_explicit(llm, "苹果")
        assert ch.intent == "苹果"
    
    def test_from_robocasa_reads_obj_main(self):
        from src.user_channel import FakeUserChannel
        llm = MockLLM(responses=[])
        class FakeEnv:
            def _get_obj_type_map(self):
                return {"obj_main": "peeler"}
        ch = FakeUserChannel.from_robocasa(llm, FakeEnv())
        assert "peeler" in ch.intent
    
    def test_history_format(self):
        from src.user_channel import FakeUserChannel
        llm = MockLLM(responses=["A1", "A2"])
        ch = FakeUserChannel.from_explicit(llm, "x")
        ch.ask("Q1")
        ch.ask("Q2")
        history = ch._format_history()
        assert "Q1" in history and "A1" in history
        assert "Q2" in history and "A2" in history
    
    def test_unhelpful_answer_handled(self):
        """用户答 "不知道" 时, channel 不报错, 返回原文。"""
        from src.user_channel import FakeUserChannel
        llm = MockLLM(responses=["不知道"])
        ch = FakeUserChannel.from_explicit(llm, "x")
        ans = ch.ask("您要哪个?")
        assert "不知道" in ans


class TestCLIUserChannel:
    def test_cli_reads_input(self, monkeypatch):
        from src.user_channel import CLIUserChannel
        ch = CLIUserChannel()
        monkeypatch.setattr("builtins.input", lambda *a, **k: "圆形的")
        ans = ch.ask("您要哪个?")
        assert ans == "圆形的"


class TestVoiceUserChannel:
    def test_voice_raises_not_implemented(self):
        from src.user_channel import VoiceUserChannel
        ch = VoiceUserChannel(tts=None, asr=None)
        with pytest.raises(NotImplementedError):
            ch.ask("anything")


class TestProtocol:
    def test_all_channels_have_ask(self):
        from src.user_channel import FakeUserChannel, CLIUserChannel
        # 接口一致性: 都有 ask(question) 方法
        assert hasattr(FakeUserChannel(MockLLM(responses=[]), "x"), "ask")
        assert hasattr(CLIUserChannel(), "ask")
