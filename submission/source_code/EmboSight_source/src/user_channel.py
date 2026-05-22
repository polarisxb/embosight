"""UserChannel: 与用户 (人/oracle/语音) 双向交互通道。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


# ============================================================
# Protocol
# ============================================================

class UserChannel(Protocol):
    def ask(self, question: str, context: Optional[dict] = None) -> str: ...


# ============================================================
# FakeUserChannel (LLM 扮演用户)
# ============================================================

_DEFAULT_SYSTEM_PATH = "prompts/user/fake_user_system.txt"


class FakeUserChannel:
    
    def __init__(self, llm, intent: str, system_path: str = _DEFAULT_SYSTEM_PATH):
        self.llm = llm
        self.intent = intent
        self.history: list[tuple[str, str]] = []
        self._system = self._load_system(system_path)
    
    @staticmethod
    def _load_system(path: str) -> str:
        p = Path(path)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return "你是一名视障用户, 简短回答。"
    
    @classmethod
    def from_query(cls, llm, query: str) -> "FakeUserChannel":
        prompt = f"用户说: {query!r}。请用 1 个词提取他真实想要的物体名 (中文): "
        intent = llm.generate(prompt).strip()
        return cls(llm, intent)
    
    @classmethod
    def from_explicit(cls, llm, intent: str) -> "FakeUserChannel":
        return cls(llm, intent)
    
    @classmethod
    def from_robocasa(cls, llm, env) -> "FakeUserChannel":
        type_map = env._get_obj_type_map()
        obj_type = type_map.get("obj_main", "unknown")
        return cls(llm, f"我想要那个 {obj_type}")
    
    def ask(self, question: str, context: Optional[dict] = None) -> str:
        prompt = (
            f"你的真实意图: {self.intent}\n\n"
            f"对话历史:\n{self._format_history()}\n\n"
            f"机器人问: {question}\n你的回答:"
        )
        ans = self.llm.generate(prompt, system=self._system).strip()
        self.history.append((question, ans))
        return ans
    
    def _format_history(self) -> str:
        if not self.history:
            return "(无)"
        return "\n".join(f"  Q: {q}\n  A: {a}" for q, a in self.history)


# ============================================================
# CLIUserChannel
# ============================================================

class CLIUserChannel:
    def ask(self, question: str, context: Optional[dict] = None) -> str:
        print(f"\n[Agent] {question}")
        return input("[You] ").strip()


# ============================================================
# VoiceUserChannel (占位)
# ============================================================

class VoiceUserChannel:
    """v1 留接口, 不实现。"""
    
    def __init__(self, tts, asr):
        self.tts = tts
        self.asr = asr
    
    def ask(self, question: str, context: Optional[dict] = None) -> str:
        raise NotImplementedError("VoiceUserChannel: 留 v2 接 Whisper/STT/TTS")
