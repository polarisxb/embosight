"""跨测试文件复用的 mock。"""
from typing import Any


class MockLLM:
    """模拟 LLMBackend.generate。"""
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def generate(self, prompt: str, system: str = "", **kw) -> str:
        self.calls.append((prompt, kw))
        if not self._responses:
            raise RuntimeError("MockLLM out of responses")
        return self._responses.pop(0)


class MockVLM:
    """模拟 VLMBackend.describe。"""
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def describe(self, image_path: str, prompt: str = "") -> str:
        self.calls.append((image_path, prompt))
        if not self._responses:
            raise RuntimeError("MockVLM out of responses")
        return self._responses.pop(0)
