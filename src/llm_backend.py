"""LLM 后端封装（DeepSeek API）

兼容 OpenAI 接口标准，可随时切换到其他 OpenAI 兼容 API。

环境变量:
    DEEPSEEK_API_KEY  必需
    DEEPSEEK_BASE_URL 默认 https://api.deepseek.com/v1
    DEEPSEEK_MODEL    默认 deepseek-chat

使用示例:
    >>> from src.llm_backend import LLMBackend
    >>> llm = LLMBackend()
    >>> response = llm.generate("你好", system="你是一个助手")
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class LLMBackend:
    """LLM 客户端（基于 OpenAI SDK 调用 DeepSeek API）"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        timeout: float = 60.0,
    ) -> None:
        """
        Args:
            api_key: API key（默认从环境变量 DEEPSEEK_API_KEY 读取）
            base_url: API 基础 URL（默认 https://api.deepseek.com/v1）
            model: 模型名（默认 deepseek-chat）
            max_tokens: 最大输出 tokens
            temperature: 采样温度（任务分解建议 0.0-0.3）
            timeout: 请求超时（秒）
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "请先安装 openai: pip install openai>=1.30.0"
            ) from e

        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "未设置 DEEPSEEK_API_KEY 环境变量，请在 .env 中配置"
            )

        self.base_url = base_url or os.environ.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
        )
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

        logger.info(f"LLMBackend 初始化: model={self.model}, base_url={self.base_url}")

    def generate(
        self,
        user_message: str,
        system: Optional[str] = None,
        json_mode: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """生成文本响应

        Args:
            user_message: 用户输入
            system: 系统 Prompt（可选）
            json_mode: 是否强制 JSON 输出
            temperature: 采样温度（覆盖默认值）

        Returns:
            生成的文本
        """
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})

        kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            logger.debug(f"LLM 响应长度: {len(content)} chars")
            return content
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("[LLMBackend] 测试调用 DeepSeek API")
    print("（需在 .env 中设置 DEEPSEEK_API_KEY）")

    try:
        llm = LLMBackend()
        response = llm.generate(
            user_message="将查询'帮我找药瓶'分解为 JSON 子任务列表，键名 subtasks",
            system="你是一个视障辅助任务分解助手，输出严格 JSON。",
            json_mode=True,
        )
        print("\n[响应]")
        print(response)
    except ValueError as e:
        print(f"\n配置错误: {e}")
    except Exception as e:
        print(f"\n调用失败: {e}")