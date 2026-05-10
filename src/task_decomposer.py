"""TaskDecomposer v1: query → DecomposedTask (primary_target + constraints)。

老 SubtaskType / BlindDimension / Subtask / BlindTaskTemplate / decompose() 已删除 (Phase 15)。
v1 不再使用模板 IDF 检索 / 五维度补全 / Subtask 流; 直接 LLM JSON 解析。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from src.world_belief import Constraint, DecomposedTask

logger = logging.getLogger(__name__)


class TaskDecomposer:
    """v1: decompose_v1(query) → DecomposedTask。"""

    def __init__(
        self,
        llm_client,
        prompt_path: str = "prompts/agent/decompose.txt",
    ) -> None:
        self.llm = llm_client
        self.prompt_path = Path(prompt_path)
        if self.prompt_path.exists():
            self._template = self.prompt_path.read_text(encoding="utf-8")
        else:
            logger.warning(f"Prompt 不存在: {self.prompt_path}, 用 fallback")
            self._template = None

    def decompose_v1(
        self, query: str, prompt_path: str | None = None,
    ) -> DecomposedTask:
        """主入口: query → DecomposedTask。

        失败 fallback: 把整句 query 当 primary_target。
        """
        if prompt_path is not None:
            p = Path(prompt_path)
            template = p.read_text(encoding="utf-8") if p.exists() else None
        else:
            template = self._template

        if template is not None:
            prompt = template.replace("{query}", query)
        else:
            prompt = (
                f"Query: {query}\n"
                f"Output JSON with primary_target and constraints[]."
            )

        try:
            raw = self.llm.generate(prompt, system="")
        except Exception:
            return DecomposedTask(primary_target=query.strip(), raw_query=query)

        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return DecomposedTask(primary_target=query.strip(), raw_query=query)
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return DecomposedTask(primary_target=query.strip(), raw_query=query)

        primary = str(data.get("primary_target", query.strip()))
        constraints: list[Constraint] = []
        for c in data.get("constraints", []):
            kind = c.get("kind")
            if kind not in {"avoid", "prefer_view", "max_force", "user_hint"}:
                continue
            constraints.append(Constraint(
                kind=kind,
                target_label=c.get("target_label"),
                text=c.get("text"),
                reason=c.get("reason", ""),
            ))
        # 解析同义词 (过滤过宽的词 + 过短/过长的词)
        synonyms = self._parse_synonyms(
            data.get("primary_target_synonyms", []), primary,
        )
        return DecomposedTask(
            primary_target=primary,
            constraints=constraints,
            raw_query=query,
            primary_target_synonyms=synonyms,
        )

    # 过宽的同义词黑名单 (匹配会污染所有 hypothesis)
    _SYNONYM_BLACKLIST = {
        "object", "thing", "item", "stuff", "something", "anything",
        "物体", "东西",
    }

    @staticmethod
    def _parse_synonyms(raw_list, primary: str) -> list[str]:
        """过滤+清洗同义词列表。"""
        if not isinstance(raw_list, list):
            return []
        primary_lower = primary.strip().lower()
        out: list[str] = []
        seen: set[str] = {primary_lower}
        for s in raw_list[:5]:  # 最多 5 个
            if not isinstance(s, str):
                continue
            cleaned = s.strip().lower()
            if not cleaned or len(cleaned) < 2 or len(cleaned) > 30:
                continue
            if cleaned in seen:
                continue
            if cleaned in TaskDecomposer._SYNONYM_BLACKLIST:
                logger.info("[decompose] dropping over-broad synonym '%s'", cleaned)
                continue
            out.append(cleaned)
            seen.add(cleaned)
        return out
