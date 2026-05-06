"""LLM 行动决策模块 — 判断查询是否需要物理动作 (Step 5)"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ActionPlan:
    """行动决策结果"""

    action_type: str  # "grasp" | "point" | "none"
    target_object: str = ""
    reason: str = ""
    safety_constraints: list[str] = field(default_factory=list)
    require_confirmation: bool = True

    @property
    def needs_execution(self) -> bool:
        return self.action_type in ("grasp", "point")


class ActionDecider:
    """根据查询 + 场景描述，决定是否需要物理动作"""

    def __init__(
        self,
        llm_client,
        prompt_path: str = "prompts/action_decider.txt",
    ) -> None:
        self.llm = llm_client
        self.prompt_path = Path(prompt_path)
        self._system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt not found: {self.prompt_path}")
        return self.prompt_path.read_text(encoding="utf-8")

    def decide(self, query: str, description: Any) -> ActionPlan:
        """根据查询和场景描述决定行动

        Args:
            query: 用户查询
            description: StructuredDescription 或 dict

        Returns:
            ActionPlan
        """
        if hasattr(description, "to_dict"):
            desc_dict = description.to_dict()
        elif isinstance(description, dict):
            desc_dict = description
        else:
            desc_dict = {"raw": str(description)}

        user_msg = (
            f"查询: {query}\n"
            f"场景描述: {json.dumps(desc_dict, ensure_ascii=False)}"
        )

        try:
            response = self.llm.generate(
                user_message=user_msg,
                system=self._system_prompt,
                json_mode=True,
            )
            data = json.loads(response)
            plan = ActionPlan(
                action_type=data.get("action_type", "none"),
                target_object=data.get("target_object", ""),
                reason=data.get("reason", ""),
                safety_constraints=data.get("safety_constraints", []),
                require_confirmation=data.get("require_confirmation", True),
            )
            logger.info(
                f"[ActionDecider] {plan.action_type} "
                f"target='{plan.target_object}' reason='{plan.reason}'"
            )
            return plan
        except Exception as e:
            logger.warning(f"[ActionDecider] failed, fallback to none: {e}")
            return ActionPlan(
                action_type="none",
                reason=f"决策失败: {e}",
            )


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("[ActionDecider] 模块加载测试")

    # 无 API 时只测试 prompt 加载
    prompt_path = Path("prompts/action_decider.txt")
    if prompt_path.exists():
        content = prompt_path.read_text(encoding="utf-8")
        print(f"  prompt loaded: {len(content)} chars")
        print(f"  first line: {content.splitlines()[0]}")
    else:
        print(f"  WARNING: {prompt_path} not found")

    print("\n  要测试完整 LLM 调用，请用:")
    print("    from src.llm_backend import LLMBackend")
    print("    from src.action_decider import ActionDecider")
    print("    decider = ActionDecider(LLMBackend())")
    print('    plan = decider.decide("帮我拿药瓶", {"objects": ["药瓶"]})')
