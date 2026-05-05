"""创新点①: 零样本视障任务分解器

将视障者的自然语言查询分解为结构化的子任务序列。
基于 Few-shot Prompt + 视障专属任务模板库。

主要设计:
    1. 视障关键维度强制编码（方位/距离/触觉/安全/行动）
    2. Few-shot 检索模板示例作为上下文
    3. 严格 JSON 输出 + 验证

使用示例:
    >>> from src.task_decomposer import TaskDecomposer
    >>> from src.llm_backend import LLMBackend
    >>> llm = LLMBackend()
    >>> decomposer = TaskDecomposer(llm)
    >>> subtasks = decomposer.decompose("我的药瓶在哪？")
    >>> for t in subtasks:
    ...     print(t)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ============================================================
# 枚举类型
# ============================================================

class SubtaskType(str, Enum):
    """子任务类型"""

    IDENTIFY = "identify"     # 识别物体存在
    LOCATE = "locate"         # 定位物体精确方位
    DESCRIBE = "describe"     # 描述场景/物体
    ALERT = "alert"           # 安全提示
    GUIDE = "guide"           # 行动指引


class BlindDimension(str, Enum):
    """视障关键维度"""

    POSITION = "position"     # 方位
    DISTANCE = "distance"     # 距离
    TACTILE = "tactile"       # 触觉特征
    SAFETY = "safety"         # 安全提示
    ACTION = "action"         # 行动建议


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Subtask:
    """单个子任务

    Attributes:
        type: 子任务类型
        target: 目标物体或区域
        priority: 优先级（1 最高，5 最低）
        blind_dimension: 关注的视障关键维度
        output_format: 期望的输出格式描述
        coverage_status: 是否已被某个观察覆盖
    """

    type: SubtaskType
    target: str
    priority: int = 3
    blind_dimension: BlindDimension = BlindDimension.POSITION
    output_format: str = ""
    coverage_status: bool = False

    def to_dict(self) -> dict[str, Any]:
        """转换为字典（用于 JSON 序列化与日志）"""
        return {
            "type": self.type.value,
            "target": self.target,
            "priority": self.priority,
            "blind_dimension": self.blind_dimension.value,
            "output_format": self.output_format,
            "coverage_status": self.coverage_status,
        }

    def __repr__(self) -> str:
        return (
            f"Subtask(type={self.type.value}, target='{self.target}', "
            f"priority={self.priority}, dim={self.blind_dimension.value})"
        )


# ============================================================
# 视障专属任务模板库
# ============================================================

class BlindTaskTemplate:
    """视障专属任务分解模板库

    覆盖 10 大类视障常见需求，每类提供典型查询样例与子任务原型。
    模板存储在 prompts/blind_task_templates.json 中。
    """

    CATEGORIES: list[str] = [
        "find_object",          # 找物
        "describe_scene",       # 描述
        "fetch_object",         # 取物
        "navigate",             # 导引
        "alert_safety",         # 警示
        "read_text",            # 阅读
        "cooking_assist",       # 烹饪辅助
        "clothing_assist",      # 着装辅助
        "medication",           # 用药
        "social",               # 社交
    ]

    def __init__(
        self,
        templates_path: str = "prompts/blind_task_templates.json",
    ) -> None:
        self.templates_path = Path(templates_path)
        self._templates: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """从 JSON 文件加载模板"""
        if not self.templates_path.exists():
            logger.warning(
                f"任务模板文件不存在: {self.templates_path}，将使用内置默认模板"
            )
            self._templates = self._builtin_templates()
            return

        with open(self.templates_path, "r", encoding="utf-8") as f:
            self._templates = json.load(f)

    def _builtin_templates(self) -> list[dict[str, Any]]:
        """内置默认模板（最少示例，确保系统能跑通）"""
        return [
            {
                "category": "find_object",
                "query": "我的药瓶在哪里？",
                "subtasks": [
                    {
                        "type": "identify",
                        "target": "药瓶",
                        "priority": 1,
                        "blind_dimension": "position",
                        "output_format": "确认药瓶是否存在",
                    },
                    {
                        "type": "locate",
                        "target": "药瓶",
                        "priority": 2,
                        "blind_dimension": "distance",
                        "output_format": "方位 + cm 级距离",
                    },
                    {
                        "type": "guide",
                        "target": "药瓶",
                        "priority": 3,
                        "blind_dimension": "action",
                        "output_format": "如何安全伸手取物",
                    },
                ],
            },
            {
                "category": "describe_scene",
                "query": "桌上有什么？",
                "subtasks": [
                    {
                        "type": "describe",
                        "target": "桌面整体",
                        "priority": 1,
                        "blind_dimension": "position",
                        "output_format": "列出所有显著物体",
                    },
                    {
                        "type": "alert",
                        "target": "桌面危险物",
                        "priority": 1,
                        "blind_dimension": "safety",
                        "output_format": "热源/锐器/易碎",
                    },
                ],
            },
            {
                "category": "alert_safety",
                "query": "周围有危险吗？",
                "subtasks": [
                    {
                        "type": "alert",
                        "target": "环境潜在危险",
                        "priority": 1,
                        "blind_dimension": "safety",
                        "output_format": "热源/锐器/不稳定/化学品",
                    },
                ],
            },
        ]

    def retrieve_similar(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        """检索 K 个最相似模板示例（Few-shot 检索）

        Args:
            query: 视障者查询
            k: 检索数量

        Returns:
            K 个最相似模板示例

        Note:
            校赛阶段使用简单关键词匹配；省赛阶段升级为 sentence embedding。
        """
        # TODO: 校赛阶段简化为关键词匹配；省赛升级为 sentence embedding
        # 简化版: 用关键词重叠度排序
        scored: list[tuple[int, dict[str, Any]]] = []
        query_chars = set(query)
        for tpl in self._templates:
            tpl_query = tpl.get("query", "")
            overlap = len(query_chars & set(tpl_query))
            scored.append((overlap, tpl))
        scored.sort(key=lambda x: -x[0])
        return [t for _, t in scored[:k]]


# ============================================================
# 核心类: TaskDecomposer
# ============================================================

class TaskDecomposer:
    """零样本视障任务分解器

    使用 LLM (DeepSeek-V3) 将视障者自然语言查询分解为结构化子任务序列。

    Attributes:
        llm: LLM 后端客户端
        prompt_path: 系统 Prompt 模板路径
        template_lib: 视障任务模板库
    """

    def __init__(
        self,
        llm_client,
        prompt_path: str = "prompts/task_decompose.txt",
        templates_path: str = "prompts/blind_task_templates.json",
    ) -> None:
        self.llm = llm_client
        self.prompt_path = Path(prompt_path)
        self.system_prompt = self._load_prompt()
        self.template_lib = BlindTaskTemplate(templates_path)

    def _load_prompt(self) -> str:
        """加载系统 Prompt 模板"""
        if not self.prompt_path.exists():
            logger.warning(f"Prompt 文件不存在: {self.prompt_path}")
            return ""
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _build_few_shot_prompt(
        self,
        query: str,
        examples: list[dict[str, Any]],
    ) -> str:
        """构建 Few-shot Prompt"""
        lines = ["## 历史示例", ""]
        for i, ex in enumerate(examples, 1):
            lines.append(f"### 示例 {i}")
            lines.append(f"查询: {ex.get('query', '')}")
            lines.append(f"分解: {json.dumps(ex.get('subtasks', []), ensure_ascii=False, indent=2)}")
            lines.append("")
        lines.append("## 当前查询")
        lines.append(f"查询: {query}")
        lines.append("")
        lines.append("请按照系统指令的格式分解上述查询为 JSON 子任务列表。")
        return "\n".join(lines)

    def decompose(self, query: str, k_examples: int = 3) -> list[Subtask]:
        """主入口: 分解视障者查询为子任务列表

        Args:
            query: 视障者自然语言查询
            k_examples: 使用的 Few-shot 示例数量

        Returns:
            子任务列表（按优先级升序排序，1 最优先）
        """
        logger.info(f"分解查询: {query}")

        examples = self.template_lib.retrieve_similar(query, k=k_examples)
        prompt = self._build_few_shot_prompt(query, examples)

        raw_output = self.llm.generate(
            user_message=prompt,
            system=self.system_prompt,
            json_mode=True,
        )

        subtasks = self._parse_subtasks(raw_output)
        subtasks = self._validate_and_sort(subtasks)

        logger.info(f"分解出 {len(subtasks)} 个子任务")
        return subtasks

    def _parse_subtasks(self, raw_output: str) -> list[Subtask]:
        """解析 LLM 输出为 Subtask 对象列表"""
        try:
            data = json.loads(raw_output)
            items = data.get("subtasks", []) if isinstance(data, dict) else data
            subtasks: list[Subtask] = []
            for item in items:
                try:
                    subtasks.append(
                        Subtask(
                            type=SubtaskType(item["type"]),
                            target=item["target"],
                            priority=int(item.get("priority", 3)),
                            blind_dimension=BlindDimension(
                                item.get("blind_dimension", "position")
                            ),
                            output_format=item.get("output_format", ""),
                        )
                    )
                except (KeyError, ValueError) as e:
                    logger.warning(f"跳过无效子任务: {item}, 原因: {e}")
            return subtasks
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}\n原始输出: {raw_output[:500]}")
            return []

    def _validate_and_sort(self, subtasks: list[Subtask]) -> list[Subtask]:
        """验证有效性并按优先级排序"""
        valid = [t for t in subtasks if t.target.strip()]
        return sorted(valid, key=lambda t: t.priority)


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("[TaskDecomposer] 模块加载测试")
    print(f"  支持子任务类型: {[t.value for t in SubtaskType]}")
    print(f"  支持视障维度: {[d.value for d in BlindDimension]}")
    print(f"  支持模板类别: {BlindTaskTemplate.CATEGORIES}")

    template_lib = BlindTaskTemplate()
    examples = template_lib.retrieve_similar("帮我找一下水杯", k=2)
    print(f"\n  '帮我找一下水杯' 检索到 {len(examples)} 个相似模板:")
    for ex in examples:
        print(f"    - {ex.get('query')}")