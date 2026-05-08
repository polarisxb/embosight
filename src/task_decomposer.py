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
        "find_object",          # 找物 (3 templates)
        "describe_scene",       # 描述 (3 templates)
        "fetch_object",         # 取物 (4 templates)
        "navigate",             # 导引 (2 templates)
        "alert_safety",         # 警示 (2 templates)
        "medication",           # 用药 (2 templates)
        "cooking_assist",       # 烹饪辅助 (2 templates)
        "clothing_assist",      # 着装辅助 (2 templates)
        "social",               # 社交 (2 templates)
        "read_text",            # 阅读 (TODO)
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

    # 同义词扩展表: 覆盖视障者常用动词/名词表达差异
    SYNONYMS: dict[str, list[str]] = {
        "拿": ["取", "递", "给我", "帮我拿", "拿来", "抓"],
        "取": ["拿", "递", "给我"],
        "找": ["在哪", "哪里", "搜", "寻找", "找到"],
        "在哪": ["哪里", "找", "位置", "放哪"],
        "看": ["看看", "描述", "告诉我"],
        "危险": ["安全", "小心", "注意", "烫", "伤"],
        "药": ["药瓶", "药片", "药盒", "感冒药"],
        "杯": ["水杯", "杯子", "茶杯", "马克杯"],
        "刀": ["刀具", "菜刀", "水果刀", "剪刀"],
        "锅": ["炒锅", "汤锅", "平底锅", "锅具"],
        "瓶": ["瓶子", "药瓶", "水瓶", "奶瓶"],
    }

    # 背景 IDF: 常见中文高频词的惩罚权重 (越常见 IDF 越低)
    BACKGROUND_IDF: dict[str, float] = {
        "的": 0.3, "了": 0.4, "吗": 0.5, "是": 0.4, "在": 0.6,
        "有": 0.6, "我": 0.5, "帮": 0.8, "个": 0.5, "那": 0.6,
        "这": 0.6, "什么": 0.7, "上": 0.6, "里": 0.7, "面": 0.7,
        "给": 0.8, "把": 0.7, "去": 0.8, "怎么": 0.8, "不": 0.5,
    }

    def retrieve_similar(
        self,
        query: str,
        k: int = 3,
        use_idf: bool = True,
        use_synonyms: bool = True,
    ) -> list[dict[str, Any]]:
        """检索 K 个最相似模板示例（Few-shot 检索）

        创新点①核心: IDF加权模板检索
            1. 对 query 和每个模板的 query 分词
            2. 同义词扩展增强召回
            3. IDF 加权 Jaccard 相似度 (稀有词权重高)
            4. 类别优先级额外加分

        Args:
            query: 视障者查询
            k: 检索数量
            use_idf: 是否使用 IDF 加权 (消融开关)
            use_synonyms: 是否使用同义词扩展 (消融开关)

        Returns:
            K 个最相似模板示例 (含检索分数)
        """
        query_tokens = self._tokenize(query)
        if use_synonyms:
            query_tokens = self._expand_synonyms(query_tokens)
        if not query_tokens:
            return self._templates[:k]

        idf = self._compute_idf() if use_idf else {}

        scored: list[tuple[float, dict[str, Any]]] = []
        for tpl in self._templates:
            tpl_tokens = self._tokenize(tpl.get("query", ""))
            if use_synonyms:
                tpl_tokens = self._expand_synonyms(tpl_tokens)
            if not tpl_tokens:
                scored.append((0.0, tpl))
                continue

            intersection = query_tokens & tpl_tokens
            union = query_tokens | tpl_tokens

            if use_idf:
                weighted_inter = sum(idf.get(w, self.BACKGROUND_IDF.get(w, 1.5)) for w in intersection)
                weighted_union = sum(idf.get(w, self.BACKGROUND_IDF.get(w, 1.5)) for w in union)
            else:
                weighted_inter = len(intersection)
                weighted_union = len(union)

            sim = weighted_inter / weighted_union if weighted_union > 0 else 0.0
            category_bonus = self._category_bonus(query, tpl.get("category", ""))
            sim += category_bonus

            scored.append((sim, tpl))

        scored.sort(key=lambda x: -x[0])

        # 检索日志: 记录 top-K 分数供消融分析
        top_k = scored[:k]
        for rank, (score, tpl) in enumerate(top_k, 1):
            logger.info(
                f"[retrieve] rank={rank} score={score:.3f} "
                f"category={tpl.get('category')} query='{tpl.get('query')}'"
            )
        if top_k:
            logger.info(
                f"[retrieve] config: use_idf={use_idf}, use_synonyms={use_synonyms}, "
                f"top1_score={top_k[0][0]:.3f}, corpus_size={len(self._templates)}"
            )

        return [t for _, t in top_k]

    def _expand_synonyms(self, tokens: set[str]) -> set[str]:
        """同义词扩展: 将 token 集合扩展为包含同义词的更大集合"""
        expanded = set(tokens)
        for token in tokens:
            if token in self.SYNONYMS:
                expanded.update(self.SYNONYMS[token])
        return expanded

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """中文分词（优先 jieba，回退到 bigram + unigram）"""
        try:
            import jieba
            return set(w for w in jieba.lcut(text) if len(w.strip()) > 0)
        except ImportError:
            tokens: set[str] = set()
            text = text.strip()
            for i in range(len(text) - 1):
                tokens.add(text[i:i+2])
            if text:
                tokens.update(set(text))
            return tokens

    def _compute_idf(self) -> dict[str, float]:
        """计算 IDF 权重: 模板语料 IDF + 背景 IDF 融合"""
        import math

        if hasattr(self, "_idf_cache"):
            return self._idf_cache

        doc_count: dict[str, int] = {}
        n = len(self._templates)
        for tpl in self._templates:
            tokens = self._tokenize(tpl.get("query", ""))
            for w in tokens:
                doc_count[w] = doc_count.get(w, 0) + 1

        corpus_idf = {w: math.log((n + 1) / (cnt + 1)) + 1 for w, cnt in doc_count.items()}

        # 融合背景 IDF: 高频通用词惩罚
        for w, bg_idf in self.BACKGROUND_IDF.items():
            if w in corpus_idf:
                corpus_idf[w] = min(corpus_idf[w], bg_idf)
            else:
                corpus_idf[w] = bg_idf

        self._idf_cache = corpus_idf
        return corpus_idf

    @staticmethod
    def _category_bonus(query: str, category: str) -> float:
        """基于关键词给类别加分"""
        CATEGORY_KEYWORDS = {
            "find_object": ["在哪", "找", "哪里", "搜", "位置", "放哪"],
            "describe_scene": ["有什么", "描述", "什么东西", "看看", "里面"],
            "fetch_object": ["拿", "取", "递", "帮我", "给我", "抓"],
            "alert_safety": ["危险", "安全", "小心", "注意", "烫", "伤"],
            "navigate": ["去", "走", "怎么", "路", "方向", "哪个方向"],
            "medication": ["药", "吃药", "服药", "感冒", "药瓶", "药片"],
            "cooking_assist": ["锅", "灶", "炒", "煮", "熟", "开了", "火"],
            "clothing_assist": ["衣服", "颜色", "穿", "袜子", "裤子", "鞋"],
            "social": ["谁", "人", "做什么", "说什么", "他", "她"],
        }
        keywords = CATEGORY_KEYWORDS.get(category, [])
        hits = sum(1 for kw in keywords if kw in query)
        return hits * 0.15


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
        """验证有效性、五维度覆盖补全、优先级排序

        核心创新：强制五维度覆盖
            1. 检查 LLM 输出是否覆盖了所有 5 个 BlindDimension
            2. 缺失维度自动补全（尤其 safety 永远不能漏）
            3. 按优先级排序
        """
        valid = [t for t in subtasks if t.target.strip()]

        covered_dims = {t.blind_dimension for t in valid}
        all_dims = set(BlindDimension)
        missing_dims = all_dims - covered_dims

        if missing_dims:
            logger.info(f"五维度缺失检测: {[d.value for d in missing_dims]}，自动补全")

        primary_target = valid[0].target if valid else "环境"

        DIM_补全_RULES: dict[BlindDimension, Subtask] = {
            BlindDimension.SAFETY: Subtask(
                type=SubtaskType.ALERT,
                target=f"{primary_target}周围潜在危险",
                priority=1,
                blind_dimension=BlindDimension.SAFETY,
                output_format="热源/锐器/易碎/不稳定/化学品",
            ),
            BlindDimension.POSITION: Subtask(
                type=SubtaskType.LOCATE,
                target=primary_target,
                priority=2,
                blind_dimension=BlindDimension.POSITION,
                output_format="方位（前/后/左/右）",
            ),
            BlindDimension.DISTANCE: Subtask(
                type=SubtaskType.LOCATE,
                target=primary_target,
                priority=3,
                blind_dimension=BlindDimension.DISTANCE,
                output_format="厘米级距离",
            ),
            BlindDimension.TACTILE: Subtask(
                type=SubtaskType.DESCRIBE,
                target=f"{primary_target}触觉特征",
                priority=4,
                blind_dimension=BlindDimension.TACTILE,
                output_format="形状/材质/温度",
            ),
            BlindDimension.ACTION: Subtask(
                type=SubtaskType.GUIDE,
                target=primary_target,
                priority=5,
                blind_dimension=BlindDimension.ACTION,
                output_format="安全取物或避开建议",
            ),
        }

        for dim in missing_dims:
            if dim in DIM_补全_RULES:
                valid.append(DIM_补全_RULES[dim])
                logger.debug(f"  补全维度: {dim.value}")

        return sorted(valid, key=lambda t: t.priority)

    # ============================================================
    # v1: decompose_v1 -> DecomposedTask (替代 decompose)
    # ============================================================

    def decompose_v1(
        self, query: str, prompt_path: str = "prompts/agent/decompose.txt",
    ):
        """v1 输出 DecomposedTask, 含 primary_target + constraints。"""
        from pathlib import Path as _P
        import json as _json
        import re as _re
        from src.world_belief import Constraint, DecomposedTask

        p = _P(prompt_path)
        if p.exists():
            template = p.read_text(encoding="utf-8")
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

        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        if not m:
            return DecomposedTask(primary_target=query.strip(), raw_query=query)
        try:
            data = _json.loads(m.group())
        except _json.JSONDecodeError:
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
        return DecomposedTask(
            primary_target=primary,
            constraints=constraints,
            raw_query=query,
        )


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