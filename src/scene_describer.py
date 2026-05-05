"""创新点③: 零样本视障友好场景描述生成器

将通用 VLM (Qwen2.5-VL) 输出改造为视障友好格式。

核心设计:
    1. 五维度强制输出（物体/方位/触觉/安全/行动）
    2. 几何后处理（深度图反推 cm 级距离）
    3. 视障友好词汇库

使用示例:
    >>> from src.scene_describer import SceneDescriber
    >>> from src.vlm_backend import VLMBackend
    >>> vlm = VLMBackend()
    >>> describer = SceneDescriber(vlm)
    >>> desc = describer.describe("kitchen.png")
    >>> print(desc.to_speech())
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ObjectPosition:
    """物体位置信息"""

    obj: str
    direction: str
    distance_cm: float
    height_cm: float = 0.0
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StructuredDescription:
    """五维度结构化描述

    Attributes:
        objects: 物体名称列表（含形状特征）
        positions: 物体位置信息列表
        tactile: 触觉特征描述列表
        safety_alerts: 安全提示列表
        actionable_advice: 行动建议列表
    """

    objects: list[str] = field(default_factory=list)
    positions: list[ObjectPosition] = field(default_factory=list)
    tactile: list[str] = field(default_factory=list)
    safety_alerts: list[str] = field(default_factory=list)
    actionable_advice: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objects": self.objects,
            "positions": [p.to_dict() for p in self.positions],
            "tactile": self.tactile,
            "safety_alerts": self.safety_alerts,
            "actionable_advice": self.actionable_advice,
        }

    def to_speech(self) -> str:
        """转化为视障者听到的语音文本

        优先级: 安全提示 > 物体位置 > 行动建议
        """
        lines: list[str] = []

        if self.safety_alerts:
            lines.append("注意：" + "；".join(self.safety_alerts))

        for p in self.positions:
            line = f"{p.direction} {p.distance_cm:.0f}厘米处有{p.obj}"
            lines.append(line)

        if self.actionable_advice:
            lines.append("建议：" + "；".join(self.actionable_advice))

        return "。".join(lines) + "。"

    def is_empty(self) -> bool:
        return not (self.objects or self.positions or self.safety_alerts)


# ============================================================
# 视障友好词汇库
# ============================================================

class BlindFriendlyVocabulary:
    """视障友好词汇库

    将通用描述转换为视障友好表达：
    - 避免单纯颜色描述（视障者不可感知）
    - 强调形状/材质/温度等触觉可感知特征
    - 提供安全等级分类
    """

    SAFETY_KEYWORDS: dict[str, list[str]] = {
        "hot": ["热", "烫", "刚煮过", "温热", "蒸汽"],
        "sharp": ["锋利", "刀", "针", "剪刀", "刃"],
        "fragile": ["易碎", "玻璃", "陶瓷", "脆"],
        "unstable": ["不稳", "悬空", "斜放", "易倒"],
        "chemical": ["清洁剂", "化学品", "溶液"],
    }

    SHAPE_KEYWORDS: list[str] = [
        "圆筒形", "长方体", "球形", "椭圆", "扁平", "立方", "锥形"
    ]

    @classmethod
    def detect_safety(cls, text: str) -> list[str]:
        """从文本中检测安全风险"""
        alerts: list[str] = []
        for category, keywords in cls.SAFETY_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                alerts.append(f"[{category}] " + text[:50])
        return alerts


# ============================================================
# 核心类: SceneDescriber
# ============================================================

class SceneDescriber:
    """零样本视障友好场景描述生成器"""

    def __init__(
        self,
        vlm_client,
        prompt_path: str = "prompts/scene_describer.txt",
        use_geometric_postprocess: bool = True,
    ) -> None:
        """
        Args:
            vlm_client: VLM 客户端
            prompt_path: 系统 Prompt 模板路径
            use_geometric_postprocess: 是否启用几何后处理（cm 级精度）
        """
        self.vlm = vlm_client
        self.prompt_path = Path(prompt_path)
        self.system_prompt = self._load_prompt()
        self.use_geo = use_geometric_postprocess

    def _load_prompt(self) -> str:
        if not self.prompt_path.exists():
            logger.warning(f"Prompt 文件不存在: {self.prompt_path}")
            return ""
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def describe(
        self,
        image_path: str,
        depth_map: Optional[Any] = None,
        subtasks: Optional[list] = None,
        viewpoint: Optional[Any] = None,
    ) -> StructuredDescription:
        """主入口: 对图像生成视障友好描述

        Args:
            image_path: RGB 图像路径
            depth_map: 深度图（可选，用于几何后处理）
            subtasks: 当前子任务列表（用于聚焦 VLM 注意力）
            viewpoint: 当前视角信息（用于估算距离）

        Returns:
            五维度结构化描述
        """
        logger.info(f"开始描述图像: {image_path}")

        prompt = self._build_describe_prompt(subtasks, viewpoint)
        raw_output = self.vlm.describe(image_path, prompt=prompt)

        desc = self._parse_description(raw_output)

        if self.use_geo and depth_map is not None:
            desc = self._geometric_postprocess(desc, depth_map, viewpoint)

        desc = self._adapt_vocabulary(desc)

        return desc

    def _build_describe_prompt(
        self,
        subtasks: Optional[list],
        viewpoint: Optional[Any],
    ) -> str:
        """构建场景描述 Prompt（视障专属）"""
        lines = [self.system_prompt]

        if viewpoint:
            lines.append(f"\n[当前视角] {viewpoint.name} - {viewpoint.purpose}")

        if subtasks:
            lines.append("\n[当前需要关注的子任务]")
            for t in subtasks:
                lines.append(f"  - {t.type.value} {t.target} (维度: {t.blind_dimension.value})")

        lines.append("\n请基于以上要求，对图像生成五维度结构化描述（严格 JSON 格式）。")
        return "\n".join(lines)

    def _parse_description(self, raw_output: str) -> StructuredDescription:
        """解析 VLM 输出为五维度结构"""
        try:
            try:
                data = json.loads(raw_output)
            except json.JSONDecodeError:
                start = raw_output.find("{")
                end = raw_output.rfind("}")
                if start >= 0 and end > start:
                    data = json.loads(raw_output[start : end + 1])
                else:
                    raise

            positions = [
                ObjectPosition(**p) for p in data.get("positions", [])
            ]
            return StructuredDescription(
                objects=data.get("objects", []),
                positions=positions,
                tactile=data.get("tactile", []),
                safety_alerts=data.get("safety_alerts", []),
                actionable_advice=data.get("actionable_advice", []),
            )
        except Exception as e:
            logger.error(f"VLM 输出解析失败: {e}\n原始输出: {raw_output[:500]}")
            return StructuredDescription()

    def _geometric_postprocess(
        self,
        desc: StructuredDescription,
        depth_map,
        viewpoint,
    ) -> StructuredDescription:
        """利用深度图校正距离（cm 级精度）

        TODO: 在仿真环境集成完成后实现：
            1. 在图像中找到每个物体的中心像素
            2. 在深度图对应位置采样深度值
            3. 反投影到 3D 坐标
            4. 计算到摄像头/视障者参考点的真实距离
        """
        logger.debug("几何后处理待实现")
        return desc

    def _adapt_vocabulary(
        self,
        desc: StructuredDescription,
    ) -> StructuredDescription:
        """将描述改造为视障友好语言"""
        for tactile_text in desc.tactile:
            extra_alerts = BlindFriendlyVocabulary.detect_safety(tactile_text)
            desc.safety_alerts.extend(extra_alerts)
        desc.safety_alerts = list(dict.fromkeys(desc.safety_alerts))
        return desc

    def aggregate(
        self,
        descriptions: list[StructuredDescription],
    ) -> StructuredDescription:
        """聚合多视角描述为最终输出

        - 物体去重
        - 距离取最优（最近视角）
        - 安全提示合并
        """
        if not descriptions:
            return StructuredDescription()

        all_objects: list[str] = []
        seen_objects: set[str] = set()
        for d in descriptions:
            for obj in d.objects:
                if obj not in seen_objects:
                    all_objects.append(obj)
                    seen_objects.add(obj)

        position_map: dict[str, ObjectPosition] = {}
        for d in descriptions:
            for p in d.positions:
                existing = position_map.get(p.obj)
                if existing is None or p.distance_cm < existing.distance_cm:
                    position_map[p.obj] = p

        all_safety: list[str] = []
        for d in descriptions:
            all_safety.extend(d.safety_alerts)
        all_safety = list(dict.fromkeys(all_safety))

        all_advice: list[str] = []
        for d in descriptions:
            all_advice.extend(d.actionable_advice)
        all_advice = list(dict.fromkeys(all_advice))

        all_tactile: list[str] = []
        for d in descriptions:
            all_tactile.extend(d.tactile)
        all_tactile = list(dict.fromkeys(all_tactile))

        return StructuredDescription(
            objects=all_objects,
            positions=list(position_map.values()),
            tactile=all_tactile,
            safety_alerts=all_safety,
            actionable_advice=all_advice,
        )


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("[SceneDescriber] 模块加载测试")

    desc = StructuredDescription(
        objects=["白色陶瓷杯", "塑料药瓶"],
        positions=[
            ObjectPosition(obj="白色陶瓷杯", direction="正前方", distance_cm=30, height_cm=8),
            ObjectPosition(obj="塑料药瓶", direction="左前方", distance_cm=25, height_cm=10),
        ],
        tactile=["白色陶瓷杯：光滑陶瓷材质，圆筒形", "塑料药瓶：磨砂塑料，圆柱形"],
        safety_alerts=["白色陶瓷杯可能温热"],
        actionable_advice=["可从左侧伸手取药瓶，距您手 25cm"],
    )

    print("\n[结构化描述]")
    print(json.dumps(desc.to_dict(), ensure_ascii=False, indent=2))

    print("\n[视障语音输出]")
    print(desc.to_speech())