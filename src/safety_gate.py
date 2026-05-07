"""SafeQuery-VLM Phase 4: 安全门控

根据物体类别风险等级 + VLM 置信度, 决定是否允许执行抓取.

核心设计:
    1. YAML 配置驱动: configs/safety_rules.yaml 定义类别→风险映射
    2. 三级门控: 一般阈值 / 高风险阈值 / 绝对拒绝
    3. 视障友好输出: 中文原因 (TTS) + 英文日志

创新点⑤: VLM 安全约束自动发现
    - 结合 VLM visible_features 文本检测潜在风险
    - 不仅依赖预定义规则, 还支持 VLM 自由文本安全提示

使用示例:
    >>> from src.safety_gate import SafetyGate
    >>> gate = SafetyGate()
    >>> decision = gate.check(grounded_object)
    >>> if decision.allow_execute:
    ...     print(f"执行: {decision.reason_user}")
    ... else:
    ...     print(f"拒绝: {decision.reason_user}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class SafetyDecision:
    """安全门控决策结果."""
    allow_execute: bool                 # 是否允许执行
    risk_level: str                     # safe/fragile/hot/sharp/high/unknown
    confidence: float                   # 触发决策时的置信度
    reason_user: str                    # TTS 给用户的中文解释
    reason_log: str                     # 日志用英文技术细节
    extra_warnings: list[str] = None    # VLM 文本检测到的额外风险

    def __post_init__(self):
        if self.extra_warnings is None:
            self.extra_warnings = []


# VLM visible_features 中的风险关键词
_FEATURE_RISK_KEYWORDS = {
    "sharp": ["sharp", "blade", "pointed", "edged", "cutting", "锋利", "刀片"],
    "hot": ["hot", "steaming", "boiling", "heated", "warm", "温热", "蒸汽", "烫"],
    "fragile": ["glass", "ceramic", "porcelain", "crystal", "玻璃", "陶瓷", "瓷"],
    "chemical": ["chemical", "cleaner", "detergent", "bleach", "清洁剂", "化学"],
}


# ============================================================
# 核心类: SafetyGate
# ============================================================

class SafetyGate:
    """安全门控: 根据物体类别 + 置信度决定是否执行抓取.

    设计原则:
        - 宁可拒绝也不误操作 (false negative > false positive)
        - 高风险物体需要更高置信度
        - 支持 VLM 文本检测额外风险 (不仅靠规则表)
    """

    def __init__(self, rules_path: str = "configs/safety_rules.yaml"):
        self._categories: dict[str, dict] = {}
        self._gates: dict[str, Any] = {}
        self._load_rules(rules_path)

    def _load_rules(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            logger.warning(f"[safety] rules file not found: {p}, using defaults")
            self._categories = {"_default": {"risk_level": "unknown", "zh_name": "未知物体", "reason": "建议手动确认"}}
            self._gates = {"min_confidence": 0.75, "high_risk_min_confidence": 0.90,
                           "high_risk_categories": ["high", "hot", "sharp"],
                           "never_execute_categories": []}
            return

        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        self._categories = data.get("categories", {})
        self._gates = data.get("gates", {})

    def _get_rule(self, category: str) -> dict:
        """查类别规则, fallback 到 _default."""
        return self._categories.get(category, self._categories.get("_default", {
            "risk_level": "unknown", "zh_name": "未知物体", "reason": "建议手动确认"
        }))

    def check(self, grounded_object) -> SafetyDecision:
        """评估 GroundedObject 是否安全可执行.

        使用 grounded_object 的以下字段:
            - matched_category 或 label: 物体类别
            - query_match_score: 匹配置信度
            - position_confidence: 定位置信度
            - per_view_features: VLM 描述文本 (用于额外风险检测)

        Returns:
            SafetyDecision
        """
        # 确定类别 (优先用 matched_category, 其次 label)
        category = getattr(grounded_object, "matched_category", "") or ""
        if not category:
            category = getattr(grounded_object, "label", "unknown")
        category = category.lower().strip()

        # 查规则
        rule = self._get_rule(category)
        risk = rule.get("risk_level", "unknown")
        zh_name = rule.get("zh_name", "未知物体")
        reason = rule.get("reason", "")

        # 综合置信度: 取 query_match 和 position 的较低者
        match_conf = getattr(grounded_object, "query_match_score", 0.0)
        pos_conf = getattr(grounded_object, "position_confidence", 0.0)
        conf = min(match_conf, pos_conf) if pos_conf > 0 else match_conf

        # VLM 文本额外风险检测
        extra_warnings = self._detect_feature_risks(grounded_object)
        if extra_warnings:
            # 如果 VLM 文本检测到高风险, 提升风险等级
            detected_risks = set()
            for w in extra_warnings:
                for risk_type in ["sharp", "hot", "chemical"]:
                    if risk_type in w.lower():
                        detected_risks.add(risk_type)
            if "chemical" in detected_risks:
                risk = "high"
            elif detected_risks & {"sharp", "hot"} and risk == "safe":
                risk = list(detected_risks & {"sharp", "hot"})[0]

        # 门控阈值
        min_conf = self._gates.get("min_confidence", 0.75)
        high_risk_cats = self._gates.get("high_risk_categories", [])
        never_cats = self._gates.get("never_execute_categories", [])

        # 绝对拒绝
        if risk in never_cats:
            return SafetyDecision(
                allow_execute=False,
                risk_level=risk,
                confidence=conf,
                reason_user=f"检测到{zh_name}, 此类物体视障场景下禁止操作。{reason}",
                reason_log=f"[safety] REJECT: never_execute category '{category}' risk={risk}",
                extra_warnings=extra_warnings,
            )

        # 高风险阈值
        if risk in high_risk_cats:
            min_conf = max(min_conf, self._gates.get("high_risk_min_confidence", 0.90))

        # 置信度检查
        if conf < min_conf:
            return SafetyDecision(
                allow_execute=False,
                risk_level=risk,
                confidence=conf,
                reason_user=f"我不太确定眼前是否为{zh_name}（置信度{conf:.0%}），建议您手动确认。{reason}",
                reason_log=f"[safety] REJECT: conf {conf:.2f} < threshold {min_conf:.2f} "
                           f"(category='{category}' risk={risk})",
                extra_warnings=extra_warnings,
            )

        # 通过 — 但如果有风险, 附加警告
        user_msg = f"已定位{zh_name}（置信度{conf:.0%}）。"
        if risk not in ("safe", "unknown"):
            user_msg += f"注意：{reason}"
        if extra_warnings:
            user_msg += " " + "；".join(extra_warnings)

        return SafetyDecision(
            allow_execute=True,
            risk_level=risk,
            confidence=conf,
            reason_user=user_msg,
            reason_log=f"[safety] PASS: {category} risk={risk} conf={conf:.2f}",
            extra_warnings=extra_warnings,
        )

    def update_object_safety(self, grounded_object) -> None:
        """将安全信息注入 GroundedObject (in-place mutation).

        设置 safety_risk 和 safety_reason 字段.
        """
        category = getattr(grounded_object, "matched_category", "") or ""
        if not category:
            category = getattr(grounded_object, "label", "unknown")
        category = category.lower().strip()

        rule = self._get_rule(category)
        grounded_object.safety_risk = rule.get("risk_level", "unknown")
        grounded_object.safety_reason = rule.get("reason", "")

    @staticmethod
    def _detect_feature_risks(grounded_object) -> list[str]:
        """从 VLM visible_features 文本中检测潜在风险.

        创新点: 不仅依赖预定义规则表, 还利用 VLM 的自由文本描述
        发现潜在风险 (如 VLM 描述 "sharp metal blade" 但类别是 "unknown").

        Returns:
            风险警告列表
        """
        warnings = []
        # 收集所有 features 文本
        features_texts = []
        if hasattr(grounded_object, "per_view_features"):
            features_texts.extend(grounded_object.per_view_features.values())
        if hasattr(grounded_object, "visible_features"):
            features_texts.append(getattr(grounded_object, "visible_features", ""))

        combined = " ".join(features_texts).lower()
        if not combined.strip():
            return warnings

        for risk_type, keywords in _FEATURE_RISK_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in combined:
                    warnings.append(f"VLM检测到{risk_type}风险特征: '{kw}'")
                    break  # 每种风险类型只报一次

        return warnings


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("[SafetyGate] 模块加载测试")

    gate = SafetyGate()

    # 模拟一个 GroundedObject (简单 mock)
    class _MockObj:
        def __init__(self, cat, match_score, pos_conf, features=None):
            self.matched_category = cat
            self.label = cat
            self.query_match_score = match_score
            self.position_confidence = pos_conf
            self.per_view_features = {"v1": features or ""}
            self.safety_risk = "unknown"
            self.safety_reason = ""

    # 测试几个场景
    for cat, score, pconf in [("apple", 0.9, 0.85), ("knife", 0.95, 0.92),
                               ("knife", 0.7, 0.8), ("unknown_thing", 0.5, 0.3)]:
        obj = _MockObj(cat, score, pconf)
        dec = gate.check(obj)
        print(f"  {cat} (conf={score}): allow={dec.allow_execute} risk={dec.risk_level} — {dec.reason_user}")
