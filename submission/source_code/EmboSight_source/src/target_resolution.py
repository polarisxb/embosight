from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def label_key(text: object) -> str:
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum())


@dataclass(frozen=True)
class TargetResolution:
    requested_label: str
    selected_label: str
    target_body: str | None
    body_category: str | None
    source: str
    confidence: float
    used_fallback: bool
    reason: str

    @property
    def status(self) -> str:
        if self.target_body is not None:
            return "fallback" if self.used_fallback else "resolved"
        return "unresolved"

    def to_diagnostic(self) -> dict[str, object]:
        return {
            "target_resolution_status": self.status,
            "target_resolution_requested_label": self.requested_label,
            "target_resolution_selected_label": self.selected_label,
            "resolved_body_name": self.target_body,
            "resolved_body_category": self.body_category,
            "target_body": self.target_body,
            "target_body_category": self.body_category,
            "target_resolution_source": self.source,
            "target_resolution_confidence": float(self.confidence),
            "target_resolution_used_fallback": bool(self.used_fallback),
            "target_resolution_reason": self.reason,
        }


def resolve_target_body(
    *,
    requested_label: str | None,
    selected_label: str | None,
    env: Any,
    allow_fallback: bool = False,
) -> TargetResolution:
    requested = str(requested_label or "").strip()
    selected = str(selected_label or "").strip()
    selected_key = label_key(selected)
    type_map = _type_map(env)

    if selected_key:
        for body, category in type_map.items():
            if selected_key == label_key(category):
                return TargetResolution(
                    requested_label=requested,
                    selected_label=selected,
                    target_body=body,
                    body_category=category,
                    source="normalized_category",
                    confidence=0.9,
                    used_fallback=False,
                    reason="matched selected label to body category",
                )

    if allow_fallback and "obj_main" in type_map:
        return TargetResolution(
            requested_label=requested,
            selected_label=selected,
            target_body="obj_main",
            body_category=type_map.get("obj_main"),
            source="fallback_obj_main",
            confidence=0.5,
            used_fallback=True,
            reason="fallback to obj_main",
        )

    return TargetResolution(
        requested_label=requested,
        selected_label=selected,
        target_body=None,
        body_category=None,
        source="unresolved",
        confidence=0.0,
        used_fallback=False,
        reason="no matching body category",
    )


def _type_map(env: Any) -> dict[str, str]:
    if env is None or not hasattr(env, "_get_obj_type_map"):
        return {}
    try:
        raw = env._get_obj_type_map()
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(body): str(category) for body, category in raw.items()}
