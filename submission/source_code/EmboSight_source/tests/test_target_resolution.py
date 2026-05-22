from __future__ import annotations

from src.target_resolution import resolve_target_body


class _Env:
    def __init__(self, mapping):
        self.mapping = mapping

    def _get_obj_type_map(self):
        return dict(self.mapping)


def test_resolve_target_body_matches_normalized_category():
    env = _Env({
        "obj_main": "glass_cup",
        "distr_counter_main": "hotdog_bun",
    })

    result = resolve_target_body(
        requested_label="glass cup",
        selected_label="glass cup",
        env=env,
    )

    assert result.target_body == "obj_main"
    assert result.body_category == "glass_cup"
    assert result.source == "normalized_category"
    assert result.confidence == 0.9
    assert result.used_fallback is False
    assert result.reason == "matched selected label to body category"


def test_resolve_target_body_records_unresolved_without_fallback():
    env = _Env({"obj_main": "tupperware"})

    result = resolve_target_body(
        requested_label="lemon",
        selected_label="lemon",
        env=env,
        allow_fallback=False,
    )

    assert result.target_body is None
    assert result.body_category is None
    assert result.source == "unresolved"
    assert result.confidence == 0.0
    assert result.used_fallback is False
    assert result.reason == "no matching body category"


def test_resolve_target_body_records_explicit_fallback():
    env = _Env({"obj_main": "tupperware"})

    result = resolve_target_body(
        requested_label="lemon",
        selected_label="lemon",
        env=env,
        allow_fallback=True,
    )

    assert result.target_body == "obj_main"
    assert result.body_category == "tupperware"
    assert result.source == "fallback_obj_main"
    assert result.confidence == 0.5
    assert result.used_fallback is True
    assert result.reason == "fallback to obj_main"


def test_target_resolution_diagnostic_keys_are_stable():
    env = _Env({"distr_counter_main": "lemon_wedge"})

    result = resolve_target_body(
        requested_label="lemon wedge",
        selected_label="lemon wedge",
        env=env,
    )

    assert result.to_diagnostic() == {
        "target_resolution_status": "resolved",
        "target_resolution_requested_label": "lemon wedge",
        "target_resolution_selected_label": "lemon wedge",
        "resolved_body_name": "distr_counter_main",
        "resolved_body_category": "lemon_wedge",
        "target_body": "distr_counter_main",
        "target_body_category": "lemon_wedge",
        "target_resolution_source": "normalized_category",
        "target_resolution_confidence": 0.9,
        "target_resolution_used_fallback": False,
        "target_resolution_reason": "matched selected label to body category",
    }
