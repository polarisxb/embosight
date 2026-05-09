from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.eval_oracle import summarize_episode  # noqa: E402

logger = logging.getLogger(__name__)


def load_scenario(config_path: str | Path, scenario_id: str) -> dict[str, Any]:
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    for scenario in data.get("scenarios", []) or []:
        if str(scenario.get("id")) == scenario_id:
            result = dict(scenario)
            result.setdefault("user_mode", "fake_from_robocasa")
            result.setdefault("max_resets", 1)
            return result
    raise KeyError(f"scenario not found: {scenario_id}")


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def reset_until_expected(
    env,
    expected_object: str | None,
    seed: int | None,
    max_resets: int,
) -> tuple[str | None, int]:
    set_global_seed(seed)
    actual_object = None
    attempts = max(1, int(max_resets))
    for attempt in range(1, attempts + 1):
        env.reset()
        actual_object = get_actual_object(env)
        logger.info(
            "[fixed_eval] reset %d/%d actual obj_main=%r expected=%r",
            attempt,
            attempts,
            actual_object,
            expected_object,
        )
        if expected_object is None or _label_key(actual_object or "") == _label_key(expected_object):
            return actual_object, attempt
    return actual_object, attempts


def get_actual_object(env) -> str | None:
    try:
        return env._get_obj_type_map().get("obj_main")
    except Exception as exc:
        logger.warning("[fixed_eval] failed to read obj_main category: %s", exc)
        return None


def rewrite_query_for_actual_object(query: str, user_mode: str, actual_object: str | None) -> str:
    if user_mode == "fake_from_robocasa" and actual_object and actual_object != "unknown":
        return f"pick up the {actual_object}"
    return query


def latest_episode_path(log_dir: str | Path) -> Path | None:
    paths = list(Path(log_dir).glob("episode_*.json"))
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def build_agent(top_cfg: dict[str, Any], agent_cfg: dict[str, Any], env, user_mode: str, query: str):
    from scripts.run_agent import _build_llm, _build_vlm
    from src.action_executor import ActionExecutor
    from src.active_planner import ActiveViewpointSelector, ViewpointLibrary
    from src.agent import EmboSightAgent
    from src.episode_logger import EpisodeLogger
    from src.grasp_planner import GraspPlanner
    from src.perception import QueryAwareGrounder
    from src.safety_gate import SafetyClassifier
    from src.task_decomposer import TaskDecomposer
    from src.user_channel import CLIUserChannel, FakeUserChannel
    from src.vlm_cache import VLMCache

    llm = _build_llm(top_cfg)
    vlm = _build_vlm(top_cfg)
    cache = VLMCache(max_size=agent_cfg["cache"]["max_size"])
    vp_lib = ViewpointLibrary(
        config_path=top_cfg.get("viewpoints_path", "configs/viewpoints.yaml"),
    )
    if user_mode == "fake_from_robocasa":
        user_channel = FakeUserChannel.from_robocasa(llm, env)
    elif user_mode == "fake_from_query":
        user_channel = FakeUserChannel.from_query(llm, query)
    else:
        user_channel = CLIUserChannel()
    return EmboSightAgent(
        task_decomposer=TaskDecomposer(llm),
        perception=QueryAwareGrounder(
            vlm=vlm,
            llm=llm,
            cache=cache,
            label_temperature=agent_cfg["perception"]["label_temperature"],
            ground_prompt_path=agent_cfg["perception"]["ground_prompt"],
            zoom_prompt_path=agent_cfg["perception"]["zoom_prompt"],
            parallax_prompt_path=agent_cfg["perception"]["parallax_prompt"],
            pose_prompt_path=agent_cfg["perception"]["pose_prompt"],
            verify_prompt_path=agent_cfg["perception"]["verify_prompt"],
            viewpoint_lib=vp_lib,
        ),
        safety_classifier=SafetyClassifier(llm=llm),
        grasp_planner=GraspPlanner(vlm=vlm, env=env),
        action_executor=ActionExecutor(scene_describer=None),
        nbv_selector=ActiveViewpointSelector(llm=llm, viewpoint_lib=vp_lib),
        user_channel=user_channel,
        episode_logger=EpisodeLogger(log_dir=agent_cfg["logger"]["log_dir"]),
        viewpoint_lib=vp_lib,
        llm=llm,
        vlm=vlm,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one deterministic EmboSight eval scenario")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--scenarios-config", default="configs/eval_scenarios.yaml")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--agent-config", default="configs/agent.yaml")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--allow-object-mismatch", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    scenario = load_scenario(args.scenarios_config, args.scenario)
    top_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    agent_cfg = yaml.safe_load(Path(args.agent_config).read_text(encoding="utf-8")) or {}
    if scenario.get("env_name"):
        top_cfg.setdefault("simulator", {})["env_name"] = scenario["env_name"]

    from scripts.run_agent import _build_env

    env = _build_env(top_cfg)
    expected_object = scenario.get("expected_object")
    actual_object, resets_used = reset_until_expected(
        env,
        expected_object=expected_object,
        seed=scenario.get("seed"),
        max_resets=int(scenario.get("max_resets", 1)),
    )
    object_match = (
        expected_object is None
        or _label_key(actual_object or "") == _label_key(str(expected_object))
    )
    if not object_match and not args.allow_object_mismatch:
        print("\n========== FIXED SCENARIO MISMATCH ==========")
        print(f"scenario: {scenario['id']}")
        print(f"expected: {expected_object}")
        print(f"actual  : {actual_object}")
        print(f"resets  : {resets_used}")
        print("Use --allow-object-mismatch to run anyway.")
        return 2

    query = rewrite_query_for_actual_object(
        str(scenario.get("query", "pick up anything")),
        str(scenario.get("user_mode", "fake_from_robocasa")),
        actual_object,
    )
    agent = build_agent(
        top_cfg,
        agent_cfg,
        env,
        str(scenario.get("user_mode", "fake_from_robocasa")),
        query,
    )
    result = agent.run(query, env)
    print("\n========== EPISODE RESULT ==========")
    print(f"scenario: {scenario['id']}")
    print(f"success : {result.success}")
    print(f"speech  : {result.speech}")
    print(f"steps   : {result.n_steps}")
    print(f"time    : {result.elapsed_seconds:.1f}s")
    if not result.success:
        print(f"reason  : {result.failure_reason}")

    episode_path = latest_episode_path(agent_cfg["logger"]["log_dir"])
    if episode_path is not None:
        summary = summarize_episode(
            episode_path,
            scenario_id=str(scenario["id"]),
            expected_object=expected_object,
            actual_object=actual_object,
        )
        print("\n========== ORACLE SUMMARY ==========")
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        print(f"episode: {episode_path}")
    else:
        logger.warning("[fixed_eval] no episode JSON found in %s", agent_cfg["logger"]["log_dir"])
    return 0 if result.success else 1


def _label_key(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


if __name__ == "__main__":
    sys.exit(main())
