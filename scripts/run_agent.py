"""EmboSight Agent v1 入口: 从 query 开始跑一个 episode。

Usage:
    python scripts/run_agent.py --query "拿苹果" \
        --config configs/default.yaml \
        --agent-config configs/agent.yaml

Backend 构造签名 (2026-05-08 已 grep 验证):
    LLMBackend(api_key, base_url, model, max_tokens, temperature, timeout)
    VLMBackend(model_id, device, torch_dtype, max_new_tokens, cache_dir)
    EnvWrapper(EnvConfig(env_name, robots, image_width, image_height, camera_names, ...))
    ViewpointLibrary(config_path)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_dotenv

# 优先加载项目根 .env (含 DEEPSEEK_API_KEY 等)
load_dotenv(str(Path(__file__).parent.parent / ".env"))

from src.action_executor import ActionExecutor  # noqa: E402
from src.active_planner import ActiveViewpointSelector, ViewpointLibrary  # noqa: E402
from src.agent import EmboSightAgent  # noqa: E402
from src.env_wrapper import EnvConfig, EnvWrapper  # noqa: E402
from src.episode_logger import EpisodeLogger  # noqa: E402
from src.grasp_planner import GraspPlanner  # noqa: E402
from src.llm_backend import LLMBackend  # noqa: E402
from src.perception import QueryAwareGrounder  # noqa: E402
from src.clip_scorer import CLIPScorer  # noqa: E402
from src.safety_gate import SafetyClassifier  # noqa: E402
from src.task_decomposer import TaskDecomposer  # noqa: E402
from src.user_channel import CLIUserChannel, FakeUserChannel  # noqa: E402
from src.vlm_backend import VLMBackend  # noqa: E402
from src.vlm_cache import VLMCache  # noqa: E402


def _build_llm(cfg: dict) -> LLMBackend:
    llm_cfg = cfg.get("llm", {})
    return LLMBackend(
        api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=llm_cfg.get("base_url"),
        model=llm_cfg.get("model", "deepseek-chat"),
        max_tokens=llm_cfg.get("max_tokens", 2048),
        temperature=llm_cfg.get("temperature", 0.1),
        timeout=llm_cfg.get("timeout", 60.0),
    )


def _build_vlm(cfg: dict) -> VLMBackend:
    vlm_cfg = cfg.get("vlm", {})
    return VLMBackend(
        model_id=vlm_cfg.get("model_id", "./checkpoints/Qwen2.5-VL-7B-Instruct"),
        device=vlm_cfg.get("device", "cuda"),
        torch_dtype=vlm_cfg.get("torch_dtype", "bfloat16"),
        max_new_tokens=vlm_cfg.get("max_new_tokens", 1024),
    )


def _build_env(cfg: dict) -> EnvWrapper:
    sim_cfg = cfg.get("simulator", {})
    cams = sim_cfg.get("camera_names")
    env_cfg = EnvConfig(
        env_name=sim_cfg.get("env_name", "PickPlaceCounterToCabinet"),
        robots=sim_cfg.get("robots", "PandaMobile"),
        image_width=sim_cfg.get("image_width", 256),
        image_height=sim_cfg.get("image_height", 256),
        camera_names=tuple(cams) if cams else EnvConfig.__dataclass_fields__["camera_names"].default,
    )
    return EnvWrapper(env_cfg)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="EmboSight Agent v1 entry point",
    )
    parser.add_argument("--query", required=True, help="用户自然语言指令")
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="顶层 LLM/VLM/sim 配置",
    )
    parser.add_argument(
        "--agent-config", default="configs/agent.yaml",
        help="agent v1 阈值/perception/cache/logger 配置",
    )
    parser.add_argument(
        "--user-mode", default="fake_from_robocasa",
        choices=["fake_from_robocasa", "fake_from_query", "cli"],
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    top_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    agent_cfg = yaml.safe_load(Path(args.agent_config).read_text(encoding="utf-8"))

    # 实例化依赖
    llm = _build_llm(top_cfg)
    vlm = _build_vlm(top_cfg)
    cache = VLMCache(max_size=agent_cfg["cache"]["max_size"])
    env = _build_env(top_cfg)
    vp_lib = ViewpointLibrary(
        config_path=top_cfg.get("viewpoints_path", "configs/viewpoints.yaml"),
    )

    # 必须先 env.reset() 才能读到真实 ep_meta (fake_from_robocasa 依赖)
    # 否则 _get_obj_type_map 返空 dict, intent 退化为 "我想要那个 unknown"
    env.reset()

    # fake_from_robocasa 模式: 读 sim 实际物体, 用它替换 query 中的物体名,
    # 避免 "query说要 apple, sim 里是 lettuce" 的失配。
    query = args.query
    if args.user_mode == "fake_from_robocasa":
        try:
            tmap = env._get_obj_type_map()
            real_obj = tmap.get("obj_main")
            if real_obj and real_obj != "unknown":
                import logging as _lg
                _lg.getLogger(__name__).info(
                    f"[run_agent] sim real target={real_obj!r}, "
                    f"rewriting query to match (was: {query!r})"
                )
                query = f"pick up the {real_obj}"
        except Exception as e:
            import logging as _lg
            _lg.getLogger(__name__).warning(
                f"[run_agent] failed to read sim obj_type: {e}; "
                f"using original query {query!r}"
            )

    # User channel
    if args.user_mode == "fake_from_robocasa":
        user_channel = FakeUserChannel.from_robocasa(llm, env)
    elif args.user_mode == "fake_from_query":
        user_channel = FakeUserChannel.from_query(llm, query)
    else:
        user_channel = CLIUserChannel()

    clip_scorer = CLIPScorer(device="cpu")
    agent = EmboSightAgent(
        task_decomposer=TaskDecomposer(llm),
        perception=QueryAwareGrounder(
            vlm=vlm, llm=llm, cache=cache,
            label_temperature=agent_cfg["perception"]["label_temperature"],
            ground_prompt_path=agent_cfg["perception"]["ground_prompt"],
            zoom_prompt_path=agent_cfg["perception"]["zoom_prompt"],
            parallax_prompt_path=agent_cfg["perception"]["parallax_prompt"],
            pose_prompt_path=agent_cfg["perception"]["pose_prompt"],
            verify_prompt_path=agent_cfg["perception"]["verify_prompt"],
            viewpoint_lib=vp_lib,
            clip_scorer=clip_scorer,
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

    result = agent.run(query, env)
    print("\n========== EPISODE RESULT ==========")
    print(f"success: {result.success}")
    print(f"speech : {result.speech}")
    print(f"steps  : {result.n_steps}")
    print(f"time   : {result.elapsed_seconds:.1f}s")
    if not result.success:
        print(f"reason : {result.failure_reason}")
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
