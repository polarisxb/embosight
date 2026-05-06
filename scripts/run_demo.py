"""EmboSight Demo 运行脚本

用法:
    python scripts/run_demo.py --query "我的药瓶在哪？"
    python scripts/run_demo.py --query "桌上有什么" --baseline
    python scripts/run_demo.py --query "我面前有什么危险" --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="EmboSight Demo")
    parser.add_argument("--query", type=str, required=True, help="视障者查询")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--baseline", action="store_true", help="跑 baseline（固定视角扫描）")
    parser.add_argument("--output", type=str, default="results/demo.json")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    from src.utils import load_dotenv, setup_logging

    load_dotenv()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("run_demo")

    logger.info("=" * 60)
    logger.info(f"EmboSight Demo")
    logger.info(f"Query:    {args.query}")
    logger.info(f"Config:   {args.config}")
    logger.info(f"Mode:     {'Baseline' if args.baseline else 'Ours'}")
    logger.info("=" * 60)

    from src.env_wrapper import EnvWrapper, EnvConfig
    from src.pipeline import EmboSightPipeline
    from src.utils import load_yaml

    cfg = load_yaml(args.config)
    sim_cfg = cfg.get("simulator", {})
    out_cfg = cfg.get("output", {})

    env_config = EnvConfig(
        sim_type=sim_cfg.get("type", "robocasa"),
        env_name=sim_cfg.get("env_name", "PickPlaceCounterToCabinet"),
        robots=sim_cfg.get("robots", "PandaMobile"),
        image_width=sim_cfg.get("image_width", 256),
        image_height=sim_cfg.get("image_height", 256),
        camera_names=tuple(sim_cfg.get("camera_names", [
            "robot0_agentview_center", "robot0_agentview_left",
            "robot0_agentview_right", "robot0_frontview",
            "robot0_robotview", "robot0_eye_in_hand",
        ])),
        output_dir=out_cfg.get("observation_dir", "./results/observations"),
    )

    pipeline = EmboSightPipeline(args.config)
    env = EnvWrapper(env_config)

    try:
        env.reset()
        result = pipeline.run(args.query, env)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"\n[最终输出（视障者听到的）]\n{result['speech']}")
        logger.info(f"\n完整结果保存至: {output_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()