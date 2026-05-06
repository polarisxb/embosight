"""端到端具身能力测试

用法:
    python scripts/test_embodied.py --query "帮我拿药瓶"
    python scripts/test_embodied.py --query "桌上有什么"
    python scripts/test_embodied.py --visualize  # 开 MuJoCo viewer
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("test_embodied")


def test_arm_control(env) -> None:
    """测试手臂移动收敛性"""
    log.info("=== Test 1: arm control ===")
    start = env.get_eef_pos()
    target = start + np.array([0.0, 0.0, 0.05])
    ok = env.move_arm_to(target)
    end = env.get_eef_pos()
    dist = float(np.linalg.norm(end - target))
    log.info(f"  start={start}, end={end}, ok={ok}, dist={dist:.4f}m")
    assert ok, f"move_arm_to failed, dist={dist:.4f}m"
    log.info("  PASS")


def test_grounding(env, target_name: str = "药瓶"):
    """测试 object grounding"""
    log.info(f"=== Test 2: ground_object('{target_name}') ===")
    g = env.ground_object(target_name)
    if g is not None:
        log.info(f"  grounding: {g}")
        log.info("  PASS")
    else:
        log.warning(f"  WARN: '{target_name}' not found, trying 'bowl'")
        g = env.ground_object("bowl")
        if g is not None:
            log.info(f"  fallback grounding: {g}")
    return g


def test_grasp(env, grounding) -> bool:
    """测试抓取"""
    log.info("=== Test 3: grasp_at ===")
    if grounding is None:
        log.warning("  SKIP: no grounding available")
        return False
    ok = env.grasp_at(grounding.position_m)
    log.info(f"  grasp ok={ok}")
    return ok


def test_observe_refresh(env) -> None:
    """测试 observe 刷新"""
    log.info("=== Test 4: observe refresh ===")
    vp = env.eye_in_hand_viewpoint()
    obs1 = env.observe(vp)
    env.move_arm_to(env.get_eef_pos() + np.array([0.0, 0.05, 0.0]))
    obs2 = env.observe(vp)
    log.info(f"  img1={obs1.image_path}")
    log.info(f"  img2={obs2.image_path}")
    assert obs1.image_path != obs2.image_path, "同一路径，刷新可能失败"
    log.info("  PASS")


def test_pipeline(query: str, config: str, visualize: bool) -> dict:
    """端到端 pipeline 测试"""
    log.info(f"=== Test 5: full pipeline | query='{query}' ===")
    from src.env_wrapper import EnvConfig, EnvWrapper
    from src.pipeline import EmboSightPipeline
    from src.utils import load_dotenv

    load_dotenv()

    env = EnvWrapper(EnvConfig(has_renderer=visualize))
    env.reset()

    try:
        # 子测试
        test_arm_control(env)
        g = test_grounding(env)
        test_observe_refresh(env)

        # 完整 pipeline
        pipeline = EmboSightPipeline(config)
        result = pipeline.run(query, env)

        out = Path("results/test_embodied_result.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        log.info(f"  saved: {out}")
        log.info(f"  speech: {result['speech'][:200]}")
        log.info(f"  action_plan: {result.get('action_plan')}")
        log.info(f"  action_result: {result.get('action_result')}")
        log.info("  PASS")
        return result
    finally:
        env.close()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="EmboSight 具身能力测试")
    parser.add_argument("--query", default="帮我拿药瓶")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    test_pipeline(args.query, args.config, args.visualize)


if __name__ == "__main__":
    main()
