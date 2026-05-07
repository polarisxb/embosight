"""SafeQuery-VLM 集成测试脚本 (Phase 2-5 端到端验证)

在服务器上运行, 验证完整信息流:
    VLM open-listing → query match → 3D projection → SceneModel → SafetyGate → grasp

运行:
    MUJOCO_GL=egl PYTHONUNBUFFERED=1 python scripts/test_safequery_integration.py
    MUJOCO_GL=egl python scripts/test_safequery_integration.py --query "帮我拿杯子"
    MUJOCO_GL=egl python scripts/test_safequery_integration.py --query "cup" --layout -1 --style -1
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("safequery_test")


def main():
    parser = argparse.ArgumentParser(description="SafeQuery-VLM integration test")
    parser.add_argument("--query", default="帮我拿削皮器", help="User query")
    parser.add_argument("--query-gt", action="store_true",
                        help="Auto-generate query targeting obj_main from GT")
    parser.add_argument("--layout", type=int, default=None, help="Layout ID (-1=random)")
    parser.add_argument("--style", type=int, default=None, help="Style ID (-1=random)")
    parser.add_argument("--cameras", default="robot0_agentview_center,robot0_agentview_left,robot0_agentview_right",
                        help="Comma-separated camera names")
    parser.add_argument("--skip-grasp", action="store_true", help="Skip grasp execution")
    args = parser.parse_args()

    cameras = tuple(args.cameras.split(","))

    # ================================================================
    # 1. 初始化环境
    # ================================================================
    from src.env_wrapper import EnvWrapper, EnvConfig

    cfg = EnvConfig(
        camera_names=cameras,
        layout_ids=args.layout,
        style_ids=args.style,
    )
    env = EnvWrapper(cfg)
    obs = env.reset()
    logger.info(f"Environment reset. action_dim={env._env.action_dim}")

    # GT 类别
    gt_types = env._get_obj_type_map()
    logger.info(f"GT object types: {gt_types}")

    # --query-gt: 自动生成查询 (目标 obj_main)
    if args.query_gt:
        obj_main_type = gt_types.get('obj_main', '')
        # 反向查 alias: 英文 → 中文
        import yaml
        alias_path = Path("configs/object_aliases.yaml")
        zh_name = obj_main_type  # fallback to English
        if alias_path.exists():
            with open(alias_path, "r", encoding="utf-8") as f:
                alias_data = yaml.safe_load(f)
            for zh, en_list in alias_data.get("aliases", {}).items():
                if obj_main_type in en_list or obj_main_type == zh:
                    zh_name = zh
                    break
        args.query = f"帮我拿{zh_name}"
        logger.info(f"[--query-gt] obj_main='{obj_main_type}' → query='{args.query}'")

    # ================================================================
    # 2. 初始化 VLM + Grounding + Safety
    # ================================================================
    from src.vlm_backend import VLMBackend
    from src.vlm_grounding import VLMGrounder
    from src.scene_model import SceneModel
    from src.safety_gate import SafetyGate

    vlm = VLMBackend()

    # LLM 用于 Level 5 语义匹配 fallback (当文本启发式失败时)
    llm = None
    try:
        from src.llm_backend import LLMBackend
        llm = LLMBackend(max_tokens=256, temperature=0.0)
        logger.info("LLM backend loaded (for semantic matching fallback)")
    except Exception as e:
        logger.warning(f"LLM backend unavailable: {e} (will skip Level 5)")

    grounder = VLMGrounder(vlm, llm_backend=llm)
    safety = SafetyGate()

    # ================================================================
    # 3. 多视角 VLM Grounding
    # ================================================================
    scene = SceneModel()

    for cam in cameras:
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing camera: {cam}")
        logger.info(f"{'='*40}")

        # 获取 RGB 图像
        img_key = f"{cam}_image"
        img = obs.get(img_key)
        if img is None:
            logger.warning(f"  No image for {cam}, skip")
            continue

        # 保存图像
        import imageio.v2 as imageio
        img_path = f"results/observations/safequery_{cam}.png"
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        imageio.imwrite(img_path, img)
        logger.info(f"  Image saved: {img_path}")

        # VLM Grounding (Prompt D 开放式检测)
        candidates = grounder.ground(img_path)
        logger.info(f"  VLM detected {len(candidates)} candidates:")
        for c in candidates:
            cat_str = f" cat={c.likely_category}" if c.likely_category else ""
            logger.info(f"    {c.label}: bbox={c.bbox_2d} conf={c.confidence}{cat_str} feat={c.visible_features[:50]}")

        # Query 匹配
        candidates = grounder.match_query(candidates, args.query, gt_types)
        logger.info(f"  After query match ('{args.query}'):")
        for c in candidates:
            logger.info(f"    {c.label}: score={c.query_match_score:.2f} method={c.match_method} cat={c.matched_category}")

        # 3D 投影
        projector = env.make_projector(cam)
        if projector is None:
            logger.warning(f"  Projector unavailable for {cam}")

        # 加入 SceneModel
        scene.add_view(cam, candidates, projector)

    # ================================================================
    # 4. SceneModel 结果
    # ================================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"SCENE MODEL RESULTS ({len(scene)} objects)")
    logger.info(f"{'='*60}")

    for obj in scene.objects:
        logger.info(
            f"  {obj.object_id}: '{obj.label}' "
            f"pos={obj.position_3d} conf={obj.position_confidence:.2f} "
            f"views={obj.observed_in_views} "
            f"match_score={obj.query_match_score:.2f} method={obj.match_method}"
        )

    # ================================================================
    # 5. Safety Gate
    # ================================================================
    logger.info(f"\n{'='*60}")
    logger.info("SAFETY GATE CHECK")
    logger.info(f"{'='*60}")

    for obj in scene.objects:
        safety.update_object_safety(obj)

    best = scene.get_best_match()
    if best is not None:
        decision = safety.check(best)
        logger.info(f"  Best match: '{best.label}' (score={best.query_match_score:.2f})")
        logger.info(f"  Risk: {best.safety_risk} — {best.safety_reason}")
        logger.info(f"  Decision: allow={decision.allow_execute}")
        logger.info(f"  User msg: {decision.reason_user}")
        logger.info(f"  Log msg: {decision.reason_log}")
        if decision.extra_warnings:
            logger.info(f"  Extra warnings: {decision.extra_warnings}")
    else:
        logger.warning(f"  No match found for query: {args.query}")

    # ================================================================
    # 6. 可选: 执行抓取
    # ================================================================
    if not args.skip_grasp and best is not None:
        decision = safety.check(best)
        if decision.allow_execute:
            logger.info(f"\n{'='*60}")
            logger.info("GRASP EXECUTION")
            logger.info(f"{'='*60}")

            # 获取 body name
            grounding = env.ground_object(args.query, allow_fallback=True)
            body_name = grounding.sim_body_name if grounding else "obj_main"
            target_pos = best.position_3d

            if best.position_confidence < 0.3 and grounding:
                target_pos = np.asarray(grounding.position_m, dtype=np.float32)
                logger.info(f"  Low 3D conf, using env grounding: {target_pos}")

            logger.info(f"  Target: body={body_name} pos={target_pos}")
            grasp_ok = env.grasp_at(
                np.asarray(target_pos, dtype=np.float32),
                target_body=body_name,
            )
            logger.info(f"  Grasp result: {grasp_ok}")
        else:
            logger.info(f"  SKIPPED (safety rejected): {decision.reason_user}")

    # ================================================================
    # 7. 总结
    # ================================================================
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Query: {args.query}")
    logger.info(f"  GT types: {gt_types}")
    logger.info(f"  Scene objects: {len(scene)}")
    if best:
        logger.info(f"  Best match: '{best.label}' score={best.query_match_score:.2f} risk={best.safety_risk}")
    else:
        logger.info(f"  Best match: NONE")
    logger.info(f"  Cameras: {cameras}")

    env.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
