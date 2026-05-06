"""Mock 端到端 Pipeline 测试（无需 API Key / GPU）

用法:
    python scripts/test_pipeline_mock.py

功能:
    用 Mock LLM / Mock VLM / Mock Env 跑通完整 pipeline，
    验证所有模块接口对齐，确保真实环境就位后可直接切换。
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_mock")

OK = "\033[92m[OK]\033[0m"
ERR = "\033[91m[FAIL]\033[0m"


# ============================================================
# Mock LLM: 模拟 DeepSeek API 返回
# ============================================================

class MockLLMBackend:
    """模拟 LLM 后端，返回固定 JSON"""

    def generate(
        self,
        user_message: str = "",
        system: str = "",
        json_mode: bool = False,
        temperature: float = 0.1,
    ) -> str:
        if "子任务" in user_message or "分解" in user_message or "subtask" in user_message.lower():
            return json.dumps({
                "subtasks": [
                    {
                        "type": "identify",
                        "target": "桌面物体",
                        "priority": 1,
                        "blind_dimension": "position",
                        "output_format": "列出所有物体",
                    },
                    {
                        "type": "locate",
                        "target": "目标物体",
                        "priority": 2,
                        "blind_dimension": "distance",
                        "output_format": "方位+距离",
                    },
                    {
                        "type": "alert",
                        "target": "潜在危险",
                        "priority": 1,
                        "blind_dimension": "safety",
                        "output_format": "安全提示",
                    },
                ]
            }, ensure_ascii=False)

        if "viewpoint_idx" in user_message or "视角" in user_message:
            if json_mode:
                return json.dumps({"viewpoint_idx": -1, "reason": "mock: 信息已足够"})
            return "yes, sufficient"

        return "mock response"


# ============================================================
# Mock VLM: 模拟 Qwen2.5-VL 返回
# ============================================================

class MockVLMBackend:
    """模拟 VLM 后端，返回固定 JSON 描述"""

    def describe(self, image: str, prompt: str = "") -> str:
        return json.dumps({
            "objects": ["白色陶瓷杯", "塑料药瓶", "不锈钢锅"],
            "positions": [
                {"obj": "白色陶瓷杯", "direction": "正前方", "distance_cm": 30, "height_cm": 8},
                {"obj": "塑料药瓶", "direction": "左前方", "distance_cm": 25, "height_cm": 10},
                {"obj": "不锈钢锅", "direction": "右侧", "distance_cm": 45, "height_cm": 15},
            ],
            "tactile": [
                "白色陶瓷杯：光滑陶瓷材质，圆筒形",
                "塑料药瓶：磨砂塑料，圆柱形",
                "不锈钢锅：金属材质，温热",
            ],
            "safety_alerts": ["不锈钢锅可能温热，注意避免烫伤"],
            "actionable_advice": ["可从左侧伸手取药瓶，距您手约25厘米"],
        }, ensure_ascii=False)

    def _ensure_loaded(self):
        pass


# ============================================================
# Mock Env: 模拟仿真环境
# ============================================================

class MockEnvWrapper:
    """模拟仿真环境，不需要 MuJoCo / RoboCasa"""

    def __init__(self):
        self._step = 0
        self.output_dir = "./results/observations"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def reset(self):
        self._step = 0
        logger.info("[MockEnv] 环境重置")
        return {}

    def move_arm_to(self, pose):
        return True

    def observe(self, viewpoint):
        from src.active_planner import Observation

        self._step += 1
        image_path = f"{self.output_dir}/mock_step_{self._step:03d}_{viewpoint.name}.png"

        # 生成一张纯色测试图
        try:
            import numpy as np
            import imageio.v2 as imageio

            color = [(200, 100, 100), (100, 200, 100), (100, 100, 200),
                     (200, 200, 100), (200, 100, 200), (100, 200, 200)]
            c = color[self._step % len(color)]
            img = np.full((256, 256, 3), c, dtype=np.uint8)
            imageio.imwrite(image_path, img)
        except ImportError:
            Path(image_path).touch()

        return Observation(viewpoint=viewpoint, image_path=image_path)

    def close(self):
        logger.info("[MockEnv] 环境关闭")


# ============================================================
# 测试函数
# ============================================================

def test_task_decomposer():
    """测试 TaskDecomposer 接口"""
    from src.task_decomposer import TaskDecomposer

    llm = MockLLMBackend()
    decomposer = TaskDecomposer(llm)
    subtasks = decomposer.decompose("我的药瓶在哪里？")

    assert len(subtasks) > 0, "subtasks 为空"
    for t in subtasks:
        assert hasattr(t, "type"), "subtask 缺少 type"
        assert hasattr(t, "target"), "subtask 缺少 target"
        assert hasattr(t, "blind_dimension"), "subtask 缺少 blind_dimension"

    print(f"  {OK} TaskDecomposer: {len(subtasks)} 个子任务")
    for t in subtasks:
        print(f"      [{t.type.value}] {t.target} (dim={t.blind_dimension.value})")
    return subtasks


def test_active_planner(subtasks):
    """测试 ActivePlanner 接口"""
    from src.active_planner import ActivePlanner, ViewpointLibrary

    llm = MockLLMBackend()
    vp_lib = ViewpointLibrary("configs/viewpoints.yaml")

    assert len(vp_lib) > 0, "视角库为空"
    print(f"  {OK} ViewpointLibrary: {len(vp_lib)} 个视角")

    planner = ActivePlanner(llm_client=llm, viewpoint_lib=vp_lib, max_viewpoints=3)
    env = MockEnvWrapper()
    env.reset()

    observations = planner.plan(subtasks, env)
    assert len(observations) > 0, "observations 为空"

    print(f"  {OK} ActivePlanner: {len(observations)} 个视角观察")
    for obs in observations:
        print(f"      {obs.viewpoint.name} -> {obs.image_path}")
    return observations


def test_scene_describer(observations):
    """测试 SceneDescriber 接口"""
    from src.scene_describer import SceneDescriber

    vlm = MockVLMBackend()
    describer = SceneDescriber(vlm)

    descriptions = []
    for obs in observations:
        desc = describer.describe(
            image_path=obs.image_path,
            viewpoint=obs.viewpoint,
        )
        descriptions.append(desc)
        assert not desc.is_empty(), f"描述为空: {obs.viewpoint.name}"

    print(f"  {OK} SceneDescriber: {len(descriptions)} 个描述")

    final = describer.aggregate(descriptions)
    assert not final.is_empty(), "聚合描述为空"

    speech = final.to_speech()
    assert len(speech) > 0, "语音文本为空"

    print(f"  {OK} 聚合描述: {len(final.objects)} 物体, {len(final.positions)} 位置")
    print(f"  {OK} 语音文本: {speech[:80]}...")
    return final, speech


def test_full_pipeline():
    """测试完整 pipeline（使用 mock 组件）"""
    from src.active_planner import ActivePlanner, ViewpointLibrary
    from src.scene_describer import SceneDescriber
    from src.task_decomposer import TaskDecomposer

    llm = MockLLMBackend()
    vlm = MockVLMBackend()
    env = MockEnvWrapper()

    decomposer = TaskDecomposer(llm)
    vp_lib = ViewpointLibrary("configs/viewpoints.yaml")
    planner = ActivePlanner(llm_client=llm, viewpoint_lib=vp_lib, max_viewpoints=4)
    describer = SceneDescriber(vlm)

    query = "我面前桌上有什么？"
    logger.info(f"Pipeline 查询: {query}")

    # Step 1: 任务分解
    subtasks = decomposer.decompose(query)

    # Step 2: 主动视角规划
    env.reset()
    observations = planner.plan(subtasks, env)

    # Step 3: 场景描述
    descriptions = []
    for obs in observations:
        desc = describer.describe(image_path=obs.image_path, viewpoint=obs.viewpoint, subtasks=subtasks)
        descriptions.append(desc)
        obs.description = desc.to_speech()

    # Step 4: 聚合
    final_desc = describer.aggregate(descriptions)
    speech = final_desc.to_speech()

    # 构建结果
    result = {
        "query": query,
        "subtasks": [s.to_dict() for s in subtasks],
        "observations": [
            {
                "viewpoint": {"name": o.viewpoint.name},
                "image_path": o.image_path,
                "description": o.description,
            }
            for o in observations
        ],
        "description": final_desc.to_dict(),
        "speech": speech,
    }

    # 保存
    output_path = Path("results/mock_demo.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  {OK} Pipeline 端到端: {output_path}")
    print(f"  {OK} 语音输出: {speech}")
    env.close()
    return result


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("EmboSight Mock Pipeline 测试")
    print("=" * 60)

    results = {}

    print("\n[1] TaskDecomposer")
    try:
        subtasks = test_task_decomposer()
        results["task_decomposer"] = True
    except Exception as e:
        print(f"  {ERR} TaskDecomposer: {e}")
        results["task_decomposer"] = False
        subtasks = []

    print("\n[2] ActivePlanner + ViewpointLibrary")
    try:
        observations = test_active_planner(subtasks)
        results["active_planner"] = True
    except Exception as e:
        print(f"  {ERR} ActivePlanner: {e}")
        results["active_planner"] = False
        observations = []

    print("\n[3] SceneDescriber + Aggregate")
    try:
        if observations:
            final, speech = test_scene_describer(observations)
        else:
            raise RuntimeError("无 observations，跳过")
        results["scene_describer"] = True
    except Exception as e:
        print(f"  {ERR} SceneDescriber: {e}")
        results["scene_describer"] = False

    print("\n[4] Full Pipeline (端到端)")
    try:
        result = test_full_pipeline()
        results["pipeline"] = True
    except Exception as e:
        print(f"  {ERR} Pipeline: {e}")
        import traceback
        traceback.print_exc()
        results["pipeline"] = False

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    for name, ok in results.items():
        status = OK if ok else ERR
        print(f"  {status} {name}")

    if all(results.values()):
        print(f"\n{OK} 所有接口验证通过！API Key / VLM 就位后可直接跑真实 pipeline。")
        return 0
    else:
        print(f"\n{ERR} 有失败项，请排查后重试。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
