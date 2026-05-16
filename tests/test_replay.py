"""4 层契约 replay 测试 (F7)。

L1 终态: success 一致
L2 action 集合: action.kind 集合一致
L3 步数同量级: 不超过 golden 的 1.5x
L4 zoom 命中: 若 golden 触发了 zoom_in re_observe, replay 也必须

Golden 数据由真 sim 录制 (Phase 14.2 / v1.1) 或手工编排 (Phase 14.3)。
"""
import glob
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from PIL import Image


GOLDEN_DIR = Path(__file__).parent / "episodes" / "golden"
GOLDEN_DIR.mkdir(parents=True, exist_ok=True)


def _make_test_image() -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.new("RGB", (256, 256), (200, 100, 50)).save(tmp.name)
    return tmp.name


_TEST_IMAGE = _make_test_image()


class FakeVPLib:
    def __init__(self, n: int = 3) -> None:
        self.viewpoints = [
            type("VP", (), {"name": f"v{i}"})() for i in range(n)
        ]

    def __len__(self) -> int:
        return len(self.viewpoints)

    def __getitem__(self, i: int):
        return self.viewpoints[i]

    def __iter__(self):
        return iter(self.viewpoints)


class FakeEnv:
    """最小 env 实现, 满足 perception/grasp_planner/action_executor 接口。"""

    def observe(self, vp):
        return type("O", (), {"image_path": _TEST_IMAGE})()

    def viewpoint_intrinsics(self, vp):
        return None

    def is_reachable(self, p, d):
        return True

    def move_to_pre_grasp(self, c):
        return True

    def descend(self, p, target_label=None, **kwargs):
        return True, float(np.asarray(p)[2])

    def approach(self, p, approach_dir, target_label=None, **kwargs):
        return True, float(np.asarray(p)[2])

    def close_gripper(self, target_label=None):
        return True

    def open_gripper(self):
        return True

    def lift(self, height_m: float = 0.10, **kwargs):
        return True, 0.05

    def get_eef_pos(self):
        return np.array([0.5, 0, 0.95])

    def move_arm_to(self, p, **kw):
        return True

    def eye_in_hand_viewpoint(self):
        return type("VP", (), {"name": "eye_in_hand"})()

    def _get_obj_type_map(self):
        return {"obj_main": "apple"}


class _SmartVLM:
    """根据 prompt 关键字路由到 ground/zoom/verify mock 池。

    perception 用一个 vlm 实例同时调 observe/zoom/verify; replay 时
    各自的 mock pool 不能复用同一 _MockFromRecord。
    """

    def __init__(self, ground, zoom, verify) -> None:
        self.ground = ground
        self.zoom = zoom
        self.verify = verify

    def describe(self, image_path: str, prompt: str = "") -> str:
        # zoom_disambiguate.txt: "这是放大裁切的物体特写"
        if "放大裁切" in prompt or "alternatives_top3" in prompt or "Zoom" in prompt:
            return self.zoom.describe(image_path, prompt)
        # verify_grasp.txt: "这是 eye-in-hand 相机看到的画面"
        if "eye-in-hand" in prompt or "is_match" in prompt or "夹爪当前" in prompt:
            return self.verify.describe(image_path, prompt)
        # default: ground
        return self.ground.describe(image_path, prompt)


def _make_test_factory(mocks: dict):
    """构造 replay agent: 把 vlm/llm 替换成 record-based mock。"""
    from src.action_executor import ActionExecutor
    from src.active_planner import ActiveViewpointSelector
    from src.agent import EmboSightAgent
    from src.grasp_planner import GraspPlanner
    from src.perception import QueryAwareGrounder
    from src.safety_gate import SafetyClassifier
    from src.task_decomposer import TaskDecomposer
    from src.user_channel import FakeUserChannel
    from src.vlm_cache import VLMCache

    vp_lib = FakeVPLib()
    cache = VLMCache(max_size=0)
    smart_vlm = _SmartVLM(
        ground=mocks["vlm_ground"],
        zoom=mocks["vlm_zoom"],
        verify=mocks["vlm_verify"],
    )
    return (
        EmboSightAgent(
            task_decomposer=TaskDecomposer(mocks["llm_decompose"]),
            perception=QueryAwareGrounder(
                vlm=smart_vlm,
                llm=mocks["llm_decompose"],
                cache=cache, label_temperature=1.0,
                viewpoint_lib=vp_lib,
            ),
            safety_classifier=SafetyClassifier(llm=mocks["llm_safety"]),
            grasp_planner=GraspPlanner(vlm=smart_vlm, env=FakeEnv()),
            action_executor=ActionExecutor(scene_describer=None),
            nbv_selector=ActiveViewpointSelector(
                llm=mocks["llm_decompose"], viewpoint_lib=vp_lib,
            ),
            user_channel=FakeUserChannel.from_explicit(
                mocks["user_answer"], "apple",
            ),
            episode_logger=None,
            viewpoint_lib=vp_lib,
            llm=mocks["llm_decompose"],
            vlm=smart_vlm,
        ),
        FakeEnv(),
    )


GOLDEN_FILES = sorted(glob.glob(str(GOLDEN_DIR / "*.json")))


@pytest.mark.skipif(not GOLDEN_FILES, reason="no golden episodes yet")
@pytest.mark.parametrize("episode_path", GOLDEN_FILES)
def test_replay_decision_consistency(episode_path: str) -> None:
    """4 层契约 (F7): L1/L2/L3/L4。"""
    from src.episode_logger import EpisodeLogger
    record = EpisodeLogger.load(episode_path)
    result = EpisodeLogger.replay(episode_path, _make_test_factory)

    # L1: 终态 success 一致
    if record.final_result is not None:
        assert result.success == record.final_result.success, (
            f"L1: success differs (expected={record.final_result.success}, "
            f"got={result.success})"
        )

    # L2: action.kind 集合一致 (replay 的 superset OK, 但应包含 golden 全部)
    golden_kinds = {a.kind for a in record.actions}
    actual_kinds = {a.kind for a in result.action_history}
    missing = golden_kinds - actual_kinds
    assert not missing, (
        f"L2: replay missing actions {missing}. "
        f"golden={golden_kinds}, actual={actual_kinds}"
    )

    # L3: 步数同量级 (仅当 golden 是 success 时严格;
    # 失败 episode 通常以 max_steps 结尾, 步数比对无意义)
    if record.final_result is not None and record.final_result.success:
        upper = max(int(len(record.actions) * 1.5), len(record.actions) + 3)
        assert len(result.action_history) <= upper, (
            f"L3: step count blew up: "
            f"{len(result.action_history)} > {upper} "
            f"(golden={len(record.actions)})"
        )

    # L4: zoom 命中
    golden_has_zoom = any(
        a.kind == "re_observe" and a.strategy == "zoom_in"
        for a in record.actions
    )
    if golden_has_zoom:
        assert any(
            a.kind == "re_observe" and a.strategy == "zoom_in"
            for a in result.action_history
        ), "L4: golden zoomed but replay didn't"


def test_replay_module_loads() -> None:
    """smoke: replay/_MockFromRecord 不会 import error。"""
    from src.episode_logger import EpisodeLogger, _MockFromRecord
    assert hasattr(EpisodeLogger, "replay")
    assert _MockFromRecord is not None
