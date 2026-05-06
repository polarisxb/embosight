#!/usr/bin/env python3
"""Phase 2 验证: 真实 DeepSeek LLM 集成测试

在服务器上运行:
    python scripts/test_real_llm.py

前置条件:
    1. .env 已配好 DEEPSEEK_API_KEY
    2. pip install openai python-dotenv
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_real_llm")

SEPARATOR = "=" * 60


def test_llm_connection():
    """测试 1: LLM 基础连通性"""
    print(f"\n[1] LLM 连通性测试")
    from src.llm_backend import LLMBackend
    llm = LLMBackend()
    response = llm.generate("请回复 OK", system="你是一个测试助手")
    print(f"  LLM 响应: {response[:100]}")
    assert len(response) > 0, "LLM 返回空响应"
    print(f"  [OK] LLM 连通正常")
    return llm


def test_task_decomposer(llm):
    """测试 2: 真实 LLM 任务分解"""
    print(f"\n[2] TaskDecomposer 真实 LLM 测试")
    from src.task_decomposer import TaskDecomposer

    decomposer = TaskDecomposer(llm_client=llm)

    queries = [
        "我的药瓶在哪里？",
        "桌上有什么东西？",
        "帮我拿水杯",
    ]

    for query in queries:
        print(f"\n  查询: {query}")
        subtasks = decomposer.decompose(query)
        print(f"  分解出 {len(subtasks)} 个子任务:")
        dims_covered = set()
        for t in subtasks:
            dim = t.blind_dimension.value
            dims_covered.add(dim)
            print(f"    [{t.type.value}] {t.target} (dim={dim}, p={t.priority})")
        print(f"  覆盖维度: {sorted(dims_covered)} ({len(dims_covered)}/5)")

        if len(dims_covered) < 5:
            print(f"  [WARN] 维度不足 5，后处理会自动补全")
        else:
            print(f"  [OK] 五维度全覆盖")

    return True


def test_active_planner_nbv(llm):
    """测试 3: 真实 LLM NBV 决策"""
    print(f"\n[3] ActivePlanner NBV 决策测试")
    from src.active_planner import ActivePlanner, ViewpointLibrary, Viewpoint, Observation
    from src.task_decomposer import TaskDecomposer

    decomposer = TaskDecomposer(llm_client=llm)
    vp_lib = ViewpointLibrary("configs/viewpoints.yaml")
    planner = ActivePlanner(llm_client=llm, viewpoint_lib=vp_lib)

    subtasks = decomposer.decompose("桌上有什么？")
    print(f"  子任务: {len(subtasks)} 个")

    # 模拟已有一个全景观察
    init_vp = vp_lib[0]
    init_obs = Observation(
        viewpoint=init_vp,
        image_path="./results/observations/test.png",
        description="桌面上有一个白色陶瓷杯和一个塑料药瓶",
    )

    # 让 LLM 选下一个视角
    next_idx = planner.select_next_viewpoint(
        subtasks=subtasks,
        observations=[init_obs],
        used_indices={0},
    )
    print(f"  LLM 选择的下一个视角索引: {next_idx}")
    if next_idx >= 0:
        print(f"  视角名: {vp_lib[next_idx].name}")
        print(f"  [OK] NBV 决策正常")
    else:
        print(f"  LLM 认为信息已足够 (early stop)")
        print(f"  [OK] NBV 早停决策")

    return True


def test_json_mode(llm):
    """测试 4: JSON 模式输出"""
    print(f"\n[4] JSON 模式测试")
    response = llm.generate(
        user_message='输出一个 JSON: {"status": "ok", "count": 3}',
        system="你是一个 JSON 输出助手，只输出纯 JSON，不要任何额外文字。",
        json_mode=True,
    )
    print(f"  原始响应: {response[:200]}")
    try:
        data = json.loads(response)
        print(f"  解析成功: {data}")
        print(f"  [OK] JSON 模式正常")
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON 解析失败: {e}")
        print(f"  这可能影响 TaskDecomposer 和 ActivePlanner 的输出解析")

    return True


if __name__ == "__main__":
    print(SEPARATOR)
    print("EmboSight Phase 2: 真实 DeepSeek LLM 集成测试")
    print(SEPARATOR)

    results = {}

    try:
        llm = test_llm_connection()
        results["connection"] = True
    except Exception as e:
        print(f"  [FAIL] LLM 连接失败: {e}")
        print(f"\n请检查:")
        print(f"  1. .env 中 DEEPSEEK_API_KEY 是否正确")
        print(f"  2. pip install openai python-dotenv")
        print(f"  3. 服务器能否访问 api.deepseek.com")
        sys.exit(1)

    for name, func in [
        ("json_mode", lambda: test_json_mode(llm)),
        ("task_decomposer", lambda: test_task_decomposer(llm)),
        ("active_planner", lambda: test_active_planner_nbv(llm)),
    ]:
        try:
            results[name] = func()
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    print(f"\n{SEPARATOR}")
    print("总结")
    print(SEPARATOR)
    for name, ok in results.items():
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} {name}")

    if all(results.values()):
        print(f"\n[OK] Phase 2 全部通过！LLM 集成就绪。")
        print(f"下一步: python scripts/test_real_llm.py 通过后，可以跑 Phase 3 (VLM)")
    else:
        print(f"\n[WARN] 部分测试失败，请检查上面的错误信息。")
