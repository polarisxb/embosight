"""探测 RoboCasa 环境中 obj_main / distr_*_main 的真实类型元数据.

目的: grounding 模块目前 100% 依赖 body name, 但 body name 只是通用容器名
(obj_main / distr_counter_main / distr_cab_main), 不含类型信息. 需要找到
RoboCasa 暴露类型的 API.

运行: MUJOCO_GL=egl python scripts/probe_robocasa_obj.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env_wrapper import EnvWrapper, EnvConfig


def inspect(obj, name, max_depth=1, prefix=""):
    """打印对象属性, 只看一层 (避免递归爆炸)"""
    print(f"\n{prefix}--- {name} ({type(obj).__name__}) ---")
    if obj is None:
        print(f"{prefix}  None")
        return
    attrs = [a for a in dir(obj) if not a.startswith("_")]
    interesting = [
        a for a in attrs
        if any(kw in a.lower() for kw in [
            "obj", "cat", "cfg", "model", "name", "class", "type",
            "fixture", "item"
        ])
    ]
    for a in interesting[:40]:
        try:
            v = getattr(obj, a)
            if callable(v):
                continue
            s = str(v)
            if len(s) > 200:
                s = s[:200] + "..."
            print(f"{prefix}  .{a} = {s}")
        except Exception as e:
            print(f"{prefix}  .{a} = <ERROR: {e}>")


def main():
    env = EnvWrapper(EnvConfig())
    env.reset()
    base = env._env

    print("=" * 60)
    print("ROBOCASA OBJECT METADATA PROBE")
    print("=" * 60)

    # 1) 环境本身
    inspect(base, "env._env")

    # 2) task_objs 暴露的 body name
    sim_body_names = list(base.sim.model.body_names)
    task_objs = [b for b in sim_body_names if b.startswith("obj_") or b.startswith("distr_")]
    print(f"\n--- task_objs (body names) ---")
    for b in task_objs:
        print(f"  {b}")

    # 3) 常见的 RoboCasa 属性
    for attr in [
        "objects", "object_cfgs", "fixtures", "_objects",
        "obj_cfgs", "object_name_to_obj", "object_body_names",
        "obj_name_to_body_id", "object_names",
    ]:
        if hasattr(base, attr):
            v = getattr(base, attr)
            print(f"\n--- env.{attr} ({type(v).__name__}) ---")
            if isinstance(v, dict):
                for k, val in list(v.items())[:10]:
                    inspect(val, f"{attr}[{k!r}]", prefix="  ")
            elif isinstance(v, (list, tuple)):
                for i, val in enumerate(v[:10]):
                    inspect(val, f"{attr}[{i}]", prefix="  ")
            else:
                print(f"  {v}")

    # 4) 尝试任务描述 / 语言
    for attr in ["lang", "get_ep_meta", "_ep_meta", "ep_meta"]:
        if hasattr(base, attr):
            v = getattr(base, attr)
            try:
                result = v() if callable(v) else v
                print(f"\n--- env.{attr} ---\n  {result}")
            except Exception as e:
                print(f"  (calling {attr} failed: {e})")

    env.close()


if __name__ == "__main__":
    main()
