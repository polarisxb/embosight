"""scripts/run_agent.py 的 import-only 测试 (确保脚本不语法错误)。

不真 exec (内有 argparse + 真实 backend 构造); 仅验证 spec 可加载。
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_run_agent_module_loads():
    spec = importlib.util.spec_from_file_location(
        "run_agent",
        str(Path(__file__).parent.parent / "scripts" / "run_agent.py"),
    )
    assert spec is not None
    assert spec.loader is not None


def test_run_agent_argparse_exists():
    """argparse 配置可读 (不实际 parse, 只读 source)。"""
    src = (Path(__file__).parent.parent / "scripts" / "run_agent.py").read_text(
        encoding="utf-8",
    )
    assert "--query" in src
    assert "--config" in src
    assert "--agent-config" in src
    assert "--user-mode" in src
