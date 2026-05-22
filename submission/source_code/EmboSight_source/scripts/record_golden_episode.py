"""录制一个 golden episode (真 sim 跑 agent + EpisodeLogger)。

Usage:
    python scripts/record_golden_episode.py --query "我要那个削皮器" \\
        --output tests/episodes/golden/02_zoom_disambiguate_peeler.json

依赖: 真实 LLM/VLM/RoboCasa (即 scripts/run_agent.py 能跑)。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record a golden episode by running run_agent and copying output.",
    )
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--output", required=True,
        help="目标 golden 文件路径, 例如 tests/episodes/golden/02_xxx.json",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--agent-config", default="configs/agent.yaml")
    parser.add_argument("--user-mode", default="fake_from_robocasa")
    args = parser.parse_args()

    # 跑 run_agent.py 录到 logs/episodes/
    cmd = [
        sys.executable, "scripts/run_agent.py",
        "--query", args.query,
        "--config", args.config,
        "--agent-config", args.agent_config,
        "--user-mode", args.user_mode,
    ]
    print(f"[record] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode not in (0, 1):
        # 0 = success, 1 = give_up — 都算合法 episode
        print(f"[record] run_agent.py failed with code {proc.returncode}")
        return proc.returncode

    log_dir = REPO_ROOT / "logs" / "episodes"
    candidates = sorted(log_dir.glob("episode_*.json"))
    if not candidates:
        print(f"ERROR: no episode generated in {log_dir}")
        return 2
    latest = candidates[-1]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(latest, out)
    print(f"[record] golden → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
