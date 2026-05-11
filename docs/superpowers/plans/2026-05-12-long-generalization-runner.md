# Long-Run Pure Generalization Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an overnight-safe pure generalization evaluator that runs 50 random-seed discover scenarios with parallelism 4, isolated per-seed memory, checkpoint/resume, per-run logs, and final summaries.

**Architecture:** Add a new orchestration script `eval/run_long_generalization.py` around the existing `eval/run_fixed.py` single-scenario runner. The orchestrator generates scenarios, runs each seed in an isolated subprocess, appends one result per completed seed to `results.jsonl`, and supports resume by skipping completed scenario IDs. Add optional `--memory-dir` support to `run_fixed.py` so pure generalization runs do not read/write the shared project `memory/` directory.

**Tech Stack:** Python standard library (`argparse`, `json`, `subprocess`, `concurrent.futures`, `datetime`, `pathlib`, `time`, `os`, `sys`), PyYAML, pytest, existing EmboSight `MemoryManager` and eval oracle output parsing.

---

## File Structure

- Create: `eval/run_long_generalization.py`
  - Owns long-run orchestration, scenario generation, subprocess execution, JSONL checkpointing, resume, summaries, and CLI.
- Modify: `eval/run_fixed.py`
  - Adds optional `--memory-dir` CLI argument.
  - Adds `create_memory_manager(memory_dir)` helper.
  - Passes `MemoryManager(memory_dir=...)` into `EmboSightAgent` only when provided.
- Create: `tests/test_long_generalization_runner.py`
  - Tests pure helper functions in the new runner without launching RoboCasa.
- Modify: `tests/test_run_fixed_eval.py`
  - Tests `create_memory_manager()` behavior for `None` and a custom directory.

---

### Task 1: Add `--memory-dir` support to `eval/run_fixed.py`

**Files:**
- Modify: `eval/run_fixed.py`
- Modify: `tests/test_run_fixed_eval.py`

- [ ] **Step 1: Write failing tests for memory manager creation**

Append these tests to `tests/test_run_fixed_eval.py`:

```python
def test_create_memory_manager_none_returns_none():
    module = _load_module()

    assert module.create_memory_manager(None) is None


def test_create_memory_manager_uses_custom_dir(tmp_path):
    module = _load_module()

    mm = module.create_memory_manager(tmp_path)

    assert mm is not None
    assert mm.memory_dir == tmp_path
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_run_fixed_eval.py -q
```

Expected before implementation:

```text
AttributeError: module 'run_fixed_eval' has no attribute 'create_memory_manager'
```

- [ ] **Step 3: Add helper in `eval/run_fixed.py`**

Insert this helper after `latest_episode_path()` and before `build_agent()`:

```python
def create_memory_manager(memory_dir: str | Path | None):
    if memory_dir is None:
        return None
    from src.memory_manager import MemoryManager
    return MemoryManager(memory_dir=Path(memory_dir))
```

- [ ] **Step 4: Thread memory manager through `build_agent()`**

Change the signature from:

```python
def build_agent(top_cfg: dict[str, Any], agent_cfg: dict[str, Any], env, user_mode: str, query: str):
```

to:

```python
def build_agent(
    top_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    env,
    user_mode: str,
    query: str,
    memory_dir: str | Path | None = None,
):
```

Inside `build_agent()`, before `return EmboSightAgent(...)`, add:

```python
    memory_manager = create_memory_manager(memory_dir)
```

Then add this keyword to the `EmboSightAgent(...)` call:

```python
        memory_manager=memory_manager,
```

- [ ] **Step 5: Add CLI argument and pass it to `build_agent()`**

In `main()`, add this parser argument after `--allow-object-mismatch`:

```python
    parser.add_argument("--memory-dir", default=None,
                        help="Optional per-run memory directory for isolated eval")
```

Change the `build_agent(...)` call from:

```python
    agent = build_agent(
        top_cfg,
        agent_cfg,
        env,
        str(scenario.get("user_mode", "fake_from_robocasa")),
        query,
    )
```

to:

```python
    agent = build_agent(
        top_cfg,
        agent_cfg,
        env,
        str(scenario.get("user_mode", "fake_from_robocasa")),
        query,
        memory_dir=args.memory_dir,
    )
```

- [ ] **Step 6: Run tests and commit**

Run:

```bash
python -m pytest tests/test_run_fixed_eval.py -q
```

Expected:

```text
6 passed
```

Commit:

```bash
git add eval/run_fixed.py tests/test_run_fixed_eval.py
git commit -m "feat(eval): allow isolated memory dir for fixed runs"
```

---

### Task 2: Add pure helper functions for the long-run runner

**Files:**
- Create: `eval/run_long_generalization.py`
- Create: `tests/test_long_generalization_runner.py`

- [ ] **Step 1: Write failing tests for helper functions**

Create `tests/test_long_generalization_runner.py`:

```python
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


RUNNER_PATH = Path(__file__).parent.parent / "eval" / "run_long_generalization.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_long_generalization", str(RUNNER_PATH))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_seed_scenarios():
    module = _load_module()

    scenarios = module.generate_seed_scenarios(seed_start=101, count=3)

    assert [s["id"] for s in scenarios] == [
        "random_seed_101", "random_seed_102", "random_seed_103",
    ]
    assert [s["seed"] for s in scenarios] == [101, 102, 103]
    assert all(s["query"] == "pick up anything" for s in scenarios)
    assert all(s["expected_object"] is None for s in scenarios)
    assert all(s["user_mode"] == "fake_from_robocasa" for s in scenarios)


def test_load_completed_results_ignores_corrupt_lines(tmp_path):
    module = _load_module()
    path = tmp_path / "results.jsonl"
    path.write_text(
        json.dumps({"scenario_id": "random_seed_101", "success": True}) + "\n"
        "not-json\n"
        + json.dumps({"scenario_id": "random_seed_102", "success": False}) + "\n",
        encoding="utf-8",
    )

    completed = module.load_completed_results(path)

    assert set(completed) == {"random_seed_101", "random_seed_102"}
    assert completed["random_seed_101"]["success"] is True
    assert completed["random_seed_102"]["success"] is False


def test_summarize_results_counts_failures_strategies_objects_and_slowest():
    module = _load_module()
    results = [
        {
            "scenario_id": "random_seed_101", "success": True, "error": None,
            "failure_reason": None, "grasp_failure_mode": None,
            "grasp_strategy": "strategy_top_down", "actual_object": "apple",
            "steps": 4, "time_s": 10.0,
        },
        {
            "scenario_id": "random_seed_102", "success": False, "error": None,
            "failure_reason": "MAX_STEPS reached", "grasp_failure_mode": None,
            "grasp_strategy": "strategy_top_down", "actual_object": "wine",
            "steps": 12, "time_s": 20.0,
        },
        {
            "scenario_id": "random_seed_103", "success": False, "error": "timeout",
            "failure_reason": "timeout", "grasp_failure_mode": None,
            "grasp_strategy": None, "actual_object": None,
            "steps": None, "time_s": 900.0,
        },
    ]

    summary = module.summarize_results(results)

    assert summary["total"] == 3
    assert summary["successes"] == 1
    assert summary["success_rate"] == 1 / 3
    assert summary["errors"] == 1
    assert summary["timeouts"] == 1
    assert summary["failure_breakdown"] == {
        "MAX_STEPS reached": 1,
        "timeout": 1,
    }
    assert summary["strategy_usage"] == {"strategy_top_down": 2}
    assert summary["object_distribution"] == {"apple": 1, "wine": 1}
    assert summary["slowest_runs"][0]["scenario_id"] == "random_seed_103"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_long_generalization_runner.py -q
```

Expected before implementation:

```text
FileNotFoundError: ... eval/run_long_generalization.py
```

- [ ] **Step 3: Create initial `eval/run_long_generalization.py` with helpers**

Create the file with these imports and helper functions:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def generate_seed_scenarios(seed_start: int, count: int) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for seed in range(seed_start, seed_start + count):
        scenarios.append({
            "id": f"random_seed_{seed}",
            "env_name": "PickPlaceCounterToCabinet",
            "seed": seed,
            "query": "pick up anything",
            "expected_object": None,
            "user_mode": "fake_from_robocasa",
            "max_resets": 1,
        })
    return scenarios


def load_completed_results(results_path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not results_path.exists():
        return completed
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            scenario_id = item.get("scenario_id")
            if scenario_id:
                completed[str(scenario_id)] = item
    return completed


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    successes = sum(1 for r in results if r.get("success"))
    errors = sum(1 for r in results if r.get("error"))
    timeouts = sum(1 for r in results if r.get("failure_reason") == "timeout")
    step_values = [r["steps"] for r in results if isinstance(r.get("steps"), int)]
    time_values = [float(r["time_s"]) for r in results if r.get("time_s") is not None]

    failure_breakdown: dict[str, int] = {}
    strategy_usage: dict[str, int] = {}
    object_distribution: dict[str, int] = {}
    failed_runs: list[dict[str, Any]] = []

    for r in results:
        if not r.get("success"):
            reason = (
                r.get("grasp_failure_mode")
                or r.get("failure_reason")
                or r.get("error")
                or "unknown"
            )
            failure_breakdown[str(reason)] = failure_breakdown.get(str(reason), 0) + 1
            failed_runs.append(r)
        strategy = r.get("grasp_strategy")
        if strategy:
            strategy_usage[str(strategy)] = strategy_usage.get(str(strategy), 0) + 1
        actual_object = r.get("actual_object")
        if actual_object:
            object_distribution[str(actual_object)] = object_distribution.get(str(actual_object), 0) + 1

    slowest_runs = sorted(
        results,
        key=lambda r: float(r.get("time_s") or 0.0),
        reverse=True,
    )[:10]

    return {
        "total": total,
        "completed": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "errors": errors,
        "timeouts": timeouts,
        "avg_steps": sum(step_values) / len(step_values) if step_values else 0.0,
        "avg_time_s": sum(time_values) / len(time_values) if time_values else 0.0,
        "failure_breakdown": dict(sorted(failure_breakdown.items(), key=lambda x: (-x[1], x[0]))),
        "strategy_usage": dict(sorted(strategy_usage.items(), key=lambda x: (-x[1], x[0]))),
        "object_distribution": dict(sorted(object_distribution.items(), key=lambda x: (-x[1], x[0]))),
        "slowest_runs": slowest_runs,
        "failed_runs": failed_runs,
    }
```

- [ ] **Step 4: Run helper tests and commit**

Run:

```bash
python -m pytest tests/test_long_generalization_runner.py -q
```

Expected:

```text
3 passed
```

Commit:

```bash
git add eval/run_long_generalization.py tests/test_long_generalization_runner.py
git commit -m "feat(eval): add long generalization runner helpers"
```

---

### Task 3: Implement subprocess execution, parsing, logs, and isolated memory prep

**Files:**
- Modify: `eval/run_long_generalization.py`
- Modify: `tests/test_long_generalization_runner.py`

- [ ] **Step 1: Add tests for parsing and isolated memory setup**

Append these tests to `tests/test_long_generalization_runner.py`:

```python
def test_parse_run_fixed_output_extracts_oracle_and_episode_result():
    module = _load_module()
    stdout = '''
========== EPISODE RESULT ==========
scenario: random_seed_101
success : True
speech  : 我来拿apple
steps   : 4
time    : 12.3s

========== ORACLE SUMMARY ==========
{
  "success": true,
  "failure_reason": null,
  "grasp_failure_mode": "success",
  "grasp_candidate_source": "strategy_top_down",
  "action_sequence": ["observe", "classify_safety", "plan_grasp_candidates", "grasp"],
  "actual_object": "apple"
}
episode: logs/episodes/episode_1.json
'''

    result = module.parse_run_fixed_output(
        scenario_id="random_seed_101",
        seed=101,
        returncode=0,
        stdout=stdout,
        stderr="",
        elapsed=12.34,
    )

    assert result["scenario_id"] == "random_seed_101"
    assert result["seed"] == 101
    assert result["success"] is True
    assert result["steps"] == 4
    assert result["speech"] == "我来拿apple"
    assert result["grasp_failure_mode"] == "success"
    assert result["grasp_strategy"] == "strategy_top_down"
    assert result["action_sequence"] == [
        "observe", "classify_safety", "plan_grasp_candidates", "grasp",
    ]
    assert result["actual_object"] == "apple"


def test_prepare_memory_dir_writes_empty_index_and_domains(tmp_path):
    module = _load_module()
    memory_dir = tmp_path / "memory" / "random_seed_101"

    module.prepare_memory_dir(memory_dir)

    assert (memory_dir / "index.yaml").exists()
    assert (memory_dir / "grasp_experience.yaml").exists()
    assert (memory_dir / "recognition_hints.yaml").exists()
    index_text = (memory_dir / "index.yaml").read_text(encoding="utf-8")
    assert "grasp" in index_text
    assert "recognition" in index_text
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_long_generalization_runner.py -q
```

Expected before implementation:

```text
AttributeError: module 'run_long_generalization' has no attribute 'parse_run_fixed_output'
```

- [ ] **Step 3: Implement output parser**

Add this function to `eval/run_long_generalization.py` after `append_jsonl()`:

```python
def parse_run_fixed_output(
    scenario_id: str,
    seed: int,
    returncode: int,
    stdout: str,
    stderr: str,
    elapsed: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "scenario_id": scenario_id,
        "seed": seed,
        "success": returncode == 0,
        "error": None,
        "steps": None,
        "time_s": round(elapsed, 1),
        "failure_reason": None,
        "grasp_failure_mode": None,
        "grasp_strategy": None,
        "action_sequence": [],
        "speech": None,
        "actual_object": None,
    }

    try:
        oracle_start = stdout.find("ORACLE SUMMARY")
        if oracle_start >= 0:
            json_start = stdout.find("{", oracle_start)
            json_end = stdout.find("\nepisode:", json_start)
            if json_end < 0:
                json_end = stdout.rfind("}")
            if json_start >= 0 and json_end >= 0:
                oracle_json = stdout[json_start:json_end + 1]
                oracle = json.loads(oracle_json)
                result["success"] = oracle.get("success", False)
                result["failure_reason"] = oracle.get("failure_reason")
                result["grasp_failure_mode"] = oracle.get("grasp_failure_mode")
                result["grasp_strategy"] = oracle.get("grasp_candidate_source")
                result["action_sequence"] = oracle.get("action_sequence", [])
                result["actual_object"] = oracle.get("actual_object")

        ep_start = stdout.find("EPISODE RESULT")
        if ep_start >= 0:
            for line in stdout[ep_start:ep_start + 500].splitlines():
                if line.startswith("steps"):
                    try:
                        result["steps"] = int(line.split(":", 1)[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("speech"):
                    result["speech"] = line.split(":", 1)[1].strip() if ":" in line else None
                elif line.startswith("reason") and not result["failure_reason"]:
                    result["failure_reason"] = line.split(":", 1)[1].strip()
    except Exception as e:
        if returncode != 0 and not result["error"]:
            tail = (stderr or stdout)[-500:] if (stderr or stdout) else "unknown"
            result["error"] = f"parse_failed ({type(e).__name__}: {e}): {tail[-200:]}"

    if returncode != 0 and not result["failure_reason"] and not result["error"]:
        result["failure_reason"] = "subprocess_nonzero_exit"
        result["error"] = stderr[-300:] if stderr else "no stderr"

    return result
```

- [ ] **Step 4: Implement isolated memory directory preparation**

Add this function after `parse_run_fixed_output()`:

```python
def prepare_memory_dir(memory_dir: Path) -> None:
    memory_dir.mkdir(parents=True, exist_ok=True)
    grasp_path = memory_dir / "grasp_experience.yaml"
    recognition_path = memory_dir / "recognition_hints.yaml"
    index_path = memory_dir / "index.yaml"

    if not grasp_path.exists():
        grasp_path.write_text(yaml.dump({"entries": []}, allow_unicode=True), encoding="utf-8")
    if not recognition_path.exists():
        recognition_path.write_text(yaml.dump({"entries": []}, allow_unicode=True), encoding="utf-8")
    index_path.write_text(
        yaml.dump({
            "version": 1,
            "domains": {
                "grasp": str(grasp_path),
                "recognition": str(recognition_path),
            },
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
```

- [ ] **Step 5: Implement subprocess runner**

Add this function after `prepare_memory_dir()`:

```python
def run_one_seed_subprocess(
    scenario: dict[str, Any],
    run_dir: Path,
    scenarios_config: Path,
    gpu_id: int,
    config: str,
    agent_config: str,
    log_level: str,
    timeout_s: int,
) -> dict[str, Any]:
    scenario_id = str(scenario["id"])
    seed = int(scenario["seed"])
    per_run_log_dir = run_dir / "per_run_logs"
    per_run_log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = per_run_log_dir / f"{scenario_id}.stdout.txt"
    stderr_path = per_run_log_dir / f"{scenario_id}.stderr.txt"
    memory_dir = run_dir / "memory" / scenario_id
    prepare_memory_dir(memory_dir)

    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, "eval/run_fixed.py",
        "--scenario", scenario_id,
        "--scenarios-config", str(scenarios_config),
        "--config", config,
        "--agent-config", agent_config,
        "--log-level", log_level,
        "--allow-object-mismatch",
        "--memory-dir", str(memory_dir),
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=env_vars,
            timeout=timeout_s,
        )
        elapsed = time.time() - t0
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        result = parse_run_fixed_output(
            scenario_id=scenario_id,
            seed=seed,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            elapsed=elapsed,
        )
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout if isinstance(e.stdout, str) else ""
        stderr = e.stderr if isinstance(e.stderr, str) else ""
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        result = {
            "scenario_id": scenario_id,
            "seed": seed,
            "success": False,
            "error": f"subprocess timeout ({timeout_s}s)",
            "time_s": float(timeout_s),
            "steps": None,
            "failure_reason": "timeout",
            "grasp_failure_mode": None,
            "grasp_strategy": None,
            "action_sequence": [],
            "speech": None,
            "actual_object": None,
        }
    except Exception as e:
        elapsed = time.time() - t0
        result = {
            "scenario_id": scenario_id,
            "seed": seed,
            "success": False,
            "error": f"{type(e).__name__}: {e}",
            "time_s": round(elapsed, 1),
            "steps": None,
            "failure_reason": "subprocess_error",
            "grasp_failure_mode": None,
            "grasp_strategy": None,
            "action_sequence": [],
            "speech": None,
            "actual_object": None,
        }

    result["stdout_path"] = str(stdout_path)
    result["stderr_path"] = str(stderr_path)
    result["memory_dir"] = str(memory_dir)
    return result
```

- [ ] **Step 6: Run tests and commit**

Run:

```bash
python -m pytest tests/test_long_generalization_runner.py -q
```

Expected:

```text
5 passed
```

Commit:

```bash
git add eval/run_long_generalization.py tests/test_long_generalization_runner.py
git commit -m "feat(eval): add long-run subprocess parsing and isolated memory"
```

---

### Task 4: Implement CLI orchestration, checkpoint/resume, and summaries

**Files:**
- Modify: `eval/run_long_generalization.py`
- Modify: `tests/test_long_generalization_runner.py`

- [ ] **Step 1: Add tests for summary formatting and `--count 0` behavior**

Append this test to `tests/test_long_generalization_runner.py`:

```python
def test_format_summary_contains_key_metrics():
    module = _load_module()
    summary = {
        "total": 2,
        "completed": 2,
        "successes": 1,
        "success_rate": 0.5,
        "errors": 0,
        "timeouts": 0,
        "avg_steps": 5.5,
        "avg_time_s": 123.4,
        "failure_breakdown": {"MAX_STEPS reached": 1},
        "strategy_usage": {"strategy_top_down": 2},
        "object_distribution": {"apple": 1, "wine": 1},
        "slowest_runs": [],
        "failed_runs": [],
    }

    text = module.format_summary(summary)

    assert "LONG GENERALIZATION SUMMARY" in text
    assert "TOTAL: 1/2 (50.0%)" in text
    assert "avg_steps=5.5" in text
    assert "MAX_STEPS reached" in text
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_long_generalization_runner.py -q
```

Expected before implementation:

```text
AttributeError: module 'run_long_generalization' has no attribute 'format_summary'
```

- [ ] **Step 3: Add summary formatting**

Add this function after `summarize_results()`:

```python
def format_summary(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("                         LONG GENERALIZATION SUMMARY")
    lines.append("=" * 80)
    total = int(summary.get("total", 0))
    successes = int(summary.get("successes", 0))
    rate = float(summary.get("success_rate", 0.0)) * 100.0
    lines.append(
        f"TOTAL: {successes}/{total} ({rate:.1f}%)  "
        f"avg_steps={float(summary.get('avg_steps', 0.0)):.1f}  "
        f"avg_time={float(summary.get('avg_time_s', 0.0)):.1f}s  "
        f"errors={int(summary.get('errors', 0))}  "
        f"timeouts={int(summary.get('timeouts', 0))}"
    )
    lines.append("")
    if summary.get("failure_breakdown"):
        lines.append(f"Failure breakdown: {summary['failure_breakdown']}")
    if summary.get("strategy_usage"):
        lines.append(f"Strategy usage:    {summary['strategy_usage']}")
    if summary.get("object_distribution"):
        lines.append(f"Object distribution: {summary['object_distribution']}")
    failed_runs = summary.get("failed_runs") or []
    if failed_runs:
        lines.append("")
        lines.append("Failed runs:")
        for r in failed_runs:
            reason = r.get("grasp_failure_mode") or r.get("failure_reason") or r.get("error") or "unknown"
            lines.append(f"  - {r.get('scenario_id')}: {reason}")
    slowest_runs = summary.get("slowest_runs") or []
    if slowest_runs:
        lines.append("")
        lines.append("Slowest runs:")
        for r in slowest_runs[:10]:
            lines.append(f"  - {r.get('scenario_id')}: {r.get('time_s')}s")
    lines.append("=" * 80)
    return "\n".join(lines)
```

- [ ] **Step 4: Add CLI parser and run directory setup**

Add these functions near the bottom of `eval/run_long_generalization.py`:

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overnight pure generalization runner")
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--log-level", default="WARNING")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--agent-config", default="configs/agent.yaml")
    parser.add_argument("--output-root", default="logs/long_generalization")
    return parser.parse_args(argv)


def make_run_id() -> str:
    return "overnight_" + datetime.now().strftime("%Y%m%d_%H%M%S")
```

- [ ] **Step 5: Add `main()` orchestration**

Add this `main()` implementation:

```python
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = args.run_id or make_run_id()
    run_dir = Path(args.output_root) / run_id
    results_path = run_dir / "results.jsonl"
    scenarios_path = run_dir / "scenarios.yaml"
    manifest_path = run_dir / "manifest.json"
    summary_json_path = run_dir / "summary.json"
    summary_txt_path = run_dir / "summary.txt"

    if run_dir.exists() and not args.resume and results_path.exists():
        print(f"Run directory already has results: {run_dir}", file=sys.stderr)
        print("Use --resume or choose a new --run-id.", file=sys.stderr)
        return 2

    run_dir.mkdir(parents=True, exist_ok=True)
    scenarios = generate_seed_scenarios(args.seed_start, args.count)
    scenarios_path.write_text(
        yaml.dump({"scenarios": scenarios}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed_start": args.seed_start,
        "count": args.count,
        "parallel": args.parallel,
        "timeout_s": args.timeout_s,
        "config": args.config,
        "agent_config": args.agent_config,
        "scenarios": [s["id"] for s in scenarios],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    completed = load_completed_results(results_path) if args.resume else {}
    pending = [s for s in scenarios if s["id"] not in completed]

    print(f"Run directory: {run_dir}", flush=True)
    print(
        f"Long generalization: {len(scenarios)} seeds, "
        f"completed={len(completed)}, pending={len(pending)}, parallel={args.parallel}",
        flush=True,
    )

    if pending:
        workers = max(1, min(args.parallel, len(pending)))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_scenario = {}
            for idx, scenario in enumerate(pending):
                gpu_id = idx % workers
                future = executor.submit(
                    run_one_seed_subprocess,
                    scenario=scenario,
                    run_dir=run_dir,
                    scenarios_config=scenarios_path,
                    gpu_id=gpu_id,
                    config=args.config,
                    agent_config=args.agent_config,
                    log_level=args.log_level,
                    timeout_s=args.timeout_s,
                )
                future_to_scenario[future] = scenario

            done_count = len(completed)
            total = len(scenarios)
            for future in as_completed(future_to_scenario):
                scenario = future_to_scenario[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "scenario_id": scenario["id"],
                        "seed": scenario["seed"],
                        "success": False,
                        "error": f"parallel_error: {type(e).__name__}: {e}",
                        "time_s": None,
                        "steps": None,
                        "failure_reason": "parallel_error",
                        "grasp_failure_mode": None,
                        "grasp_strategy": None,
                        "action_sequence": [],
                        "speech": None,
                        "actual_object": None,
                    }
                append_jsonl(results_path, result)
                done_count += 1
                status = "✓ SUCCESS" if result.get("success") else "✗ FAILED"
                t = f"{result.get('time_s')}s" if result.get("time_s") is not None else "?"
                print(f"  [{done_count}/{total}] {result['scenario_id']} → {status} ({t})", flush=True)

    all_results = list(load_completed_results(results_path).values())
    all_results.sort(key=lambda r: int(r.get("seed") or 0))
    summary = summarize_results(all_results)
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_text = format_summary(summary)
    summary_txt_path.write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text, flush=True)
    print(f"\nDetailed JSONL: {results_path}", flush=True)
    print(f"Summary JSON:   {summary_json_path}", flush=True)
    print(f"Summary TXT:    {summary_txt_path}", flush=True)

    return 0 if all(r.get("success") for r in all_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run tests and a zero-count smoke test**

Run:

```bash
python -m pytest tests/test_long_generalization_runner.py -q
python eval/run_long_generalization.py --count 0 --run-id smoke_count0 --output-root logs/long_generalization_test
```

Expected pytest:

```text
6 passed
```

Expected smoke command:

```text
TOTAL: 0/0 (0.0%)
```

- [ ] **Step 7: Commit**

Commit:

```bash
git add eval/run_long_generalization.py tests/test_long_generalization_runner.py
git commit -m "feat(eval): implement resumable long generalization runner"
```

---

### Task 5: Full verification, cleanup, and usage instructions

**Files:**
- Modify only if tests reveal an issue.

- [ ] **Step 1: Run focused eval tests**

Run:

```bash
python -m pytest tests/test_run_fixed_eval.py tests/test_long_generalization_runner.py -q
```

Expected:

```text
12 passed
```

- [ ] **Step 2: Run full test suite**

Run:

```bash
python -m pytest tests/ --tb=short
```

Expected:

```text
all tests pass
```

- [ ] **Step 3: Check git status**

Run:

```bash
git status --short
```

Expected allowed state:

```text
 M memory/grasp_experience.yaml
```

The memory file may contain local evaluation artifacts from earlier runs and should not be committed with this feature unless the user explicitly asks.

- [ ] **Step 4: Push implementation commits**

Run:

```bash
git push
```

Expected:

```text
main -> main
```

- [ ] **Step 5: Provide server command to user**

Final overnight command:

```bash
cd ~/embodied-AI-one && git pull
nohup python eval/run_long_generalization.py \
  --seed-start 101 \
  --count 50 \
  --parallel 4 \
  --run-id overnight_$(date +%Y%m%d_%H%M%S) \
  > long_generalization.out 2>&1 &
```

Progress check:

```bash
tail -40 long_generalization.out
```

Find run directory:

```bash
grep "Run directory:" long_generalization.out | tail -1
```

Resume example:

```bash
python eval/run_long_generalization.py \
  --seed-start 101 \
  --count 50 \
  --parallel 4 \
  --run-id <same_run_id> \
  --resume
```

---

## Self-Review

### Spec Coverage

- New long-run script: Task 2-4 create `eval/run_long_generalization.py`.
- `--seed-start`, `--count`, `--parallel`, `--run-id`, `--resume`, `--timeout-s`, config args: Task 4 CLI.
- Per-seed isolated memory: Task 1 `--memory-dir`, Task 3 `prepare_memory_dir()` and subprocess argument.
- Durable `results.jsonl`: Task 2 `append_jsonl()`, Task 4 parent appends after each future.
- Resume: Task 2 `load_completed_results()`, Task 4 skips completed scenarios.
- Per-run logs: Task 3 writes stdout/stderr paths.
- Final summaries: Task 2 `summarize_results()`, Task 4 `format_summary()` and summary files.
- Existing tests: Task 5 full pytest.
- Smoke run: Task 4 zero-count smoke test.

### Placeholder Scan

No `TBD`, `TODO`, `fill in later`, or unspecified test steps are present. Every task lists exact files, commands, expected outputs, and code snippets for implementation.

### Type Consistency

- `scenario_id` is consistently a string such as `random_seed_101`.
- `seed` is consistently an integer.
- `run_dir`, `scenarios_config`, and `memory_dir` are `Path` values in helpers.
- Result dictionaries consistently include `scenario_id`, `seed`, `success`, `error`, `steps`, `time_s`, `failure_reason`, `grasp_failure_mode`, `grasp_strategy`, `action_sequence`, `speech`, and `actual_object`.
