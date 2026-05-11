---
title: Long-Run Pure Generalization Runner Design
date: 2026-05-12
status: approved-for-planning
---

# Long-Run Pure Generalization Runner Design

## Goal

Add an overnight-safe evaluation runner for EmboSight pure generalization testing.

The runner should execute many random-seed discover scenarios for long periods without losing progress if the terminal, SSH session, subprocess, or one scenario fails. The initial target is 50 random seeds with parallelism 4.

This runner evaluates pure generalization, not cross-episode memory learning. Each seed must use an isolated memory directory so results are not affected by previous scenarios.

## Non-Goals

- Do not replace `eval/run_batch.py`.
- Do not implement memory-learning evaluation in this phase.
- Do not tune grasping, perception, or decision policies here.
- Do not require manual editing of `configs/eval_scenarios.yaml` for each run.

## Recommended Approach

Create a new script:

```text
eval/run_long_generalization.py
```

The new script will act as a durable long-run orchestrator around `eval/run_fixed.py`. It will generate random discover scenarios, run them in subprocesses, write per-seed results immediately, and support resume by skipping completed seed IDs.

This keeps normal batch evaluation simple while giving overnight runs stronger reliability guarantees.

## Architecture

### Components

1. **Long-run orchestrator**: `eval/run_long_generalization.py`
   - Parses long-run CLI arguments.
   - Creates a run directory under `logs/long_generalization/<run-id>/`.
   - Generates a scenario manifest for the seed range.
   - Launches seed runs with bounded parallelism.
   - Writes one JSONL result immediately after each seed completes.
   - Writes final `summary.json` and `summary.txt`.

2. **Single-scenario runner**: existing `eval/run_fixed.py`
   - Continues to run exactly one scenario.
   - Needs a `--memory-dir` CLI option for pure generalization isolation.
   - Passes that memory directory into `EmboSightAgent` through `MemoryManager`.

3. **Result parser**
   - Reuses the existing parsing pattern from `eval/run_batch.py`.
   - Extracts oracle summary, episode result, action sequence, strategy, failure reason, actual object, steps, and time.

4. **Run directory**
   - Stores durable artifacts for inspection and resume.

## Run Directory Layout

Each long run creates:

```text
logs/long_generalization/<run-id>/
  manifest.json
  scenarios.yaml
  results.jsonl
  summary.json
  summary.txt
  per_run_logs/
    random_seed_101.stdout.txt
    random_seed_101.stderr.txt
    random_seed_102.stdout.txt
    random_seed_102.stderr.txt
  memory/
    random_seed_101/
    random_seed_102/
```

### File meanings

- `manifest.json`: immutable run configuration and task list.
- `scenarios.yaml`: generated scenario config consumed by `run_fixed.py`.
- `results.jsonl`: append-only checkpoint file; one completed run per line.
- `summary.json`: machine-readable aggregate results.
- `summary.txt`: human-readable aggregate summary.
- `per_run_logs/`: stdout/stderr for each seed subprocess.
- `memory/`: isolated memory directory for each seed.

## CLI

Default overnight command:

```bash
python eval/run_long_generalization.py \
  --seed-start 101 \
  --count 50 \
  --parallel 4 \
  --run-id overnight_$(date +%Y%m%d_%H%M%S)
```

Resume command:

```bash
python eval/run_long_generalization.py \
  --seed-start 101 \
  --count 50 \
  --parallel 4 \
  --run-id overnight_20260512_0119 \
  --resume
```

Recommended `nohup` usage:

```bash
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

## CLI Options

- `--seed-start`: first random seed, default `101`.
- `--count`: number of seeds, default `50`.
- `--parallel`: number of worker subprocesses, default `4`.
- `--run-id`: run directory name. If omitted, generate timestamp ID.
- `--resume`: reuse existing run directory and skip completed seeds.
- `--timeout-s`: per-seed subprocess timeout, default `900`.
- `--log-level`: sub-process log level, default `WARNING`.
- `--config`: simulator config path, default `configs/default.yaml`.
- `--agent-config`: agent config path, default `configs/agent.yaml`.
- `--output-root`: output root, default `logs/long_generalization`.

## Data Flow

1. Generate seeds from `[seed_start, seed_start + count)`.
2. Generate discover scenarios:
   - `id`: `random_seed_<seed>`
   - `env_name`: `PickPlaceCounterToCabinet`
   - `seed`: seed
   - `query`: `pick up anything`
   - `expected_object`: null
   - `user_mode`: `fake_from_robocasa`
   - `max_resets`: 1
3. Write `scenarios.yaml` once per run.
4. Load existing completed IDs from `results.jsonl` if `--resume` is enabled.
5. Submit unfinished seeds to a `ProcessPoolExecutor` with `max_workers=parallel`.
6. Each worker runs `eval/run_fixed.py` as a subprocess with:
   - generated scenario config
   - per-seed memory dir
   - per-seed stdout/stderr capture
   - timeout
7. Parent process appends each finished result to `results.jsonl` immediately.
8. Parent process prints compact progress lines.
9. At the end, aggregate all JSONL entries and write summaries.

## Pure Generalization Memory Isolation

Pure generalization requires no cross-seed memory transfer.

Therefore `run_fixed.py` needs a new optional argument:

```bash
--memory-dir <path>
```

When provided, `build_agent()` should construct:

```python
MemoryManager(memory_dir=Path(memory_dir))
```

and pass it to `EmboSightAgent(memory_manager=...)`.

The long-run orchestrator will set:

```text
logs/long_generalization/<run-id>/memory/random_seed_<seed>/
```

for each seed.

If `--memory-dir` is not provided, existing behavior remains unchanged.

## Checkpoint and Resume Semantics

`results.jsonl` is the source of truth for completed seeds.

- A seed is considered complete if a valid JSON line exists with matching `scenario_id`.
- On resume, completed seeds are skipped.
- Incomplete or corrupt trailing lines should be ignored rather than crashing the resume process.
- The runner should append and flush after every completed seed.

## Failure Handling

A single seed failure must not stop the long run.

Failure categories:

- `timeout`: subprocess exceeded `timeout_s`.
- `subprocess_nonzero_exit`: subprocess exited non-zero and oracle parsing did not provide a better reason.
- `parse_failed`: stdout/stderr did not contain expected JSON.
- `parallel_error`: worker raised an unexpected exception.
- normal task failure: oracle says `success=false` with a `failure_reason`.

All failure records should include:

- `scenario_id`
- `seed`
- `success`
- `error`
- `failure_reason`
- `grasp_failure_mode`
- `grasp_strategy`
- `action_sequence`
- `actual_object`
- `steps`
- `time_s`
- stdout/stderr log paths

## Summary Metrics

`summary.json` and `summary.txt` should include:

- total runs
- completed runs
- success count and success rate
- error count
- timeout count
- average steps
- average time
- failure breakdown
- strategy usage
- actual object distribution
- slowest 10 runs
- failed run list

## Testing Strategy

### Unit-level tests

Add tests for pure Python helper logic where practical:

- seed list generation
- scenario generation
- JSONL completed-ID loading
- corrupt-line tolerance
- summary aggregation

### Smoke tests

Add a dry or tiny smoke path if implementation cost is low:

- `--count 0` should create run directory and empty summary.
- `--count 1 --parallel 1` can be manually run on the server for end-to-end verification.

### Regression checks

Run existing test suite after changes:

```bash
python -m pytest tests/ --tb=short
```

## Risks and Mitigations

### Risk: Memory pollution across seeds

Mitigation: per-seed `--memory-dir` from the long-run orchestrator.

### Risk: Progress loss after SSH disconnect or process interruption

Mitigation: append result JSONL immediately after each seed completes; use `nohup` for the parent process.

### Risk: One scenario hangs forever

Mitigation: per-seed timeout, default 900 seconds.

### Risk: Concurrent writes corrupt checkpoint

Mitigation: only parent process writes `results.jsonl`; workers return result objects.

### Risk: Existing batch runner behavior changes

Mitigation: create a new long-run script and keep `run_batch.py` unchanged except optional shared helpers only if needed.

## Acceptance Criteria

The implementation is complete when:

1. `eval/run_long_generalization.py` exists and supports the CLI above.
2. `eval/run_fixed.py` accepts optional `--memory-dir` without changing default behavior.
3. Each seed writes isolated memory under the run directory.
4. `results.jsonl` is updated after each completed seed.
5. Re-running with `--resume` skips completed seeds.
6. A final summary is written to `summary.json` and `summary.txt`.
7. Existing tests pass.
8. A small local or server smoke run demonstrates the script starts, writes outputs, and summarizes results.
