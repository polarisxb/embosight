#!/usr/bin/env python3
"""Mock run_fixed.py for testing the long generalization runner orchestration.
Instantly prints fake EPISODE RESULT and ORACLE SUMMARY, then exits 0.
"""
import argparse
import json
import random
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--scenarios-config", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--agent-config", default="")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--allow-object-mismatch", action="store_true")
    parser.add_argument("--memory-dir", default=None)
    args = parser.parse_args()

    # Simulate variable timing (0.1-0.5s)
    time.sleep(random.uniform(0.1, 0.5))

    # Random success/failure
    success = random.random() < 0.7
    steps = random.randint(3, 10)
    objects = ["apple", "wine", "cup", "plate", "tupperware"]
    obj = random.choice(objects)
    strategies = ["strategy_top_down", "vlm_top_grasp", "gentle_side"]
    strategy = random.choice(strategies)

    print("\n========== EPISODE RESULT ==========")
    print(f"scenario: {args.scenario}")
    print(f"success : {success}")
    print(f"speech  : mock speech for {obj}")
    print(f"steps   : {steps}")
    print(f"time    : {steps * 2.5:.1f}s")

    if not success:
        reasons = ["MAX_STEPS reached", "hit_z_floor", "grasp_slip"]
        print(f"reason  : {random.choice(reasons)}")

    oracle = {
        "success": success,
        "failure_reason": None if success else "MAX_STEPS reached",
        "grasp_failure_mode": "success" if success else "hit_z_floor",
        "grasp_candidate_source": strategy,
        "action_sequence": ["observe", "classify_safety", "plan_grasp_candidates", "grasp"],
        "actual_object": obj,
    }

    print("\n========== ORACLE SUMMARY ==========")
    print(json.dumps(oracle, indent=2))
    print(f"episode: logs/episodes/mock_episode.json")


if __name__ == "__main__":
    main()
