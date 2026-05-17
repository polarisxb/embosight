#!/usr/bin/env python3
"""CLI tool to inspect, clean, and migrate EmboSight memory files.

Usage examples:
    # Show summary of all grasp memory entries
    python scripts/clean_memory.py status

    # Retire entries with stale code_version (interactive)
    python scripts/clean_memory.py retire-stale

    # Migrate all entries to v2 schema in-place
    python scripts/clean_memory.py migrate

    # Purge retired entries from file
    python scripts/clean_memory.py purge-retired

    # Reset a specific object's grasp history
    python scripts/clean_memory.py reset --object wooden_spoon
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from src.memory_manager import (  # noqa: E402
    GRASP_CODE_VERSION,
    GRASP_SCHEMA_VERSION,
    _FAIL_BAN_THRESHOLD,
    MemoryManager,
)


def _load_grasp_file(memory_dir: Path) -> tuple[Path, dict]:
    idx_path = memory_dir / "index.yaml"
    if not idx_path.exists():
        sys.exit(f"[error] index.yaml not found in {memory_dir}")
    with open(idx_path, encoding="utf-8") as f:
        idx = yaml.safe_load(f) or {}
    grasp_path = Path(idx.get("domains", {}).get("grasp", ""))
    if not grasp_path.exists():
        sys.exit(f"[error] grasp file not found: {grasp_path}")
    with open(grasp_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return grasp_path, data


def _save(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    print(f"  [saved] {path}")


# ── Commands ──


def cmd_status(args: argparse.Namespace) -> None:
    """Print summary of grasp memory entries."""
    _, data = _load_grasp_file(args.memory_dir)
    file_cv = data.get("code_version", "(none)")
    file_sv = data.get("schema_version", "(none)")
    stale = file_cv != GRASP_CODE_VERSION
    print(f"File code_version : {file_cv}  (current: {GRASP_CODE_VERSION})")
    print(f"File schema_version: {file_sv}  (current: {GRASP_SCHEMA_VERSION})")
    print(f"Stale             : {'YES' if stale else 'no'}")
    print()

    entries = data.get("entries", [])
    if not entries:
        print("  (no entries)")
        return

    for i, e in enumerate(entries):
        obj = e.get("object_type", "?")
        retired = e.get("retired", False)
        tag = " [RETIRED]" if retired else ""
        print(f"  [{i}] {obj}{tag}")
        strategies = e.get("strategies", {})
        if strategies:
            for sname, sdata in strategies.items():
                s = sdata.get("successes", 0)
                f = sdata.get("failures", 0)
                fbr = sdata.get("failures_by_reason", {})
                banned = any(c >= _FAIL_BAN_THRESHOLD for c in fbr.values())
                ban_tag = " ** BANNED **" if banned and not retired and not stale else ""
                reasons = ", ".join(f"{r}:{c}" for r, c in fbr.items()) if fbr else "-"
                print(f"      {sname}: ok={s} fail={f} reasons=[{reasons}]{ban_tag}")
        else:
            failed = e.get("failed", [])
            for f in failed:
                print(f"      (v1) {f.get('strategy','?')}: {f.get('reason','?')} x{f.get('count',1)}")
        print()


def cmd_retire_stale(args: argparse.Namespace) -> None:
    """Mark entries from a stale code_version as retired."""
    path, data = _load_grasp_file(args.memory_dir)
    file_cv = data.get("code_version", "")
    if file_cv == GRASP_CODE_VERSION:
        print("File code_version matches current — nothing to retire.")
        return

    entries = data.get("entries", [])
    count = 0
    for e in entries:
        if not e.get("retired"):
            e["retired"] = True
            count += 1

    if count == 0:
        print("All entries already retired.")
        return

    if not args.yes:
        ans = input(f"Retire {count} entries from code_version={file_cv}? [y/N] ")
        if ans.lower() != "y":
            print("Aborted.")
            return

    _save(path, data)
    print(f"  Retired {count} entries.")


def cmd_migrate(args: argparse.Namespace) -> None:
    """Migrate all entries to v2 schema in-place."""
    path, data = _load_grasp_file(args.memory_dir)
    entries = data.get("entries", [])
    migrated = 0
    for i, e in enumerate(entries):
        before = copy.deepcopy(e)
        after = MemoryManager._migrate_grasp_entry_v1_to_v2(e)
        entries[i] = after
        if after != before:
            migrated += 1

    data["schema_version"] = GRASP_SCHEMA_VERSION
    data["code_version"] = GRASP_CODE_VERSION
    _save(path, data)
    print(f"  Migrated {migrated}/{len(entries)} entries to schema v{GRASP_SCHEMA_VERSION}.")


def cmd_purge_retired(args: argparse.Namespace) -> None:
    """Remove retired entries from the file permanently."""
    path, data = _load_grasp_file(args.memory_dir)
    entries = data.get("entries", [])
    before_count = len(entries)
    kept = [e for e in entries if not e.get("retired")]
    removed = before_count - len(kept)

    if removed == 0:
        print("No retired entries to purge.")
        return

    if not args.yes:
        ans = input(f"Permanently remove {removed} retired entries? [y/N] ")
        if ans.lower() != "y":
            print("Aborted.")
            return

    data["entries"] = kept
    _save(path, data)
    print(f"  Purged {removed} entries, {len(kept)} remaining.")


def cmd_reset(args: argparse.Namespace) -> None:
    """Remove all grasp history for a specific object."""
    if not args.object:
        sys.exit("[error] --object is required for reset command")
    path, data = _load_grasp_file(args.memory_dir)
    entries = data.get("entries", [])
    obj_key = args.object.lower().strip()
    kept = [e for e in entries
            if e.get("object_type", "").lower().strip() != obj_key]
    removed = len(entries) - len(kept)

    if removed == 0:
        print(f"No entries found for '{args.object}'.")
        return

    if not args.yes:
        ans = input(f"Remove {removed} entries for '{args.object}'? [y/N] ")
        if ans.lower() != "y":
            print("Aborted.")
            return

    data["entries"] = kept
    _save(path, data)
    print(f"  Removed {removed} entries for '{args.object}'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect, clean, and migrate EmboSight memory files.",
    )
    parser.add_argument(
        "--memory-dir", type=Path, default=Path("memory"),
        help="Path to the memory directory (default: memory/)",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip confirmation prompts",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show summary of grasp memory entries")
    sub.add_parser("retire-stale", help="Mark stale entries as retired")
    sub.add_parser("migrate", help="Migrate entries to v2 schema")
    sub.add_parser("purge-retired", help="Remove retired entries permanently")
    p_reset = sub.add_parser("reset", help="Reset grasp history for an object")
    p_reset.add_argument("--object", required=True, help="Object type to reset")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    {
        "status": cmd_status,
        "retire-stale": cmd_retire_stale,
        "migrate": cmd_migrate,
        "purge-retired": cmd_purge_retired,
        "reset": cmd_reset,
    }[args.command](args)


if __name__ == "__main__":
    main()
