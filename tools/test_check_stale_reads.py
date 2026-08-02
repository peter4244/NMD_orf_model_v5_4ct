#!/usr/bin/env python3
"""Run check_stale_reads.py from a worktree OTHER than its own.

This test exists because the bug it catches was invisible to any test living
beside the tool. `ROOT` was derived from the script's location, so every
invocation inspected the script's own worktree and reported all-clear from a
stale one. A self-test in that tree would have passed forever.

So the assertion is not "does it report correctly" but "does it report on the
CALLER". It is skipped, not failed, when there is only one worktree — a test
that silently passes for the wrong reason is what we are guarding against.
"""
import subprocess, sys, json
from pathlib import Path

TOOL = Path(__file__).resolve().parent / "check_stale_reads.py"

wt = [l.split()[0] for l in subprocess.run(
    ["git", "worktree", "list"], capture_output=True, text=True,
    cwd=TOOL.parent).stdout.splitlines()]
if len(wt) < 2:
    print("  SKIP: only one worktree; this test needs at least two."); sys.exit(0)

fails = 0
for tree in wt:
    name = Path(tree).name
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, cwd=tree).stdout.strip()
    out = subprocess.run([sys.executable, str(TOOL)], capture_output=True,
                         text=True, cwd=tree).stdout
    first = out.splitlines()[0] if out else ""
    ok_name = f"worktree {name}" in first
    ok_head = f"HEAD {head}" in first
    print(f"  {'ok  ' if ok_name and ok_head else 'FAIL'}  invoked from {name}: {first.strip()}")
    if not ok_name:
        print(f"          reported a worktree other than the caller ({name})")
    if not ok_head:
        print(f"          reported a HEAD other than the caller's ({head})")
    fails += not (ok_name and ok_head)

print(f"\n  {len(wt)-fails}/{len(wt)} worktrees reported on themselves.")
sys.exit(1 if fails else 0)
