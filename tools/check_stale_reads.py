#!/usr/bin/env python3
"""Are the files I am about to cite stale in this worktree?

WHY NOT "AM I BEHIND". Four stale-read incidents on 2026-08-02, every one caught
by another window and none by the reader:

  * a 364-line copy of a 910-line method document, which produced an adversarial
    review reporting a retracted result as still asserted;
  * a sync command failing silently for hours behind `-q 2>/dev/null`, 60 commits
    behind, found only because a cited file did not exist;
  * a producer reported as absent from master when it had already been merged;
  * a count sentence read from an older head after it had been fixed.

A worktree gives no signal that it is behind. But "behind by N" is the wrong
signal: master moves DURING a turn, so N is almost always non-zero and a check
that fires constantly is one nobody reads. The question that matters is narrower
and almost always answers no:

    of the files I am about to talk about, which differ from master?

That converts an ignorable condition into an actionable one. Stale-by-two-commits
is normal; stale ON THE FILE YOU ARE QUOTING is the defect.

USAGE

    tools/check_stale_reads.py PATH [PATH ...]   # the ones you are about to cite
    tools/check_stale_reads.py                   # everything that differs
    tools/check_stale_reads.py --against origin/master

Exit 1 if any named path differs from the reference. With no paths, exit 0 and
just report — a survey is not a failure.

NOTE the reference is a LOCAL ref by default. If nobody has fetched, master is
itself stale and this check will cheerfully pass. `--fetch` refreshes first.
"""
import argparse
import subprocess
import sys
from pathlib import Path

def _toplevel():
    """The CALLER's worktree, never the script's own.

    THIS LINE WAS THE BUG. It read Path(__file__).resolve().parent.parent, which
    is the worktree the script LIVES in, so every invocation reported on that one
    regardless of where it was run. Run from a stale worktree it printed "nothing
    differs from the reference" while four files, including both documents that
    window had been reading all session, were behind.

    A stale-read checker that passes while the reader is stale is the failure it
    exists to prevent, and the same shape as the sync command in this file's own
    docstring: a check that cannot fail is worse than no check, because it turns
    an unmonitored risk into a monitored one reporting all-clear.

    Found by the interpretability window running it from their tree — which is
    also the point of the cross-worktree test below: a self-test living beside
    the thing it tests shares its assumptions and would have passed forever.
    """
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                       capture_output=True, text=True)
    if r.returncode or not r.stdout.strip():
        sys.exit("FATAL: not inside a git worktree")
    return Path(r.stdout.strip())


ROOT = _toplevel()


def git(*args, check=False):
    r = subprocess.run(("git", "-C", str(ROOT)) + args,
                       capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"FATAL: git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="files you are about to cite")
    ap.add_argument("--against", default="master", help="reference ref (default master)")
    ap.add_argument("--fetch", action="store_true",
                    help="git fetch first; without it the reference may itself be stale")
    args = ap.parse_args()

    if args.fetch:
        git("fetch", "--quiet")

    ref = args.against
    if not git("rev-parse", "--verify", "--quiet", ref):
        sys.exit(f"FATAL: no such ref: {ref}")

    head = git("rev-parse", "--short", "HEAD")
    tip = git("rev-parse", "--short", ref)
    behind = git("rev-list", "--count", f"HEAD..{ref}")
    ahead = git("rev-list", "--count", f"{ref}..HEAD")
    print(f"  worktree {ROOT.name}   HEAD {head}   {ref} {tip}   behind {behind}, ahead {ahead}")
    if not args.fetch:
        print(f"  (reference is the LOCAL {ref}; run with --fetch if nobody has fetched recently)")

    # Files that differ between this worktree's HEAD and the reference.
    changed = set(git("diff", "--name-only", f"HEAD...{ref}").splitlines())

    if not args.paths:
        if not changed:
            print("  nothing differs from the reference.")
            return 0
        print(f"\n  {len(changed)} file(s) differ from {ref}:")
        for f in sorted(changed):
            print(f"    {f}")
        print("\n  A survey is not a failure — name the paths you are citing to get a verdict.")
        return 0

    stale = []
    for p in args.paths:
        rel = p
        try:
            rel = str(Path(p).resolve().relative_to(ROOT))
        except ValueError:
            pass
        exists_here = (ROOT / rel).exists()
        in_ref = bool(git("cat-file", "-e", f"{ref}:{rel}") == "" and
                      subprocess.run(("git", "-C", str(ROOT), "cat-file", "-e",
                                      f"{ref}:{rel}"), capture_output=True).returncode == 0)
        if rel in changed:
            why = ("differs from " + ref if exists_here and in_ref else
                   "MISSING HERE, present in " + ref if in_ref else
                   "present here, absent from " + ref)
            stale.append((rel, why))
        elif not exists_here and not in_ref:
            stale.append((rel, "does not exist in either"))

    print()
    for p in args.paths:
        rel = str(Path(p).resolve().relative_to(ROOT)) if Path(p).exists() else p
        hit = next((s for s in stale if s[0] in (rel, p)), None)
        print(f"  {'STALE' if hit else 'ok   '}  {rel}" + (f"   — {hit[1]}" if hit else ""))

    if stale:
        print(f"\n  {len(stale)} of {len(args.paths)} cited path(s) are stale here.")
        print(f"  Do not quote them. `git merge {ref}` first.")
        return 1
    print(f"\n  All {len(args.paths)} cited path(s) match {ref}. Safe to quote.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
