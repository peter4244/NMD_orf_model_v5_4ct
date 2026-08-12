#!/usr/bin/env python3
"""check_docs.py — this repository's reader-facing documents may not name what is not here.

WHY. Three independent seats reproduced the reproduction package and this repository over
2026-08-11/12 and returned 48 findings, none of them a defect in the analysis. Two were ours and
both are this shape: the README's build order named `drivers/slurm_determinism.sh` after that
driver was deleted at v3.1.0, and the reproduction package told readers a wrapper ran the uORF
metrics when it runs inference only. A reader following either lands on something absent.

TWO CHECKS, AND THE SECOND IS THE ONE THAT MATTERS HERE.

  1. Backticked paths in reader-facing documents must resolve.
  2. BARE SCRIPT NAMES INSIDE FENCED BLOCKS must resolve. The build order is a fenced block of
     `slurm_*.sh` names with no backticks anywhere, so check 1 cannot see it — and the build order
     is precisely where a renamed or retired driver goes stale. A linter that only reads backticks
     would have passed the exact defect this file exists to catch.

THREE ROOTS. A named file may belong to this repository, to the Zenodo deposit, or to the
reproduction package. The same string is correct against one and meaningless against the others,
so each is resolved against all three before it is called missing.

SCOPE IS THE READER-FACING SET, deliberately. analysis_plans/ and superseded/ are working notes
that name run outputs and one-off artifacts; checking them reported 26 problems, none of which a
reader could ever hit. A checker that fires on things that are fine gets bypassed, and then it
protects nothing.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DEPOSIT = Path.home() / "claude_projects" / "nmd_deposit_2026"
REPRO = Path.home() / "claude_projects" / "nmd_lung_longread_2026"

# Current documents only. BUGFIX_STOP_CODON_2026-03-31.md is a DATED RECORD -- it names the
# absolute /projects/talisman paths and the v5 driver as they were in March, which is what a record
# of a past fix is for. Checking it reported three "missing" files that are missing on purpose.
READER_DOCS = ["README.md", "METHODS.md", "CLAUDE.md", "RETRAIN_ARCHITECTURE_CHANGES.md",
               "ROW_TILED_PERTURBATION_2026-08-02.md"]

# Run outputs. Gitignored by design; naming one is not a broken reference.
OUTPUT_PREFIXES = ("runs/", "results_", "deprecated_", "tmp/", "logs/", "slurm_logs/")

EXEMPT: dict[str, str] = {
    "autocorr.py":
        "CLAUDE.md names it in an ANECDOTE -- two windows once wrote this filename to the same "
        "scratch directory and one silently replaced the other. It is an example of a collision, "
        "not a reference to a file, and the checker cannot tell those apart from the string alone",
}

RX_TICKED = re.compile(r"`([A-Za-z0-9_][A-Za-z0-9_./+-]*\.(?:R|Rmd|py|sh|yml|yaml|tsv|json|h5|pt))`")
RX_BARE = re.compile(r"\b((?:slurm|submit)_[A-Za-z0-9_]+\.sh|[0-9]{2}[a-z]?_[A-Za-z0-9_]+\.py)\b")


def resolves(ref: str) -> bool:
    if ref.startswith(OUTPUT_PREFIXES) or ref in EXEMPT:
        return True
    name = Path(ref).name
    for root in (HERE, DEPOSIT, DEPOSIT / "source_data", REPRO):
        if not root.is_dir():
            continue
        if (root / ref).exists():
            return True
        try:
            if any(root.rglob(name)):
                return True
        except OSError:
            pass
    return False


def main() -> int:
    problems = []
    checked = 0
    for rel in READER_DOCS:
        doc = HERE / rel
        if not doc.exists():
            continue
        checked += 1
        in_fence = False
        for n, line in enumerate(doc.read_text(errors="replace").splitlines(), 1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            refs = {m.group(1) for m in RX_TICKED.finditer(line)}
            if in_fence:
                refs |= {m.group(1) for m in RX_BARE.finditer(line)}
            for ref in refs:
                if not resolves(ref):
                    problems.append((rel, n, ref))

    print(f"  checked {checked} reader-facing document(s)")
    print(f"    roots: this repo, {'deposit' if DEPOSIT.is_dir() else 'deposit ABSENT'}, "
          f"{'reproduction package' if REPRO.is_dir() else 'reproduction package ABSENT'}")
    if not problems:
        print("  OK — every file named in a reader-facing document resolves against one of the roots")
        return 0
    print(f"\n  {len(problems)} problem(s):\n")
    for rel, n, ref in problems:
        print(f"    {rel}:{n}\n        {ref}  — not in this repository, the deposit, or the package")
    print("\n  Fix the reference, or add it to EXEMPT with the reason.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
