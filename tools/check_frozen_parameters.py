#!/usr/bin/env python3
"""Every frozen parameter must have ONE value, everywhere it is written down.

WHY THIS AND NOT A PROSE CHECKER. The failure class is a document or a script
restating a value that is declared elsewhere, and the two copies drifting. The
interpretability window scoped the three possible checkers on 2026-08-02:

  prose vs prose   DO NOT BUILD. Within twenty lines of the band freeze,
                   ANALYSIS_SEQUENCING_PROPOSAL.md legitimately contains 4, 8
                   and 16 in four distinct roles -- primary, sweep values,
                   fallback under a pre-set conditional, and a superseded-record
                   block deliberately preserving a struck value. Separating those
                   needs sentence-level understanding, not a lint, and a noisy
                   checker trains people to ignore it.
  don't restate    Adopted as a convention, not a tool: declare a frozen
                   parameter once and reference it thereafter.
  code vs code     THIS. Code constants are unambiguous, so precision is
                   near-perfect: read the literal from the AST and compare. No
                   parsing of prose, no judgement, and a green result means
                   something.

WHAT IT CHECKS

  1. DECLARED parameters (analysis_plans/frozen_parameters.tsv) must match their
     declared value at every module-level assignment that uses the name.
  2. UNDECLARED names assigned at module level in more than one file with more
     than one distinct value are reported as candidate drift -- a warning, not a
     failure, because a name like N or WINDOW legitimately means different things
     in different scripts.

Names are matched by identity, so KOZAK_FLOOR and FLOOR are the same parameter
only if both are declared as aliases (comma-separated in the parameter column).

Exit 1 on any declared-parameter mismatch. Warnings alone exit 0.

Usage:  python3 tools/check_frozen_parameters.py [--warnings]
"""
import argparse
import ast
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DECL = ROOT / "analysis_plans" / "frozen_parameters.tsv"
SCAN = ["analysis_plans", "tools", "."]


def module_constants(path):
    """Module-level UPPERCASE literal assignments: {name: (value, lineno)}."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return {}
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Name) or not tgt.id.isupper():
            continue
        try:
            out[tgt.id] = (ast.literal_eval(node.value), node.lineno)
        except (ValueError, SyntaxError):
            continue
    return out


def same(a, b):
    """Equality that treats a rounded copy as DIFFERENT.

    A frozen parameter restated to fewer significant figures is exactly the
    defect this exists to catch, so no tolerance is applied. Tuples compare
    element-wise; ints and floats compare by value so 8 == 8.0.
    """
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))
                and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b)))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warnings", action="store_true",
                    help="also list undeclared names that differ across files")
    args = ap.parse_args()

    if not DECL.exists():
        sys.exit(f"FATAL: no declaration at {DECL.relative_to(ROOT)}")
    declared = {}
    for r in csv.DictReader(DECL.open(encoding="utf-8"), delimiter="\t"):
        if not (r.get("parameter") or "").strip():
            continue
        try:
            val = ast.literal_eval(r["value"].strip())
        except (ValueError, SyntaxError):
            sys.exit(f"FATAL: {r['parameter']}: value is not a literal: {r['value']!r}")
        for alias in (a.strip() for a in r["parameter"].split(",")):
            declared[alias] = (val, r["parameter"], r.get("authority", ""))

    seen = defaultdict(list)
    files = sorted({p for d in SCAN for p in (ROOT / d).glob("*.py")})
    for path in files:
        for name, (val, line) in module_constants(path).items():
            seen[name].append((path.relative_to(ROOT), line, val))

    failures, checked = [], 0
    for name, (want, canon, authority) in sorted(declared.items()):
        for rel, line, got in seen.get(name, []):
            checked += 1
            if not same(got, want):
                failures.append((canon, rel, line, got, want, authority))

    print(f"  {len(declared)} declared parameter names, {checked} assignments "
          f"checked across {len(files)} files")

    if failures:
        print()
        for canon, rel, line, got, want, authority in failures:
            print(f"  FAIL  {canon}")
            print(f"          {rel}:{line} has {got!r}")
            print(f"          declared          {want!r}")
            if authority:
                print(f"          authority         {authority}")
    else:
        print("  OK: every declared frozen parameter matches its declaration.")

    if args.warnings:
        drift = {n: v for n, v in seen.items()
                 if n not in declared and len({repr(x[2]) for x in v}) > 1}
        if drift:
            print(f"\n  candidate drift — undeclared names with >1 value "
                  f"({len(drift)}); declare or rename:")
            for name, hits in sorted(drift.items()):
                vals = ", ".join(sorted({repr(h[2]) for h in hits}))
                print(f"    {name:16} {vals}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
