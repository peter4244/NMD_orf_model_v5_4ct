#!/usr/bin/env python3
"""Every result that crosses the entry trigger gets a row, and a row with an empty field is visibly incomplete.

WHY A FORM AND NOT A RULE. On 2026-08-02 six results were retracted. Not one was
caught by a check; five were caught by a person reading a sentence. And every
rule written that day was broken that day, usually by whoever wrote it, often
within the hour. Memory is the weakest instrument available, so the four things
that kept going wrong become FIELDS rather than rules:

    enumeration  ->  "state the set"          1.148x had none
    bound        ->  "state the ceiling"      0.941 exceeded one nobody stated
    scope        ->  model or biology         a model result read as biology
    producer     ->  "no producer, no claim"  1.148x cited a script that cannot emit it

You cannot follow or forget a field. You can only leave it blank, and a blank
cell is visible to a reader who arrived five minutes ago.

WHAT THIS CHECKS, AND WHAT IT CANNOT. It checks FORM: fields present, producer
resolvable, enumeration containing an actual count, disposition in vocabulary.
It cannot check CONTENT. `probe_elevated_composition_profile.py` was named as
the producer of 1.148x and does not compute that quantity at all -- the file
existed, so a form check passes it. Only a person reading the script caught that.
The row catches omission; a reader catches a wrong entry; nothing catches a wrong
question.

RELATIONSHIP TO THE PAPER TABLES. docs/manuscript_claims.tsv is parsed BACKWARD
out of the manuscript, and claim_status.tsv then traces each claim to code -- 52
of 72 claims sit at `traced` and only 6 at `reproduced`, which is the cost of
reconstructing provenance after the fact. This file is the FORWARD half, written
by whoever ran the analysis while the numbers are in front of them. A paper claim
whose row exists arrives with its producer and population already recorded.

    measurement -> [ROW] -> a sentence in the paper -> manuscript_claims.tsv
                                                    -> claim_status.tsv

HOW WE WILL KNOW THIS WAS THE WRONG DESIGN. Written before first use, because a
criterion invented afterwards is a rationalisation (the project's own row
template, field 10: fix the decision rule before the run). Any one of these means
the FORMAT is wrong, not that people need reminding:

  1. Fields satisfied but empty of content -- an enumeration reading "all
     transcripts" instead of an n; a bound reading "none" where one exists.
  2. Rows written after the fact rather than at measurement -- the row's commit
     landing well after its producer's.
  3. Results reaching a shared document with no row at all. This is the 1.148
     case and it is the failure that matters most.
  4. Nobody consults them -- no narrative ever cites a row id, and the file is
     bookkeeping rather than infrastructure.

Usage:  tools/check_results.py [--file docs/results.tsv]
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

FIELDS = ["id", "assertion", "enumeration", "producer", "scope", "bound",
          "gated_by", "disposition"]
DISPOSITIONS = {"preliminary", "candidate", "in_paper", "retracted", "out"}
SCOPES = {"model", "biology", "both"}

# A count, an n, a job id -- something that pins the set to a size.
HAS_COUNT = re.compile(r"\b(n\s*[=:]?\s*)?\d[\d,]*\b")


def toplevel():
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                       capture_output=True, text=True)
    if r.returncode or not r.stdout.strip():
        sys.exit("FATAL: not inside a git worktree")
    return Path(r.stdout.strip())


def main():
    ROOT = toplevel()
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="analysis_plans/results.tsv")
    args = ap.parse_args()

    path = ROOT / args.file
    if not path.exists():
        print(f"  no results file at {args.file} — nothing to check yet.")
        return 0

    rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
    problems = []

    for r in rows:
        rid = (r.get("id") or "?").strip()

        retracted = (r.get("disposition") or "").strip() == "retracted"
        for f in FIELDS:
            if (r.get(f) or "").strip():
                continue
            # A RETRACTED row may lack an enumeration -- for 1.148x the missing
            # enumeration IS the reason it died. What it may never lack is the
            # reason, which lives in `bound`. The first version of this check had
            # the exemption backwards, and the format caught it on its first run:
            # requiring the enumeration of a result that never had one makes the
            # file permanently red, and a check that always fails is ignored.
            # A retraction is a claim (W247) and its provenance is its reason.
            if retracted and f == "enumeration":
                continue
            problems.append((rid, f, "EMPTY — the reason is required on a retracted row"
                             if retracted and f == "bound" else "EMPTY"))

        d = (r.get("disposition") or "").strip()
        if d and d not in DISPOSITIONS:
            problems.append((rid, "disposition", f"{d!r} not in {sorted(DISPOSITIONS)}"))

        s = (r.get("scope") or "").strip().split()[0].lower() if r.get("scope") else ""
        if s and s not in SCOPES:
            problems.append((rid, "scope", f"must begin model|biology|both, got {s!r}"))

        enum = (r.get("enumeration") or "").strip()
        if enum and not HAS_COUNT.search(enum):
            problems.append((rid, "enumeration", "no count in it — 'the set' is not a set"))

        # Producer must resolve to something in the tree. Cheap, and it is the
        # check that 1.148 would have survived -- so it is a floor, not a proof.
        prod = (r.get("producer") or "").strip()
        for tok in re.findall(r"[\w./-]+\.(?:py|R|sh)\b", prod):
            tok = tok.split(":")[-1]
            if not list(ROOT.rglob(Path(tok).name)):
                problems.append((rid, "producer", f"{tok} not found in the tree"))

    print(f"  {len(rows)} result row(s) in {args.file}")
    if not problems:
        print("  OK: every row is complete and every producer resolves.")
        print("  (Form only. A filled field can still be wrong — see the docstring.)")
        return 0

    for rid, f, why in problems:
        print(f"  FAIL  {rid:6} {f:12} {why}")
    print(f"\n  {len(problems)} problem(s). A row with an empty field is not ready to be cited.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
