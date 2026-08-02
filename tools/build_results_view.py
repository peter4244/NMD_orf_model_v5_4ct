#!/usr/bin/env python3
"""Generate the flat view over claim_records/. GENERATED — never hand-edited.

    python3 tools/build_results_view.py [--check]

WHY A VIEW AND NOT A LEDGER. The rows are the ledger, one JSON file each, because this repo is
worktree-split and a shared append-only table would merge inside a generated artifact (D66, and
the concurrency argument in tools/file_result.py). This produces the thing people actually read
and grep, deterministically, from those rows. Same idiom as the analysis repo: views are
generated, a stale view is regenerated rather than deleted, and hand-editing one is how the two
copies start disagreeing.

--check exits non-zero if the view on disk differs from a fresh build, which is what a
pre-commit hook wants. Without it, the tool writes.

WHAT THE VIEW DELIBERATELY OMITS: inputs and outputs. A row can carry hundreds of traced paths
and flattening them into a cell produces a table nobody can read and a diff nobody can review.
The count is shown instead; the paths stay in the row, which is where a graph builder reads
them from anyway.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RECORDS = REPO / "claim_records"
VIEW = RECORDS / "index.tsv"

COLS = ["id", "state", "rung", "assertion", "producer", "producer_sha", "run_id",
        "n_inputs", "n_outputs", "n_values", "seed", "filed", "filer", "supersedes"]


def build():
    rows = []
    for p in sorted(RECORDS.glob("R*.json")):
        d = json.loads(p.read_text())
        rows.append({
            "id": d["id"],
            "state": d.get("state", ""),
            "rung": d.get("rung", ""),
            # Tabs and newlines would break the TSV the same way a stray tab in worklog.tsv
            # silently shifted every later field. Replace, do not trust the writer.
            "assertion": " ".join(str(d.get("assertion", "")).split()),
            "producer": d.get("producer", ""),
            "producer_sha": str(d.get("producer_sha", ""))[:12],
            "run_id": d.get("run_id", ""),
            "n_inputs": str(len(d.get("inputs", []))),
            "n_outputs": str(len(d.get("outputs", []))),
            "n_values": str(len(d.get("values", []))),
            "seed": str(d.get("seed", "")),
            "filed": d.get("filed", ""),
            "filer": d.get("filer", ""),
            "supersedes": d.get("supersedes", ""),
        })
    out = ["\t".join(COLS)]
    out += ["\t".join(r[c] for c in COLS) for r in rows]
    return "\n".join(out) + "\n", len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not RECORDS.exists():
        sys.exit("claim_records/ does not exist — nothing filed yet")
    text, n = build()
    if a.check:
        cur = VIEW.read_text() if VIEW.exists() else ""
        if cur != text:
            sys.exit("STALE — run: python3 tools/build_results_view.py")
        print(f"  ok  claim_records/index.tsv current ({n} rows)")
        return
    VIEW.write_text(text)
    print(f"  wrote claim_records/index.tsv ({n} rows)")


if __name__ == "__main__":
    main()
