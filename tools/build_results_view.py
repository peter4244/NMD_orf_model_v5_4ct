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

TWO VIEWS, NOT ONE, added 2026-08-02 on Harold's finding. index.tsv is one line per R-row, so it
cannot carry `quantity` -- a row holds several values and a single cell would have to pick one.
So quantities.tsv is one line per VALUE: (quantity, value, population, n, R-id). That is the
table tools/check_quantity_identity.py joins on, and it is the shape a graph builder wants
anyway, since the claim node is the value and not the filing.

WHY THIS MATTERS MORE THAN IT LOOKS. "One quantity, two values over different sets" is the first
section of the analysis repo's CLAUDE.md -- this project's declared signature defect -- and the
forward system could not see it. `quantity` was carried from claim_emit, preserved through
filing, and keyed on by nothing.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RECORDS = REPO / "claim_records"
VIEW = RECORDS / "index.tsv"
QVIEW = RECORDS / "quantities.tsv"

COLS = ["id", "state", "rung", "assertion", "producer", "producer_sha", "run_id",
        "n_inputs", "n_outputs", "n_values", "seed", "filed", "filer", "supersedes"]
QCOLS = ["quantity", "value", "n", "population", "claim_id", "r_id", "state", "producer"]


def cell(v):
    """Tabs and newlines would shift every later field, the way one stray tab in the analysis
    repo's worklog silently misaligned a row. Normalise; do not trust the writer."""
    return " ".join(str(v).split())


def build_quantities():
    rows = []
    for p in sorted(RECORDS.glob("R*.json")):
        d = json.loads(p.read_text())
        for v in d.get("values", []):
            rows.append({
                "quantity": cell(v.get("quantity", "")),
                "value": cell(v.get("value", "")),
                "n": cell(v.get("n", "")),
                "population": cell(v.get("population", "")),
                "claim_id": cell(v.get("claim_id", "")),
                "r_id": d["id"],
                "state": cell(d.get("state", "")),
                "producer": cell(d.get("producer", "")),
            })
    rows.sort(key=lambda r: (r["quantity"], r["r_id"]))
    out = ["\t".join(QCOLS)] + ["\t".join(r[c] for c in QCOLS) for r in rows]
    return "\n".join(out) + "\n", len(rows)


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
    qtext, qn = build_quantities()
    if a.check:
        stale = [name for name, path, want in
                 (("index.tsv", VIEW, text), ("quantities.tsv", QVIEW, qtext))
                 if (path.read_text() if path.exists() else "") != want]
        if stale:
            sys.exit(f"STALE ({', '.join(stale)}) — run: python3 tools/build_results_view.py")
        print(f"  ok  claim_records/ views current ({n} rows, {qn} values)")
        return
    VIEW.write_text(text)
    QVIEW.write_text(qtext)
    print(f"  wrote claim_records/index.tsv ({n} rows)")
    print(f"  wrote claim_records/quantities.tsv ({qn} values)")


if __name__ == "__main__":
    main()
