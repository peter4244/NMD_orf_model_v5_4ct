#!/usr/bin/env python3
"""File a forward result — one R-row, recording provenance and nothing else.

    python3 tools/file_result.py --assertion "..." --producer analysis_plans/x.py \
        --run-id 8896445 --runlog <path> --trace <path> --seed 100 \
        --values /path/to/claim_values.<run>.tsv --filer interpretability

SCOPE IS PROVENANCE ONLY (D65). What goes in a row is exactly what is needed to build the
data-to-claims graph: the assertion, the producer and its commit, the files read and written,
the emitted values with their populations, and the run identity. Scientific standing -- the
null a result was measured against, the caveat that must travel with it, which gate blocks it,
its paper disposition -- is a SEPARATE review process and deliberately has no field here. Do
not add one; that decision has a D-number.

ONE FILE PER ROW, AND WHY. D66 puts this ledger in the model repo, which is worktree-split:
master, interp and results are three checkouts. A single append-only TSV here would take
concurrent writes from three windows and merge inside a generated table -- the failure D53
avoided by refusing to split the analysis repo. One JSON file per row means two windows filing
at once touch different files and there is nothing to resolve. tools/build_results_view.py
regenerates the flat view over the directory.

DIRECTORY NAME. `claim_records/`, deliberately NOT `results_rows/`. Every results_* directory
in this repo is gitignored data; a ledger that vanishes under a future glob is not a risk worth
taking for a tidier name.

INPUTS ARE TRACED, NEVER TYPED. D27: backward walking answers only which file produces a claim;
forward execution answers what it reads, completely, in one run. Hand-resolved read-sets were
wrong three times running, each miss costing a 25-agent fan-out. So this reads the trace rather
than accepting a list.

THE SHORT-TRACE GUARD, and the failure it is named after. trace_reads.py's audit hook is blind
to HDF5 -- h5py opens through the HDF5 C library and never touches CPython's file API, so ZERO
events fire. `results_4ct/nmd_orf_data.h5` is THE central input of evaluate.py,
11_kernel_shap_branches.py and infer_uorf_attention.py, so an unpatched trace of any of them
looks perfectly clean while omitting the one file that matters. h5py is patched because it was
measured; the next C extension will not be. C34 states the principle: a trace is complete only
over the call surface it wraps, so a SHORT trace is suspect. A trace with no reads is refused
here rather than warned about, because a warning on the happy path is a warning nobody reads.
"""
import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RECORDS = REPO / "claim_records"
ID_RE = re.compile(r"^R(\d{4})\.json$")


def sh(cmd, cwd=None):
    """Run and return stdout. Never suppresses stderr -- a guard that cannot fail is worse
    than no guard, and `git merge --ff-only -q 2>/dev/null` reporting success for hours while
    doing nothing is the logged instance (W240)."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        sys.exit(f"FAILED: {cmd}\n{r.stderr.strip()}")
    return r.stdout.strip()


def next_id():
    """Allocate the lowest unused R number. Scans the directory rather than keeping a counter:
    a counter file is a second place the state lives, and two worktrees would both increment it."""
    used = {int(m.group(1)) for p in RECORDS.glob("R*.json")
            if (m := ID_RE.match(p.name))}
    return f"R{(max(used) + 1 if used else 1):04d}"


def read_trace(path):
    """Split a trace into reads and writes. Columns are fn/kind/exists/path; the tracer already
    promotes any path touched by a write to `write`, so a path appears in one list only."""
    rows = [ln.split("\t") for ln in Path(path).read_text().splitlines()[1:] if ln.strip()]
    reads = sorted({r[3] for r in rows if len(r) > 3 and r[1] == "read"})
    writes = sorted({r[3] for r in rows if len(r) > 3 and r[1] == "write"})
    return reads, writes


def read_values(path):
    """Carry the emitted values across. `population` is mandatory at the emit call site and is
    the reason this is read rather than retyped -- it is known there and nowhere else."""
    lines = Path(path).read_text().splitlines()
    if not lines:
        return []
    head = lines[0].split("\t")
    keep = ("claim_id", "quantity", "value", "n", "population", "producer_file", "producer_line")
    out = []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        row = dict(zip(head, ln.split("\t")))
        out.append({k: row.get(k, "") for k in keep})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--assertion", required=True, help="what is claimed. The only human field.")
    ap.add_argument("--producer", required=True, help="repo-relative path to the script")
    ap.add_argument("--run-id", required=True, help="job id, or a run identifier")
    ap.add_argument("--runlog", required=True, help="path to what the run printed")
    ap.add_argument("--trace", required=True, help="trace_reads output for this run")
    ap.add_argument("--values", required=True, help="claim_values file for this run")
    ap.add_argument("--seed", default="")
    ap.add_argument("--filer", required=True, help="which window filed it")
    ap.add_argument("--allow-short-trace", action="store_true",
                    help="file anyway when the trace records no reads. Records WHY in the row.")
    ap.add_argument("--short-trace-reason", default="")
    a = ap.parse_args()

    if not (REPO / a.producer).exists():
        sys.exit(f"producer not found: {a.producer}")
    for f in (a.runlog, a.trace, a.values):
        if not Path(f).exists():
            sys.exit(f"not found: {f}")

    reads, writes = read_trace(a.trace)
    if not reads and not a.allow_short_trace:
        sys.exit(
            f"REFUSED: the trace records ZERO reads ({a.trace}).\n"
            "That is the exact signature of the h5py blindness -- a clean-looking trace that\n"
            "omitted the one file that mattered. Either the producer genuinely read nothing,\n"
            "or the tracer could not see its I/O route. Confirm which, then re-file with\n"
            "--allow-short-trace --short-trace-reason '...'. Do not pass the flag to move on.")

    values = read_values(a.values)
    if not values:
        sys.exit(f"REFUSED: no emitted values in {a.values}. Emission is required at filing "
                 "(D66) -- a result whose producer emitted nothing has no machine-checkable "
                 "number behind it.")

    rid = next_id()
    RECORDS.mkdir(exist_ok=True)
    row = {
        "id": rid,
        "assertion": a.assertion,
        "producer": a.producer,
        "producer_sha": sh("git rev-parse HEAD", cwd=REPO),
        "producer_dirty": bool(sh("git status --porcelain -- " + a.producer, cwd=REPO)),
        "inputs": reads,
        "outputs": writes,
        "values": values,
        "run_id": a.run_id,
        "runlog": a.runlog,
        "trace": a.trace,
        "seed": a.seed,
        "state": "live",
        "rung": "asserted",
        "supersedes": "",
        "filed": datetime.date.today().isoformat(),
        "filer": a.filer,
    }
    if not reads:
        row["short_trace_reason"] = a.short_trace_reason
    (RECORDS / f"{rid}.json").write_text(json.dumps(row, indent=2) + "\n")
    print(f"  filed {rid}  ({len(reads)} reads, {len(writes)} writes, {len(values)} values)")
    if row["producer_dirty"]:
        print("  WARNING: producer has uncommitted changes, so producer_sha does not describe "
              "the code that ran. Commit it and re-file.")
    print("  regenerate the view: python3 tools/build_results_view.py")


if __name__ == "__main__":
    main()
