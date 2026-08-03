#!/usr/bin/env python3
"""File a forward result — one R-row, recording provenance and nothing else.

    python3 tools/file_result.py --assertion "..." --producer analysis_plans/x.py \
        --run-id 8896445 --runlog <path> --trace <path> --seed 100 \
        --values /path/to/claim_values.<run>.tsv --filer hi

SCOPE IS PROVENANCE ONLY (D65). What goes in a row is exactly what is needed to build the
data-to-claims graph: the assertion, the producer and its commit, the files read and written,
the emitted values with their populations, and the run identity. Scientific standing -- the
null a result was measured against, the caveat that must travel with it, which gate blocks it,
its paper disposition -- is a SEPARATE review process and deliberately has no field here. Do
not add one; that decision has a D-number.

## WHY THE REFUSALS ARE THE WHOLE SECURITY MODEL

Rewritten 2026-08-02 after Harold's review, which found the defect that matters: **filing a
result silences check_d50.py.** He filed the assertion "keto ratio is 1.148x" with --producer
pointing at THIS script, which computes no such thing, and a hand-written one-line values file
with population, producer_file and producer_line all empty. Nothing refused it, and 1.148
disappeared from the gate's report.

That is exactly what 1.148x was -- a number carrying authority it never earned -- so the gate
failed against its own founding case. And the adoption test ("filing must be faster than not
filing") makes the silencing path the CHEAP one.

The gate resolves document numbers against filed values, so **the gate's integrity is entirely
this file's refusals.** claim_emit.emit() already raises on a blank population; the hole was
that this script accepted any TSV. So: a values file must be demonstrably claim_emit output,
its run_id must match the run being filed, and every row must carry the fields whose absence
is claim_emit's signature of not having produced it.

Six refusals, each named after something measured:

  values not from claim_emit   producer_file/producer_line empty is the signature. Harold's
                               fabricated row had both empty and filed green.
  run_id mismatch              D66(b) keys claim_values by run_id, and claim_emit's docstring
                               warns that appending to a stale file silently mixes vintages.
  blank population             Unstated populations are the dominant defect class here, about
                               one per manuscript section. The spec promised this guard as
                               "an empty field is visibly incomplete" and did not implement it.
  producer dirty               Was a warning that still wrote, which contradicted this file's
                               own two calibrations -- "a warning on the happy path is a warning
                               nobody reads" and "a guard that cannot fail is worse than no
                               guard". If the producer has uncommitted changes then producer_sha
                               does not describe the code that ran, and the row records a
                               version it does not have.
  producer_file mismatch       Added on Harold's SECOND pass, and it is the one that closes
                               fabrication rather than slowing it. With a valid header, a matching
                               run_id and every field populated, a hand-written TSV still filed
                               cleanly against a producer computing nothing of the sort -- because
                               producer_file was never compared to --producer. claim_emit writes
                               producer_file from the ACTUAL CALL FRAME, so a disagreement is
                               close to proof.
  trace misses a real .h5      See below.

## THE SHORT-TRACE GUARD

trace_reads.py's audit hook is blind to HDF5 -- h5py opens through the HDF5 C library and never
touches CPython's file API, so ZERO events fire. results_4ct/nmd_orf_data.h5 is THE central
input of evaluate.py, 11_kernel_shap_branches.py and infer_uorf_attention.py, so an unpatched
trace of any of them looks perfectly clean while omitting the one file that matters.

Harold's correction: refusing only at zero reads is the wrong threshold, because the failure
produces zero events only when the h5 is the producer's SOLE input -- one config file alongside
it gives a trace of length 1 that passes. So the check is now AIMED AT THE MEASURED FAILURE
rather than at a guessed floor: if the producer actually READS HDF5, an HDF5 file must appear in
the trace. The zero-read refusal is kept as the coarse backstop.

NARROWED AGAIN on Harold's second pass, because his own steer implemented literally false-
positived: matching any mention of h5 refused a producer whose only reference was the docstring
explaining this guard. It now requires an I/O site -- an h5py.File call, a read_hdf, or a quoted
path literal ending .h5/.hdf5 -- with comments and triple-quoted blocks stripped first, so prose
can never trigger it.

## IDENTITY, AND WHY IDS CARRY A FILER

D66 puts this ledger in the model repo, which is worktree-split: master, interp and results are
three checkouts. One file per row defeats CONTENT conflict. Harold showed it does not defeat
IDENTITY conflict -- RECORDS resolves per checkout, so two windows filing concurrently both
scan their own directory, both allocate R0007, and both write that path with different content.

Pete's ruling, 2026-08-02: **filer-prefixed R-numbers.** Each filer allocates inside its own
namespace, so two windows never collide. The residual is one filer working in two worktrees at
once, which is why the write is O_EXCL rather than a plain open -- if the path exists, stop.
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RECORDS = REPO / "claim_records"

# Namespacing follows the repo convention for cluster jobs and outputs: hi = interpretability,
# md = model. gu = guardian/organizer, st = storyteller.
FILERS = {"hi": "interpretability", "md": "model", "gu": "guardian", "st": "storyteller"}
ID_RE = re.compile(r"^R-([a-z]{2})-(\d{4})\.json$")

# claim_emit's own columns. A file missing any of these was not written by it.
EMIT_COLS = ("claim_id", "quantity", "value", "n", "population",
             "producer_file", "producer_line", "run_id")
# NARROWED 2026-08-02. The first version matched any mention of h5 anywhere in the source, and so
# refused a producer whose only reference was a DOCSTRING explaining this very guard. Harold's
# steer, his defect, and he reported it. Now it looks for an actual I/O site: an h5py call, or a
# quoted path ending .h5/.hdf5. A comment or a prose mention no longer counts.
H5_HINT = re.compile(
    r"""h5py\s*\.\s*File            # h5py.File(...)
      | \bpd\s*\.\s*read_hdf       # pandas.read_hdf
      | ['"][^'"\n]*\.(?:h5|hdf5)['"]  # a quoted path literal
    """, re.X | re.I)


def sh(cmd, cwd=None, check=True):
    """Run and return stdout. NEVER suppresses stderr -- `git merge --ff-only -q 2>/dev/null`
    reporting success for hours while doing nothing is the logged instance (W240)."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if check and r.returncode != 0:
        sys.exit(f"FAILED: {cmd}\n{r.stderr.strip()}")
    return r.stdout.strip()


def next_id(filer):
    """Allocate the lowest id not yet used BY THIS FILER.

    Scans the directory rather than keeping a counter: a counter file is a second place the
    state lives, and two worktrees would both increment it. Genuinely lowest-unused, not
    max+1 -- the previous docstring said one and the code did the other, which is the exact
    class of defect this project exists to complain about."""
    used = {int(m.group(2)) for p in RECORDS.glob(f"R-{filer}-*.json")
            if (m := ID_RE.match(p.name))}
    n = 1
    while n in used:
        n += 1
    return f"R-{filer}-{n:04d}"


def read_trace(path):
    """Split a trace into reads and writes. Columns are fn/kind/exists/path; the tracer already
    promotes any path touched by a write to `write`, so a path appears in one list only."""
    rows = [ln.split("\t") for ln in Path(path).read_text().splitlines()[1:] if ln.strip()]
    reads = sorted({r[3] for r in rows if len(r) > 3 and r[1] == "read"})
    writes = sorted({r[3] for r in rows if len(r) > 3 and r[1] == "write"})
    return reads, writes


def read_values(path, run_id, producer):
    """Carry the emitted values across, refusing anything that is not claim_emit output.

    `population` is read rather than retyped because it is known at the emit call site and
    nowhere else -- claim_emit's docstring makes that argument and raises on a blank one. This
    function is the second half of that guarantee: it verifies the file it was handed actually
    came from there."""
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    if len(lines) < 2:
        sys.exit(f"REFUSED: {path} has no value rows.")
    head = lines[0].split("\t")
    missing = [c for c in EMIT_COLS if c not in head]
    if missing:
        sys.exit(f"REFUSED: {path} is not claim_emit output — missing column(s): "
                 f"{', '.join(missing)}.\nEmission is required at filing (D66). Wire "
                 "claim_emit.emit() into the producer and re-run it; do not hand-write this file.")
    # run_id is carried PER VALUE, not just on the row. Harold, 2026-08-02: a values file can span
    # two runs, so the row-level run_id is not a substitute, and check_unfiled_values.py cannot key
    # on (run_id, quantity) unless it is here.
    keep = ("claim_id", "quantity", "value", "n", "population",
            "producer_file", "producer_line", "run_id")
    out, bad = [], []
    for i, ln in enumerate(lines[1:], 2):
        row = dict(zip(head, ln.split("\t")))
        for col in ("producer_file", "producer_line", "population", "quantity", "value"):
            if not str(row.get(col, "")).strip():
                bad.append(f"  line {i}: blank `{col}`")
        rr = str(row.get("run_id", "")).strip()
        if rr and rr != str(run_id):
            bad.append(f"  line {i}: run_id {rr!r} != --run-id {run_id!r}")
        # THE CHEAPEST REMAINING REFUSAL, and it closes the fabrication path Harold left open on
        # his second pass. With a valid header, a matching run_id and every field populated, a
        # hand-written TSV still filed cleanly against a producer that computes nothing of the
        # sort -- because producer_file was never compared to --producer. claim_emit writes
        # producer_file from the ACTUAL CALL FRAME, so a disagreement is close to proof that the
        # named producer did not emit these values.
        pf = str(row.get("producer_file", "")).strip()
        if pf and Path(pf).name != Path(producer).name:
            bad.append(f"  line {i}: producer_file {pf!r} is not --producer {producer!r}")
        out.append({k: row.get(k, "") for k in keep})
    if bad:
        sys.exit("REFUSED: the values file does not carry what claim_emit writes.\n"
                 + "\n".join(bad[:12])
                 + "\n\nBlank producer_file/producer_line is the signature of a hand-written "
                   "row. A blank population is the dominant defect class in this manuscript. "
                   "A run_id mismatch means the file mixes vintages, which claim_emit's own "
                   "docstring warns about.")
    return out


def check_trace_reaches_h5(producer, reads):
    """If the producer's source mentions HDF5, an HDF5 file must appear in the trace.

    Aimed at the measured failure rather than a guessed floor. The audit hook records nothing
    for h5py, so a producer whose central input is an .h5 yields a trace that looks clean."""
    src = (REPO / producer).read_text(errors="replace")
    # Remove comments and triple-quoted blocks first: a docstring is prose, not an I/O site.
    stripped = re.sub(r'(?s)""".*?"""|\'\'\'.*?\'\'\'', "", src)
    stripped = re.sub(r"(?m)#.*$", "", stripped)
    if not H5_HINT.search(stripped):
        return
    if not any(p.lower().endswith((".h5", ".hdf5")) for p in reads):
        sys.exit(
            f"REFUSED: {producer} references HDF5 but its trace records no .h5 read.\n"
            "trace_reads.py's audit hook does not fire for h5py — it opens through the HDF5 C\n"
            "library and never touches CPython's file API — so this is the signature of a\n"
            "trace that looks complete and omitted the producer's central input.\n"
            "Confirm the h5py wrapper is active, then re-run the trace.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--assertion", required=True, help="what is claimed. The only human field.")
    ap.add_argument("--producer", required=True, help="repo-relative path to the script")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runlog", required=True, help="path to what the run printed")
    ap.add_argument("--trace", required=True, help="trace_reads output for this run")
    ap.add_argument("--values", required=True, help="claim_emit output for this run")
    ap.add_argument("--filer", required=True, choices=sorted(FILERS),
                    help="; ".join(f"{k}={v}" for k, v in sorted(FILERS.items())))
    ap.add_argument("--seed", default="")
    ap.add_argument("--supersedes", default="", help="an R-number this replaces")
    ap.add_argument("--allow-short-trace", action="store_true")
    ap.add_argument("--short-trace-reason", default="")
    a = ap.parse_args()

    if not (REPO / a.producer).exists():
        sys.exit(f"producer not found: {a.producer}")
    for f in (a.runlog, a.trace, a.values):
        if not Path(f).exists():
            sys.exit(f"not found: {f}")

    if sh(f"git status --porcelain -- {a.producer}", cwd=REPO):
        sys.exit(f"REFUSED: {a.producer} has uncommitted changes, so the recorded "
                 "producer_sha would not describe the code that ran. Commit it, then file.")

    reads, writes = read_trace(a.trace)
    check_trace_reaches_h5(a.producer, reads)
    if not reads and not a.allow_short_trace:
        sys.exit(f"REFUSED: the trace records ZERO reads ({a.trace}). Either the producer "
                 "genuinely read nothing, or the tracer could not see its I/O route. Confirm "
                 "which, then re-file with --allow-short-trace --short-trace-reason '...'. "
                 "Do not pass the flag to move on.")

    values = read_values(a.values, a.run_id, a.producer)

    RECORDS.mkdir(exist_ok=True)
    rid = next_id(a.filer)
    row = {
        "id": rid,
        "assertion": a.assertion,
        "producer": a.producer,
        "producer_sha": sh("git rev-parse HEAD", cwd=REPO),
        "inputs": reads,
        "outputs": writes,
        "values": values,
        "run_id": a.run_id,
        "runlog": a.runlog,
        "trace": a.trace,
        "seed": a.seed,
        "state": "live",
        "rung": "asserted",
        "supersedes": a.supersedes,
        "filed": datetime.date.today().isoformat(),
        "filer": a.filer,
    }
    if not reads:
        row["short_trace_reason"] = a.short_trace_reason

    # O_EXCL, not a plain write. Filer-prefixed ids stop two WINDOWS colliding; this stops one
    # filer in two worktrees from silently overwriting itself.
    target = RECORDS / f"{rid}.json"
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        sys.exit(f"REFUSED: {target.name} already exists. Another filing raced this one; "
                 "re-run and it will take the next id.")
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(row, indent=2) + "\n")

    print(f"  filed {rid}  ({len(reads)} reads, {len(writes)} writes, {len(values)} values)")
    print("  regenerate the view: python3 tools/build_results_view.py")


if __name__ == "__main__":
    main()
