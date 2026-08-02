#!/usr/bin/env python
"""
probe_bank_floor_chunk_invariance.py — what is the mutagenesis bank's actual
noise floor, and does it differ for the two branch columns?

WHY. The interpretability handoff of 2026-08-01 raised three precision questions
that must be settled before `vals_capture` / `vals_decay` mean anything, because
the reported median |effect through capture| (1.3e-06) sits at the same order as
float32 spacing at a log-odds of 10 (9.5e-07) and as a previously measured
cross-batch baseline floor (5.96e-07). Two of the three are answerable by reading
build_ism_bank.py. The third is not: nothing in the bank measures the floor of the
BRANCH columns, only of the total logit (`chunk_offset`).

THE OPERATOR. Rebuild the same transcripts with the same checkpoint at several
`--chunk-rows`, and compare the shards entry by entry. Chunk size is supposed to
bound memory and change nothing else -- build_ism_bank.py says so at its OOM
backoff: "Chunk size bounds memory and nothing else, so shrinking it changes wall
time and not a number." Any entry that moves is the measurement's own noise, since
the model, the weights, the inputs and the arithmetic are identical.

WHY THIS IS THE RIGHT NULL AND A NO-OP IS NOT. Substituting the observed base for
itself is the reference the bank already differences against, so it is exactly
zero by construction and can never expose the floor. The floor lives in the
DIFFERENCE between two rows of the same chunk, and only a change of chunk shape
moves it while holding the answer fixed.

WHY CHUNK SIZE IS NOT A CONTRIVED PERTURBATION. It happens in production. The OOM
backoff at build_ism_bank.py:640 halves `chunk_rows` and never restores it, so
every transcript after the first OOM in a run is built at a different chunk shape
from every transcript before it -- and `chunk_rows` is not among the fields the
shard records.

READ THE SPREAD, NOT THE MAXIMUM. One transcript's floor is not the bank's floor:
the K=1 arm crosses a batch-size kernel regime and the K=4 arm does not.

  python analysis_plans/probe_bank_floor_chunk_invariance.py \
      --chunk-rows 4096,512,97,4 \
      --only ENST00000466568.1,ENST00000492005.1
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# Every float array the bank ships per transcript. Named explicitly rather than
# discovered from the file, so a field added later shows up as a KeyError here
# instead of silently going unchecked.
FLOAT_KEYS = ["vals", "vals_capture", "vals_decay", "dsel", "dstart",
              "chunk_offset", "mass"]
EXACT_KEYS = ["obs", "valid", "dgc", "fill_count"]
SCALARS = ["base_logit", "base_logit_training", "noop", "n_floor"]


def build(out, chunk_rows, only, tensor, checkpoint, split, device):
    """One bank build. Its own --out, so its own shard directory: a rerun that
    shared shards with an earlier chunk size would compare a build against
    itself and pass no matter what."""
    shards = Path(str(out) + ".shards")
    if shards.exists():
        raise SystemExit(f"{shards} exists; refusing to reuse shards from an "
                         f"earlier build -- the comparison would be vacuous")
    cmd = [sys.executable, str(REPO / "build_ism_bank.py"),
           "--tensor", tensor, "--checkpoint", checkpoint, "--split", split,
           "--out", str(out), "--chunk-rows", str(chunk_rows),
           "--device", device, "--only", only]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:]); print(r.stderr[-4000:])
        raise SystemExit(f"build failed at chunk_rows={chunk_rows}")
    for line in r.stdout.splitlines():
        if "floor=" in line or "batch-shape offset" in line:
            print("   " + line.strip())
    return shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-rows", default="4096,512,97,4")
    ap.add_argument("--only", required=True)
    ap.add_argument("--tensor", default="results_tensor_v6")
    ap.add_argument("--checkpoint",
                    default="results_interp_all/v6_checkpoints/b8_s100.pt")
    ap.add_argument("--split", default="results_ism_v6/ism_subset.tsv")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workdir", required=True,
                    help="scratch directory for the builds; must be empty of "
                         "prior shard directories for these chunk sizes")
    ap.add_argument("--compare-only", action="store_true",
                    help="reuse the builds already in --workdir. For re-cutting "
                         "the same builds; it cannot be used to make the "
                         "comparison itself pass, since it compares whatever is "
                         "there and says which chunk sizes it found")
    args = ap.parse_args()

    sizes = [int(x) for x in args.chunk_rows.split(",")]
    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)
    if args.compare_only:
        shard_dirs = {c: work / f"cr{c}.h5.shards" for c in sizes}
        for c, d in shard_dirs.items():
            if not d.is_dir():
                raise SystemExit(f"--compare-only: {d} is not there")
        print(f"comparing existing builds at chunk_rows {sizes}")
    else:
        shard_dirs = {c: build(work / f"cr{c}.h5", c, args.only, args.tensor,
                               args.checkpoint, args.split, args.device)
                      for c in sizes}

    ref = sizes[0]
    txs = sorted(p.stem for p in shard_dirs[ref].glob("*.npz"))
    print(f"\n{len(txs)} transcripts, reference chunk_rows={ref}\n")

    for tx in txs:
        R = np.load(shard_dirs[ref] / f"{tx}.npz")
        K = len(R["p_capture"])
        print(f"{tx}   K={K}   base_logit {float(R['base_logit']):+.6f}   "
              f"shipped chunk_offset max {R['chunk_offset'].max():.3e}")
        print(f"  {'array':<14} {'vs cr':>7} {'n finite':>9} {'n differ':>9} "
              f"{'max|diff|':>11} {'x chunk_offset':>15}")
        off_max = float(R["chunk_offset"].max())
        for c in sizes[1:]:
            Q = np.load(shard_dirs[c] / f"{tx}.npz")
            for k in EXACT_KEYS:
                n = int((np.atleast_1d(R[k]) != np.atleast_1d(Q[k])).sum())
                if n:
                    print(f"  {k:<14} {c:>7} {'':>9} {n:>9,}   EXACT FIELD MOVED")
            for k in FLOAT_KEYS:
                a, b = np.atleast_1d(R[k]), np.atleast_1d(Q[k])
                fin = np.isfinite(a) & np.isfinite(b)
                # A NaN on one side and a number on the other is a differing
                # entry, not an entry to drop: dropping it would hide the worst
                # possible disagreement behind the word "finite".
                mism = int((np.isfinite(a) != np.isfinite(b)).sum())
                d = np.abs(a[fin] - b[fin]) if fin.any() else np.zeros(1)
                nd, mx = int((d > 0).sum()) + mism, float(d.max())
                rel = mx / off_max if off_max > 0 else float("nan")
                print(f"  {k:<14} {c:>7} {int(fin.sum()):>9,} {nd:>9,} "
                      f"{mx:>11.3e} {rel:>15.2f}")
            for k in SCALARS:
                if float(R[k]) != float(Q[k]):
                    print(f"  {k:<14} {c:>7}   {float(R[k]):.6e} -> {float(Q[k]):.6e}")
        print()

    # ---- the floor in absolute terms, and what it censors -------------------
    # `chunk_offset` is the WRONG normalizer and the table above shows why: the
    # measured floor runs 1.5x it on one transcript and 20x on another, because
    # chunk_offset measures the distance to the batch-K base pass -- which is
    # large exactly when K is small -- and not the chunk-to-chunk variation the
    # difference actually carries.
    #
    # What does bound it on every transcript here is a few float32 ulp of the
    # transcript's own logit. That is where the resolution is lost: aggregate()
    # runs in float64, but z_p and z_d come out of the encoders in float32, and
    # casting them up afterwards cannot recover what the encoder already rounded.
    print("=" * 78)
    print("absolute floor, and the censoring it implies\n")
    print(f"  {'transcript':<20} {'K':>3} {'|logit|':>8} {'3 ulp32':>10} "
          f"{'floor vals':>11} {'floor cap':>10} {'floor dec':>10}")
    pooled = {k: [] for k in ("vals", "vals_capture", "vals_decay")}
    for tx in txs:
        R = np.load(shard_dirs[ref] / f"{tx}.npz")
        others = [np.load(shard_dirs[c] / f"{tx}.npz") for c in sizes[1:]]
        fl = {k: max(float(np.nanmax(np.abs(R[k] - Q[k]))) for Q in others)
              for k in pooled}
        bl = abs(float(R["base_logit"]))
        print(f"  {tx:<20} {len(R['p_capture']):>3} {bl:>8.3f} "
              f"{3 * np.spacing(np.float32(bl)):>10.2e} {fl['vals']:>11.3e} "
              f"{fl['vals_capture']:>10.3e} {fl['vals_decay']:>10.3e}")
        m = np.isfinite(R["vals"])
        for k in pooled:
            pooled[k].append(np.abs(R[k][m]))
    n = len(np.concatenate(pooled["vals"]))
    print(f"\n  pooled over {n:,} substitutions at chunk_rows={ref}")
    for k, xs in pooled.items():
        x = np.concatenate(xs)
        print(f"    {k:<13} median {np.median(x):.3e}   "
              f"above 3e-06 {100 * (x > 3e-6).mean():5.1f}%   "
              f"above 1e-05 {100 * (x > 1e-5).mean():5.1f}%")
    print("\n  Read the capture row against the other two. The split is sound "
          "arithmetically;\n  what differs is dynamic range, and that is what "
          "decides whether it is readable.")


if __name__ == "__main__":
    main()
