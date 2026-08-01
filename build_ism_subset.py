#!/usr/bin/env python
"""
build_ism_subset.py — the stratified subset the mutagenesis bank is built over.

REPLACES the label-stratified random prefix of plan §9.3 step 2, and the reason
is decisive. The mechanism cell the section is about — degraded WITHOUT a
premature stop in the main reading frame — is 381 transcripts in ~42,000, under
1%. A random 1,000 draws nine of them and a random 5,000 draws forty-five.
Neither supports a stratified analysis, so the prefix design answered the wrong
question however tidy its extension property was.

Instead: take the scarce cells WHOLE, sample the abundant ones, and record a
SAMPLING WEIGHT per transcript so any population-level estimate can be
reweighted. Without the weight the bank silently describes a population nobody
has.

THE CELLS. Main ORF is the GENCODE-annotated start projected into the isoform's
own exon coordinates (results_ism_v6/gencode_candidate_flags.tsv), not the
expression-derived reference — the latter is label-adjacent and this
stratification is about the label. "Main-ORF stop" means that candidate has a
junction more than 50 bases past its stop codon.

Transcripts with no GENCODE annotation are their own cells rather than dropped.
36.5% of the pool has none, and that is where most NMD lives; a stratification
that silently excluded them would describe annotated transcripts and be read as
describing transcripts.

Usage:
    python build_ism_subset.py --n 5000 --out results_ism_v6
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
POOL = REPO / "results_pool_v6" / "orf_pool.tsv"
FLAGS = REPO / "results_ism_v6" / "gencode_candidate_flags.tsv"
SPLIT = REPO / "results_ism_v6" / "discovery_confirmation_split.tsv"
SEED = 20260801


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", default="results_ism_v6")
    args = ap.parse_args()
    t0 = time.time()

    sp = pd.read_csv(SPLIT, sep="\t")                    # 41,765, the tensor's own set
    pool = pd.read_csv(POOL, sep="\t",
                       usecols=["isoform_id", "slot", "n_downstream_ejc"])
    fl = pd.read_csv(FLAGS, sep="\t",
                     usecols=["isoform_id", "slot", "is_gencode_start"])
    m = pool.merge(fl, on=["isoform_id", "slot"], how="left")

    # the annotated main ORF, and whether its own stop has a downstream junction
    main = m[m.is_gencode_start == 1][["isoform_id", "n_downstream_ejc"]]
    main = main.drop_duplicates("isoform_id").set_index("isoform_id")
    sp["has_annotation"] = sp.isoform_id.isin(main.index)
    sp["main_orf_stop"] = sp.isoform_id.map(main.n_downstream_ejc).gt(0)

    def cell(r):
        lab = "NMD" if r.is_nmd == 1 else "control"
        if not r.has_annotation:
            return f"{lab} / no annotation"
        return f"{lab} / {'main-ORF stop' if r.main_orf_stop else 'NO main-ORF stop'}"
    sp["cell"] = sp.apply(cell, axis=1)

    counts = sp.cell.value_counts().sort_index()
    print(f"population {len(sp):,} transcripts (the tensor's own set)")
    print(f"  with an admitted GENCODE start: {int(sp.has_annotation.sum()):,} "
          f"({100*sp.has_annotation.mean():.1f}%)\n")
    print(f"  {'cell':<34} {'N':>8}  {'% of pool':>10}")
    for c, n in counts.items():
        print(f"  {c:<34} {n:>8,}  {100*n/len(sp):>9.2f}%")

    # ---- take the scarce cells whole, sample the abundant ones -------------
    rng = np.random.default_rng(SEED)
    # a cell at or below this is taken entire; the rest share what is left
    SCARCE = max(1, args.n // len(counts))
    take, weights = [], {}
    small = {c: n for c, n in counts.items() if n <= SCARCE}
    budget = args.n - sum(small.values())
    big = {c: n for c, n in counts.items() if n > SCARCE}
    per_big = budget // max(len(big), 1)

    print(f"\n  cells at or below {SCARCE:,} taken WHOLE; the rest share "
          f"{budget:,} at {per_big:,} each")
    print(f"\n  {'cell':<34} {'N':>8} {'taken':>8} {'weight':>9}")
    for c, n in counts.items():
        ids = sp.loc[sp.cell == c, "isoform_id"].to_numpy()
        k = n if c in small else min(n, per_big)
        pick = ids if k == n else rng.choice(ids, k, replace=False)
        take.append(pd.DataFrame({"isoform_id": pick, "cell": c}))
        w = n / k
        weights[c] = w
        print(f"  {c:<34} {n:>8,} {k:>8,} {w:>9.3f}"
              + ("   WHOLE" if k == n else ""))

    sub = pd.concat(take, ignore_index=True)
    sub["sampling_weight"] = sub.cell.map(weights)
    # tx_length travels with the subset: the bank needs it and a consumer that
    # has to go back to tx_summary for one column will eventually join it wrong.
    sub = sub.merge(sp[["isoform_id", "gene_id", "chr", "tx_length", "is_nmd",
                        "arm", "has_annotation", "main_orf_stop"]], on="isoform_id")
    sub = sub.sort_values("isoform_id", kind="stable").reset_index(drop=True)

    print(f"\n  total {len(sub):,} transcripts, {sub.gene_id.nunique():,} genes")
    print(f"  discovery {int((sub.arm=='discovery').sum()):,} / "
          f"confirmation {int((sub.arm=='confirmation').sum()):,}")
    print(f"  genes in discovery {sub.loc[sub.arm=='discovery','gene_id'].nunique():,} / "
          f"confirmation {sub.loc[sub.arm=='confirmation','gene_id'].nunique():,}")
    print(f"  weighted total reconstructs the pool: "
          f"{sub.sampling_weight.sum():,.0f} against {len(sp):,}")

    out = Path(args.out) / "ism_subset.tsv"
    sub.to_csv(out, sep="\t", index=False)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    (Path(args.out) / "ism_subset_provenance.json").write_text(json.dumps(dict(
        built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        script="build_ism_subset.py", seed=SEED, sha256=sha,
        n=len(sub), n_genes=int(sub.gene_id.nunique()),
        cells={c: dict(population=int(counts[c]),
                       taken=int((sub.cell == c).sum()),
                       weight=float(weights[c])) for c in counts.index},
        note=("Scarce cells taken whole, abundant cells sampled, sampling_weight "
              "recorded per transcript. Any population-level estimate must be "
              "reweighted or it describes this subset rather than the pool. Main "
              "ORF is the GENCODE-projected start, not the expression-derived "
              "reference; transcripts with no annotation are their own cells "
              "rather than dropped.")), indent=2) + "\n")
    print(f"\nwrote {out}  sha256 {sha[:32]}")
    print(f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
