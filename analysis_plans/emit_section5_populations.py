#!/usr/bin/env python
"""
emit_section5_populations.py — the producer for every population number the §5
Methods states, emitting each through the shared claim contract.

WHY THIS EXISTS. The §5 Methods draft states n = 28,775 isoforms carrying a
reference-AUG-traced candidate, and the internal summary states the projection
failure breakdown and the window-fill statistics. Every one of those numbers was
computed in a throwaway `python -c` invocation while drafting. A number in a
manuscript with no producer is the exact defect `tools/claim_emit.py` was written
to prevent -- it is the same shape as the published branch percentages, which
exist only as stdout from a script that writes them nowhere.

So this file is the producer. It recomputes each number from source and emits it
with its population attached, at the moment it is computed.

CLAIM IDs ARE PROVISIONAL AND DELIBERATELY OUT OF THE WAY. The v6 §5 has no
final sentence numbering, and the existing 5.1.1-5.6.9 in claim_status.tsv all
describe the OLD model, which the redesign replaces. Reusing them would key a v6
number to a sentence about a different model. The 5.90.x block is used instead:
it sorts cleanly under the checker's integer parse, cannot collide with any
plausible real paragraph number, and is one find-and-replace away from the real
IDs once §5's sentences are fixed.

NOTHING HERE DECLARES `published`. These are supporting numbers -- populations,
counts, exclusions -- not values the paper prints. Declaring a `published` value
for them would make the checker compare a subgroup size against a claim's value
list, which it explicitly warns is a category error.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
TRACK_A = Path.home() / "claude_projects" / "nmd_lung_longread_2026"
sys.path.insert(0, str(TRACK_A / "tools"))
sys.path.insert(0, str(REPO))

from claim_emit import emit                    # noqa: E402
from tensor_io import decode_windows           # noqa: E402

FLOOR = -1.2507921188400943
POOL = REPO / "results_pool_v6" / "orf_pool.tsv"
REFCDS = REPO / "results_4ct_dn" / "ref_cds_features.tsv"

# Categories in ref_cds_features.tsv that mean the reference AUG was successfully
# traced into the target isoform. Read off 05t_ref_cds_features.R, where every
# other category is an early `next` before the trace completes.
SUCCESS = ("no_downstream_ejc", "effectively_ptc", "truncated_no_ejc")
FAILURE = ("ref_atg_lost", "no_ref_isoform", "not_atg_in_target", "no_stop_in_target")


def main():
    sys.stdout.reconfigure(line_buffering=True)

    # ---- reference-AUG projection outcomes --------------------------------
    ref = pd.read_csv(REFCDS, sep="\t", usecols=["isoform_id", "category"])
    n_iso = len(ref)
    vc = ref.category.value_counts()
    n_ok = int(sum(vc.get(c, 0) for c in SUCCESS))
    unknown = set(vc.index) - set(SUCCESS) - set(FAILURE)
    if unknown:
        raise SystemExit(f"unclassified ref_cds category: {sorted(unknown)} — "
                         "resolve against 05t_ref_cds_features.R before emitting")

    emit("5.90.1", "isoforms with a reference-AUG-traced candidate", n_ok,
         n=n_iso,
         population="all isoforms in ref_cds_features.tsv; traced = the reference "
                    "start codon is exonic in the target, maps to a transcript "
                    "position, and that position reads AUG")
    for c in FAILURE:
        emit("5.90.1", f"projection failures: {c}", int(vc.get(c, 0)), n=n_iso,
             population=f"all isoforms in ref_cds_features.tsv; category {c}")

    # ---- the pool ----------------------------------------------------------
    pool = pd.read_csv(POOL, sep="\t",
                       usecols=["isoform_id", "slot", "is_ref_cds", "kozak_score",
                                "admitted_by", "orf_length"])
    is_ref = pool.is_ref_cds.to_numpy() == 1
    per_tx = pool.groupby("isoform_id").is_ref_cds.sum()

    emit("5.90.2", "candidates in the pool", len(pool), n=len(per_tx),
         population="all admitted candidate ORFs across all pooled transcripts")
    emit("5.90.2", "transcripts in the pool", len(per_tx), n=len(per_tx),
         population="transcripts with at least one admitted candidate")
    emit("5.90.2", "transcripts carrying a reference-AUG-traced candidate",
         int((per_tx == 1).sum()), n=len(per_tx),
         population="pooled transcripts; a transcript carries at most one")
    emit("5.90.2", "reference candidates below the initiation floor",
         int((is_ref & (pool.kozak_score.to_numpy() < FLOOR)).sum()),
         n=int(is_ref.sum()),
         population=f"reference-AUG-traced candidates scoring below the MANE "
                    f"fifth-percentile floor {FLOOR:.4f}, admitted only by the "
                    f"reference exemption")
    emit("5.90.2", "non-reference candidates below the initiation floor",
         int((~is_ref & (pool.kozak_score.to_numpy() < FLOOR)).sum()),
         n=int((~is_ref).sum()),
         population="non-reference candidates below the floor; should be ~0 if "
                    "the floor binds")
    emit("5.90.2", "agreement between reference-AUG trace and the TD2 CDS call",
         float((pool.loc[is_ref, "is_ref_cds"].notna()
                & (pd.read_csv(POOL, sep="\t", usecols=["is_sqanti_cds"])
                   .is_sqanti_cds.to_numpy()[is_ref] == 1)).mean()),
         n=int(is_ref.sum()),
         population="reference-AUG-traced candidates; fraction also flagged as "
                    "the TD2 CDS call for that isoform")

    # ---- window geometry, on the validation chromosome ---------------------
    tensor = REPO / "results_tensor_chr2" / "nmd_tensor.h5"
    if tensor.exists():
        with h5py.File(tensor, "r") as f:
            iso = np.array([s.decode() for s in f["isoform_id"][:]])
            cnt = f["count"][:]
            o_s = f["orf_start"][:].astype(np.int64)
            o_e = f["orf_end"][:].astype(np.int64)
            codes = f["codes"][:, 0]
            L, R = int(f.attrs["atg_left"]), int(f.attrs["atg_right"])

        p2 = pool[pool.isoform_id.isin(set(iso))].copy()
        p2["tx"] = p2.isoform_id.map({s: i for i, s in enumerate(iso)})
        p2 = p2.sort_values(["tx", "slot"], kind="stable").reset_index(drop=True)
        assert np.array_equal(p2["tx"].to_numpy(),
                              np.repeat(np.arange(len(iso)), cnt))
        r2 = p2.is_ref_cds.to_numpy() == 1

        right = np.empty(len(p2), dtype=np.int32)
        for i in range(0, len(p2), 4096):
            w = decode_windows(codes[i:i + 4096], o_s[i:i + 4096], L,
                               o_s[i:i + 4096])
            right[i:i + 4096] = (w[:, 6:9].sum(1) > 0)[:, L:].sum(1)
        full = right >= R
        POP = ("candidates on chr2 (validation split), which is held out from "
               "gradient updates but was used for early stopping")

        emit("5.90.3", "reference candidates with a fully filled downstream "
             "start window", float(full[r2].mean()), n=int(r2.sum()),
             population=POP + "; reference-AUG-traced arm")
        emit("5.90.3", "non-reference candidates with a fully filled downstream "
             "start window", float(full[~r2].mean()), n=int((~r2).sum()),
             population=POP + "; non-reference arm")
        emit("5.90.3", "median ORF length, reference candidates",
             float(np.median((o_e - o_s + 1)[r2])), n=int(r2.sum()),
             population=POP + "; reference-AUG-traced arm")
        emit("5.90.3", "median ORF length, non-reference candidates",
             float(np.median((o_e - o_s + 1)[~r2])), n=int((~r2).sum()),
             population=POP + "; non-reference arm")
        print(f"chr2 geometry emitted over {len(p2):,} candidates")

    print(f"\nemitted to {Path(TRACK_A) / 'tmp' / 'claim_values.tsv'}")
    print(f"reference-AUG traced {n_ok:,} of {n_iso:,} isoforms")


if __name__ == "__main__":
    main()
