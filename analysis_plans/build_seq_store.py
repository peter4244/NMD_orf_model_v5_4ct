#!/usr/bin/env python
"""
Extract the ~42k transcript sequences the model universe needs out of the
615k-record SQANTI FASTA, once, into a compact store that Experiments 2 and 3
can load in seconds.

Experiments 2 and 3 need, per isoform:
  - the three bases of the stop codon, to confirm the table's stop_codon column
  - the bases immediately after it (+4, +5, +6 ...), for the stop-context contrast
  - the 3'UTR, for the motif scan

Rather than guess how much is enough, the store keeps the FULL sequence for the
model-universe isoforms. 42k transcripts at a mean ~2.5 kb is ~105 MB of ASCII,
which is fine in memory and on disk.

Writes  seq_store.npz  (isoform_id array + one concatenated byte blob + offsets)
alongside this script.

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python build_seq_store.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd

FASTA = os.path.expanduser(
    "~/claude_projects/nmd_deposit_2026/source_data/sqanti/nmd_lungcells_corrected.fasta"
)
TABLES = os.path.expanduser("~/claude_projects/nmd_w69_tables_2026-07-30")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seq_store.npz")


def main():
    t0 = time.time()
    tx = pd.read_csv(os.path.join(TABLES, "tx_summary.tsv"), sep="\t",
                     usecols=["isoform_id", "tx_length"])
    want = dict(zip(tx["isoform_id"], tx["tx_length"]))
    print(f"want {len(want):,} isoforms", flush=True)

    seqs = {}
    cur_id, cur = None, []
    n_rec = 0
    with open(FASTA, "r") as fh:
        for line in fh:
            if line[0] == ">":
                if cur_id is not None and cur_id in want:
                    seqs[cur_id] = "".join(cur)
                n_rec += 1
                if n_rec % 100000 == 0:
                    print(f"  {n_rec:,} records scanned, {len(seqs):,} kept "
                          f"({time.time() - t0:.0f}s)", flush=True)
                cur_id = line[1:].split()[0]
                cur = []
            else:
                cur.append(line.strip())
    if cur_id is not None and cur_id in want:
        seqs[cur_id] = "".join(cur)

    print(f"scanned {n_rec:,} records, kept {len(seqs):,} "
          f"({time.time() - t0:.0f}s)", flush=True)
    missing = set(want) - set(seqs)
    print(f"MISSING from the FASTA: {len(missing):,}", flush=True)
    if missing:
        print(f"  examples: {sorted(missing)[:5]}", flush=True)

    # length agreement is the check that we pulled the right records
    bad = [(i, len(seqs[i]), want[i]) for i in seqs if len(seqs[i]) != want[i]]
    print(f"LENGTH MISMATCH vs tx_summary.tx_length: {len(bad):,} of {len(seqs):,}",
          flush=True)
    if bad:
        print(f"  examples (id, fasta_len, table_len): {bad[:5]}", flush=True)

    ids = np.array(sorted(seqs), dtype=object)
    blob = "".join(seqs[i] for i in ids).encode("ascii")
    lens = np.array([len(seqs[i]) for i in ids], dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(lens)])
    np.savez_compressed(
        OUT,
        ids=ids.astype(str),
        blob=np.frombuffer(blob, dtype=np.uint8),
        offsets=off,
    )
    print(f"wrote {OUT}  ({os.path.getsize(OUT) / 1e6:.0f} MB, "
          f"{time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
