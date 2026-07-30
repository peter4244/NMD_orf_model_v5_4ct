#!/usr/bin/env python3
"""stamp_dataset_provenance.py — record WHICH HDF5 produced the contents of a results directory.

THE PROBLEM THIS SOLVES, and why it is a directory-level record rather than a filename change.
No output filename in this repository says which data vintage produced it.
`metrics_atg500_stop500_seed100_val_clean.json` is name-identical whether it came from the
pre-clip encoding or the post-clip one. On 2026-07-30 that was handled by physically moving 12 GB
into `superseded_2026-07-29/` -- by directory, because the names could not carry it.

The rule this repo now works to: *a filename must distinguish any two files whose CONTENT can
differ; everything else belongs in a sidecar.* Member seed failed that test and went into the
names. Data vintage fails it too, but a hash in every filename is unreadable and would break every
consumer, so it goes here -- one record per results directory, the same pattern as
`tx_summary_provenance.json`.

WHAT IT RECORDS, and what it deliberately does not. The HDF5's identity (path, size, mtime), its
split composition and label counts, the window groups it carries, and -- if the tree it was built
from still has one -- the `tx_summary_provenance.json` that produced its labels. That chains the
record back to the ORFik scan. It does NOT hash the 2.9 GB file: reading it costs minutes and the
size+mtime+composition triple already distinguishes every vintage this project has produced.

    python3 stamp_dataset_provenance.py --results-dir results_4ct_sweep \\
        --h5 results_4ct_dn/nmd_orf_data.h5
    python3 stamp_dataset_provenance.py --results-dir results_4ct_dn --check
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import h5py
import numpy as np

NAME = "dataset_provenance.json"


def describe(h5_path: Path, source_tree: Path | None):
    st = h5_path.stat()
    with h5py.File(h5_path, "r") as f:
        splits = np.array([s.decode() if isinstance(s, bytes) else s for s in f["split"][:]])
        labels = f["labels"][:]
        rec = {
            "hdf5_path": str(h5_path.resolve()),
            "hdf5_bytes": st.st_size,
            "hdf5_mtime": __import__("datetime").datetime.fromtimestamp(st.st_mtime).isoformat(),
            "n_transcripts": int(len(splits)),
            "n_nmd": int(labels.sum()),
            "n_non_nmd": int((labels == 0).sum()),
            "splits": dict(sorted(collections.Counter(splits.tolist()).items())),
            "window_groups": sorted(k for k in f.keys() if k.startswith("w")),
        }
    # Chain back to the labels' own provenance if it is still beside the tables.
    if source_tree is not None:
        tp = Path(source_tree) / "tx_summary_provenance.json"
        rec["tx_summary_provenance"] = json.loads(tp.read_text()) if tp.exists() else None
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                    help="directory whose CONTENTS this record describes")
    ap.add_argument("--h5", help="the HDF5 those contents were produced from")
    ap.add_argument("--check", action="store_true",
                    help="verify an existing stamp still matches the HDF5 it names")
    a = ap.parse_args()

    rd = Path(a.results_dir)
    if not rd.is_dir():
        sys.exit(f"{rd} is not a directory")
    out = rd / NAME

    if a.check:
        if not out.exists():
            sys.exit(f"no {NAME} in {rd} -- the vintage of everything here is unrecorded")
        rec = json.loads(out.read_text())
        h5 = Path(rec["hdf5_path"])
        if not h5.exists():
            sys.exit(f"stamp names {h5}, which no longer exists")
        cur = describe(h5, None)
        drift = [k for k in ("hdf5_bytes", "n_transcripts", "n_nmd", "n_non_nmd")
                 if rec.get(k) != cur.get(k)]
        if drift:
            print(f"STALE: {rd}/{NAME} disagrees with {h5} on {drift}")
            for k in drift:
                print(f"  {k}: stamped {rec.get(k)}  now {cur.get(k)}")
            return 1
        print(f"ok: {rd}/{NAME} still describes {h5}")
        print(f"  {cur['n_transcripts']:,} transcripts, splits {cur['splits']}")
        return 0

    if not a.h5:
        sys.exit("--h5 is required unless --check")
    h5 = Path(a.h5)
    if not h5.exists():
        sys.exit(f"{h5} does not exist")
    rec = describe(h5, h5.parent)
    rec["stamped_at"] = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    rec["describes"] = str(rd.resolve())
    out.write_text(json.dumps(rec, indent=2))
    print(f"wrote {out}")
    print(f"  {h5.name}: {rec['n_transcripts']:,} transcripts, "
          f"{rec['n_nmd']:,} NMD, groups {rec['window_groups']}")
    print(f"  splits: {rec['splits']}")
    if rec.get("tx_summary_provenance"):
        p = rec["tx_summary_provenance"]
        print(f"  labels from: {p.get('written_by')} @ {p.get('written_at')}, "
              f"scan {p.get('source_rds', '?')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
