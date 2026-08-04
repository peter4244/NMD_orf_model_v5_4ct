#!/usr/bin/env python
"""verify_packed_roundtrip.py — the licence for storing windows packed.

Section 5's numbers were computed from the DENSE nine-channel arrays. The only claim that
licenses replacing them is: what the loader hands the model is the same, element for element.
build_tensor.py's "verified to decode identically" is a claim about ITS tensor, and
verify_tensor_encoding.py checks encoding against the transcript SEQUENCE -- a different
question. This checks ours.

TWO MODES, and the second is the one that matters.

  --mode memory  pack and unpack a dense file's windows in RAM. Tests the CODEC only.
  --mode files   decode a PACKED file and compare against a retained DENSE file. Tests the
                 codec AND the write path -- batch offsets, phase arrays, chunking -- none of
                 which the memory mode can see, because it never goes through a file.

WHY ROWS, NOT WINDOW GROUPS. The earlier version sampled "all four window groups and both
anchors" and read `[:n_tx]`, i.e. only the first write batch. The write path is indexed by
ROWS, in batches of 1000 (data_prep.py), so a batch-offset error or a mishandled final partial
batch (41,776 rows -> 42 batches, last one 776) is invisible to any sample that never straddles
a boundary. This samples boundary rows explicitly.

WHY isoform_id, NOT POSITION. data_prep re-derives row order and ORF selection on every run, so
comparing position i to position i assumes a determinism this test would otherwise be asserting
rather than checking. Rows are matched by id.

PADDED SLOTS ARE CHECKED, NOT SKIPPED. The earlier version did `if not d.any(): continue`,
which skipped exactly the phase = -1 path the fillvalue design rests on. An unfilled slot must
decode to all zeros, and that is now an assertion.

DTYPE. Decoding is done at float16 and upcast, which is what utils._read_windows does. Decoding
straight to float32 does NOT reproduce the dense values: data_prep stores channel 5 as
float32-computed GC rounded to float16, while unpack recomputes it in the caller's dtype --
measured 445 of 452 filled positions differing, max 2.4e-4. Small, and beside the point: the
claim is bit-identity.
"""
import argparse
import sys

import h5py
import numpy as np

from window_codec import pack_window, unpack_window


def boundary_rows(n_tx, batch=1000, extra=32, seed=0):
    """Rows that straddle write-batch boundaries, plus a random stratum."""
    rows = {0, n_tx - 1}
    for b in range(batch, n_tx, batch):
        rows.update({b - 1, b, min(b + 1, n_tx - 1)})
    rng = np.random.default_rng(seed)
    rows.update(int(x) for x in rng.choice(n_tx, size=min(extra, n_tx), replace=False))
    return np.array(sorted(r for r in rows if 0 <= r < n_tx))


def windows_of(grp, anchor, rows):
    """(len(rows), K, 9, W) float16 from either layout, decoded the way the loader does."""
    if f"{anchor}_windows" in grp:
        return grp[f"{anchor}_windows"][rows], "dense"
    codes = grp[f"{anchor}_codes"][rows]
    phase = grp[f"{anchor}_phase"][rows]
    n, K, W = codes.shape
    out = np.zeros((n, K, 9, W), dtype=np.float16)
    for i in range(n):
        for k in range(K):
            out[i, k] = unpack_window(codes[i, k], phase[i, k], dtype=np.float16)
    return out, "packed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("memory", "files"), required=True)
    ap.add_argument("--dense", required=True, help="HDF5 holding the nine dense channels")
    ap.add_argument("--packed", help="HDF5 holding uint8 codes (--mode files)")
    a = ap.parse_args()
    if a.mode == "files" and not a.packed:
        ap.error("--mode files requires --packed")

    total = mismatch = pad_checked = pad_bad = 0
    examples = []

    with h5py.File(a.dense, "r") as fd:
        ids_d = np.array([s.decode() if isinstance(s, bytes) else s
                          for s in fd["isoform_id"][:]])
        rows_d = boundary_rows(len(ids_d))
        groups = sorted(k for k in fd.keys() if k.startswith("w"))
        print("dense  %s\n  %d rows, groups %s, sampling %d rows"
              % (a.dense, len(ids_d), groups, len(rows_d)))

        fp = h5py.File(a.packed, "r") if a.mode == "files" else None
        rows_p = None
        if fp is not None:
            ids_p = np.array([s.decode() if isinstance(s, bytes) else s
                              for s in fp["isoform_id"][:]])
            pos = {t: i for i, t in enumerate(ids_p)}
            missing = [t for t in ids_d[rows_d] if t not in pos]
            if missing:
                sys.exit("FAIL: %d sampled ids absent from the packed file, e.g. %s"
                         % (len(missing), missing[:3]))
            rows_p = np.array([pos[t] for t in ids_d[rows_d]])
            print("packed %s\n  %d rows, matched by isoform_id" % (a.packed, len(ids_p)))

        for g in groups:
            for anchor in ("atg", "stop"):
                dref, _ = windows_of(fd[g], anchor, rows_d)
                if fp is None:
                    got = np.zeros_like(dref)
                    for i in range(dref.shape[0]):
                        for k in range(dref.shape[1]):
                            c, ph = pack_window(dref[i, k])
                            got[i, k] = unpack_window(c, ph, dtype=np.float16)
                else:
                    got, kind = windows_of(fp[g], anchor, rows_p)
                    if kind != "packed":
                        sys.exit("FAIL: %s/%s holds dense arrays, not codes" % (a.packed, g))

                bad = 0
                for i in range(dref.shape[0]):
                    for k in range(dref.shape[1]):
                        d, r = dref[i, k], got[i, k]
                        if not d.any():                      # padded slot
                            pad_checked += 1
                            if r.any():
                                pad_bad += 1
                            continue
                        total += 1
                        if not np.array_equal(r, d):
                            bad += 1
                            if len(examples) < 3:
                                examples.append((g, anchor, int(rows_d[i]), k,
                                                 [c for c in range(9)
                                                  if not np.array_equal(r[c], d[c])]))
                mismatch += bad
                print("  %-6s %-5s mismatched=%d" % (g, anchor, bad))
        if fp is not None:
            fp.close()

    print("\nfilled windows compared : %d" % total)
    print("bit-identical           : %d" % (total - mismatch))
    print("MISMATCHED              : %d" % mismatch)
    print("padded slots checked    : %d  non-zero (must be 0): %d" % (pad_checked, pad_bad))
    for g, anchor, row, k, ch in examples:
        print("   %s/%s row %d orf %d -> channels %s" % (g, anchor, row, k, ch))
    ok = mismatch == 0 and pad_bad == 0 and total > 0
    print("\nRESULT: " + ("PASS" if ok else "FAIL - do NOT adopt packing"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
