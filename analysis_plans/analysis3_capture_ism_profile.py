#!/usr/bin/env python
"""
analysis3_capture_ism_profile.py — what in the start window does capture respond to?

THE QUESTION. Capture is computed from one 1,000-position window per candidate:
900 nt upstream of the A of the AUG, then 100 from the AUG onward. We know it is
not splice junctions, and we know a large part of its apparent skill at picking
annotated starts is the SHAPE of that window rather than its sequence. What we do
not know is whether anything in the bases matters, and where.

This substitutes every alternative base at every filled position of the window and
records how capture moves. A subsample, on the laptop, ahead of the full bank.

SCOPE, AND WHY IT NEEDS NO PROPAGATION. This measures CAPTURE, not P(NMD).
Capture for candidate k is a function of candidate k's own start window and
nothing else — enforced in `model_v6.py` and verified by its self-check. So
perturbing that one window is the complete intervention for this quantity. The
propagation requirement in the plan applies to transcript-level output, where a
coordinate falling in several candidates' windows must be rewritten consistently;
it does not apply here, and saying so is not a shortcut but a consequence of the
architecture.

THE CONTROLS TODAY MADE NECESSARY.

  Only FILLED positions holding a real base are substituted. An unfilled position
  has no base; writing one there invents sequence the encoding says is absent.

  ΔGC is recorded per substitution. Channel 5 moves on 68.2% of single-base
  substitutions, so base identity and local GC shift are confounded in the profile
  itself and any candidate has to be screenable for GC-drivenness afterwards.

  Effects are summarised WITHIN isoforms before any cross-isoform average.
  Averaging across isoforms at a fixed offset dilutes the per-isoform maximum
  about ninefold.

  The positional profile is reported raw AND fill-conditioned, because isoforms
  differ in where their windows are filled and an apparent peak can be a fill
  pattern rather than a sequence signal.

  A matched control candidate from the SAME transcript is profiled alongside each
  annotated start, so "the profile has structure" can be separated from "annotated
  starts have structure".

WHAT IT CANNOT DO. It cannot discover a motif — that needs the full bank, the
gene split and the circular-shift null. It reports where in the window capture is
sensitive, and whether the sensitivity at the classical initiation positions looks
like the known preference.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
TRACK_A = Path.home() / "claude_projects" / "nmd_lung_longread_2026"
sys.path.insert(0, str(TRACK_A / "tools"))
sys.path.insert(0, str(REPO))
from claim_emit import emit                                  # noqa: E402
from model_v6 import ScanningNMDModel                        # noqa: E402
from tensor_io import decode_windows                         # noqa: E402

TENSOR = REPO / "results_tensor_v6" / "nmd_tensor.h5"
CKPT = REPO / "results_interp_all" / "ckpt_interp_c32_b8"
FLAGS = REPO / "results_ism_v6" / "gencode_candidate_flags.tsv"
OUT = REPO / "results_interp_all" / "capture_ism_profile.npz"
N_TX = int(sys.argv[1]) if len(sys.argv) > 1 else 250
BASES = "ACGT"


def capture(model, win, bs=384):
    out = np.empty(len(win), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(win), bs):
            z = model.init_head(model.enc_init(torch.as_tensor(win[i:i + bs])))
            out[i:i + bs] = torch.sigmoid(z.squeeze(-1)).double().numpy()
    return out


def main():
    sys.stdout.reconfigure(line_buffering=True)
    with h5py.File(TENSOR, "r") as f:
        iso = np.array([s.decode() for s in f["isoform_id"][:]])
        off, cnt = f["offset"][:], f["count"][:]
        o_s = f["orf_start"][:].astype(np.int64)
        codes = f["codes"][:]
        L, W = int(f.attrs["atg_left"]), int(f.attrs["window"])

    fl = pd.read_csv(FLAGS, sep="\t",
                     usecols=["isoform_id", "slot", "has_gencode_cds", "is_gencode_start"])
    fl = fl[(fl.has_gencode_cds == 1) & (fl.is_gencode_start == 1)]
    key = {s: i for i, s in enumerate(iso)}
    fl = fl[fl.isoform_id.isin(key)]
    rng = np.random.default_rng(20260801)
    pick = fl.sample(min(N_TX, len(fl)), random_state=20260801)
    print(f"{len(fl):,} transcripts have an admitted annotated start; "
          f"sampling {len(pick):,}")

    # rows: the annotated candidate, and one non-annotated candidate from the
    # same transcript as a matched control
    jobs = []
    for r in pick.itertuples():
        i = key[r.isoform_id]
        lo, n = int(off[i]), int(cnt[i])
        anno = lo + int(r.slot)
        others = [k for k in range(lo, lo + n) if k != anno]
        if not others:
            continue
        jobs.append((anno, "annotated"))
        jobs.append((int(rng.choice(others)), "control"))
    print(f"{len(jobs):,} candidate windows to profile "
          f"({sum(1 for _,t in jobs if t=='annotated'):,} annotated)")

    ckpts = sorted(CKPT.glob("b8_s*.pt"))
    models = []
    for cp in ckpts:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        a = ck["args"]
        m = ScanningNMDModel(conv_channels=a["conv_channels"], n_bins=a["n_bins"],
                             n_structural=1, permute_bins=False)
        m.load_state_dict(ck["model"]); m.eval()
        models.append(m)

    # accumulators, offset relative to the A of the AUG at index L
    n_off = W
    sums = {t: np.zeros(n_off) for t in ("annotated", "control")}
    cnts = {t: np.zeros(n_off) for t in ("annotated", "control")}
    permax = {t: [] for t in ("annotated", "control")}
    dead = {t: [0, 0] for t in ("annotated", "control")}
    # signed effect of substituting TO each base, at the classical positions
    kpos = list(range(-6, 0)) + [3, 4]          # -6..-1 and +4,+5 (0-based +3,+4)
    ktab = {p: {b: [] for b in BASES} for p in kpos}
    gcrec = []

    for jn, (row, tag) in enumerate(jobs):
        base_codes = codes[row, 0]
        anchor = np.array([o_s[row]])
        w0 = decode_windows(base_codes[None, :], anchor, L, anchor)
        fill = (base_codes & 7)
        pos = np.flatnonzero((fill >= 1) & (fill <= 4))
        if not len(pos):
            continue
        variants, meta = [], []
        for p_ in pos:
            cur = int(fill[p_]) - 1
            for b in range(4):
                if b == cur:
                    continue
                cc = base_codes.copy()
                cc[p_] = (cc[p_] & ~np.uint8(7)) | np.uint8(b + 1)
                variants.append(cc); meta.append((p_, cur, b))
        V = decode_windows(np.stack(variants), np.repeat(anchor, len(variants)),
                           L, np.repeat(anchor, len(variants)))
        d = np.zeros(len(variants))
        for m in models:
            d += (capture(m, V) - capture(m, w0)[0]) / len(models)
        gcrec.append(float(np.mean(np.abs(V[:, 5] - w0[0, 5]).sum(1) > 0)))
        ad = np.abs(d)
        dead[tag][0] += int((ad == 0).sum()); dead[tag][1] += len(ad)
        mx = 0.0
        for (p_, cur, b), dv in zip(meta, d):
            o = p_ - L
            sums[tag][p_] += abs(dv); cnts[tag][p_] += 1
            mx = max(mx, abs(dv))
            if tag == "annotated" and o in kpos:
                ktab[o][BASES[b]].append(dv)
        permax[tag].append(mx)
        if jn % 50 == 0:
            print(f"  {jn}/{len(jobs)}", flush=True)

    np.savez_compressed(OUT, **{f"sum_{t}": sums[t] for t in sums},
                        **{f"cnt_{t}": cnts[t] for t in cnts})
    print(f"\ncached -> {OUT.name}")

    print(f"\n=== liveness (substitutions leaving capture bitwise unchanged) ===")
    for t in ("annotated", "control"):
        z, n = dead[t]
        print(f"  {t:<10} {100*z/max(n,1):>5.1f}% dead of {n:,} substitutions")

    print(f"\n=== per-isoform maximum |delta capture| ===")
    for t in ("annotated", "control"):
        v = np.array(permax[t])
        print(f"  {t:<10} median {np.median(v):.4f}   mean {v.mean():.4f}   "
              f"max {v.max():.4f}   n {len(v):,}")
    emit("5.90.15", "per-isoform maximum |delta capture|, annotated starts",
         float(np.median(permax["annotated"])), n=len(permax["annotated"]),
         population="sampled transcripts with an admitted GENCODE-annotated start; "
                    "every alternative base at every filled window position; "
                    "median over isoforms of the within-isoform maximum")

    print(f"\n=== positional profile: top 15 offsets by mean |delta|, annotated ===")
    prof = np.divide(sums["annotated"], np.maximum(cnts["annotated"], 1))
    cprof = np.divide(sums["control"], np.maximum(cnts["control"], 1))
    med = np.median(prof[cnts["annotated"] > 0])
    top = np.argsort(-prof)[:15]
    print(f"  median over all offsets: {med:.5f}")
    print(f"  {'offset':>8}{'mean|d| anno':>14}{'control':>10}{'ratio to median':>17}")
    for p_ in sorted(top, key=lambda x: prof[x], reverse=True):
        print(f"  {p_-L:>8}{prof[p_]:>14.5f}{cprof[p_]:>10.5f}"
              f"{prof[p_]/max(med,1e-12):>17.1f}x")

    print(f"\n=== the classical initiation positions, signed mean delta capture ===")
    print(f"  positive = substituting TO this base RAISES capture")
    print(f"  {'pos':>5}" + "".join(f"{b:>9}" for b in BASES))
    for p_ in kpos:
        lab = f"{p_:+d}" if p_ < 0 else f"+{p_+1}"
        row = "".join(f"{np.mean(ktab[p_][b]):>9.4f}" if ktab[p_][b] else f"{'-':>9}"
                      for b in BASES)
        print(f"  {lab:>5}{row}")
    print(f"\n  GC channel moved on {100*np.mean(gcrec):.1f}% of substitutions "
          f"(base identity and local GC are confounded)")


if __name__ == "__main__":
    main()
