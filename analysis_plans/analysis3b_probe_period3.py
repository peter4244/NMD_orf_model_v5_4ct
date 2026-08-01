#!/usr/bin/env python
"""
analysis3b_probe_period3.py — is there signal above noise, before anything is interpreted?

PETE'S QUESTION, and it comes first. A period-3 base preference was read off the
eight classical initiation positions: G at codon phase 0, A at phase 1, C at
phase 2, eight for eight. Before interpreting that, we need evidence that anything
in this window exceeds background. A pattern in noise will still look like a
pattern if you only look at eight numbers.

FOUR TESTS, three of them free from the same forward passes.

1. THE METHOD'S OWN ZERO. Substituting a base for itself must return exactly 0.
   The model is deterministic in eval mode, so this is a bitwise check, not a
   statistical one. If it is not exactly zero the whole measurement is
   instrumentation error and nothing below matters.

2. CROSS-SEED AGREEMENT. Five independently trained models. Noise does not
   replicate across independent initialisations; a learned feature does. Reported
   as the correlation between per-seed position-by-base effect matrices, and as
   sign agreement at the classical positions. This is check 1 of the six.

3. EFFECT AGAINST ITS OWN STANDARD ERROR. Each cell is a mean over isoforms, so
   it has a standard error. How many positions exceed three of them, and where do
   the classical positions sit in the distribution of ALL positions.

4. THE PHASE-SHIFT NULL, which is the specific null for the specific claim. Phase
   is (offset mod 3). Re-aggregate the identical data under (offset+1) mod 3 and
   (offset+2) mod 3. If the G/A/C rule is genuinely frame-locked it should hold at
   the true phase and break under both shifts. If it holds under a shifted
   labelling too, the rule is an artifact of aggregation rather than a property of
   the model. This costs nothing — it is the same numbers relabelled.

Only after those does it report the regional breakdown and the AUG-destruction
result.
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
OUT = REPO / "results_interp_all" / "period3_probe.npz"
N_TX = int(sys.argv[1]) if len(sys.argv) > 1 else 150
BASES = "ACGT"
KPOS = [-6, -5, -4, -3, -2, -1, 3, 4]
REGIONS = [("deep upstream", -900, -200), ("mid upstream", -199, -50),
           ("proximal upstream", -49, -1), ("proximal downstream", 3, 30),
           ("deeper downstream", 31, 99)]
EXPECT = {0: "G", 1: "A", 2: "C"}


def capture(model, win, bs=384):
    out = np.empty(len(win), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(win), bs):
            z = model.init_head(model.enc_init(torch.as_tensor(win[i:i + bs])))
            out[i:i + bs] = torch.sigmoid(z.squeeze(-1)).double().numpy()
    return out


def rule_hits(S, N, W, L, phase_shift=0):
    """How many region x phase cells pick the base the G/A/C rule predicts."""
    hits = tot = 0
    for _, lo, hi in REGIONS:
        for ph in (0, 1, 2):
            sel = [L + o for o in range(lo, hi + 1)
                   if (o + phase_shift) % 3 == ph and 0 <= L + o < W]
            if not sel:
                continue
            v = S[sel].sum(0) / np.maximum(N[sel].sum(0), 1)
            hits += int(BASES[int(np.argmax(v))] == EXPECT[ph]); tot += 1
    return hits, tot


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
    pick = fl.sample(min(N_TX, len(fl)), random_state=20260801)
    rng = np.random.default_rng(20260801)

    jobs = []
    for r in pick.itertuples():
        i = key[r.isoform_id]
        lo, n = int(off[i]), int(cnt[i])
        anno = lo + int(r.slot)
        others = [k for k in range(lo, lo + n) if k != anno]
        if others:
            jobs.append((anno, "annotated"))
            jobs.append((int(rng.choice(others)), "control"))
    print(f"{len(jobs):,} windows ({sum(1 for _,t in jobs if t=='annotated'):,} annotated)")

    ck_paths = sorted(CKPT.glob("b8_s*.pt"))
    models = []
    for cp in ck_paths:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        a = ck["args"]
        m = ScanningNMDModel(conv_channels=a["conv_channels"], n_bins=a["n_bins"],
                             n_structural=1, permute_bins=False)
        m.load_state_dict(ck["model"]); m.eval()
        models.append(m)
    NS = len(models)

    arms = ("annotated", "control")
    S = {t: np.zeros((NS, W, 4)) for t in arms}      # per-seed signed sum
    S2 = {t: np.zeros((W, 4)) for t in arms}         # sum of squares, 5-seed mean
    N = {t: np.zeros((W, 4)) for t in arms}
    augd = {t: [] for t in arms}
    self_max = 0.0

    for jn, (row, tag) in enumerate(jobs):
        bc = codes[row, 0]
        anchor = np.array([o_s[row]])
        w0 = decode_windows(bc[None, :], anchor, L, anchor)
        fill = (bc & 7)
        pos = np.flatnonzero((fill >= 1) & (fill <= 4))
        if not len(pos):
            continue
        variants, meta = [], []
        for p_ in pos:
            cur = int(fill[p_]) - 1
            for b in range(4):
                cc = bc.copy()
                cc[p_] = (cc[p_] & ~np.uint8(7)) | np.uint8(b + 1)
                variants.append(cc); meta.append((p_, b, b == cur))
        V = decode_windows(np.stack(variants), np.repeat(anchor, len(variants)),
                           L, np.repeat(anchor, len(variants)))
        per = np.stack([capture(m, V) - capture(m, w0)[0] for m in models])
        mean = per.mean(0)
        for j, (p_, b, is_self) in enumerate(meta):
            if is_self:                                   # test 1: the method's zero
                self_max = max(self_max, float(np.abs(per[:, j]).max()))
                continue
            S[tag][:, p_, b] += per[:, j]
            S2[tag][p_, b] += mean[j] ** 2
            N[tag][p_, b] += 1
        aug = [mean[j] for j, (p_, b, s_) in enumerate(meta)
               if not s_ and 0 <= p_ - L <= 2]
        if aug:
            augd[tag].append(float(np.mean(aug)))
        if jn % 50 == 0:
            print(f"  {jn}/{len(jobs)}", flush=True)

    np.savez_compressed(OUT, **{f"S_{t}": S[t] for t in arms},
                        **{f"N_{t}": N[t] for t in arms})
    M = {t: S[t].sum(0) / NS / np.maximum(N[t], 1) for t in arms}   # 5-seed mean

    print(f"\n{'='*72}\n1. THE METHOD'S OWN ZERO\n{'='*72}")
    print(f"  max |delta capture| over all base-for-itself substitutions: {self_max:.3e}")
    print(f"  {'PASS — the floor is exact zero' if self_max == 0 else 'FAIL'}")

    print(f"\n{'='*72}\n2. CROSS-SEED AGREEMENT (five independent initialisations)\n{'='*72}")
    A = S["annotated"] / np.maximum(N["annotated"], 1)
    ok = N["annotated"].sum(1) > 0
    flat = A[:, ok, :].reshape(NS, -1)
    cors = [np.corrcoef(flat[i], flat[j])[0, 1]
            for i in range(NS) for j in range(i + 1, NS)]
    print(f"  pairwise correlation of per-seed position-by-base effect matrices:")
    print(f"    median {np.median(cors):.3f}   range [{min(cors):.3f}, {max(cors):.3f}]"
          f"   over {len(cors)} pairs")
    agree = []
    for o in KPOS:
        for b in range(4):
            v = A[:, L + o, b]
            agree.append(np.mean(np.sign(v) == np.sign(v.mean())))
    print(f"  sign agreement across seeds at the 32 classical cells: "
          f"{100*np.mean(agree):.1f}%  (chance 50%)")
    emit("5.90.18", "median pairwise cross-seed correlation of ISM effect matrices",
         float(np.median(cors)), n=len(cors),
         population="annotated-start windows; per-seed mean signed delta capture "
                    "at every position-by-base cell with data")

    print(f"\n{'='*72}\n3. EFFECT AGAINST ITS OWN STANDARD ERROR\n{'='*72}")
    se = np.sqrt(np.maximum(S2["annotated"] / np.maximum(N["annotated"], 1)
                            - M["annotated"] ** 2, 0) / np.maximum(N["annotated"], 1))
    z = np.divide(M["annotated"], np.maximum(se, 1e-12))
    valid = N["annotated"] > 5
    print(f"  cells with data: {int(valid.sum()):,}")
    print(f"  |z| > 3 : {int((np.abs(z) > 3).sum()):,} "
          f"({100*(np.abs(z) > 3).sum()/max(valid.sum(),1):.1f}%)")
    print(f"  |z| > 5 : {int((np.abs(z) > 5).sum()):,}")
    kz = [abs(z[L + o, b]) for o in KPOS for b in range(4)]
    allz = np.abs(z[valid])
    print(f"  the 32 classical cells: median |z| {np.median(kz):.2f}, "
          f"against {np.median(allz):.2f} for all cells")
    print(f"  classical cells sit at the {100*np.mean(allz < np.median(kz)):.1f}th "
          f"percentile of all cells")

    print(f"\n{'='*72}\n4. THE PHASE-SHIFT NULL — the specific null for the claim\n{'='*72}")
    for t in arms:
        line = []
        for sh in (0, 1, 2):
            h, tot = rule_hits(S[t].sum(0) / NS, N[t], W, L, sh)
            line.append(f"shift {sh}: {h}/{tot}")
        print(f"  {t:<10} " + "   ".join(line))
    h0, tot0 = rule_hits(S["annotated"].sum(0) / NS, N["annotated"], W, L, 0)
    emit("5.90.19", "region-phase cells following G/A/C at true phase, annotated",
         float(h0 / max(tot0, 1)), n=tot0,
         population="annotated-start windows; five regions x three phases; "
                    "compared against the same data relabelled at phase shift 1 and 2")

    print(f"\n{'='*72}\n5. REGIONAL BREAKDOWN (only meaningful if the above passed)\n{'='*72}")
    for t in arms:
        print(f"\n  --- {t} ---")
        print(f"  {'region':<22}{'phase':>6}" + "".join(f"{b:>9}" for b in BASES) + "   best")
        for name, lo, hi in REGIONS:
            for ph in (0, 1, 2):
                sel = [L + o for o in range(lo, hi + 1) if o % 3 == ph and 0 <= L + o < W]
                if not sel:
                    continue
                v = (S[t].sum(0) / NS)[sel].sum(0) / np.maximum(N[t][sel].sum(0), 1)
                print(f"  {name if ph==0 else '':<22}{ph:>6}"
                      + "".join(f"{x:>9.4f}" for x in v)
                      + f"   {BASES[int(np.argmax(v))]}")

    print(f"\n{'='*72}\n6. DESTROYING THE AUG ITSELF\n{'='*72}")
    for t in arms:
        v = np.array(augd[t])
        print(f"  {t:<10} mean signed delta {v.mean():+.4f}   median {np.median(v):+.4f}"
              f"   lowers capture in {100*np.mean(v<0):.0f}% of windows   n {len(v)}")


if __name__ == "__main__":
    main()
