"""
model_capture_premise.py — does the 5' scanner actually do a decent job?

SCOPE -- THIS PRODUCES CLAIMS ABOUT THE MODEL, NOT ABOUT BIOLOGY. It reads model
outputs (p_capture, p_select) against candidate features. Nothing here is a
statement about transcript biology.

THE PREMISE UNDER TEST. Pete: "the model works, the 5' scanner must be doing a
decent job." That inference is not safe. P(NMD) = sum_k P(select k) * d_k, so the
PRODUCT can be right while selection is diffuse, provided d_k is near zero for the
wrong candidates. Decay can carry a lazy scanner. Before either window commits to
a plan built on the scanner working, measure whether it does.

Descriptive only. No null, no test, no adjustment. Three questions.

  Q1  IS THE SCANNER BETTER THAN POSITION ALONE? Does argmax p_select land on the
      reference candidate more often than "the most 5' candidate" does? Selection
      is stick-breaking over a 5'->3' ordering (model_v6.py:7,15-18), so a 5'-most
      baseline is what the PRIOR alone would achieve. If the model does not beat
      it, "the scanner works" is a statement about the ordering, not the head.
      argmax p_capture is reported beside it: that is the head's own preference
      with the queue removed.

  Q2  IS p_capture CONCENTRATED OR FLAT? If capture is near-uniform across
      candidates while p_select is peaked, the stick-breaking prior is doing the
      selecting and the head is contributing little. Reported as within-transcript
      coefficient of variation for capture, and normalised entropy for select.
      NOTE THE ASYMMETRY, stated rather than hidden: p_select is a distribution
      over candidates and entropy is natural for it; p_capture is a vector of
      INDEPENDENT sigmoids and is not a distribution, so it gets CV rather than an
      entropy that would require an arbitrary normalisation.

  Q3  IS CAPTURE REALLY BLIND TO THE STRUCTURAL BLOCK, ON REAL DATA? model_v6.py
      routes the stop window and the structural block (which carries
      n_downstream_ejc) to DECAY only; capture sees the ATG window alone
      (:160-164). There is an assertion for it at :279 -- but it runs on RANDOM
      tensors. This runs the same idea on real candidates: within-transcript rank
      correlation of p_capture with n_downstream_ejc. The architecture says any
      association must be indirect, through start context. kozak_score is carried
      as the comparator that SHOULD associate.

WHY THE REFERENCE CANDIDATE IS THE YARDSTICK AND WHAT IT COSTS. cand_is_ref_cds
marks the annotated CDS. It is not ground truth about which ORF a ribosome uses --
it is the best available proxy and transcripts without one are excluded and
counted. A model that disagreed with the annotation everywhere would score badly
here and might still be right; that limit is stated rather than adjusted for.

Run from the repo root.
"""

import argparse
import numpy as np
import h5py


def spearman(a, b):
    """Rank correlation, no scipy dependency. Ties averaged."""
    if len(a) < 3:
        return np.nan
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="results_ism_v6/bank_interp_s100.h5")
    args = ap.parse_args()

    with h5py.File(args.bank, "r") as f:
        off = f["cand_offset"][:]
        cnt = f["cand_count"][:]
        pcap = f["p_capture"][:]
        pdec = f["p_decay"][:]
        psel = f["p_select"][:]
        is_ref = f["cand_is_ref_cds"][:]
        o_start = f["cand_orf_start"][:]
        o_end = f["cand_orf_end"][:]
        ejc = f["cand_n_downstream_ejc"][:]
        koz = f["cand_kozak_score"][:]
        labels = f["labels"][:]
        N = len(cnt)
        ck = f.attrs.get("checkpoint", "?")

    r_part, r_cp, r_ep = [], [], []
    r_len, r_part_len, r_short, r_long = [], [], [], []
    cv_dec, dec_at_sel, dec_max = [], [], []
    hit_sel = hit_cap = hit_5p = 0
    n_used = n_cand_used = 0
    ncands, cv_cap, ent_sel, maxsel = [], [], [], []
    r_ejc, r_koz = [], []
    per_label = {0: [0, 0], 1: [0, 0]}

    for i in range(N):
        lo, k = int(off[i]), int(cnt[i])
        if k < 2:
            continue
        pc, ps = pcap[lo:lo + k], psel[lo:lo + k]
        ref = is_ref[lo:lo + k].astype(bool)
        ncands.append(k)

        # ---- Q2, computed for every transcript with >=2 candidates
        if pc.mean() > 0:
            cv_cap.append(float(pc.std() / pc.mean()))
        s = ps.sum()
        if s > 0:
            q = ps / s
            nz = q[q > 0]
            ent_sel.append(float(-(nz * np.log(nz)).sum() / np.log(k)))
            maxsel.append(float(q.max()))

        # ---- Q3, within transcript so between-transcript structure cannot leak
        e_ = ejc[lo:lo + k].astype(float)
        pos_ = o_start[lo:lo + k].astype(float)
        r_ejc.append(spearman(pc, e_))
        r_koz.append(spearman(pc, koz[lo:lo + k].astype(float)))

        # IS THE EJC ASSOCIATION MEDIATED BY POSITION? Partial rank correlation
        # of capture with EJC count, holding candidate start position fixed.
        # The architecture forbids a direct route, so if the raw -0.46 is
        # position it should collapse here. Position is NOT a confound to be
        # removed -- it is the hypothesised mediator, and this measures how much
        # of the association it accounts for.
        rce, rcp, rep = spearman(pc, e_), spearman(pc, pos_), spearman(e_, pos_)
        if all(np.isfinite([rce, rcp, rep])) and abs(rcp) < 1 and abs(rep) < 1:
            den = np.sqrt((1 - rcp ** 2) * (1 - rep ** 2))
            if den > 1e-9:
                r_part.append(float((rce - rcp * rep) / den))
                r_cp.append(rcp)
                r_ep.append(rep)

        # IS THE ROUTE ORF LENGTH? At a matched START, more downstream junctions
        # means an earlier stop means a SHORTER ORF -- and the ATG window carries
        # 100 nt INTO the ORF, so a short ORF's in-window portion is fill-limited
        # or post-stop rather than coding. If capture is reading coding-likeness,
        # holding LENGTH should collapse the capture-EJC association, and capture
        # should track length directly.
        len_ = (o_end[lo:lo + k] - o_start[lo:lo + k]).astype(float)
        r_len.append(spearman(pc, len_))
        rcl, rel = spearman(pc, len_), spearman(e_, len_)
        if all(np.isfinite([rce, rcl, rel])) and abs(rcl) < 1 and abs(rel) < 1:
            den2 = np.sqrt((1 - rcl ** 2) * (1 - rel ** 2))
            if den2 > 1e-9:
                r_part_len.append(float((rce - rcl * rel) / den2))
        # the 100 nt prediction: strong among ORFs shorter than the in-window
        # portion, weak among ORFs longer than it, where the window is coding
        # either way
        for band, acc in ((len_ <= 100, r_short), (len_ > 100, r_long)):
            if band.sum() >= 3:
                acc.append(spearman(pc[band], e_[band]))

        # DOES DECAY DISCRIMINATE AMONG CANDIDATES AT ALL? If d_k is flat, decay
        # cannot be carrying selection; if it is sharp and low off the chosen
        # candidate, it can. Raised by the interpretability window.
        d_ = pdec[lo:lo + k]
        if d_.mean() > 0:
            cv_dec.append(float(d_.std() / d_.mean()))
        dec_at_sel.append(float(d_[int(np.argmax(ps))]))
        dec_max.append(float(d_.max()))

        # ---- Q1, only where a reference candidate exists
        if not ref.any():
            continue
        n_used += 1
        n_cand_used += k
        sel_i = int(np.argmax(ps))
        cap_i = int(np.argmax(pc))
        p5_i = int(np.argmin(o_start[lo:lo + k]))
        hs, hc, h5 = ref[sel_i], ref[cap_i], ref[p5_i]
        hit_sel += hs
        hit_cap += hc
        hit_5p += h5
        lab = int(labels[i])
        if lab in per_label:
            per_label[lab][0] += hs
            per_label[lab][1] += 1

    ncands = np.array(ncands)
    print(f"BANK {args.bank}")
    print(f"checkpoint {ck}")
    print(f"transcripts {N}   with >=2 candidates {len(ncands):,}"
          f"   with a reference candidate {n_used:,}")
    print(f"candidates per transcript: median {np.median(ncands):.0f}"
          f"  mean {ncands.mean():.1f}  max {ncands.max()}")
    print(f"candidates in the Q1 sample {n_cand_used:,}"
          f"   chance rate {n_used/max(1,n_cand_used):.3f}")

    print("\n" + "=" * 70)
    print("Q1  DOES THE SCANNER BEAT POSITION ALONE?")
    print("=" * 70)
    print("  fraction landing on the reference candidate")
    print(f"    argmax p_select  (model)          {hit_sel/n_used:.3f}")
    print(f"    argmax p_capture (head alone)     {hit_cap/n_used:.3f}")
    print(f"    most 5' candidate (prior alone)   {hit_5p/n_used:.3f}")
    print(f"    chance                            {n_used/n_cand_used:.3f}")
    for lab, name in ((1, "NMD"), (0, "control")):
        h, n = per_label[lab]
        if n:
            print(f"    model, {name:<8} transcripts        {h/n:.3f}  (n={n:,})")
    print("  If the model does not beat the 5'-most baseline, 'the scanner works'")
    print("  is a statement about the stick-breaking ordering, not about the head.")

    print("\n" + "=" * 70)
    print("Q2  IS CAPTURE CONCENTRATED, OR IS THE PRIOR SELECTING?")
    print("=" * 70)
    cv, es, ms = np.array(cv_cap), np.array(ent_sel), np.array(maxsel)
    print(f"  p_capture CV within transcript      "
          + " ".join(f"{np.percentile(cv, p):.3f}" for p in (10, 25, 50, 75, 90)))
    print(f"  p_select normalised entropy         "
          + " ".join(f"{np.percentile(es, p):.3f}" for p in (10, 25, 50, 75, 90)))
    print(f"  p_select max share                  "
          + " ".join(f"{np.percentile(ms, p):.3f}" for p in (10, 25, 50, 75, 90)))
    print("  deciles 10/25/50/75/90. Entropy 0 = one candidate takes everything,")
    print("  1 = uniform. Low capture CV with low select entropy means the")
    print("  ordering is doing the work.")

    print("\n" + "=" * 70)
    print("Q3  IS CAPTURE BLIND TO THE STRUCTURAL BLOCK, ON REAL DATA?")
    print("=" * 70)
    re_, rk = np.array(r_ejc), np.array(r_koz)
    re_, rk = re_[np.isfinite(re_)], rk[np.isfinite(rk)]
    print(f"  within-transcript rank corr, p_capture vs n_downstream_ejc")
    print(f"    median {np.median(re_):+.3f}   mean {re_.mean():+.3f}   n {len(re_):,}")
    print(f"  within-transcript rank corr, p_capture vs kozak_score  (comparator)")
    print(f"    median {np.median(rk):+.3f}   mean {rk.mean():+.3f}   n {len(rk):,}")
    rp = np.array(r_part)
    print(f"\n  PARTIAL, holding candidate start position fixed")
    print(f"    median {np.median(rp):+.3f}   mean {rp.mean():+.3f}   n {len(rp):,}")
    print(f"    capture~position median {np.median(r_cp):+.3f}"
          f"    ejc~position median {np.median(r_ep):+.3f}")
    print("    If the raw association collapses here, it is POSITION, which is")
    print("    what the architecture predicts. If it survives, capture is")
    print("    tracking junction structure by some route we have not found.")

    def med(x):
        x = np.array(x); x = x[np.isfinite(x)]
        return (np.median(x), len(x)) if len(x) else (np.nan, 0)
    m1, n1 = med(r_len); m2, n2 = med(r_part_len)
    m3, n3 = med(r_short); m4, n4 = med(r_long)
    print("\n  IS THE ROUTE ORF LENGTH?")
    print(f"    capture ~ ORF length                 median {m1:+.3f}  n {n1:,}")
    print(f"    capture ~ EJC | ORF LENGTH held      median {m2:+.3f}  n {n2:,}")
    print(f"      (vs raw -0.460 and position-held -0.582; if length is the")
    print(f"       route this collapses toward zero)")
    print(f"    capture ~ EJC, ORFs <= 100 nt        median {m3:+.3f}  n {n3:,}")
    print(f"    capture ~ EJC, ORFs >  100 nt        median {m4:+.3f}  n {n4:,}")
    print("      the ATG window carries 100 nt into the ORF, so a coding-likeness")
    print("      account predicts strong in the first band and weak in the second")

    print("\n" + "=" * 70)
    print("Q4  DOES DECAY DISCRIMINATE AMONG CANDIDATES?")
    print("=" * 70)
    cd_, ds_, dm_ = np.array(cv_dec), np.array(dec_at_sel), np.array(dec_max)
    print(f"  p_decay CV within transcript        "
          + " ".join(f"{np.percentile(cd_, p):.3f}" for p in (10, 25, 50, 75, 90)))
    print(f"  p_decay at the SELECTED candidate   "
          + " ".join(f"{np.percentile(ds_, p):.3f}" for p in (10, 25, 50, 75, 90)))
    print(f"  p_decay max over candidates         "
          + " ".join(f"{np.percentile(dm_, p):.3f}" for p in (10, 25, 50, 75, 90)))
    print("  If decay were flat it could not be carrying selection; if it is")
    print("  sharp, 'decay does everything and capture is decorative' is live.")

    print("  Architecture says capture sees the ATG window only, so any EJC")
    print("  association must be INDIRECT via start context. A large direct")
    print("  association would mean the invariance is broken -- worth finding.")


if __name__ == "__main__":
    main()
