"""
model_capture_beyond_length.py — STEP 0. Is the capture head anything beyond ORF length?

SCOPE -- CLAIMS ABOUT THE MODEL, NOT ABOUT BIOLOGY. Reads model outputs
(p_capture, p_select) against candidate features. `p_capture` IS the network's own
quantity; reading it from a cached bank does not make this a statement about
transcript biology.

--------------------------------------------------------------------------------
WHY THIS EXISTS. C11 measured that a one-line heuristic -- "take the longest
candidate" -- recovers the annotated start 0.678 of the time against the model's
0.697. I wrote that into the sentence as "the model's selection IS approximately a
longest-ORF heuristic." The interpretability window rejected it and is right:

    C11 is an ACCURACY comparison. That is a MECHANISM claim.

Two methods can score alike and be right about DIFFERENT transcripts. If they are,
"approximately a longest-ORF heuristic" is false while every number in C11 stays
true. The 2x2 is the object, not the difference of marginals -- the same error class
as the run-length null and the 1.148 figure.

This step licenses that clause or refutes it, before anything expensive runs.

--------------------------------------------------------------------------------
PRIMARY -- THE NOT-LONGEST SUBSET. Assumption-free.

Restrict to transcripts where the reference candidate is NOT the longest. There the
longest-ORF heuristic scores ZERO by construction, so whatever the model scores is
its contribution beyond length, with no model of the length relationship required.

  COMPARATOR, fixed before the run (interpretability window's amendment, and it
  closes a hole I left): NOT "above chance". Chance is 0.055 and a 12% recovery
  would clear it while meaning nothing. The bar is the best LENGTH-FREE heuristic
  available inside the same subset -- MOST-5' WITHIN THE SUBSET, which is what
  "the queue with capture switched off" means, and which can actually fail.

  DECISION, registered:
    model <= most-5' within subset  -> the mechanisms coincide; clause 1 is earned;
                                       the scanner story closes as a length
                                       heuristic plus the geometry leak we made
    model >  most-5' within subset  -> the head does something length cannot
                                       express; clause 1 is WRONG and the sentence
                                       becomes "matches a length heuristic in
                                       aggregate accuracy while differing in
                                       mechanism"

  The 2x2 of model-correct against longest-correct is printed over ALL transcripts
  as well, because agreement in the margins with disagreement in the cells is
  exactly what the objection was about and it should be visible rather than derived.

--------------------------------------------------------------------------------
COMPANION -- THE RANK RESIDUAL. Rank-specified, not linear.

+0.760 was a RANK correlation, so a linear residual could leave length structure
behind and any discrimination found would be unmodelled length rather than
something beyond it. So: compare the reference candidate's rank by p_capture
against its rank by ORF length, within transcript. A head that is only length
puts them at the same rank.

--------------------------------------------------------------------------------
THE SIGN PREDICTION -- registered before the run, and it is the only direct test
of C3 either window has proposed.

C3 says the head gates by staying QUIET upstream, which is what lets the
stick-breaking queue reach the main ORF. That predicts a specific, signed,
positionally-asymmetric discrepancy:

  FOR NON-REFERENCE CANDIDATES UPSTREAM OF THE REFERENCE, the model's rank of that
  candidate is WORSE than its length rank -- suppressed beyond what length accounts
  for.

  Downstream non-reference candidates are the internal control: gating has no
  reason to suppress them, so the discrepancy should be smaller there.

  IF the discrepancy is symmetric in position, or runs the other way, C3's gating
  reading does not survive. Stated so it can fail.

--------------------------------------------------------------------------------
SECOND ARM -- DOES CAPTURE TRACK DECAY, OR ANNOTATION? Added before the run, after
the interpretability window found that the forward-pass separation is given back by
the loss.

  P(NMD) = sum_k p_k * d_k, and the loss is BCE on that product
  (model_v6.py:199-200, train_v6.py:8,358). So dL/dz_p_k is scaled by d_k: the
  capture head cannot OBSERVE termination and is TRAINED TO PREDICT IT from the
  start window. And log_pass = cumsum(log_q) - log_q is EXCLUSIVE (:194), so
  candidate j's logit carries d_k for every downstream k -- the head is trained to
  suppress an upstream candidate when downstream candidates carry decay structure.

  CONSEQUENCE FOR THE PRIMARY, and it is asymmetric rather than fatal:
    POSITIVE branch INTACT -- recovering the reference above the most-5' bar where
      length cannot holds whatever objective the head was trained on.
    NEGATIVE branch COMPROMISED -- failing to recover it is now ambiguous between
      "only length" and "correctly optimising something else."
  This arm disambiguates that branch and nothing else.

  THE MEASUREMENT: within transcript, rank correlation of p_capture against d_k,
  and against reference status. PARTIALLED ON ORF LENGTH, raw reported beside it.
  The partial is not optional: d_k is higher for early-stopping candidates, which
  are SHORT ORFs, and C8 has capture tracking length at +0.760 with length fully
  mediating the junction association. Without the partial this arm re-derives C8
  and reports it as new -- caught by the model window before the run.

  DECISION, fixed here so it cannot be chosen after the numbers:
    DECAY-PREDICTOR   median |partial with d_k| > median |partial with reference|
                      AND the per-transcript comparison favours d_k in > 60% of
                      transcripts
    INITIATION-LIKE   the reverse on both
    LENGTH AGAIN      both partials collapse toward zero, as the junction partial
                      did (-0.009), meaning neither target survives length
    SPLIT             the two criteria disagree -> unresolved, and it is resolved
                      by more seeds rather than by looking

--------------------------------------------------------------------------------
Run from the repo root.
"""

import argparse
import numpy as np
import h5py


def rank_desc(x):
    """1 = largest. Ties get the average rank."""
    order = np.argsort(np.argsort(-x))
    return order.astype(float) + 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="results_ism_v6/bank_interp_s100.h5")
    args = ap.parse_args()

    with h5py.File(args.bank, "r") as f:
        off, cnt = f["cand_offset"][:], f["cand_count"][:]
        pcap, psel = f["p_capture"][:], f["p_select"][:]
        is_ref = f["cand_is_ref_cds"][:]
        o_s, o_e = f["cand_orf_start"][:], f["cand_orf_end"][:]
        pdec = f["p_decay"][:]
        N = len(cnt)
        ck = f.attrs.get("checkpoint", "?")

    # all-transcript 2x2 of model-correct against longest-correct
    cell = np.zeros((2, 2), dtype=int)      # [model][longest]
    sub_n = sub_model = sub_5p = sub_long = sub_cand = 0
    ref_rank_cap, ref_rank_len = [], []
    disc_up, disc_down = [], []
    rd_raw, rd_par, rr_raw, rr_par, favours_d = [], [], [], [], []

    for i in range(N):
        lo, k = int(off[i]), int(cnt[i])
        if k < 2:
            continue
        ref = is_ref[lo:lo + k].astype(bool)
        if not ref.any():
            continue
        ps, pc = psel[lo:lo + k], pcap[lo:lo + k]
        start = o_s[lo:lo + k].astype(float)
        length = (o_e[lo:lo + k] - o_s[lo:lo + k]).astype(float)

        sel_i = int(np.argmax(ps))
        lng_i = int(np.argmax(length))
        p5_i = int(np.argmin(start))
        cell[int(ref[sel_i]), int(ref[lng_i])] += 1

        # companion: where does the reference sit by capture vs by length
        rc, rl = rank_desc(pc), rank_desc(length)
        ref_i = int(np.flatnonzero(ref)[0])
        ref_rank_cap.append(rc[ref_i])
        ref_rank_len.append(rl[ref_i])

        # sign prediction: capture rank MINUS length rank, positive = suppressed
        # beyond length. Split by position relative to the reference.
        d = rc - rl
        up = (start < start[ref_i]) & (~ref)
        dn = (start > start[ref_i]) & (~ref)
        if up.any():
            disc_up.append(float(d[up].mean()))
        if dn.any():
            disc_down.append(float(d[dn].mean()))

        # SECOND ARM: capture against decay, and against annotation, both
        # partialled on ORF length so neither is C8 in disguise.
        d_ = pdec[lo:lo + k].astype(float)
        rf = ref.astype(float)
        def sp(a, b):
            ra, rb = rank_desc(a), rank_desc(b)
            if ra.std() == 0 or rb.std() == 0:
                return np.nan
            return float(np.corrcoef(ra, rb)[0, 1])
        def partial(a, b, c):
            rab, rac, rbc = sp(a, b), sp(a, c), sp(b, c)
            if not all(np.isfinite([rab, rac, rbc])):
                return np.nan
            den = np.sqrt((1 - rac ** 2) * (1 - rbc ** 2))
            return float((rab - rac * rbc) / den) if den > 1e-9 else np.nan
        pd_ = partial(pc, d_, length)
        pr_ = partial(pc, rf, length)
        rd_raw.append(sp(pc, d_)); rr_raw.append(sp(pc, rf))
        rd_par.append(pd_); rr_par.append(pr_)
        if np.isfinite(pd_) and np.isfinite(pr_):
            favours_d.append(abs(pd_) > abs(pr_))

        # PRIMARY: transcripts where the reference is NOT the longest
        if ref[lng_i]:
            continue
        sub_n += 1
        sub_cand += k
        sub_model += ref[sel_i]
        sub_5p += ref[p5_i]
        sub_long += ref[lng_i]          # zero by construction; printed as a check

    print(f"BANK {args.bank}\ncheckpoint {ck}")
    print(f"transcripts with a reference candidate and >=2 candidates: "
          f"{cell.sum():,}")

    print("\n" + "=" * 70)
    print("THE 2x2 -- do the model and the heuristic agree on WHICH transcripts?")
    print("=" * 70)
    tot = cell.sum()
    print(f"                      longest RIGHT   longest WRONG")
    print(f"    model RIGHT       {cell[1,1]:>12,}   {cell[1,0]:>13,}")
    print(f"    model WRONG       {cell[0,1]:>12,}   {cell[0,0]:>13,}")
    print(f"  model marginal   {cell[1].sum()/tot:.3f}"
          f"    longest marginal {cell[:,1].sum()/tot:.3f}")
    agree = (cell[1, 1] + cell[0, 0]) / tot
    print(f"  agreement {agree:.3f}   model-right-heuristic-wrong "
          f"{cell[1,0]/tot:.3f}   heuristic-right-model-wrong {cell[0,1]/tot:.3f}")
    print("  Similar marginals with a large off-diagonal would mean the two")
    print("  score alike while being right about different transcripts.")

    print("\n" + "=" * 70)
    print("PRIMARY -- NOT-LONGEST SUBSET (length alone cannot succeed here)")
    print("=" * 70)
    if sub_n == 0:
        print("  EMPTY SUBSET -- the reference is always the longest candidate.")
    else:
        print(f"  transcripts {sub_n:,}   candidates {sub_cand:,}"
              f"   chance {sub_n/sub_cand:.3f}")
        print(f"    model  (argmax p_select)          {sub_model/sub_n:.3f}")
        print(f"    most-5' WITHIN subset  [the bar]  {sub_5p/sub_n:.3f}")
        print(f"    longest WITHIN subset  [check=0]  {sub_long/sub_n:.3f}")
        verdict = ("BEYOND LENGTH -- clause 1 is wrong"
                   if sub_model / sub_n > sub_5p / sub_n
                   else "COINCIDES -- clause 1 earned, story closes")
        print(f"  -> {verdict}")

    print("\n" + "=" * 70)
    print("COMPANION -- reference candidate's rank by capture vs by length")
    print("=" * 70)
    rc_, rl_ = np.array(ref_rank_cap), np.array(ref_rank_len)
    print(f"  median rank by capture {np.median(rc_):.1f}"
          f"   by length {np.median(rl_):.1f}")
    print(f"  capture rank better than length rank: "
          f"{(rc_ < rl_).mean():.3f}   worse: {(rc_ > rl_).mean():.3f}"
          f"   equal: {(rc_ == rl_).mean():.3f}")

    print("\n" + "=" * 70)
    print("SIGN PREDICTION -- C3 says upstream non-reference candidates are")
    print("suppressed BEYOND length. positive = suppressed.")
    print("=" * 70)
    du, dd = np.array(disc_up), np.array(disc_down)
    print(f"  UPSTREAM   of reference: mean {du.mean():+.3f}"
          f"   median {np.median(du):+.3f}   n {len(du):,}")
    print(f"  DOWNSTREAM of reference: mean {dd.mean():+.3f}"
          f"   median {np.median(dd):+.3f}   n {len(dd):,}")
    print(f"  asymmetry (up - down) {du.mean() - dd.mean():+.3f}")
    print("  C3 survives only if upstream is positive AND larger than downstream.")

    print("\n" + "=" * 70)
    print("SECOND ARM -- does capture track DECAY or ANNOTATION, beyond length?")
    print("=" * 70)
    def m(x):
        x = np.array(x, float); x = x[np.isfinite(x)]
        return (np.median(x), len(x)) if len(x) else (np.nan, 0)
    a, na = m(rd_raw); b, nb = m(rd_par)
    c, nc = m(rr_raw); d, nd = m(rr_par)
    fav = float(np.mean(favours_d)) if favours_d else np.nan
    print(f"  capture ~ d_k          raw {a:+.3f}   partial on length {b:+.3f}  n {nb:,}")
    print(f"  capture ~ reference    raw {c:+.3f}   partial on length {d:+.3f}  n {nd:,}")
    print(f"  transcripts where |partial d_k| > |partial reference|: {fav:.3f}")
    if abs(b) < 0.05 and abs(d) < 0.05:
        v = "LENGTH AGAIN -- neither target survives length"
    elif abs(b) > abs(d) and fav > 0.60:
        v = "DECAY-PREDICTOR -- the head predicts decay from the start window"
    elif abs(d) > abs(b) and fav < 0.40:
        v = "INITIATION-LIKE -- training coupling did not dominate"
    else:
        v = "SPLIT -- unresolved; resolve by more seeds, not by looking"
    print(f"  -> {v}")


if __name__ == "__main__":
    main()
