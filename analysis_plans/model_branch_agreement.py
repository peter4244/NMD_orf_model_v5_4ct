"""
model_branch_agreement.py — have the two heads collapsed onto the same features?

SCOPE -- A CLAIM ABOUT THE MODEL, NOT ABOUT BIOLOGY.

THE QUESTION (Pete). P(NMD) = sum_k p_select_k * d_k, and the loss only ever sees
the PRODUCT. A product does not identify its factors -- p*d = (p*c)*(d/c) -- and the
gradients co-adapt, capture's scaled by d and decay's by p. So do p and d stay
distinct after training, or does capture become a decay-predictor within its
information limit?

WHY THE SCALAR CORRELATION IS NOT ENOUGH. Correlating p_capture against d_k asks
whether the two OUTPUTS move together, which cannot separate "the heads read the
same sequence" from "both happen to track ORF length." This measures the thing
directly: FOR EVERY POSITION, DO THE TWO BRANCHES RESPOND TO THE SAME BASES?

  vals_capture  what a substitution here does through the INITIATION branch alone
  vals_decay    the mirror, holding capture fixed
  (build_ism_bank.py attrs `branch_attribution`)

RESTRICTED TO THE ATG WINDOW, which is the only region capture can see at all
(model_v6.py:160-161). Positions covered by no candidate's ATG window are carried
as a CONTROL: capture cannot respond there, so agreement must vanish, and if it
does not the measurement is picking up something other than what it claims.

THE NOISE-FLOOR PROBLEM, AND IT IS DOCUMENTED. The bank's own `branch_resolution`
note: vals_capture is roughly a THOUSANDFOLD smaller than vals_decay, so "it is the
column where resolution could be mistaken for signal", and "the two columns have
DIFFERENT noise floors and a single liveness threshold across them would be wrong
in one." A naive correlation would partly measure which positions clear which
floor. So each column gets ITS OWN floor, measured IN SAMPLE from positions that
are noise by construction -- selection mass below 1e-8, where a substitution
cannot move the output. That is the in-sample noise null of section 3.2, not an
analytic threshold.

Rank correlation, within transcript, so between-transcript structure cannot leak
in.

DECISION, REGISTERED BEFORE THE RUN:

  >= 0.70   COLLAPSED. The heads read the same bases; the two-head decomposition
            is largely bookkeeping and the architecture's separation is nominal.
  0.20-0.70 PARTIALLY REDUNDANT. Capture has learned some of decay's job within
            its information limit, which is what the gradient argument predicts.
  <  0.20   GENUINELY SEPARATE. The factorisation means what it claims, and the
            architecture's information asymmetry is holding against the loss.

MY PREDICTION, recorded so it can be wrong: PARTIALLY REDUNDANT, and toward the
lower end. Capture is information-limited -- it cannot see the stop window or the
structural block -- and stick-breaking constrains it in a way decay is not
constrained, so full collapse is impossible. But capture already tracks downstream
junction count at -0.46, which it cannot see, so some of decay's job has been
learned by proxy.

>>> FIRST RUN FAILED ITS OWN SANITY CHECK (job 8899766). Sub-floor agreement was
>>> +0.294, HIGHER than the +0.266 among real positions, and out-of-window
>>> agreement was +0.109 rather than 0. Diagnosis: both columns scale with
>>> SELECTION MASS -- |vals| ~ mass x sensitivity in each branch -- so they
>>> correlate arithmetically wherever mass varies, whether or not the heads read
>>> the same bases. The v1 number measured mass co-scaling.
>>>
>>> FIX: STRATIFY BY MASS, do not adjust it away. Mass is the architecture
>>> (toolbox axis 1) and dividing it out asks a counterfactual the model never
>>> computes. Within a narrow log-mass band the common multiplicative factor is
>>> held, so residual agreement is FEATURE agreement. Both sanity checks are
>>> re-run within band and must now pass there.

SANITY CHECKS, both of which must pass or the number means nothing:
  - agreement among SUB-FLOOR positions must be ~0 (that is noise against noise)
  - agreement OUTSIDE any ATG window must be ~0 (capture cannot respond there)

Run from the repo root.
"""

import argparse
import numpy as np
import h5py

DEAD_CUT = 1e-8


def spearman(a, b):
    if len(a) < 5:
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

    f = h5py.File(args.bank, "r")
    spans = f["spans"][:]
    off, cnt = f["cand_offset"][:], f["cand_count"][:]
    N = len(cnt)

    # ---- pass 1: the two in-sample noise floors, from positions that cannot respond
    cap_dead, dec_dead = [], []
    for i in range(0, N, 7):                       # every 7th transcript is plenty
        lo, k = int(off[i]), int(cnt[i])
        b = spans[lo:lo + k]
        P = int(max(b[:, 3].max(), b[:, 5].max()))
        if P < 50:
            continue
        m = f["mass"][i, :P].astype(np.float64)
        dead = np.flatnonzero((m < DEAD_CUT) & f["valid"][i, :P].astype(bool))
        if not len(dead):
            continue
        with np.errstate(invalid="ignore"):
            cap_dead.append(np.nanmax(np.abs(f["vals_capture"][i, :P][dead]), 1))
            dec_dead.append(np.nanmax(np.abs(f["vals_decay"][i, :P][dead]), 1))
    cd = np.concatenate(cap_dead); dd = np.concatenate(dec_dead)
    cd, dd = cd[np.isfinite(cd)], dd[np.isfinite(dd)]
    floor_cap, floor_dec = np.percentile(cd, 95), np.percentile(dd, 95)

    print(f"BANK {args.bank}")
    print(f"checkpoint {f.attrs.get('checkpoint','?')}")
    print(f"\nIN-SAMPLE NOISE FLOORS, 95th percentile among positions with mass "
          f"< {DEAD_CUT:g}")
    print(f"  capture {floor_cap:.3e}   decay {floor_dec:.3e}"
          f"   ratio {floor_dec/max(floor_cap,1e-300):.1f}x")

    # ---- pass 2: agreement, in the ATG window, above both floors
    # global log-mass quantile bands, so a band means the same thing everywhere
    pool = []
    for i in range(0, N, 7):
        lo, k = int(off[i]), int(cnt[i])
        b = spans[lo:lo + k]
        P = int(max(b[:, 3].max(), b[:, 5].max()))
        if P < 50:
            continue
        m = f["mass"][i, :P].astype(np.float64)
        liv = m[(m >= DEAD_CUT) & f["valid"][i, :P].astype(bool)]
        if len(liv):
            pool.append(np.log10(liv).astype(np.float32))
    edges = np.quantile(np.concatenate(pool), np.linspace(0, 1, 9))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    print(f"  8 global log-mass bands, edges "
          + " ".join(f"{e:.1f}" for e in edges))

    r_in, r_sub, r_out = [], [], []
    n_used = n_pos = 0
    for i in range(N):
        lo, k = int(off[i]), int(cnt[i])
        b = spans[lo:lo + k]
        P = int(max(b[:, 3].max(), b[:, 5].max()))
        if P < 50:
            continue
        with np.errstate(invalid="ignore"):
            ec = np.nanmax(np.abs(f["vals_capture"][i, :P]), 1)
            ed = np.nanmax(np.abs(f["vals_decay"][i, :P]), 1)
        ok = f["valid"][i, :P].astype(bool) & np.isfinite(ec) & np.isfinite(ed)

        # positions covered by ANY candidate's ATG window (spans cols 2,3)
        atg = np.zeros(P, bool)
        for a0, a1 in b[:, 2:4]:
            atg[max(0, int(a0)):min(P, int(a1) + 1)] = True

        m = f["mass"][i, :P].astype(np.float64)
        band = np.full(P, -1, np.int16)
        lv = (m >= DEAD_CUT) & ok
        if lv.any():
            band[lv] = np.digitize(np.log10(m[lv]), edges[1:-1])

        # WITHIN MASS BAND, so the common mass factor is held rather than removed
        for bi in range(8):
            inb = band == bi
            live = inb & atg & (ec > floor_cap) & (ed > floor_dec)
            if live.sum() >= 20:
                r_in.append(spearman(ec[live], ed[live]))
                n_pos += int(live.sum())
            sub = inb & atg & (ec <= floor_cap) & (ed <= floor_dec)
            if sub.sum() >= 20:
                r_sub.append(spearman(ec[sub], ed[sub]))
            out = inb & (~atg) & (ed > floor_dec)
            if out.sum() >= 20:
                r_out.append(spearman(ec[out], ed[out]))
        n_used += 1
    f.close()

    def rep(name, x):
        x = np.array(x, float); x = x[np.isfinite(x)]
        if not len(x):
            print(f"  {name:<38} n=0")
            return np.nan
        print(f"  {name:<38} median {np.median(x):+.3f}   mean {x.mean():+.3f}"
              f"   n {len(x):,}")
        return float(np.median(x))

    print(f"\n{'='*70}\nBRANCH AGREEMENT -- do the two heads read the same bases?\n{'='*70}")
    print(f"  transcripts {n_used:,}   (transcript x mass band) cells scored"
          f"   positions {n_pos:,}")
    m_in = rep("IN ATG window, above floors, WITHIN BAND", r_in)
    print("\n  SANITY CHECKS -- both must be ~0 or the number above means nothing")
    rep("sub-floor positions (noise vs noise)", r_sub)
    rep("outside any ATG window (capture blind)", r_out)

    v = ("COLLAPSED -- the decomposition is largely bookkeeping" if m_in >= 0.70
         else "PARTIALLY REDUNDANT -- capture has learned some of decay's job"
         if m_in >= 0.20
         else "GENUINELY SEPARATE -- the factorisation means what it claims")
    print(f"\n  -> {v}")


if __name__ == "__main__":
    main()
