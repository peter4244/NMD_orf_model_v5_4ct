"""
model_cancellation_channel.py — is the -0.453 / +0.403 cancellation ONE relationship
seen twice with opposite sign?

SCOPE: a claim ABOUT THE MODEL.

THE HYPOTHESIS (interpretability window, offered as reasoning not measurement).
C14 found p_capture ~ ejc = -0.453 and p_select ~ ejc = -0.050, so the 5'->3'
ordering contributes +0.403 and the two arms nearly cancel. Their proposal: this
may be a SINGLE ORF-LENGTH RELATIONSHIP SEEN TWICE WITH OPPOSITE SIGN --

  the HEAD favours long ORFs, whose stops sit near the 3' end and therefore carry
  FEWER downstream junctions  -> capture ~ ejc negative
  the QUEUE favours upstream candidates, which are short and carry MORE downstream
  junctions                    -> ordering contribution positive

If so, holding ORF length fixed collapses BOTH arms and there is no second channel.

WHY A LENGTH PARTIAL IS LEGITIMATE HERE, WHEN PETE RULED IT OUT ELSEWHERE. His
correction was that length is upstream of window content, so conditioning on it
blocks the causal path being measured. That applies when the question is "how big
is the effect." It does NOT apply here, because the question is "ARE THESE TWO ARMS
THE SAME RELATIONSHIP" -- and that is exactly what a partial answers. Stated so the
distinction is on the record rather than assumed.

REGISTERED BEFORE THE RUN:

  ONE CHANNEL   both partials collapse toward zero (|r| < 0.10). The cancellation
                is one length relationship seen twice, and there is nothing else.
  TWO CHANNELS  at least one partial survives. Something beyond length drives one
                arm, and which arm survives says which.
  INVERTED      a partial grows rather than collapses, as happened in C7 when
                position was held. Then length was SUPPRESSING the relationship and
                the account is wrong in a third way.

Run from the repo root.
"""
import numpy as np, h5py

f = h5py.File("results_ism_v6/bank_interp_s100.h5", "r")
off, cnt = f["cand_offset"][:], f["cand_count"][:]
ps_, pc_ = f["p_select"][:], f["p_capture"][:]
ej_, os_, oe_ = f["cand_n_downstream_ejc"][:], f["cand_orf_start"][:], f["cand_orf_end"][:]

def sp(x, y):
    if len(x) < 4: return np.nan
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])

def partial(a, b, c):
    rab, rac, rbc = sp(a, b), sp(a, c), sp(b, c)
    if not all(np.isfinite([rab, rac, rbc])): return np.nan
    d = np.sqrt((1 - rac**2) * (1 - rbc**2))
    return float((rab - rac*rbc)/d) if d > 1e-9 else np.nan

R = {k: [] for k in ("cap_raw","cap_par","sel_raw","sel_par","len_ejc","cap_len","sel_len")}
for i in range(len(cnt)):
    lo, k = int(off[i]), int(cnt[i])
    if k < 4: continue
    pc, ps = pc_[lo:lo+k], ps_[lo:lo+k]
    ej = ej_[lo:lo+k].astype(float); ln = (oe_[lo:lo+k]-os_[lo:lo+k]).astype(float)
    R["cap_raw"].append(sp(pc, ej));  R["cap_par"].append(partial(pc, ej, ln))
    R["sel_raw"].append(sp(ps, ej));  R["sel_par"].append(partial(ps, ej, ln))
    R["len_ejc"].append(sp(ln, ej));  R["cap_len"].append(sp(pc, ln))
    R["sel_len"].append(sp(ps, ln))

def rep(n, key):
    x = np.array(R[key], float); x = x[np.isfinite(x)]
    print(f"  {n:<46} median {np.median(x):+.3f}   n {len(x):,}")
    return float(np.median(x))

print("="*76)
print("IS THE CANCELLATION ONE LENGTH RELATIONSHIP SEEN TWICE?")
print("="*76)
cr = rep("p_capture ~ ejc                    raw", "cap_raw")
cp = rep("p_capture ~ ejc   | ORF LENGTH HELD", "cap_par")
print()
sr = rep("p_select  ~ ejc                    raw", "sel_raw")
spp = rep("p_select  ~ ejc   | ORF LENGTH HELD", "sel_par")
print()
rep("ORF length ~ ejc          (the shared driver?)", "len_ejc")
rep("p_capture ~ ORF length", "cap_len")
rep("p_select  ~ ORF length", "sel_len")

one = abs(cp) < 0.10 and abs(spp) < 0.10
inv = abs(cp) > abs(cr) or abs(spp) > abs(sr)
v = ("INVERTED -- length was SUPPRESSING it; the account is wrong a third way" if inv
     else "ONE CHANNEL -- the cancellation is one length relationship seen twice" if one
     else "TWO CHANNELS -- something beyond length drives at least one arm")
print(f"\n  -> {v}")
