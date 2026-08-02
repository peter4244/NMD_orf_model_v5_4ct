"""
model_route_to_ptc.py — does the model ROUTE to premature-stop-bearing frames?

SCOPE: a claim ABOUT THE MODEL.

THE GAP. Pete's founding hypothesis is that the model chooses ORFs by whether they
encode a premature stop. Everything we have said about it rests on a CHAIN --
p_capture ~ d, p_select ~ d, capture ~ ejc among short candidates -- and the
DIRECT link, `p_select ~ n_downstream_ejc`, has never been measured. Raised by the
interpretability window; the gap is mine.

REGISTERED BEFORE THE RUN.

  p_capture ~ ejc is -0.460 overall (job 8898926): the HEAD scores junction-bearing
  candidates LOWER. But p_select = p_capture x survival over a 5'->3' queue, and
  upstream candidates have more sequence behind them and therefore more downstream
  junctions. So the ordering should push the other way.

  PREDICTED: p_select ~ ejc is POSITIVE, and the gap between it and the -0.460 of
  p_capture ~ ejc is the ORDERING's contribution to routing toward premature stops.

  IF NEGATIVE: the model routes AWAY from junction-bearing frames and Pete's
  hypothesis fails at the routing step, whatever the product does downstream.
  IF ~ZERO: routing is indifferent to junction structure and the alignment
  measured at the product comes entirely from d.

Reported by label, and with the posterior (p_select * d) beside the prior, since
the posterior is what the prediction is about.

Run from the repo root.
"""
import numpy as np, h5py

f = h5py.File("results_ism_v6/bank_interp_s100.h5", "r")
off, cnt = f["cand_offset"][:], f["cand_count"][:]
ps_, pc_, pd_ = f["p_select"][:], f["p_capture"][:], f["p_decay"][:]
ej_, os_ = f["cand_n_downstream_ejc"][:], f["cand_orf_start"][:]
lab = f["labels"][:]

def sp(x, y):
    if len(x) < 4: return np.nan
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])

r = {k: [] for k in ("sel", "cap", "post", "pos", "sel_nmd", "sel_ctl")}
for i in range(len(cnt)):
    lo, k = int(off[i]), int(cnt[i])
    if k < 4: continue
    ps, pc, pdc = ps_[lo:lo+k], pc_[lo:lo+k], pd_[lo:lo+k]
    ej = ej_[lo:lo+k].astype(float); st = os_[lo:lo+k].astype(float)
    a = sp(ps, ej)
    r["sel"].append(a); r["cap"].append(sp(pc, ej))
    r["post"].append(sp(ps*pdc, ej)); r["pos"].append(sp(st, ej))
    (r["sel_nmd"] if int(lab[i]) == 1 else r["sel_ctl"]).append(a)

def rep(n, x):
    x = np.array(x, float); x = x[np.isfinite(x)]
    print(f"  {n:<48} median {np.median(x):+.3f}   n {len(x):,}")
    return float(np.median(x))

print("=" * 76)
print("DOES THE MODEL ROUTE TO PREMATURE-STOP-BEARING FRAMES?")
print("=" * 76)
c = rep("p_capture ~ n_downstream_ejc   [the head]", r["cap"])
s = rep("p_select  ~ n_downstream_ejc   [THE ANSWER]", r["sel"])
p = rep("posterior ~ n_downstream_ejc   [p_select * d]", r["post"])
rep("start position ~ ejc  (5' = more junctions?)", r["pos"])
print()
rep("p_select ~ ejc, NMD transcripts", r["sel_nmd"])
rep("p_select ~ ejc, control transcripts", r["sel_ctl"])
print(f"\n  ordering's contribution = p_select - p_capture = {s - c:+.3f}")
v = ("ROUTES TOWARD premature stops -- Pete's hypothesis holds at the routing step"
     if s > 0.10 else
     "ROUTES AWAY -- the hypothesis fails at routing whatever the product does"
     if s < -0.10 else
     "INDIFFERENT -- routing ignores junction structure; the product's alignment is all d")
print(f"  -> {v}")
