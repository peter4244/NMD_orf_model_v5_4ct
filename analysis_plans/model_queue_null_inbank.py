"""In-bank queue-geometry null, at the interpretability window's request.

Their argument: p_select_k = p_capture_k * PROD_{j<k}(1 - p_capture_j). With every
p_capture set to a constant, p_select is strictly DECREASING in slot, so its rank
equals that of -slot for any c. Queue index is 5'->3' start order, and a more 5'
candidate has more transcript downstream and therefore MORE junctions. So
p_select ~ ejc is positive BY CONSTRUCTION, with no model in it.

Zero is therefore the wrong reference for this rank-product quantity. The right
one is the queue-only null: sp(-slot, ejc).

Same population, same estimator as model_cancellation_channel.py.
"""
import numpy as np, h5py
f = h5py.File("results_ism_v6/bank_interp_s100.h5","r")
off, cnt = f["cand_offset"][:], f["cand_count"][:]
ps_, ej_, os_, oe_ = f["p_select"][:], f["cand_n_downstream_ejc"][:], f["cand_orf_start"][:], f["cand_orf_end"][:]

def sp(x,y):
    if len(x)<4: return np.nan
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    if rx.std()==0 or ry.std()==0: return np.nan
    return float(np.corrcoef(rx,ry)[0,1])
def par(a,b,c):
    rab,rac,rbc=sp(a,b),sp(a,c),sp(b,c)
    if not all(np.isfinite([rab,rac,rbc])): return np.nan
    d=np.sqrt((1-rac**2)*(1-rbc**2)); return float((rab-rac*rbc)/d) if d>1e-9 else np.nan

R={k:[] for k in ("q_raw","q_len","m_raw","m_len","len_ejc","slot_start")}
for i in range(len(cnt)):
    lo,k=int(off[i]),int(cnt[i])
    if k<4: continue
    ej=ej_[lo:lo+k].astype(float); ln=(oe_[lo:lo+k]-os_[lo:lo+k]).astype(float)
    ps=ps_[lo:lo+k]; q=-np.arange(k,dtype=float)          # queue-only null
    R["q_raw"].append(sp(q,ej));  R["q_len"].append(par(q,ej,ln))
    R["m_raw"].append(sp(ps,ej)); R["m_len"].append(par(ps,ej,ln))
    R["len_ejc"].append(sp(ln,ej)); R["slot_start"].append(sp(q,-os_[lo:lo+k].astype(float)))

def m(k):
    x=np.array(R[k],float); x=x[np.isfinite(x)]; return np.median(x), len(x)
print(f"{'':<34}{'raw':>10}{'length held':>14}")
for lab,a,b in (("queue only, NO MODEL", "q_raw","q_len"), ("the real model (C16)","m_raw","m_len")):
    (ra,na),(rb,nb) = m(a), m(b)
    print(f"  {lab:<32}{ra:>+10.3f}{rb:>+14.3f}   n {na:,}")
(le,_)=m("len_ejc"); (ss,_)=m("slot_start")
print(f"\n  length ~ ejc  {le:+.3f}   (their external run: -0.524)")
print(f"  slot vs start-order sanity (must be +1.000): {ss:+.3f}")
ma,_=m("m_len"); qa,_=m("q_len")
print(f"\n  model {ma:+.3f} against queue-only null {qa:+.3f}  ->  deficit {ma-qa:+.3f}")
