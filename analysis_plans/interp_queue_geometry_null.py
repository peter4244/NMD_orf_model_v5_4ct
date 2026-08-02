"""
interp_queue_geometry_null.py — does C16's +0.442 need a model at all?

SCOPE: an INSTRUMENT claim. Nothing here touches the network. It asks what a
stick-breaking queue produces on this candidate set when the head is REMOVED.

THE OBJECTION BEING TESTED (interpretability window, 2026-08-02).

C16 reports p_select ~ n_downstream_ejc = -0.050 raw, +0.442 holding ORF length,
and reads it as "at matched length the model routes toward junction-bearing
candidates." But p_select is not a head output. It is

    p_select_k = p_capture_k * PROD_{j<k} (1 - p_capture_j)

whose second factor is a SURVIVAL TERM that decreases monotonically with queue
index k. And queue index is 5'->3' start order, while n_downstream_ejc counts
junctions downstream of the candidate's STOP -- so a more 5' candidate has more
transcript left downstream and carries MORE junctions.

    earlier in queue -> larger survival term -> higher p_select
    earlier in queue -> more downstream junctions

That is a POSITIVE p_select ~ ejc association produced by the ordering we imposed,
with no model behaviour in it. Holding ORF LENGTH does not hold POSITION: at fixed
length, candidates still sit anywhere along the transcript. So the length partial
leaves this channel completely intact.

C14 already measured the queue's contribution at +0.403. C16's partial is +0.442.
The parsimonious reading is that C16 did not find a new relationship -- it removed
the head's length-driven negative arm and thereby EXPOSED the queue arm C14 had
already quantified.

THE NULL. Set every p_capture to the same constant c. Then

    p_select_k = c * (1-c)^(k-1)

is a strictly DECREASING function of slot, so its Spearman correlation with
anything equals that of -slot, for any c in (0,1). The degenerate null therefore
needs no parameter and no model: it is exactly partial(-slot, ejc, length).

WHAT THIS CAN AND CANNOT ESTABLISH. It CANNOT show C16 is only geometry -- the
real p_select uses real capture values and sits somewhere between this null and a
genuine head effect. It CAN supply the reference point C16 has no value for:
whether the ordering ALONE already produces a partial of that size. C16 is
currently reported against an implicit reference of zero, and zero is the wrong
reference for a quantity built from a monotone rank product.

REGISTERED BEFORE THE RUN:

  GEOMETRY SUFFICES   null partial >= ~+0.35. The ordering alone reproduces most
                      of C16's +0.442. C16's number is then not evidence of a
                      routing preference, and the retracted "the queue's order
                      correlates with them" framing is what it re-derives.
  GEOMETRY PARTIAL    null partial +0.10 to +0.35. Real but insufficient; the gap
                      is the part needing a model, and C16 should be restated
                      against this floor rather than against zero.
  GEOMETRY IRRELEVANT null partial < +0.10. My objection fails. C16 stands as
                      written and the routing preference is a model behaviour.
  WRONG SIGN          null partial negative. My premise about position and
                      junction count is backwards; withdraw the objection.

Data: results_pool_v6/orf_pool.tsv, the candidate pool -- local, 802,035 rows.
Statistic and estimator deliberately copied from model_cancellation_channel.py
(sha 29cd4de1...): within-transcript Spearman, first-order partial by the same
formula, median across transcripts, k>=4. A different estimator would confound
"different answer" with "different method".

POPULATION. C16 ran on bank_interp_s100.h5, which covers the 4,999-transcript ISM
subset (results_ism_v6/ism_subset.tsv), giving n = 4,815 at k>=4. The full pool has
42,043 transcripts. Both are reported below and the MATCHED one is the comparison;
running the null on the full pool while comparing to a subset result would be the
same wrong-reference error this script exists to point out.
"""
import numpy as np, pandas as pd, sys, hashlib, pathlib

ROOT = pathlib.Path("/Users/petecastaldi/claude_projects/NMD_orf_model_v5_4ct")
POOL = ROOT / "results_pool_v6/orf_pool.tsv"

print("=== code provenance ===")
me = pathlib.Path(__file__).resolve()
print(f"{hashlib.sha256(me.read_bytes()).hexdigest()}  {me.name}")

# ---------------------------------------------------------------- estimators
# Copied verbatim from model_cancellation_channel.py so the comparison is like
# for like. Any change here breaks comparability with C16 and must be stated.
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

# ---------------------------------------------------------------- self-test
print("\n=== self-test, must pass before any data is read ===")
rng = np.random.default_rng(0)

# 1. perfect monotone agreement
assert abs(sp(np.arange(20), np.arange(20)*3.0) - 1.0) < 1e-9, "sp monotone"
assert abs(sp(np.arange(20), -np.arange(20)*3.0) + 1.0) < 1e-9, "sp antitone"

# 2. a strictly decreasing survival term ranks exactly like -slot, for any c.
#    This is the identity the whole null rests on, so it is asserted not assumed.
for c in (0.05, 0.3, 0.9):
    k = np.arange(1, 15)
    psel = c * (1 - c) ** (k - 1)
    ej = rng.normal(size=len(k))
    assert abs(sp(psel, ej) - sp(-k, ej)) < 1e-9, f"survival != -slot at c={c}"
print("  survival term ranks as -slot for c in {0.05, 0.3, 0.9}   OK")

# 3. partial removes a genuine common cause
cc = rng.normal(size=400)
aa = cc + 0.05*rng.normal(size=400)
bb = cc + 0.05*rng.normal(size=400)
assert abs(partial(aa, bb, cc)) < 0.35, f"partial failed to remove common cause: {partial(aa,bb,cc):+.3f}"
print(f"  partial removes a common cause: raw {sp(aa,bb):+.3f} -> partial {partial(aa,bb,cc):+.3f}   OK")

# 4. partial LEAVES a relationship that does not run through c
dd = rng.normal(size=400)
ee = dd + 0.05*rng.normal(size=400)
ff = rng.normal(size=400)
assert partial(dd, ee, ff) > 0.9, "partial destroyed an unrelated-to-c relationship"
print(f"  partial preserves a relationship independent of c: {partial(dd,ee,ff):+.3f}   OK")
print("  SELF-TEST PASSED")

# ---------------------------------------------------------------- data
print(f"\n=== data ===\n{POOL}")
df = pd.read_csv(POOL, sep="\t", usecols=["isoform_id","slot","orf_start","orf_length","n_downstream_ejc"])
print(f"  {len(df):,} candidates, {df.isoform_id.nunique():,} transcripts")

# Is slot actually 5'->3' start order? The whole objection assumes it. Assert it.
chk = df.groupby("isoform_id", sort=False).apply(
    lambda g: sp(g.slot.values, g.orf_start.values) if len(g) >= 4 else np.nan,
    include_groups=False)
chk = chk[np.isfinite(chk)]
print(f"  slot ~ orf_start   median {np.median(chk):+.4f}  over {len(chk):,} transcripts")
assert np.median(chk) > 0.99, "slot is NOT 5'->3' start order; the objection's premise fails"

# ---------------------------------------------------------------- the null
SUB = ROOT / "results_ism_v6/ism_subset.tsv"
keep = set(pd.read_csv(SUB, sep="\t", usecols=["isoform_id"]).isoform_id)
print(f"  ISM subset (what the bank covers): {len(keep):,} transcripts")

def run(frame, tag):
    R = {k: [] for k in ("pos_ejc","null_raw","null_par","len_ejc","pos_len","kk")}
    for _, g in frame.groupby("isoform_id", sort=False):
        if len(g) < 4: continue
        slot = g.slot.values.astype(float)
        ej   = g.n_downstream_ejc.values.astype(float)
        ln   = g.orf_length.values.astype(float)
        st   = g.orf_start.values.astype(float)
        R["pos_ejc"].append(sp(st, ej))          # premise: 5' start -> more downstream EJC
        R["null_raw"].append(sp(-slot, ej))      # queue-only p_select, no model
        R["null_par"].append(partial(-slot, ej, ln))   # <-- THE NUMBER
        R["len_ejc"].append(sp(ln, ej))          # C16 measured -0.575 here
        R["pos_len"].append(sp(st, ln))
        R["kk"].append(len(g))

    def rep(name, key):
        x = np.array(R[key], float); x = x[np.isfinite(x)]
        print(f"  {name:<52} median {np.median(x):+.3f}   n {len(x):,}")
        return float(np.median(x))

    print(f"\n{'='*76}\n{tag}\n{'='*76}")
    kk = np.array(R["kk"])
    print(f"  candidates per transcript: median {np.median(kk):.0f}  "
          f"q1 {np.percentile(kk,25):.0f}  q3 {np.percentile(kk,75):.0f}  "
          f"(Spearman is coarse at small k -- why medians land on round values)")
    print("\n  -- premise checks --")
    rep("orf_start ~ ejc        (5' start, more downstream EJC?)", "pos_ejc")
    le = rep("ORF length ~ ejc       (C16 measured -0.575)", "len_ejc")
    rep("orf_start ~ ORF length (is length a proxy for position?)", "pos_len")
    print("\n  -- the null: p_select with every p_capture equal --")
    nr = rep("QUEUE-ONLY p_select ~ ejc          raw", "null_raw")
    npar = rep("QUEUE-ONLY p_select ~ ejc   | LENGTH HELD", "null_par")
    return nr, npar, le

run(df, "FULL POOL -- context only, NOT the comparison")
nr, np_, le = run(df[df.isoform_id.isin(keep)], "MATCHED TO THE BANK -- this is the comparison")

print("\n" + "="*76)
print(f"  C16 measured, with the real model:  raw -0.050   | length held  +0.442")
print(f"  this null, with no model at all:    raw {nr:+.3f}   | length held  {np_:+.3f}")

v = ("GEOMETRY SUFFICES -- the ordering alone reproduces most of C16's +0.442"
     if np_ >= 0.35 else
     "GEOMETRY PARTIAL -- real but insufficient; C16 should be restated against this floor"
     if np_ >= 0.10 else
     "GEOMETRY IRRELEVANT -- my objection fails, C16 stands as written"
     if np_ >= 0 else
     "WRONG SIGN -- my premise is backwards, withdraw the objection")
print(f"\n  -> {v}")
print(f"\n  reference check: length~ejc here {le:+.3f} against C16's -0.575")
