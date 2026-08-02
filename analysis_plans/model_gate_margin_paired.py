"""
model_gate_margin_paired.py — is the queue's margin over pure position real?

SCOPE: a claim ABOUT THE MODEL.

WHY THIS EXISTS. Job 8900942 re-scored section 1 against GENCODE's NMD call, as
Pete required. The gate result survived -- head's own argmax 0.304, queue built
from that same head 0.793. But the POSITION baseline rose from 0.424 (main-ORF
target) to 0.702, so the queue's margin over NO MODEL AT ALL is +0.091, not the
+0.273 the wrong target suggested.

Section 1 says the queue is "far better than either." Against the head that is
still true by +0.489. Against position it is +0.091 and UNTESTED, and I am not
writing a margin into the narrative that I have not measured. This measures it.

WHY A PAIRED TEST AND NOT TWO PROPORTIONS. Both arms are scored on the SAME 1,099
transcripts, so they are paired and the independent-proportions interval is the
wrong one -- it ignores that the two arms agree on most transcripts and would
overstate the uncertainty. McNemar uses only the DISCORDANT transcripts, which is
where the entire difference lives.

WHY ALSO A GENE-CLUSTERED BOOTSTRAP. Transcripts from one gene share sequence,
architecture and often the same annotated CDS, so they are not independent draws.
McNemar assumes they are. The gene-clustered bootstrap resamples GENES with
replacement and is the honest interval; McNemar is reported beside it as the
optimistic bound. If they disagree materially, the clustering matters and the
bootstrap is the one to quote.

REGISTERED BEFORE THE RUN:

  M1  queue > position, bootstrap CI excludes 0
        -> section 1 keeps a margin claim over position, stated as +0.091 with
           its interval, and "far better than either" is replaced by the two
           different margins: large over the head, modest over position.
  M2  CI includes 0
        -> THE QUEUE IS NOT MEASURABLY BETTER THAN THE MOST 5' CANDIDATE on the
           right target. Section 1 then claims only the gate signature -- routing
           the head through stick-breaking is what makes it useful -- and must
           state plainly that the resulting queue does not beat pure position.
           That is a materially weaker section 1 and it should say so.

  MY PREDICTION: M1, but not by much. 0.091 * 1099 is about 100 transcripts net,
  which should clear zero; gene clustering may widen it enough to matter. I have
  already been wrong once today about this comparison, in the direction of
  assuming the model looked better than it does, so I am registering the weaker
  expectation: the interval will be wide and the honest sentence will be "modest."

  ALSO REGISTERED, the part that is not about section 1: the head's own argmax
  scores 0.304 on NMD-biotype and 0.304 on protein_coding -- identical to three
  decimals. Reported here as raw counts so it can be seen as the coincidence it
  probably is rather than repeated as a finding.

Run from the repo root.
"""
import numpy as np, h5py, pandas as pd

BANK = "results_ism_v6/bank_interp_s100.h5"
N_BOOT = 10000
SEED = 20260802

bio = pd.read_csv("analysis_plans/gencode_biotype_bank.tsv", sep="\t")
bmap = dict(zip(bio.isoform_id, bio.gencode_biotype))

f = h5py.File(BANK, "r")
tx = np.array([s.decode() for s in f["transcript_id"][:]])
gid = np.array([s.decode() for s in f["gene_id"][:]])
off, cnt = f["cand_offset"][:], f["cand_count"][:]
pcap_, ps_, pd_ = f["p_capture"][:], f["p_select"][:], f["p_decay"][:]
st_, en_ = f["cand_orf_start"][:], f["cand_orf_end"][:]
gen_ = f["cand_is_gencode_start"][:]

recs = {"NMD_biotype": [], "protein_coding": []}
for i in range(len(cnt)):
    bt = bmap.get(tx[i])
    grp = ("NMD_biotype" if bt == "nonsense_mediated_decay"
           else "protein_coding" if bt == "protein_coding" else None)
    if grp is None:
        continue
    lo, k = int(off[i]), int(cnt[i])
    if k < 2:
        continue
    a = gen_[lo:lo + k] == 1                      # NEVER astype(bool)
    if not a.any():
        continue
    pcap, ps, pdc = pcap_[lo:lo + k], ps_[lo:lo + k], pd_[lo:lo + k]
    st, en = st_[lo:lo + k].astype(np.int64), en_[lo:lo + k].astype(np.int64)
    recs[grp].append((gid[i],
                      bool(a[int(np.argmax(pcap))]),      # head
                      bool(a[int(np.argmin(st))]),        # position
                      bool(a[int(np.argmax(en - st))]),   # length
                      bool(a[int(np.argmax(ps))]),        # queue
                      bool(a[int(np.argmax(ps * pdc))]))) # posterior
f.close()

ARM = {"head": 1, "position": 2, "length": 3, "queue": 4, "posterior": 5}


def paired(rows, x, y, rng):
    """McNemar plus gene-clustered bootstrap of the accuracy difference x - y."""
    g = np.array([r[0] for r in rows])
    xa = np.array([r[ARM[x]] for r in rows])
    ya = np.array([r[ARM[y]] for r in rows])
    b = int((xa & ~ya).sum())      # x right, y wrong
    c = int((~xa & ya).sum())      # y right, x wrong
    diff = xa.mean() - ya.mean()

    # exact McNemar, two-sided, on the discordant pairs only
    n = b + c
    if n:
        from math import comb
        k = min(b, c)
        p_exact = min(1.0, 2.0 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)
    else:
        p_exact = 1.0

    # gene-clustered bootstrap: resample GENES, carry all their transcripts
    ug, inv = np.unique(g, return_inverse=True)
    idx_by_gene = [np.flatnonzero(inv == j) for j in range(len(ug))]
    boots = np.empty(N_BOOT)
    for t in range(N_BOOT):
        pick = rng.integers(0, len(ug), len(ug))
        sel = np.concatenate([idx_by_gene[j] for j in pick])
        boots[t] = xa[sel].mean() - ya[sel].mean()
    lo_, hi_ = np.percentile(boots, [2.5, 97.5])
    return diff, b, c, p_exact, lo_, hi_, len(ug)


rng = np.random.default_rng(SEED)
print("=" * 78)
print("IS THE QUEUE'S MARGIN OVER PURE POSITION REAL?")
print("=" * 78)
print(f"bank {BANK}   bootstrap {N_BOOT:,} gene-clustered replicates   seed {SEED}")

for grp in ("NMD_biotype", "protein_coding"):
    rows = recs[grp]
    if not rows:
        continue
    ng = len(set(r[0] for r in rows))
    print(f"\n  {grp}   n = {len(rows):,} transcripts in {ng:,} genes")
    print(f"    {'comparison':<24}{'diff':>8}{'b':>7}{'c':>7}"
          f"{'McNemar p':>12}{'gene-clustered 95% CI':>26}")
    for x, y in (("queue", "position"), ("queue", "head"),
                 ("queue", "length"), ("posterior", "queue")):
        d, b, c, p, lo_, hi_, _ = paired(rows, x, y, rng)
        star = "" if (lo_ <= 0 <= hi_) else "  *"
        print(f"    {x+' - '+y:<24}{d:>+8.3f}{b:>7}{c:>7}{p:>12.2e}"
              f"{'['+format(lo_,'+.3f')+', '+format(hi_,'+.3f')+']':>26}{star}")

print("\n  * = gene-clustered interval excludes zero")

print("\n" + "=" * 78)
print("THE REGISTERED DECISION, on NMD_biotype")
print("=" * 78)
rows = recs["NMD_biotype"]
d, b, c, p, lo_, hi_, ng = paired(rows, "queue", "position", np.random.default_rng(SEED))
print(f"  queue - position = {d:+.3f}   gene-clustered 95% CI [{lo_:+.3f}, {hi_:+.3f}]")
print(f"  discordant transcripts: queue-only {b}, position-only {c}")
if lo_ > 0:
    print(f"  -> M1. The queue beats pure position, and the honest word is 'modest',")
    print(f"     not 'far better'. Section 1 states two different margins.")
else:
    print(f"  -> M2. THE QUEUE IS NOT MEASURABLY BETTER THAN THE MOST 5' CANDIDATE.")
    print(f"     Section 1 claims the gate signature only, and says so plainly.")

print("\n" + "=" * 78)
print("THE 0.304 / 0.304 COINCIDENCE, as raw counts")
print("=" * 78)
for grp in ("NMD_biotype", "protein_coding"):
    rows = recs[grp]
    h = sum(r[ARM["head"]] for r in rows)
    print(f"  {grp:<18} {h:>6} / {len(rows):<6} = {h/len(rows):.4f}")
print("  Equal to three decimals is a coincidence of two ratios, not a shared count.")
