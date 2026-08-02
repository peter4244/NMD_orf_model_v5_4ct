"""
model_reweight_exposure.py — how much could reweighting move each benchmark row?

SCOPE: a claim ABOUT THE INSTRUMENT.

WHY THIS EXISTS. The ISM bank is a stratified subset with a per-transcript
`sampling_weight`, and no recovery producer applies it, so every level we quote
(0.883, 0.793, 0.702, 0.460, 0.304) is an unweighted bank statistic. Pete's call
was that this does not change the conclusion and is not worth chasing. This
tests that call instead of accepting or overriding it, and it runs LOCALLY --
`ism_subset.tsv` and `gencode_biotype_bank.tsv` are both in the repo, so no bank
and no cluster login are needed.

THE KEY REALISATION, which is why this is cheap. Reweighting can only move a
WITHIN-GROUP mean to the extent the weights VARY WITHIN THAT GROUP. If every
transcript scored in a row carries the same weight, the weighted and unweighted
means are identical and the row is exactly safe. So the question is not "is the
bank stratified" -- it is -- but "do the strata cut across the rows we report?"

THE BOUND. For a 0/1 outcome x with normalised weights p, the shift is
|sum(p*x) - mean(x)|. Maximised adversarially by concentrating the outcome on the
heaviest or lightest transcripts, which gives the WORST CASE the row could move
under any possible pattern of hits. That is a bound, not an estimate: the real
shift needs the bank. A small bound settles the row; a large bound means the row
must be recomputed with weights before it is quoted.

NOT REGISTERED AS A PREDICTION, because the answer is arithmetic rather than
empirical -- the weights are known and the bound follows from them. Recorded
instead: my expectation was that NMD_biotype would be nearly flat, because
GENCODE's NMD biotype is close to coextensive with the `main_orf_stop` strata and
those took weight ~1.

Run from the repo root.
"""
import pandas as pd, numpy as np

SUB = "results_ism_v6/ism_subset.tsv"
BIO = "analysis_plans/gencode_biotype_bank.tsv"

sub = pd.read_csv(SUB, sep="\t")
bio = pd.read_csv(BIO, sep="\t")
m = sub.merge(bio, on="isoform_id", how="left")
m["grp"] = np.where(m.gencode_biotype == "nonsense_mediated_decay", "NMD_biotype",
                    np.where(m.gencode_biotype == "protein_coding", "protein_coding",
                             "other/none"))


def worst_case_shift(w):
    """Max |weighted mean - unweighted mean| over all 0/1 outcome patterns."""
    w = np.asarray(w, float)
    p = w / w.sum()
    n = len(w)
    o = np.argsort(w)
    worst = 0.0
    for k in range(1, n):
        x = np.zeros(n); x[o[-k:]] = 1.0        # hits on the heaviest transcripts
        worst = max(worst, abs((p * x).sum() - x.mean()))
        x = np.zeros(n); x[o[:k]] = 1.0         # hits on the lightest
        worst = max(worst, abs((p * x).sum() - x.mean()))
    return worst


print("=" * 74)
print("REWEIGHTING EXPOSURE, BY BENCHMARK ROW")
print("=" * 74)
print(f"subset {SUB}\nbiotype {BIO}")
print(f"\nbank {len(sub):,} transcripts   weights sum to {sub.sampling_weight.sum():,.0f}"
      f"   ({sub.sampling_weight.sum()/len(sub):.2f}x)")

for g in ("NMD_biotype", "protein_coding"):
    d = m[m.grp == g]
    w = d.sampling_weight
    print(f"\n{'-'*74}\n{g}   n = {len(d):,}")
    print(f"  {'stratum':<34}{'n':>6}{'share':>9}{'weight':>9}")
    for c, k in d.cell.value_counts().items():
        ww = d.loc[d.cell == c, "sampling_weight"].iloc[0]
        print(f"  {c:<34}{k:>6}{k/len(d)*100:>8.1f}%{ww:>9.2f}")
    share_flat = (w <= 1.1).mean()
    print(f"\n  weights within this group: {len(w.unique())} distinct, "
          f"{w.min():.2f} to {w.max():.2f}   cv {w.std()/w.mean():.3f}")
    print(f"  share of the group at weight <= 1.1: {share_flat*100:.1f}%")
    print(f"  ⇒ WORST-CASE SHIFT FROM REWEIGHTING: {worst_case_shift(w):+.4f}")

print(f"\n{'='*74}\nWHAT THIS SETTLES\n{'='*74}")
nmd = m[m.grp == "NMD_biotype"].sampling_weight
pc = m[m.grp == "protein_coding"].sampling_weight
print(f"  Every headline number in the narrative is scored on NMD_biotype, whose")
print(f"  worst case is {worst_case_shift(nmd):+.4f} and whose realistic shift is far")
print(f"  smaller -- {(nmd <= 1.1).mean()*100:.1f}% of the group sits at weight ~1, and only")
print(f"  {(nmd > 1.1).sum()} transcripts carry the heavy weights at all.")
print(f"\n  The protein_coding CONTRAST row is a different matter: worst case")
print(f"  {worst_case_shift(pc):+.4f}, with {(pc > 10).mean()*100:.1f}% of the group at weight > 10.")
print(f"  Any claim resting on that row -- including the posterior's -0.253 --")
print(f"  needs weights before it is quoted.")
