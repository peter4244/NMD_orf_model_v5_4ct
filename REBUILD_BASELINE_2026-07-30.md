# Rebuild baseline — what to check the new HDF5 against

Written 2026-07-30, **before** the rebuild, so the check is a check and not a rationalisation.

**REWRITTEN THE SAME DAY. The first version of this file was wrong in its headline**, and it was
wrong in the one place that mattered: it told you to expect ~39,660 transcripts and said the
handoff's ~41,765 was an error. The reverse is true. §4 keeps the retraction rather than deleting
it, because this document is the one that gets opened at rebuild time and a silently-corrected
baseline is worse than a wrong one.

---

## 1. What the rebuild should produce

The rebuild runs `sbatch slurm_build_h5_dn.sh`, which passes `--results-dir results_4ct_dn`, so it
builds on the **deposit-native** tree against **fresh Stage 3 tables** (Pete's call, 2026-07-30 —
not the 2026-07-27 vintage).

| check | expectation |
|---|---|
| `Master transcript list: N isoforms` | **N = 42,043 − k**, exactly, where `k` is the count on the `Excluding read-through loci:` line |
| total transcripts | **41,765** if `k` = 278; the arithmetic must close either way |
| `Excluding read-through loci:` | ~278 transcripts on ~233 composite gene ids, ~0.66% |
| `test_paralog` | **below 122** — two of its genes were read-through loci |
| `val_paralog` | **> 0**; expected near 56 isoforms from 19 val-side genes |
| `train + val + val_paralog + test + test_paralog` | **= N** (asserted in code as of `420a264`; raises otherwise) |
| NMD fraction | ~22% overall and in each split |
| `label provenance:` line | must appear, naming `export_rds.R` and the scan it read |
| `gene_id present for all N transcripts` | ideally yes; if not, the count is printed and those transcripts skipped **both** gene-level screens |
| `transcripts had NO chromosome` | ideally absent; if printed, they went to **train** by fall-through |

The strongest of these is `N = 42,043 − k`, because it is verified **inside the log** rather than
against anyone's recollection. The read-through count is printed two lines above the master total.

**Stop if:** `test_paralog` goes **up**, `val_paralog` comes back **0**, the master total does not
equal 42,043 − k, or the label-provenance line is missing.

## 2. The three universes

Not three vintages of one quantity — three quantities at different pipeline stages:

| size | what it is |
|---|---|
| **42,063** | isoforms **eligible** for the ORF scan |
| **42,043** | isoforms the scan **returned ORFs for** — the deposit-native scaffold |
| **39,938** | the **published** model's labeled universe (8,840 NMD + 31,098 non-NMD) |

42,063 − 42,043 = **20 transcripts with no ORF at all**, carried as `no_orfs` in the scan's own
metadata rather than lost (Track A, `05s` sentinel `NO_ORFS_IN_BATCH`, `e0aabd7`/C47).

On the deposit-native scaffold the relabel step is a **verified no-op** — 42,043 rows in and out,
0 dropped, 0 label disagreements, log preserved at
`nmd_lung_longread_2026/verification/phaseC/relabel_is_noop.out` — so nothing is filtered between
scan and training there, and 39,938 never enters the deposit-native arithmetic.

## 3. The published tree, measured — reference only, NOT the rebuild target

Counted from `results_4ct/uorf_attention_predictions.tsv`, a tracked artifact carrying the
published HDF5's own `split` and `label` columns:

```bash
awk -F'\t' 'NR>1{c[$2]++} END{for(s in c) printf "  %-14s %6d\n", s, c[s]}' results_4ct/uorf_attention_predictions.tsv | sort
```

train 25,441 · val 4,236 · test 10,131 · test_paralog 130 · no val_paralog category · total 39,938.

**These belong to the published tree and must not be used as rebuild expectations.** Two are worth
carrying forward anyway:

- `test_paralog` is **130** here against the deposit-native **122**. Not a contradiction — different
  trees — but any claim quoting either number should say which. Track A is landing that as a
  claim-level finding on 5.6.4.
- Labels here are **8,833 NMD / 31,105 non-NMD**, against **8,840 / 31,098** in `METHODS.md`,
  `README.md:44` and `CLAUDE.md:16`. A 7-isoform swap in opposite directions, total unchanged. Not
  NMD/non-NMD overlap — `relabel_tx_summary_4ct.R:87-96` guards that the sets are disjoint — so
  most likely a mashr vintage difference. **Still open; needs the cluster CSVs.**

## 4. What the first version of this file got wrong, and why

It claimed the expected count was ~39,660 rather than ~41,765, reasoning that
`relabel_tx_summary_4ct.R:132` drops neither-class rows and the published universe is 39,938.

The error: **applying the published tree's universe to a deposit-native rebuild.** 39,938 and
42,043 are two different trees, not two stages of one pipeline. The relabel drops nothing on the
deposit-native scaffold, so the 39,938 figure never applies there.

It also inverted a second diagnosis. `METHODS.md`'s "0.70% against the labeled universe of 39,938"
was praised as correct; it is the figure that is **wrong** for a deposit-native rebuild, and
`data_prep.py`'s 42,0xx denominator was the right family all along. METHODS now states both, each
labelled with its tree.

Recorded because the failure mode is the one this project keeps paying for: the number was
arithmetically fine and the **population it belonged to** was never stated. An unstated population
is not a detail — it is the defect.
