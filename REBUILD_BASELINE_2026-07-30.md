# Rebuild baseline — what to check the new HDF5 against

Written 2026-07-30, **before** the rebuild, so the check is a check and not a rationalisation.

Supersedes the expected counts in `nmd_lung_longread_2026/docs/TRACK_B_HANDOFF_2026-07-29.md`
section 2, which are wrong — see §3.

---

## 1. The published baseline, measured (not remembered)

Counted from `results_4ct/uorf_attention_predictions.tsv`, a **tracked artifact in this repo**
that carries the HDF5's own `split` and `label` columns for every isoform. Reproduce with:

```bash
awk -F'\t' 'NR>1{c[$2]++} END{for(s in c) printf "  %-14s %6d\n", s, c[s]}' results_4ct/uorf_attention_predictions.tsv | sort
```

| quantity | published value |
|---|---|
| transcripts in the HDF5 | **39,938** (39,938 unique isoform_ids) |
| train | **25,441** |
| val | **4,236** |
| test | **10,131** |
| test_paralog | **130** |
| val_paralog | **absent** — the category did not exist |
| labels in the HDF5 | **8,833 NMD / 31,105 non-NMD** |

Cross-checks that hold: the five splits sum to 39,938; `n_test` in
`results_4ct/metrics_atg500_stop500.json` is 10,131 and `predictions_atg500_stop500.tsv` has
10,131 rows; NMD prevalence is 21–22% in every split.

**Two of these contradict what was written down.**

- `test_paralog` is **130**, not the 122 the handoff states.
- The HDF5 holds **8,833 / 31,105**, not the **8,840 / 31,098** in `METHODS.md:21-23`,
  `README.md:44` and `CLAUDE.md:16`. A 7-isoform disagreement, in opposite directions, totals
  unchanged. Not the NMD/non-NMD overlap — `relabel_tx_summary_4ct.R:87-96` guards that the two
  sets are disjoint on this vintage — so most likely a mashr vintage difference between the run
  that built the HDF5 and the run the documented counts came from. **Needs the mashr CSVs on the
  cluster to settle; logged, not chased.**

`val_paralog` being absent confirms the validation-side screen is genuinely new, so it has no
published baseline to "move off". The handoff's "val_paralog moves off 56" has no referent.

## 2. What the rebuild should produce

The exclusion code has **never run** — the HDF5 has not been rebuilt since `cf19dd6`. So the
278 / 233 read-through figures come from an ad-hoc analysis, not from this pipeline.

That matters for the denominator. `data_prep.py:628`'s comment says *"0.66% of 42,063"*, and
278/42,063 = 0.661% — a three-digit match, so the 278 was counted over a **42,063-row** table.
`data_prep.py` reads `tx_summary.tsv`, which relabel has already cut to the labeled universe.
So the number of read-through transcripts **inside** the labeled set is `k ≤ 278`, and is not
known in advance.

**Therefore the honest prediction is a range with an exact internal consistency requirement,
not a single number:**

| check | expectation |
|---|---|
| `Master transcript list: N isoforms` | **N = 39,938 − k**, exactly, where `k` is the count on the `Excluding read-through loci:` line |
| total transcripts | between **39,660** and **39,938** |
| `test_paralog` | **< 130** (two of its genes were read-through loci) |
| `val_paralog` | **> 0**, and new — no baseline |
| `train + val + val_paralog + test + test_paralog` | **= N** (now asserted in code, raises otherwise) |
| NMD fraction | ~22% overall and in each split |
| `gene_id present for all N transcripts` | ideally yes; if not, the count is printed and those transcripts skipped **both** gene-level screens |
| `transcripts had NO chromosome` | ideally absent; if printed, they went to **train** by fall-through |
| `label provenance:` line | must appear, naming `relabel_tx_summary_4ct.R` and the mashr date |

This is a stronger check than a remembered number, because `N = 39,938 − k` is verified **within
the log itself** rather than against anyone's recollection.

**Stop if:** `test_paralog` goes **up**, `val_paralog` comes back **0**, the master total does not
equal 39,938 − k, or the label-provenance line is missing.

## 3. Why the handoff's ~41,765 is wrong

It computes 42,043 − 278. But 42,043 is a **pre-relabel scaffold** count, and the read-through
exclusion runs on the **post-relabel** table:

```
export_rds.R          -> tx_summary_prelabel.tsv   (ORFik scaffold, ~42,0xx rows)
relabel_*.R           -> tx_summary.tsv            (drops neither-class rows -> 39,938)   :132
data_prep.load_tx_summary <- tx_summary.tsv                                                :~299
data_prep.py          tx_ids = tx_summary ids ∩ FASTA ids                                  :608-609
data_prep.py          removes read-through from THAT set                                   :632-638
```

`METHODS.md:95-96` already states this correctly (*"divides by the isoform count after the FASTA
intersection; against the labeled universe of 39,938 … that is 0.70%"*). The error is in the
`data_prep.py:628` code comment, and the handoff inherited it.

The measured published count of 39,938 also settles a second thing: **no labeled isoform is
missing from the FASTA**, so that term drops out of the arithmetic entirely.

## 4. The three universe sizes, resolved

They are not three vintages of one quantity. They are two quantities at different pipeline stages:

- **39,938** — the labeled universe: relabel's output, data_prep's input, and (measured above)
  exactly what the published HDF5 contains.
- **42,043 / 42,063** — pre-relabel scaffold counts. The 20-row gap between them is still open,
  but it is upstream of the model and does not affect any model number.

Only 39,938 should be quoted in the manuscript for the model's universe, and after the rebuild it
becomes 39,938 − k.
