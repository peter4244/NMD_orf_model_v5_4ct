# Rebuild baseline — what to check the new HDF5 against

Written 2026-07-30, **before** the rebuild, so the check is a check and not a rationalisation.

**REWRITTEN THE SAME DAY. The first version of this file was wrong in its headline**, and it was
wrong in the one place that mattered: it told you to expect ~39,660 transcripts and said the
handoff's ~41,765 was an error. The reverse is true. §4 keeps the retraction rather than deleting
it, because this document is the one that gets opened at rebuild time and a silently-corrected
baseline is worse than a wrong one.

---

## 0. Check the ENCODING first, before any count

```bash
python3 verify_clip_in_h5.py results_4ct_dn/nmd_orf_data.h5 --window 2000
```

This is step zero because the three 2026-07-29 data_prep fixes have **never run**, and the
pipeline's history is of being wrong at exit 0. The assay reads the clip's arithmetic signature out
of the frame channels — `atg_filled + stop_filled == W + (L-3)` where both windows clip, versus
`2W` everywhere pre-clip — so it needs no anchor arrays and no downstream output. Proven against
synthetic pre- and post-clip files built with the real encoder: post-clip 75.7% of ORFs below 2W
(PASS), pre-clip 0% (FAIL).

**Do not assay the clip by checking whether the branch decomposition moved.** Every section-5
deposit-native value in the ledger was measured 2026-07-27, before the fixes landed, so those are
**pre-clip baselines, not targets**. A retrain that reproduces 61.6/26.1/12.3 is evidence the clip
did not take effect. Stasis is the alarm.

## 1. What the rebuild should produce

The rebuild runs `sbatch slurm_build_h5_dn.sh`, which passes `--results-dir results_4ct_dn`, so it
builds on the **deposit-native** tree against **fresh Stage 3 tables** (Pete's call, 2026-07-30 —
not the 2026-07-27 vintage).

**`k` IS NOW KNOWN EXACTLY (2026-07-30 02:xx), computed from the fresh tables before the rebuild
ran.** Track A's Stage 3 output is at `~/claude_w69_tables_2026-07-30/` (a throwaway projection per
D16). Joining `tx_summary.tsv` to `ref_cds_features.tsv` on `isoform_id` and counting composite
`ENSGa.v::ENSGb.v` gene ids gives **278 transcripts on 233 loci, 0.66%** — so every expectation
below is an exact number rather than a range. Reproduce with the snippet in §5.

| check | expectation |
|---|---|
| `Master transcript list: N isoforms` | **41,765**, exactly (42,043 − 278) |
| `Excluding read-through loci:` | **278** transcripts on **233** composite gene ids, **0.66%** |
| `gene_id present for all N transcripts` | **yes** — measured 0 of 42,043 missing, so the counter must report the all-present line |
| `train` | **26,711** |
| `val` | **4,356** |
| `val_paralog` | **56** — UNCHANGED, see below |
| `test` | **10,520** |
| `test_paralog` | **122** — UNCHANGED, see below |
| `train + val + val_paralog + test + test_paralog` | **= 41,765** (asserted in code as of `420a264`; raises otherwise) |
| NMD fraction | ~22% overall and in each split |
| `label provenance:` line | must appear: **42,043 rows, NMD=9,425**, `export_rds.R` @ 2026-07-30T01:57:47, scan mtime 2026-07-30T01:56:14 |
| NMD count after exclusion | **≤ 9,425**; the fresh scaffold is 9,425 / 32,618 = 22.4% NMD |
| `gene_id present for all N transcripts` | ideally yes; if not, the count is printed and those transcripts skipped **both** gene-level screens |
| `transcripts had NO chromosome` | ideally absent; if printed, they went to **train** by fall-through |

The strongest of these is `N = 42,043 − k`, because it is verified **inside the log** rather than
against anyone's recollection. The read-through count is printed two lines above the master total.

### The paralog stop condition was BACKWARDS, and this corrects it

The Track B handoff says *"`test_paralog` moves off 122 — lower, because two of its genes were
read-through loci"*, `data_prep.py` carried the same claim in a comment, and an earlier version of
**this file** said "below 122". All three are wrong, and the reason is the very defect the exclusion
was introduced to remove.

A composite gene id is `ENSGa.v::ENSGb.v`. It never equals a plain `ENSG` in `paralog_genes.tsv`,
so a composite-locus transcript was **never in `test_paralog` to begin with** — that invisibility is
the bug. Excluding it can therefore only shrink `train`/`val`/`test`. Measured on the fresh tables,
before the rebuild ran:

| | pre-exclusion | post-exclusion | delta |
|---|---|---|---|
| train | 26,887 | **26,711** | −176 |
| val | 4,381 | **4,356** | −25 |
| val_paralog | 56 | **56** | **0** |
| test | 10,597 | **10,520** | −77 |
| test_paralog | 122 | **122** | **0** |
| total | 42,043 | **41,765** | −278 |

Of the 278 excluded transcripts, **zero** have a gene in either paralog list. The leakage is removed
by deleting the twin **sequence**, not by reassigning a split — which is exactly what METHODS
"Isoform universe" says, and METHODS was right while the handoff and the code comment were not.

**So do not read an unchanged `test_paralog` as evidence the exclusion failed.** Unchanged is
correct, and 122 is also what claim 5.6.4 rests on, so it staying put is convenient rather than
suspicious. `data_prep.py`'s comment is corrected.

**Stop if:** any of the five split counts differs from the table above, the master total is not
41,765, the read-through line is not 278 / 233, or the label-provenance line is missing.

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


---

## 5. Verified against the fresh tables, before the rebuild

Track A's Stage 3 finished 2026-07-30 ~01:57 into `~/claude_projects/nmd_w69_tables_2026-07-30/`.
Everything checkable without the cluster was checked there, and all of it passed:

| quantity | measured | expected |
|---|---|---|
| `tx_summary.tsv` rows | **42,043** | 42,043 — the deposit-native scaffold |
| `ref_cds_features.tsv` rows | **42,063** | 42,063 — the *eligible* set |
| difference | **20** | C47's 20 no-ORF transcripts, now visible directly in the tables |
| `paralog_genes.tsv` | **56** genes | 56 test-side |
| `val_paralog_genes.tsv` | **19** genes | 19 val-side |
| sidecar vs file | 42,043 rows / 9,425 NMD, **match** | the guard's whole purpose |
| NMD fraction | **22.4%** | ~22% |
| `SHA256SUMS.txt` | **OK** | — |
| read-through in scaffold | **278 tx / 233 loci** | ~278 / ~233 |
| isoforms with no `gene_id` | **0** of 42,043 | ideally 0 |

One alarm raised and resolved: `paralog_genes.tsv` is 1.2 KB here against 9.8 MB in the 2026-07-27
staging tree. That is a **format** change, not data loss — the new file is the flagged gene list
(56 rows) and the counts match Track A's stated 56/19 exactly.

```bash
# recompute k from the tables, no cluster needed
python3 - <<'EOF'
import csv
gene = {r["isoform_id"]: r["gene_id"] for r in
        csv.DictReader(open("ref_cds_features.tsv"), delimiter="\t")}
tx = [r["isoform_id"] for r in csv.DictReader(open("tx_summary.tsv"), delimiter="\t")]
rt = [t for t in tx if "::" in str(gene.get(t, ""))]
print(len(tx), len(rt), len({gene[t] for t in rt}), len(tx) - len(rt))
EOF
```

**The labels changed and this is where it shows.** The fresh scaffold is 9,425 NMD / 32,618
non-NMD over 42,043. The published tree was 8,833 / 31,105 over 39,938, and METHODS/README/CLAUDE
document 8,840 / 31,098. Three different pairs. The fresh one is authoritative for the retrain and
its provenance is now recorded; the published-vs-documented 7-isoform gap remains open and is a
question about the *published* record, not about this build.
