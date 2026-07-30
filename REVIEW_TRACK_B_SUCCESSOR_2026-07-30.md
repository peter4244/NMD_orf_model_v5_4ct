# Track B successor — independent review, 2026-07-30

Review pass required by `docs/TRACK_B_HANDOFF_2026-07-29.md` section 0, before any rebuild work.
Written by the Track B successor window. **No ledger rows written.** This file is prose for
whoever holds the ledger token.

Everything below was verified locally against the code. Nothing needed Explorer. Test scripts are
in the session scratchpad; the two that matter are reproduced in §7.

Repo state on arrival: `master` @ `3f96643`, clean except the one deliberate uncommitted file.
`tools/check.py` in the analysis repo: **53/53 green**, as handed over.

---

## 1. Blocking — these sit in front of the next action

### 1.1 The handoff's pre-registered rebuild counts are wrong by ~2,100

The handoff states, in advance, that the rebuild should give **~41,765 transcripts (42,043 − 278)**,
and says: *"If `test_paralog` moves up, or `val_paralog` comes back 0, stop."*

The 41,765 is wrong. Traced through the code:

| step | file:line | what it does | count |
|---|---|---|---|
| `export_rds.R` | `:137` | writes `tx_summary.tsv` | its own header says 61K rows |
| `relabel_tx_summary_4ct.R` | `:132` | **drops neither-class rows**, writes `tx_summary.tsv` | 39,938 = 8,840 + 31,098, exact |
| `data_prep.load_tx_summary` | — | reads `tx_summary.tsv` | 39,938 |
| `data_prep.py` | `:608-609` | `tx_ids` ∩ FASTA ids | ≤ 39,938 |
| `data_prep.py` | `:632-638` | removes 278 read-through | **≈ 39,660** |

The 278 exclusion applies to the **post-relabel labeled** set, not to the 42,043-row pre-relabel
scaffold, which `data_prep.py` never sees.

`METHODS.md:95-96` **gets this right**: *"data_prep.py divides by the isoform count after the FASTA
intersection; against the labeled universe of 39,938 (above) that is 0.70%."* The wrong denominator
is in the **code comment at `data_prep.py:628`** — *"0.66% of 42,063"* — which contradicts what the
line above it actually computes (`100 * _readthrough.sum() / len(tx_ids)`). The handoff inherited
the comment's error.

**Consequence:** following the handoff's stop condition would halt a *correct* rebuild. Expect
≈39,660, and read the pre-exclusion figure off the log line `Master transcript list: N isoforms`.

### 1.2 The three universe sizes are two different quantities — and the dangerous reading is unexamined

They are not three vintages of one number:

- **39,938** = post-relabel **labeled** universe (relabel's output → data_prep's input)
- **42,043 / 42,063** = **pre-relabel scaffold** counts

The unexamined possibility, and it is the serious one: `export_rds.R` and
`relabel_tx_summary_4ct.R` **write the same path**, `tx_summary.tsv`. If
`results_4ct_dn/tx_summary.tsv` has ~42,043 rows, relabel was **never run against the dn tree**, and
the deposit-native HDF5 was built from whatever `is_nmd` column `export_rds.R` carried — plausible
values, exit 0, wrong labels. `data_prep.py:828` reads `is_nmd` with no vintage check whatsoever.

**Nothing in `data_prep.py` guards that `tx_summary.tsv` has been relabeled.** Cheap fix in the
"rules in code, not in docs" style: have relabel emit a sibling `tx_summary_provenance.json`
(mashr dir, row count, class counts) and have `data_prep.py` assert it exists and matches.

**Check on the cluster, one command, no interpretation needed:**

```bash
wc -l results_4ct/tx_summary.tsv results_4ct_dn/tx_summary.tsv
```

39,939 lines (with header) = relabeled. ~42,044 = not relabeled, and the dn HDF5 is suspect.

### 1.3 `evaluate.py`'s test-set gate has a hole: `--split all`

```
evaluate.py:32    FINAL_SPLITS = {"test", "test_clean", "test_all", "test_paralog"}
evaluate.py:179   choices=[... "test_paralog", "all"]
utils.py:176      elif split == "all":  mask = np.ones(len(splits), dtype=bool)
```

`--split all` selects **every isoform, including chr1/3/5/7**, and is not in `FINAL_SPLITS`, so it
scores the test set with **no `--final`**. Worse, `evaluate.py:196` makes it *impossible* to mark
such a run as final — `--final` with a non-test split is an error.

`METHODS.md:279-289` says *"`--final` marks the one evaluation that is allowed to touch chr1,3,5,7."*
That is false as written.

`all` has a legitimate use (full-cohort interpretation — per project convention, interpretability
uses all data while only AUC/AUPRC are test-only). So the fix is a named exemption, not removing the
choice: either require `--final` for `all`, or add a separate `--full-cohort` affirmation, and stop
`evaluate.py` writing an `auc`/`auprc` for it.

---

## 2. Defects the correction commit introduced into itself

### 2.1 Six `file:line` citations in the reviewed section were broken by `3f96643`

`3f96643` added +6 lines to `model.py` and +15 to `03_train.py`, then did not re-check the citations
pointing past the insertions:

| METHODS | cites | actually at | now points to |
|---|---|---|---|
| `:53` | `model.py:95` | 101, 103 | a blank line |
| `:57` | `model.py:203-208` | 209-214 | `)` and a comment |
| `:58` | `model.py:84` (ORFEncoder) | 90 | `x = self.mid_pool(x)` |
| `:58` | `model.py:164` (NMDOrfModel) | 170 | inside `AttentionAggregator` |
| `:240` | `03_train.py:173` | 188 | a comment line |
| `:240` | `:212` | 227 | a dict literal |

The tell: `03_train.py:140` and `:144`, in the *same paragraph*, **were** updated. The author fixed
the citations they were actively editing and left the neighbours.

Whole-file audit: **18 of 36 citations resolve correctly.** A second stale cluster (offset +20) sits
in the unreviewed "Per-ORF Structural Features" section — `model.py:85,106` (real sites 105, 126),
`model.py:199` (real site 219), `data_prep.py:46` (real site 51), `data_prep.py:47-53` (real list
52-58), plus `11_kernel_shap_branches.py:39` and `:33-118`, and
`09b_export_subgroup_profiles.py:30-43` (cited twice).

This is the same failure class as the G2 view staleness in handoff item 4 — except the ledger's G2
views are checked by `tools/check.py` and METHODS' citations are checked by nothing. A ~30-line
linter over METHODS would close it permanently.

### 2.2 The `[CHANGED]` markers stop at the section boundary

The rewrite marked "Model Training and Evaluation" thoroughly. Every universe- or encoding-dependent
number **outside** it is unmarked and now stale:

- `:21` Dataset summary — 39,938 / 8,840 / 31,098 / ratio 1:3.5
- `:498` DeepSHAP background — *"~22% NMD prevalence (8,840 / 39,938)"*
- **`:549` branch decomposition — *"60.0 / 29.1 / 10.9 over 39,938"*** ← the important one
- `:576` — *"mean −1.1233, sd 0.5417 over 39,938 rows"*
- `:657` uORF-attention — *"full v5_4ct labeled universe (39,938 = 8,840 + 31,098)"*, val AUC 0.9376

`:549` matters most. The clip exists **because** that number is confounded — `data_prep.py:255-260`
says so: *"The published branch decomposition (60.7% structural / 28.8% stop / 10.5% ATG) treats the
ATG branch and the stop branch as two independent Shapley players. When they read the same bases …
the attribution is split between inputs that are partly the same input."* The section reporting it
carries no marker at all.

### 2.3 One paragraph contradicts itself

`METHODS.md:96` computes 0.70% *"against the labeled universe of 39,938"*; `:97` then says the three
universe sizes *"are different vintages; the discrepancy is unresolved and worth settling before any
of them is quoted."* It quotes a number it declares unsettled, in the next clause. (Per §1.2 the
resolution is that 39,938 is the right one here — but the paragraph should say so, not hedge.)

---

## 3. The uncommitted file — do not land it, and the handoff's check is not sufficient

`relabel_tx_summary_4ct.R`, repointing `tx_summary_6ct.tsv` → `tx_summary_4ct.tsv`.

1. **The diff's own comment is wrong, independent of the cluster.** It says
   `tx_summary_4ct.tsv` is *"written by export_rds.R"*. `export_rds.R:137` writes
   **`tx_summary.tsv`**, and the string `4ct` appears nowhere in its outputs.
2. **Neither filename is produced by anything in this repo.** The repoint moves from one
   hand-made untracked artifact to another, so it cannot be validated from the repo at all.
3. **The docs already disagree three ways** — `README.md:82` ("bootstrap from `tx_summary_6ct.tsv`"),
   `README.md:113-114` ("`export_rds.R` … `tx_summary.tsv` is then relabeled"), and the committed
   script (`tx_summary_6ct.tsv`). The handoff says to fix `README.md:82` in the same commit;
   **`README.md:113-114` needs it too.**
4. **There is a design question under the filename question.** If `README.md:113` is right and the
   input is `tx_summary.tsv`, then relabel reads and writes the same path — non-rerunnable (a second
   run relabels already-relabeled data) and it destroys the pre-relabel scaffold. A distinct input
   name presumably exists to prevent exactly that. This is Pete's call about pipeline shape, not a
   basename lookup.
5. It hardcodes **42,043**, one of the disputed numbers, into a comment.

---

## 4. `collect_sweep.py` — a second defect, on top of the known one

The known one (anti-conservative `gap <= max(sd_a, sd_b)`) is documented and unfixed. Also present:

**4.1 The leader is chosen without requiring complete seeds.** `best = rows[0]` after sorting, with
no `n_seeds` filter (`:73-74`, and `:116-117` for the metric-choice block). Demonstrated with
synthetic cells — 11 configs at 5 seeds plus one config with a single lucky seed:

```
  atg1000_stop2000         0.99990        --   1
  atg1000_stop1000         0.92540   0.00032   5  ?    CANNOT ASSESS (needs >=2 seeds on both)
  ... all 11 fully-sampled configs report CANNOT ASSESS ...
  leader: atg1000_stop2000

=== the metric choice (D-B3.2) ===
  AUC   would select: atg1000_stop2000
  AUPRC would select: atg1000_stop2000
  Both metrics agree, so the metric choice does not change the winner here.
```

One missing cell makes the entire sweep uninterpretable, and the bottom block names the 1-seed
config as the winner under both metrics with no caveat. Since the sweep is deliberately 60
*individual* jobs (arrays are penalised on this cluster), a missing cell is the **expected** case.

**4.2 No check that the seed sets match across configurations.** The handoff's replacement spec
calls for it; it is absent. Per-config means can be formed over different seed sets and compared as
if paired.

**4.3** `pooled = max(best[skey], r[skey])` is named "pooled" but is a max — reinforces the known
anti-conservatism rather than mitigating it.

**4.4** (minor) `sorted(v.get("best_epoch") …)` at `:59` raises `TypeError` if any metrics JSON
lacks `best_epoch`. Current `evaluate.py:113` emits it, so fresh sweeps are fine; legacy files are not.

---

## 5. The fix for review "code defect 2" is narrower than METHODS states

**5.1 `verify_pool_equivalence()` runs on CPU only.** `model.py:284` is `torch.randn(n, channels, L)`
with no device, and `verify_determinism.py:131` calls it before any device is selected. The
substitution exists *because* `adaptive_max_pool2d_backward_cuda` has no deterministic kernel — the
entire concern is CUDA — and the check never touches CUDA. `METHODS.md:74-76` does not scope it.

**5.2 `verify_determinism.py` never exercises mixed precision.** `03_train.py` trains under
`autocast` + `GradScaler` on CUDA (`:62,66,165-166`); `one_run` does not. A PASS therefore does not
establish determinism of the actual GPU training path. Moot if canonical training moves to CPU
(which the handoff argues for on other grounds) — but `METHODS.md:258-259` does not say so.

**5.3** `one_run` uses plain `Adam(model.parameters())`, not `make_optimizer`'s differential weight
decay. Low materiality, but the docstring claims it builds things "EXACTLY as `03_train.py` does".

**5.4** `verify_determinism.py:40` cites `03_train.py:130-132`; the real site is `149-150`.

**5.5** The docstring rationale *"lengths that are not multiples of the adaptive output size"* is
vacuous — the output size is 1, and every integer is a multiple of 1.

**5.6** No tied inputs are ever constructed (`randn`), so the tie-routing property the docstring says
it protects is untested.

---

## 6. Lower priority, but they bear on the rebuild

**6.1 Silent-absence defaults in `data_prep.py`.** `_gene_of.get(tid, "")` (`:632`) — a transcript
missing from `ref_cds_features.tsv` is neither read-through-excluded nor paralog-flagged, silently,
and nothing counts them. `chr_map.get(tid, "")` (`:677`) — a transcript with no chromosome falls
through both branches to **train**, silently. Both are two-line instrumentation fixes and both bear
directly on whether the predicted counts mean anything.

**6.2 `model.py`'s defaults are stale.** `ORFEncoder(n_orf_features=4)` and
`config.get("n_orf_features", 4)` while `config.yaml:53` sets 5. Verified: a model built from an
empty config gets a 4-feature structural branch and **34,018** parameters instead of 34,050.
`verify_determinism.py:42-46` already documents this trap and works around it — better to make the
default 5, or remove it so a partial config raises.

**6.3 The `val_clean` guard's error message misleads after a rebuild.** `utils.py:165-171` raises
"rebuild the HDF5" when zero `val_paralog` labels exist. If the read-through exclusion drops the
val-side paralog count to 0, the message tells you to do the thing you just did.

---

## 7. What checks out — verified, not assumed

Two local test scripts, both pure-numpy/torch, no cluster data.

**7.1 Parameter counts — all exact.** Built the real `NMDOrfModel` at four grid points:
34,050 @ 500/500; 25,346 @ 100/100; 29,698 @ 100/1000; 34,050 @ 1000/2000; per-CNN 12,736 (>100) and
8,384 (=100). The correction commit's claim that "34,050/12,736 hold only where both windows exceed
100" is right.

**7.2 Every measured window number in METHODS reproduces exactly**, using the real
`encode_window_v5`:

- 5′UTR extent **249 nt for every ORF length** (L = 30, 99, 300, 498, 1200)
- ORF-side fill **14 / 49 / 149 / 248 / 251**
- total filled **263 / 298 / 398 / 497 / 500**
- unclipped reach **249/251** (start anchor), **252/248** (stop anchor)
- raw overlap `W − L + 3` → **473 / 404 / 203 / 2** at L = 30 / 99 / 300 / 501
- `mid = orf_start + (L−1)//2` — matches the code for all L in 3…2999 step 3
- matched budget at L=300: **847** (550/550) vs **749** (100/1000) vs **748** (1000/100)

  My first pass reported a mismatch here. It was my check's definitions, not METHODS — I counted
  positions either side of the *anchor*, METHODS counts them either side of *`orf_start`*. Flagging
  it because "the verification was wrong rather than the thing verified" is on the handoff's own
  list of recurring traps, and it happened again here.

**7.3 Clip inertness threshold is exactly `L >= W + 3`.** Tested at L = W, W+2, W+3, W+6: inert only
from W+3. The "off by three at the boundary" correction is right.

**7.4 Nine-channel disjointness holds, by perturbation, across 24 (W, L) combinations**
(W ∈ {100, 500, 1000, 2000} × L ∈ {30, 99, 300, 498, 1200, 3000}), with dense junctions on both
sides of the midpoint. **No base at or after the midpoint changes any channel of the start window.**
The channel-4 and channel-5 fixes are real and complete — this tests dependence, not where values
are written, which is how the original claim went wrong.

**7.5 Frame channels** — values identical across ORF lengths and start offsets, identical between
start and stop windows, exactly one-hot on the filled range and zero elsewhere. The corrected
"values carry nothing, support carries ORF length" statement is right.

**7.6 Also verified accurate:** the `val_clean` guard; `pos_weight` from the training split;
`drop_last=True`; the overfit flag being non-halting; normalization computed on the training split
over valid ORFs only (`data_prep.py:744-752`); the non-uniform `NMD_ALLOW_NONDETERMINISM`
(`joint_dn`, `atgstop_dn`, `all_dn` set it, the non-dn joint wrapper does not); and
`slurm_sweep_member.sh`'s `val_paralog` preflight plus `mkdir -p`.

---

## 8. Suggested order of operations

1. Land nothing from §3 yet — it needs a decision from Pete, not a grep.
2. Before the rebuild: add the two-line instrumentation in §6.1 and the relabel-provenance guard in
   §1.2. Both are cheap and both make the rebuild log self-verifying.
3. Correct the expected counts (§1.1) so the rebuild has a real pre-registered check:
   **≈39,660 transcripts**, `test_paralog` and `val_paralog` both moving down off 122 / 56.
4. Fix `collect_sweep.py` (§4 plus the known defect) before any sweep number is quoted.
5. §2.1 and §2.2 are METHODS edits, independent of everything else.

---

## 9. Addendum, found while landing the fixes — a blind spot in the claim graph itself

**For Track A. This is about the instrument, not the model.**

After committing, `tools/check.py` in the analysis repo went **52/53**, with
`docs/extract_edges_G2.tsv` STALE. That much is the expected fifth occurrence of the pattern in
handoff item 4. Attributing it turned up something else.

**`relabel_tx_summary_4ct.R` has 0 rows in both G2 views** — `extract_calls_G2.tsv` and
`extract_edges_G2.tsv`. The script that assigns every NMD/non-NMD training label does not appear
in the code–claim graph at all. It is not a corpus gap: `config/corpus.yml` includes `"*.R"` for
`model_repo`, so the file is in scope.

**Mechanism.** G2 extracts I/O whose path literal sits **at the call site**, and misses I/O whose
path was bound to a variable first:

| extracted | not extracted |
|---|---|
| `safe_write_tsv(df, file.path(out_dir, "tx_summary.tsv"))` — export_rds.R:137, in the graph | `write.table(tx, out_path, ...)` — relabel:159, absent |
| | `read.delim(prelabel_path, ...)` — relabel:123, absent |
| | `jsonlite::write_json(prov, prov_path, ...)` — relabel:183, absent |

**This is why the two-writers-one-file defect was invisible to the graph.** The edges view showed
exactly one writer for `tx_summary.tsv` —

```
model_repo::export_rds.R   write   tx_summary.tsv   line 137
```

— because the competing writer routed its path through a variable. A graph used as a correctness
instrument reported a single-writer file that had two writers.

It is the same shape as the basename-grep rule (a path built by `sprintf`/`paste` survives a
rename undetected), but one level up: here the path is fully determined and static, just not
*syntactically present* where the extractor looks.

**Three consequences worth landing:**

1. The extent is unknown. One confirmed instance; any variable-routed read or write anywhere in
   the corpus is equally invisible. Worth a sweep, because "no edge" currently cannot be
   distinguished from "no such edge".
2. **The rewiring committed in `420a264` will not show up in G2 either** — the new
   `export_rds.R -> tx_summary_prelabel.tsv -> relabel -> tx_summary.tsv` chain is half
   variable-routed, so regenerating the views will show `export_rds.R` writing
   `tx_summary_prelabel.tsv` and *nothing* consuming it. That absence is an artifact, not a
   dangling edge.
3. The G2 edges view also now carries one **factually wrong** row, not merely a stale line number:
   `export_rds.R write tx_summary.tsv line 137`. `export_rds.R` no longer writes that file. This
   is a graph-shape change, so regenerating is required rather than optional.

Not regenerated here — the analysis repo is Track A's, and it was live (worklog 119 → 122 while
this review ran).
