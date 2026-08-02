# Handoff — interpretability window, evening of 2026-08-02

*State, not history. What happened is in git; this is what is true and what to do.*

## Before you read anything

```sh
git merge master && git diff --name-only HEAD...master
```

Four stale reads on 2026-08-02, none caught by the reader. A worktree gives no signal that
it is behind. `tools/check_stale_reads.py` on the `results` branch is meant for this but
**currently inspects its own worktree rather than the caller's**, so it reports all-clear
from anywhere; fix pending with its owner.

## Where things stand

Pete's question was whether the model picks ORFs because they carry a premature stop.
**No.** `p_select ~ n_downstream_ejc` is −0.050 marginally and −0.067 holding length and
position jointly; the two single partials (+0.452, −0.543) are each the other confounder
leaking. The junction preference enters at the decay multiplication, not at selection.
Two independent routes, producers committed: `model_c16_retraction.py` (job 8900746) and
`interp_queue_geometry_null.py`.

Narrative: `NARRATIVE_HOW_THE_MODEL_DECIDES.md` on master. Claims: `CAPTURE_HEAD_STORY.md`.

## ⇒ Do this first

**Follow up the initiation-context finding. It is the only positive result of the day and
nobody has touched it.**

Perturbing the 25 bases immediately 5′ of the start codon moves the initiation head's logit
more than anything else in its 900-base window — **0.413 / 0.283 / 0.241 / 0.275** across
four ORF-length bands, **at the same position in every band**, in a tile **99.6% filled** in
all of them so there is no fill confound. That is where Kozak context sits.
*Jobs 8900209, 8900420; `FINDINGS_TILED_PERTURBATION_2026-08-02.md`.*

**Why this and not the louder thread.** It is the only result that got *stronger* under
scrutiny. Everything else chased that day — routing, junction structure, the whole C14/C16
sequence — turned out to be the architecture describing itself.

**What it does not yet establish.** The finding is *positional*: the head responds at −13.
It does **not** show the head reads Kozak *content*. Different claims; only the first is
earned. **Design around the receptive field, do not discover it** — the encoder's is ~42
positions and a Kozak motif is ~10 nt, so the instrument may be able to say *where* the head
responds but not *what* it reads there. `RETRAIN_ARCHITECTURE_CHANGES.md` item 3.

Tiling infrastructure exists and runs in minutes: `interp_tiled_perturbation.py`, self-test
in-file.

## Open, with priors

| | |
|---|---|
| **C15** — routing junction-biased +0.119 in NMD, −0.189 in controls; 0.31 gap the model cannot see labels to produce | All three windows called it the most interesting item. **Temper that.** Three claims reversed under conditioning that day; expect a fourth. First cut: does it survive matching NMD and control on 5′UTR architecture (uORF count, UTR length, candidate count)? If it collapses it is population composition. Needs a login |
| **No baseline under the 0.883 GENCODE recovery figure** | The 0.678 longest-ORF baseline belongs to main-ORF recovery, a different target. That number has no floor under it |
| **Fill boundary vs whatever-is-filled** | Closed to tiling — downstream tiles are already below the receptive field. Needs a retrain varying window extent; item 7 |
| **Motifs** | Nothing works. The region caller's success criterion points the wrong way |

## What will bite you

**The ISM bank is a stratified subset.** `build_ism_bank.py` writes the warning into the h5:
population estimates must be reweighted by `sampling_weight`. Thirty scripts read the bank;
two read the weights. Verified locally in `results_ism_v6/ism_subset.tsv` — weights sum to
**41,765** over **4,999** rows, and annotated transcripts are **~8×** over-represented while
recovery is scored *against* the annotation. **Exposure scales with how much your quantity
depends on `is_nmd`, `has_annotation`, `main_orf_stop`.** Recovery: maximally exposed. The
queue null: not — it moves 0.016 weighted, being within-transcript geometry. Pete has judged
this not to be a blocker; know it before you quote a bank number as a population number.

**Any candidate-level correlation is position until proven otherwise** — `CLAUDE.md`, D62.
Within-transcript ORF length ratio is **83.5× median**, so "holding length" describes a
regime the model never occupies. **Use a partial to explain why, never to state behaviour.**

**Prose is the only artifact that never gets recomputed**, and it is hard-wrapped here, so a
phrase you inserted straddles a newline and a plain `grep` false-negatives. Flatten first:
`tr '\n' ' ' < FILE | tr -s ' ' | grep -i "phrase"`. Verify a documented change by grepping
its content — a non-empty diff is not evidence the intended edit landed.

## The standard

Every number needs a producer, a runlog, and an enumeration. **A retraction is a claim** and
needs the provenance of what it retracts. **State the null:** for any claim that *the model
does X*, ask what the architecture does with no model in it — for a rank product, monotone
ordering or normalised share, zero is not the null. That is what killed C16.

And the one that outranks all of them: **a contract no consumer is obliged to read is
decoration.** The `sampling_weight` warning was machine-written, attached to the data, and
stating a precondition — and it failed because nothing read it.
