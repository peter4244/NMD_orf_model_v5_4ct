# Start here — handoff to the next window

*Written 2026-07-31 by the model-side window. Deliberately written without shorthand.*

## What you are here to do

**Be one of two windows working together to identify the best approach to use sequence-based
modelling to identify sequence elements that influence NMD susceptibility.**

Two windows work in parallel and check each other. You are the **model window** (repo
`~/claude_projects/NMD_orf_model_v5_4ct`, branch `master`). The other is the **manuscript and
reproducibility window** (repo `~/claude_projects/nmd_lung_longread_2026`). Message it with
`mcp__ccd_session_mgmt__send_message`; find its id with `list_sessions` — it is titled
"NMD reproducibility repo".

The checking is not a formality. On 2026-07-31 each window overturned several of the other's
conclusions, including three of this window's predictions in a row. Assume your own results are
wrong until the other window has failed to break them.

## The two documents that matter most

| what | where |
|---|---|
| **The plan** — five cheap experiments, then a retraining plan | `nmd_lung_longread_2026/docs/EXPERIMENT_AND_RETRAIN_PLAN_2026-07-31.md` |
| **The ranked hypotheses** — 143 proposals in four tiers | `nmd_lung_longread_2026/docs/MERGED_HYPOTHESIS_RANKING_2026-07-31.md` |

Supporting, in this repo, all committed:

- `analysis_plans/HYPOTHESES_TRACK_B_2026-07-31.tsv` — this window's 128 raw proposals, plus
  `HYPOTHESES_TRACK_B_README.md` with eight flags to apply before using any of them
- `analysis_plans/DAY_SUMMARY_2026-07-31.md` — one page on what was established
- `analysis_plans/SEQUENCE_DISCOVERY_BRIEF.md` — the biology, the model, the data's limits,
  and a list of methods that are already dead with the reason each died
- `analysis_plans/ANALYSIS5_PLAN_2026-07-31.md` — the start-codon-context analysis
- `analysis_plans/TRACK_A_HANDOFF_2026-07-31.md` — findings for the other window, section 13
  is the current one
- The other window wrote its own handoff at `nmd_lung_longread_2026/docs/TRACK_A_HANDOFF_2026-08-01.md`.
  Read both; they were written to agree and any place they do not is a thing to resolve.

## What is settled, in plain terms

**The model already predicts well and that is no longer the point.** It scores 0.931 for
discrimination on held-out chromosomes. The goal now is to use it to *discover* which pieces of
sequence make a transcript vulnerable to decay.

**Most of what the model uses is not sequence.** Each candidate reading frame carries five numbers
alongside its sequence, and those five carry most of the decision.

**The most important of those five is broken in an interesting way.** It counts exon junctions
occurring after the stop codon — but with no distance rule. The actual biology is that a junction
must be *more than about 50 nucleotides* past the stop to trigger decay. So the model was never
given the real rule.

**And it never learned it either.** Sliding a junction mark from 2 to 300 nucleotides past the stop
changes the model's output by the same amount everywhere. It has learned "a junction is present
downstream", not "a junction is far enough downstream".

**But the rule is real in the data.** Holding the junction count at exactly one, transcripts whose
junction sits more than 50 nucleotides past the stop are NMD-positive **46.8%** of the time; when it
sits closer, **10.8%**. A four-fold difference the model's inputs cannot express. This is the
strongest result anyone produced, and it needs no model at all — two spreadsheets.

**The pattern generalises, but read the conclusion carefully.** Every sequence computation the model
failed to learn corresponds to one of those five numbers that made learning unnecessary: the
junction-distance rule, the start-codon context, which reading frame is the real one, and the
geometry relating frames to each other.

That is a description. The prescription both windows drew from it — withhold all five — was **wrong
for the junction count, and Pete corrected it.** The 50-nucleotide rule is textbook biology, so
under his scoping (supply what is solved, discover what is not) it belongs on the supplied side
alongside splice-site locations and start-codon identity. Keeping it also *helps*: with that chunk
of variance accounted for, what is left over for stop context, upstream-ORF strength and 3'UTR
motifs is cleaner, and the model does not spend limited capacity re-deriving something known.

**What was broken was the feature, not the decision to have one.** Fixing it — so it implements the
distance rule instead of counting any junction at any distance — is the change that matters. One
cheap arm runs without it as a check that it is not masking something unexpected. The other three
numbers are still withheld.

**Start-codon context has a weak but real effect**, and it is strongest in transcripts where the
model is unsure which reading frame is genuine — which is what you would expect if the context is
being used to *choose* the frame rather than to score it.

**The candidate reading frames the model sees are badly chosen.** It keeps 5 of roughly 37 per
transcript, picked by a three-valued score that produces ties, broken by position in the transcript.
Half the relevant upstream frames are invisible to it. Scoring them properly with a published
position-weight matrix removes the ties completely, but does *not* fix the coverage — that needs
more slots, not better ranking.

## What is dead — do not re-propose

Listed with reasons in `SEQUENCE_DISCOVERY_BRIEF.md` section 6. The short version: gradient-based
attribution on this model (its error exceeded the thing it was measuring); shuffling three
nucleotides as a control (it is the identity operation); average-magnitude summaries across
positions; treating agreement between the five trained copies as evidence on its own; and asking
whether junction distance matters *beyond* the 50-nucleotide threshold — that rule is a threshold,
so there is no dose to respond to, and the question is empty even for a perfect model.

## Five traps that cost this project real time

1. **Agents invent runs and numbers.** Two of four summary agents cited experiments that never
   happened, formatted identically to real ones. **Never repeat a number you have not personally
   traced to a log or recomputed.**
2. **A confident negative has repeatedly turned out to be a confounder, not the model.** It happened
   at least three times. If something comes out flat in this area, assume something structural is
   producing it before you believe it.
3. **Check every claim against the code or the data, not against memory.** Numbers written from
   recollection have been wrong repeatedly, including in documents that were already committed.
4. **A statistic can look like signal and be a property of how you measured it.** Two examples: a
   slope fitted over a short range beats one fitted over a long range almost automatically; and a
   comparison that holds something fixed "by construction" is not thereby free of the thing it was
   meant to control.
5. **Do not pre-excuse a result.** This happened twice in one day and both times it stopped the
   looking. The model's flat response to junction distance was explained away as "it was handed the
   answer anyway" — which turned out to be false, and the true reason was more interesting. A weak
   start-codon signal was explained away as an artefact of dead measurements — also false. If you
   catch yourself writing the excuse before the check, do the check.

## Practical facts

- Local Python: `~/miniforge3/envs/nmd_model_local/bin/python`. The default has a broken numeric
  library.
- Trained models: `results_4ct_sweep/` has five copies each at two window widths. There is **no**
  five-copy set at the width both config files point to — check the width you load.
- Transcript sequences are local at
  `~/claude_projects/nmd_deposit_2026/source_data/sqanti/nmd_lungcells_corrected.fasta`.
- `infer_uorf_attention.py` assumes exactly five candidate frames (lines 174, 207) and will
  silently drop the rest if that number changes.
- Ask before logging into the cluster, every time.
