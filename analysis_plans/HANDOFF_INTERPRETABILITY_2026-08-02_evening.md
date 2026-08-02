# Handoff — interpretability window, evening of 2026-08-02

*Written at the close of the day, for whoever picks this window up next. Supersedes
`HANDOFF_INTERPRETABILITY_2026-08-02.md`, which is the morning version and is now
mostly spent.*

**Read this file, then `NARRATIVE_HOW_THE_MODEL_DECIDES.md` on master. Nothing else is
required to start.**

---

## FIRST, BEFORE YOU READ ANY DOCUMENT

```sh
git merge master && git diff --name-only HEAD...master
```

**Four stale reads happened on 2026-08-02 and not one was caught by the reader.** A
worktree gives no signal that it is behind. Two of the four produced adversarial reviews
of retracted content, and one had me reporting a fixed item as broken. There is a tool —
`tools/check_stale_reads.py` on the `results` branch — but **as of tonight it inspects
its own worktree rather than the caller's**, so it reports all-clear from anywhere. The
one-line fix is with its owner. Until it lands, the command above is the check.

---

## THE STATE IN ONE PARAGRAPH

Pete's founding question was whether the model picks ORFs because they carry a premature
stop. **The answer is no** — `p_select ~ n_downstream_ejc` is −0.050 marginally and −0.067
holding length and position jointly, and the two single partials (+0.452, −0.543) are each
the other confounder leaking. The junction preference enters at the decay multiplication,
not at selection. This is settled with two independent routes and committed producers
behind both. **A three-window conference on narrative process closed tonight**; the story
is agreed by all three windows and sits with Pete, who decides when it reaches the guardian
for candidate review under D60.

---

## ⇒ THE ONE THING I WOULD DO FIRST, AND WHY IT IS NOT THE OBVIOUS ONE

**Follow up the initiation-context finding. It is the only positive result of the day and
nobody has touched it since 15:45.**

Perturbing the 25 bases immediately 5′ of the start codon moves the initiation head's logit
more than perturbing anything else in its 900-base upstream window — **0.413 / 0.283 /
0.241 / 0.275** across four ORF-length bands, **at the same position in every band**, in a
tile that is **99.6% filled in all of them** so there is no fill confound. That is where
Kozak context sits. *Jobs 8900209, 8900420; `FINDINGS_TILED_PERTURBATION_2026-08-02.md`.*

**Why it matters more than it looks.** It is the only result today that got *stronger*
under scrutiny instead of dissolving. Everything else we chased — routing, junction
structure, the whole C14/C16 sequence — turned out to be the architecture describing
itself: the queue produces the correlation, the fill boundary encodes length, position
confounds all of it. True, and all about the instrument.

**The honest pattern from today: we chased the loud result and left the real one alone.**
Artifacts are large and clean, which is exactly why they are seductive. The Kozak-position
finding is smaller, quieter, and the only candidate for a statement about biology.

**What it still needs, and none of it is expensive.** The finding is *positional* — the head
responds at −13. It does **not** establish that the head reads Kozak *content*. Those are
different claims and only the first is earned. Distinguishing them is the obvious next
measurement, and the tiling infrastructure already exists (`interp_tiled_perturbation.py`,
self-test in-file, runs in minutes).

**A caveat that must travel with any follow-up:** the encoder's receptive field is ~42
positions, so **nothing below that resolution can be localised** — see
`RETRAIN_ARCHITECTURE_CHANGES.md` item 3 and the receptive-field discussion. A Kozak motif
is ~10 nt. The instrument can say *where* the head responds; it may not be able to say
*what* it reads there. Design around that rather than into it.

---

## WHAT IS SETTLED, AND WHAT KILLED THE ALTERNATIVES

| | |
|---|---|
| **Routing is indifferent to junction structure** | −0.050 marginal, −0.067 holding length+position. Producer `model_c16_retraction.py`, job 8900746 |
| **The head's junction aversion is entirely ORF length** | `p_capture ~ ejc` −0.453 → **−0.009** holding length |
| **The queue produces the apparent junction preference** | queue-only null with no model: **+0.334** raw against the model's −0.050. `interp_queue_geometry_null.py` |
| **The head reads initiation context at −13** | the one positive. Jobs 8900209, 8900420 |
| **Capture's upstream sensitivity is diffuse, not sparse** | 32 consecutive tiles within ~6%. Refuted a registered prediction |

**Three claims reversed under conditioning in one day** (C7, C16, and the length arm).
`CLAUDE.md` now carries the standing hazard: *in this model, any candidate-level correlation
is position until proven otherwise* — with the operational half, **use a partial to explain
why, never to state behaviour.** The within-transcript ORF length ratio is **83.5× median**,
so "holding length" describes a regime the model never occupies.

---

## OPEN, WITH MY HONEST PRIOR ON EACH

**C15 — routing is junction-biased +0.119 in NMD transcripts and −0.189 in controls, a 0.31
gap the model cannot see labels to produce.** All three windows called this the most
interesting open item. **I would temper that.** Three claims reversed under conditioning
today; my prior is that this is a fourth, not a mechanism. Design it *expecting* a
confound — the obvious first cut is whether it survives matching the two groups on 5′UTR
architecture (uORF count, UTR length, candidate count). If it collapses, it is population
composition. Needs an Explorer login.

**No heuristic baseline under the 0.883 GENCODE recovery figure.** Marked `[unclaimed]` in
the narrative and it is a genuine gap — the 0.678 longest-ORF baseline belongs to main-ORF
recovery, a different target. That number currently has no floor under it.

**Whether the head reads the fill boundary or whatever is filled.** Closed to tiling —
downstream tiles are already below the receptive field. Needs a retrain varying the window
extent; `RETRAIN_ARCHITECTURE_CHANGES.md` item 7 says why.

**Motifs. Nothing works.** The region caller's success criterion points the wrong way and
the proposed null fix was refuted by its own self-test.

---

## HOW THIS WINDOW WORKS WITH THE OTHER TWO

Three worktrees, one repo: **master** (model window / storyteller), **interp** (this one),
**results** (guardian). Merge master before reading anything.

**Conference protocol** is in `CLAUDE.md`. Called by Pete on a topic; while open, every
message goes to all members; it ends when the storyteller declares **and** every member has
read the story and either agrees or agrees to stop critiquing. Tonight's closed.

**Escalation to the guardian is Pete's call** (D60). Do not self-escalate and do not withhold.

**Two things I got wrong that are worth not repeating.** I headed messages *"same text to
both"* while tailoring each copy — the header asserted a content identity that did not hold,
and anyone reconstructing who knew what would be misled. Write *"to both; tailored per
recipient."* And I built a duplicate stale-read checker without looking for an existing one,
against the inventory-first rule; the duplication found a real bug in the original, but that
was luck, not method.

**The error I would most want you not to repeat:** I flagged a caveat — that my null was
computed on a different candidate set than the number I was comparing it to — and then
quoted the result as solid anyway. It was wrong by half. **Flagging a caveat is not the same
as propagating it.** Identifying the reason your number might be wrong and then using it
regardless is worse than not having noticed.

**And one failure class from the model window's own record, because it is invisible and you
will hit it.** A scripted edit — `str.replace` on a document — **missed on a leading-space
mismatch and returned its input unchanged**, while a *second* replacement in the same script
did match. So the commit was non-empty, the diff showed `1 insertion, 1 deletion`, the
message described the fix, and **the fix had not happened**. Nothing looked wrong at any
point. They caught it by grepping for the phrase itself.

⇒ **Verify a documented change by grepping for its content, never by the commit message, the
diff size, or the fact that the commit was non-empty.** `CLAUDE.md` already says a commit
message is not evidence of its diff; this is the sharper version — *a non-empty diff is not
evidence that the intended edit landed.*

**Two practical notes, both learned by getting this wrong while writing this file.** Prose
here is hard-wrapped, so a phrase you inserted will often straddle a newline and a plain
`grep "some phrase"` returns nothing for text that is present. Flatten first:
`tr '\n' ' ' < FILE | tr -s ' ' | grep -i "some phrase"`. And — the part that actually
matters — **I committed anyway while my own check was reporting MISSING**, on the reasoning
that I knew what I had written. The check was wrong that time. Next time it will not be, and
the habit of overriding it is the defect, not the grep.

---

## THE STANDARD, WHICH IS THE POINT OF ALL OF IT

Every number needs a producer, a runlog, and an enumeration. **A retraction is a claim and
needs the same provenance as the thing it retracts** — that rule exists because a retraction
was adopted on prose today while the claim it killed had a script, a runlog, a job id and
predictions registered before the run.

**And state the null.** For any claim that *the model does X*, ask what the architecture does
with **no model in it**. For anything built from a rank product, a monotone ordering or a
normalised share, **zero is not the null**. That is what killed C16, and it is the most
reusable thing this window produced.

---

## THE LAST THING, AND IT IS THE ONE I AM LEAST SURE WE WILL ACT ON

**Nothing was caught by a check on 2026-08-02.** Not one thing. Two dead sentences in prose,
four stale reads, a retraction adopted without a producer, and a silent no-op edit — every
one caught by a person reading. Even the tool built specifically to catch stale reads was
itself broken in a way its own self-test could not see, because a test that lives beside the
thing it tests inherits its assumptions.

A great deal of machinery was built today: decisions D59–D62, a conference protocol, a marker
system, three narrative rules. **The entire catch record belongs to people reading carefully.**

The one piece of process that did fire — the storyteller naming her own most-retractable
sentence and then testing it instead of hedging — worked because **it is not a check. It is a
prompt to think, aimed at the author, at the moment they are least inclined to look.**

The model window's sharpening of this is better than mine and is the sentence to keep:
**both of the things that caught errors required someone reading with the expectation of
finding something wrong, and that is not automatable.** Weight accordingly before building
the next gate. More prompts of that kind; fewer gates.

### Why the failed safeguards failed — the actionable half

*Pete's framing, 2026-08-02 evening: a tendency to bloat and confabulate that needs
structure to counteract. Kept short deliberately; a long section here would be its own
counterexample.*

**Every safeguard that did not fire was an annotation on prose** — the `[unclaimed]`
markers, the thirteen-field row template, the enumeration requirement, the claims table,
the `Depends on` block. All existed that day. None fired. A marker was *tidied away* while
its number was still unfiled, and the claims table stalled at C12 while the narrative ran
ahead using numbers past it.

**Every safeguard that worked converted a prose claim back into something that could
fail** — name your most-retractable sentence *and test it*; state the null *and measure
it*; verify a change *by grepping the artifact*.

⇒ **Text that cannot fail does not get written.** Adding prose to discipline prose is the
move that keeps not working, and it is the bloat mechanism too: every incident spawns
another document.

**⇒ AND THAT IS NOT SUFFICIENT — the model window's amendment, which is better than my
rule.** The `sampling_weight` warning is the *strongest* form of text that can fail:
machine-written, attached to the data itself, stating a precondition. **It still failed,
because nothing was obliged to read it.** So: *a contract no consumer must read is
decoration.* The medium was never the problem; the missing enforcement point was.

**Their counter would have fired where mine would not.** A script that reads the bank and
computes a mean must either apply `sampling_weight` or declare in one line that it reports
a subset statistic — **and fail if it does neither.** That triggers on the **shape of the
computation** rather than on whether a number appears in a table, which is why it also
catches the qualitative case. Note what it means for my two counters: **producer-before-prose
would not have caught this** — every one of those numbers had a producer, a runlog, a job id
and registered predictions; the producers agreed with each other and were wrong together.
**Nor would the linter** — every number was in the claims table, so it would have gone green.

**Where the drift lives — I claimed "not one script was wrong that day" and it is FALSE.
Retracted the same evening.** The narration point below stands for the six drifts it was
drawn from; **the generalisation to the scripts does not.**

⇒ **The bank is a stratified subset and almost nothing reweights it.** `build_ism_bank.py`
writes the warning into the h5 as an attribute — *scarce mechanism cells taken whole,
abundant ones sampled, so any population estimate must be reweighted by `sampling_weight`.*
**Thirty scripts read that bank; two read `sampling_weight`, neither a recovery producer.**
Verified independently against `results_ism_v6/ism_subset.tsv`: five distinct weights
summing to **41,765** over a bank of **4,999**, and annotated transcripts over-represented
about **8×** — while recovery is scored *against the annotation*. So 0.883, 0.793 and 0.697
are correct **bank** statistics quoted all day as **population** statistics. Nothing is
retracted because nothing has been recomputed; the model window holds this.

**How exposed is a given number? In proportion to how much its quantity depends on the
stratifying variables** — `is_nmd`, `has_annotation`, `main_orf_stop`. Recovery is maximally
exposed. The queue null is not: measured three ways it moves 0.016 — full pool **+0.500**,
unweighted subset **+0.501**, weighted subset **+0.485** — because it is within-transcript
queue geometry and the strata are label and annotation axes. **Use that to triage what needs
recomputing rather than recomputing everything.**

**My error, in the exact class this file warns about.** I had six observations of narration
drift, generalised to a claim about every script, and never measured the middle. **That is
A → C with B unmeasured — the rule this window filed — applied to my own summary.**

*What survives:* **prose is the only artifact that never gets recomputed.** A script fails
when it is wrong; a number gets re-derived; a sentence sits there being repeated with
growing authority. That is true of the six drifts. It is not the whole disease.

**Two counters, cheapest first.**

1. **Producer before prose.** A claim enters a document only after its producer exists,
   never the reverse. Every drift that day was written in the opposite order.
2. **Split state from history.** These documents bloat because one artifact does both
   jobs. Git stores history perfectly and nobody has to read it. Keep one short document
   for what is true now, every sentence pointing at a producer.

A linter extracting each number from the narrative and failing when it is absent from the
claims table would have caught three of the six mechanically. **It catches nothing
qualitative** — the two dead sentences contained no numbers — so it supplements reading
and does not replace it.
