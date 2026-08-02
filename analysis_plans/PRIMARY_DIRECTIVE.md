# Primary directive

*Set by Pete, 2026-08-02. Canonical copy: `NMD_orf_model_v5_4ct`, branch `master`.*

**Applies to the two claim-generating windows — the model window and the
interpretability window.** It does not govern the results-and-figures window, whose
work is coordination, the decision register and the claim ledger, and whose standards
are those artifacts rather than this one. Larry consumes claims produced under this
directive; he is not bound by it when deciding what gets drawn or how the register is
kept.

---

## The directive

**Produce true claims about how the NMD model makes decisions and which elements of
transcript sequence it uses.**

A claim is ours to make when it is **stated at the strength the evidence earns**,
**carries its bound**, and **says whether it is about the model or about the
biology**.

**Bounds and negatives are claims, and usually the durable ones.** "At most this
big," and "the instrument would have seen it and did not," outlast most positives.

---

## Why these three conditions and not others

They are not a checklist bolted on. They are the three ways our claims have actually
failed:

- **overclaimed** — "motif" was doing work the adjustments had not earned, for two
  days, across two windows
- **unbounded** — the composition profile was quoted without the ceiling that limits
  what it can be worth
- **unscoped** — a model finding read as a statement about biology. These come apart:
  the +4 composition bias after a stop codon is real in the data and absent from what
  the model reads

## Why negatives count

Five positive claims were retracted in two days; the negatives held. That is a fact
about the error process — background, denominator and reference-point mistakes all
inflate — so the evidentiary bar is asymmetric by justification rather than by taste.

A negative is only a claim when **paired with demonstrated sensitivity**. The
stop-codon result qualifies: importance is flat across ±6 *and* the +4 bias is real
and structured in the data — instrument capable, model silent. An unpaired negative
does not qualify, because a robustly recognised pattern would produce the same reading
by construction.

A localization is also a claim. When a signal turns out to live in routing rather than
in decay-head reading, that locates the mechanism rather than dissolving it — selection
mass is itself sequence-driven.

---

## Standards

All paths repo-relative in `NMD_orf_model_v5_4ct`, branch **`master`**, the canonical
copy. The `_interp` and `_results` worktrees hold mirrors that go stale between merges.
Anchors are section headings rather than line numbers, which drift.

| what | file | section |
|---|---|---|
| the pre-specification a claim needs before code is written | `analysis_plans/ANALYSIS_SEQUENCING_PROPOSAL.md` | `## The row template — thirteen fields, and an empty one blocks implementation` |
| defined terms; nothing outside this list belongs in a document, legend or commit message | `analysis_plans/SEQUENCE_ENRICHMENT_APPROACH.md` | `## 0. Defined terms` |
| how to choose adjustments, and what each licenses you to say | `analysis_plans/ADJUSTMENT_TOOLBOX.md` | `## What each adjustment licenses you to say` |
| the strength ladder — which sentence a result has actually earned | `analysis_plans/SEQUENCE_ENRICHMENT_APPROACH.md` | `#### The ladder, and where our results actually sit` |
| the standing bound every enrichment claim travels with | `analysis_plans/SEQUENCE_ENRICHMENT_APPROACH.md` | `## 6. What is not settled` — the PWM ceiling, 1.73% of importance variance at width 9, held out |

**Cite these; do not restate them.** Restating a standard in a second place is how the
two copies diverge, and it is the failure mode that produced the A4 contradiction and
the 1.148 figure. If one of them is wrong, fix it where it lives.
