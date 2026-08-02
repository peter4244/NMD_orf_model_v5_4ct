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
**carries its bound**, and **says what it is about — the model, the biology, or our
instrument**.

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

**Three scopes, not two.** Much of what we produce is about neither the model nor the
biology but about **our instrument** — that a null destroyed structure the data has
architecturally, that a region caller's criterion runs backwards, that a statistic is
scale-free and so cannot have a low-signal control. These are the most transferable
findings we have and the most likely to be re-derived by whoever comes next if they
are not written down as claims in their own right.

## Why negatives count

**Both kinds are held to the same bar. It is not the same test, because they fail
differently.** A positive has to survive its adjustments and state what it does not
license. A negative has to be **paired with demonstrated sensitivity** — a showing
that the instrument would have detected the thing had it been there.

The stop-codon result qualifies: importance is flat across ±6 *and* the +4 bias is
real and structured in the data. Instrument capable, model silent. An unpaired
negative does not qualify, because a robustly recognised pattern would produce the
same reading by construction.

> **An earlier version of this section claimed the bar should be asymmetric, on the
> grounds that "background, denominator and reference-point mistakes all inflate."
> Withdrawn 2026-08-02 — the interpretability window showed the record contradicts
> it.** At least three of our errors ran the other way: the fold-over-median elevation
> rule, where on the production banks *the null beat the data*; the stop-codon
> off-by-one, whose first pass reported the 24th percentile and "actively suppressed"
> where the corrected reading is "indistinguishable"; and criterion 1 of the region
> caller, whose direction is such that real signal fails it.
>
> Their second argument is the one that settles it: **"the negatives held" may be
> differential scrutiny rather than differential reliability** — positives got looked
> at harder because they were the interesting ones. By this directive's own standard
> the withdrawn sentence was an *unpaired negative*: nothing demonstrated we would
> have caught a wrong negative. We now know we had one, and it took the production
> banks to expose it.

A localization is also a claim. When a signal turns out to live in routing rather than
in decay-head reading, that locates the mechanism rather than dissolving it — selection
mass is itself sequence-driven. It has its own content and its own ways of being
wrong, and it is held to the bar above like anything else.

**A scheduling consequence, since it follows and is easy to miss.** With the
stop-codon control retired, SpliceAI/GT-AG is the only positive control this project
has. So **until C1 runs, no Phase C result is a claim in either direction** — not
merely no negatives. A pipeline that cannot recover a known motif produces
unreadable positives too.

## When the ledger and this directive disagree

Two definitions of "claim" are in force and they can come apart. The register's
threshold is procedural — a result becomes a claim the moment it is written into a
document another window reads. This directive's is evidentiary. A result can cross the
first without meeting the second, which is exactly what the 1.148 figure did.

**The register records it; the directive governs whether we assert it.** Such a result
enters the ledger flagged as needing repair rather than being barred from it —
barring it would leave unrecorded numbers circulating, which is the worse failure and
the one that let 1.148 live for two days.

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
