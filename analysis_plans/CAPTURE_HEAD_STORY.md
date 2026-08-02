# What the capture head is actually doing — running story and claim→code map

*Started 2026-08-02, model window. Canonical on `master`. Pete's instruction: keep
this as a coherent story AND a code-to-claims sequence, so it can be relayed to the
results window at the proper time rather than reconstructed then.*

**SCOPE: every claim below is about the model, not about transcript biology**, with
one exception marked BIOLOGY. These are read from model outputs; `vals_decay` and
`p_capture` are the network's own quantities.

---

## The story in one paragraph

A transcript carries a median of 15–17 candidate start codons. The model picks one
by a stick-breaking queue in 5′→3′ order, and the head that scores each candidate
(`p_capture`) is shown only the start window — 900 bases upstream plus 100 into the
ORF — and is architecturally blind to where translation ends and to the junction
count. It nonetheless picks the annotated start well. But its scores track ORF
**length** at +0.76, and every association it has with decay-side structure runs
through length and vanishes when length is held. The leading account is that the
head is doing **gene-finding**, possibly as simply as preferring the longest ORF —
a real and sensible signal, learned from the right window, but not the initiation
biology the head is named for.

---

## Claims, each with its producer

| # | claim | scope | evidence | producer / job |
|---|---|---|---|---|
| C1 | The capture head cannot see the stop window or the structural block; capture is computed from the ATG window alone, and the stop window and structural features (including `n_downstream_ejc`) route to decay only. | model / architecture | `model_v6.py:160-164`; assertion at `:279-285` — but that assertion runs on **random** tensors | code read |
| C2 | The model lands on the annotated start **0.697** of the time, against **0.424** for a most-5′-candidate baseline and **0.055** chance. The scanner beats position alone. | model | 3,412 transcripts with a reference candidate, 62,149 candidates | `model_capture_premise.py`, job 8898926 |
| C3 | **The head's own argmax is worse than position** — 0.305 against 0.424. It is not ranking candidates; it gates passage. Low scores upstream are how the right answer wins, which is what scanning is. | model | same run | job 8898926 |
| C4 | Capture is **not flat**: within-transcript CV median 1.23. Selection is concentrated: median normalised entropy 0.29, median max share 0.76. The stick-breaking prior is not doing the work alone. | model | same run | job 8898926 |
| C5 | Decay **does** discriminate among candidates: `p_decay` CV median 0.77, and decay at the selected candidate is sharply bimodal (deciles 0.008 / 0.033 / 0.171 / 0.807 / 0.975). | model | same run | job 8898939 |
| C6 | Capture's scores track downstream junction count at median **−0.460**, against **+0.124** for `kozak_score` — roughly four times stronger than the feature it is supposed to use. | model | within-transcript rank correlation, n=4,917 | job 8898926 |
| C7 | **Position does not mediate C6.** Holding candidate start position fixed, the partial correlation is **−0.582**, stronger than the raw −0.460. Position was suppressing the association. | model | n=4,526 | job 8898939 |
| C8 | **ORF length fully mediates C6.** `capture ~ ORF length` is **+0.760**; holding length collapses the junction association to **−0.009**. | model | n=4,917 / 4,695 | job 8899132 |
| C9 | The reading-frame channels (6–8) are written across the **entire** window, including all 900 upstream UTR positions — phase relative to a downstream AUG. So a periodic 3-cycle grid is supplied throughout the 5′UTR. | model / instrument | `data_prep.py:207-211`: `genomic_positions = arange(w_start, w_end)`, written to every filled position | code read |

## What follows from C9, and it matters for a prior observation

"ORF periodicity bleeding into the 5′UTR" is most likely **the encoding, not the
sequence.** We hand the model a perfectly periodic phase grid across the whole UTR,
so any composition preference read through those channels looks periodic upstream by
construction. Not pure artifact — the same grid lets the model spot in-frame upstream
stops and AUGs, which is legitimate — but any upstream periodicity claim must be
attributed to the channel before it is attributed to the sequence.

## The open implication (Pete, 2026-08-02) — BIOLOGY, and not yet tested

If the head is a gene-finder **and** the model correctly selects premature-stop ORFs
(which it must, to predict decay), then **NMD-susceptible ORFs must look like real
genes** to a gene-finder. That is not a necessary truth — they could have been
spurious short ORFs — so it is a biology claim derivable from a model claim. First
supporting hint: the model finds the reference ORF in 0.658 of NMD transcripts
against 0.729 of controls, so NMD ORFs are not invisible to it. **Not established.**

---

## What is NOT established, stated so it cannot be quietly assumed

- **Whether "gene-finding" is as simple as "prefer the longest ORF."** C8 is within
  transcript, and the longest candidate is usually the real one, so +0.76 may be
  that heuristic and nothing subtler. **The immediate next check is a longest-ORF
  baseline against C2's 0.697.** Until that runs, "gene-finding" is a family of
  accounts rather than one.
- **Whether the head reads UTR sequence or ORF sequence.** Pete's framing of the key
  determination. `vals_capture` exists in the banks and would answer it directly:
  is capture's sensitivity concentrated in the 900 upstream positions or the 100
  in-ORF ones? Not run.
- **Whether codon phase is used at all.** The interpretability window's proposed
  test: rotate channels 6–8 by one position. Fill mask stays bit-identical, only
  codon alignment changes — a conditioning test rather than an ablation, which
  matters because §5.3 records that both known window leaks were invisible to
  ablation. Not run.
- **C1's assertion has never been run on real inputs.** It is a unit test on random
  tensors. C6–C8 are consistent with the invariance holding, since the whole
  association is mediated, but that is inference rather than measurement.

## Errors made and corrected in this thread, kept because they shaped the result

1. I proposed **position** as the route for C6. Measurement refuted it — the
   association strengthened when position was held (C7).
2. I predicted the junction association would be **strong for ORFs ≤100 nt** and
   weak above, on the reasoning that a short ORF's in-window portion runs past its
   stop. Measured +0.176 and −0.490 — backwards. The split does not test that
   mechanism; it tests where ORF length still has range, so it re-expressed C8
   rather than probing its cause. A badly designed test, not a surprising world.
3. I cited `model.py` for architecture earlier in the day. It is the superseded v5
   file; the current model is `model_v6.py`. Caught by the interpretability window.
