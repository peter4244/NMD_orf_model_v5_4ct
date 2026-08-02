# How the model chooses a reading frame

*Written 2026-08-02 by the model window, reviewed by the interpretability window.
Canonical on `master`. Every claim below carries its producer and job.*

**SCOPE.** Every finding here is **about the model**, or **about our instrument**,
and each is labelled. None is a statement about transcript biology. The quantities
read — `p_capture`, `p_select`, `p_decay`, `vals_capture`, `vals_decay` — are the
network's own outputs; reading them from a cached bank does not make them
observations of biology.

**BANK.** `results_ism_v6/bank_interp_s100.h5`, checkpoint
`runs/interp_c32_b8_s100/best.pt`, 4,999 transcripts, median 15 candidate reading
frames per transcript (mean 17.6, max 103), 62,149 candidates in the
reference-bearing subset.

---

## The finding

**The alignment between where the model routes and what it calls decay-prone is
produced by the stick-breaking ordering, not by the head that does the routing.**

| quantity | median, within transcript | job |
|---|---|---|
| `p_select` ~ `p_decay` | **+0.399** (74.4% of transcripts positive) | 8899905 |
| `p_capture` ~ `p_decay` — the same question, queue removed | **+0.091** | 8899905 |
| alignment gain: actual mixture ÷ independent-factor baseline | **1.292** | 8899905 |
| `p_select` ~ `p_decay`, NMD transcripts | +0.549 | 8899905 |
| `p_select` ~ `p_decay`, control transcripts | +0.170 | 8899905 |

`P(NMD) = Σ_k p_select_k · d_k`. Under independent factors that sum would be about
`(Σ p_select)·(mean d)`. **The measured gain of 1.292 says the model's prediction
runs ~29% above that baseline because the factors align** — so the correlation is
not a description of the model, it is part of what the model computes.

**The mechanism.** Selection is stick-breaking over candidates ordered 5′→3′
(`model_v6.py:7,15-18`), so upstream candidates receive weight by position. Upstream
candidates are short ORFs with more sequence behind them and therefore more
downstream junctions — which are exactly the candidates the decay head scores high.
The ordering routes toward decay-prone frames without any head learning to.

**What it licenses.** The model does select reading frames by premature-stop
status. Not by observing termination — the capture head architecturally cannot
(`model_v6.py:160-164`) — and not by the head having learned to, since with the
queue removed the alignment falls to +0.091. **By the ordering.**

**What it does not license.** That the ordering was designed to do this, or that
the alignment is desirable. And `p_decay` is another head's output, so this is the
model's internal accounting rather than validation against which frame a ribosome
uses.

---

## What the capture head is doing, and it is not initiation

| # | claim | job |
|---|---|---|
| **C2** | The model lands on the annotated start **0.697** of the time, against **0.424** for a most-5′-candidate baseline and **0.055** chance. | 8898926 |
| **C11** | A one-line heuristic — *take the longest candidate* — reaches **0.678**. The head's contribution over it is **1.9 points**. | 8899353 |
| **C3** | **The head's own argmax scores 0.305 — worse than position.** It is not ranking candidates; it gates passage, and its *low* scores upstream are what let the queue reach the main ORF. | 8898926 |
| **C8** | `p_capture` ~ ORF length is **+0.760**. Holding length collapses its association with junction count from −0.460 to **−0.009**. | 8899132 |
| **C6** | `p_capture` ~ downstream junction count is **−0.460**, against **+0.124** for `kozak_score` — roughly four times stronger than the feature it is supposed to use. | 8898926 |

**C3 is architecturally expected rather than a curiosity** (interpretability
window). The loss is BCE on the product (`train_v6.py:8,358`), so `∂L/∂z_p_k` is
scaled by `d_k`, and `log_pass` is exclusive (`model_v6.py:194`) — candidate *j*'s
gradient carries `d_k` for every downstream *k*. **The head is trained to suppress
an upstream candidate when downstream candidates carry decay structure.** Gating is
the shape the gradient pushes toward.

**And C11 measures accuracy, not mechanism.** Two methods can score alike while
being right about different transcripts. C11 bounds what the head *adds* over a
heuristic; it does not establish what the head *does*.

### The sentence this yields

> **The separation the architecture buys in the forward pass is given back by the
> loss.** Three encoders, a stop-window invariance test, and a comment stating that
> all of it exists to license reading `p_k` as initiation — and then a
> decay-weighted gradient trains that head anyway.

---

## The two heads read different bases

Per-position agreement between `vals_capture` and `vals_decay` inside the ATG
window, **within mass band** (job 8899820):

    in ATG window, above both floors      +0.093
    noise vs noise (must be ~0)           +0.019   PASS
    outside any ATG window (capture blind)+0.075   control

The in-window figure sits barely above a control where capture cannot respond at
all. Genuine agreement is on the order of 0.02.

**Reading different bases is not the same as representing independent quantities**,
and the evidence is against independence: capture's *output* tracks downstream
junction count at −0.460, information it cannot see. The two heads read different
regions and arrive at correlated conclusions.

**Instrument limit:** ISM sees only what is fragile to single substitutions, so a
feature both heads encode robustly would be invisible to both profiles and would
not register as agreement. This bounds detectable shared *reading*, not shared
representation.

---

## Findings about our instrument

| # | finding | job / source |
|---|---|---|
| **C10/C12** | The ATG window's downstream fill is clipped at the ORF midpoint (`build_tensor.py:271`, `limit_hi=mid`) and runs 100 nt past the AUG, so **fill = min(100, length/2)** — below 200 nt the fill boundary encodes ORF length exactly. Measured: the length association is **+0.442** below 200 nt and **+0.200** at or above. A contributor, not the whole account. | 8899353 |
| **C9** | The reading-frame channels are written across the **entire** window including all 900 upstream UTR positions (`data_prep.py:207-211`). A periodic 3-cycle grid is supplied throughout the 5′UTR, so **prior observations of "ORF periodicity bleeding into the 5′UTR" are most likely the encoding rather than the sequence.** | code |
| — | **Selection mass confounds any two-branch comparison.** Both `vals` columns scale with mass, so a naive correlation between them measures mass co-scaling. Job 8899766 failed its own sanity check on exactly this: noise-vs-noise agreement +0.294, *higher* than the +0.266 among real positions. Fixed by stratifying on mass, never dividing it out. | 8899766 → 8899820 |
| — | **The capture branch's exclusion from the programme was correct**, and its real reason was never written down. Median `\|vals\|` is 6,861× smaller than decay's while its own floor is only 23× lower, so **capture's signal-to-noise is 296× worse**. 64% of live positions clear their own floor against decay's 99%. | 8899965 |

---

## Bounds, and what is not established

- **The PWM ceiling stands over everything**: a single position weight matrix
  explains **1.73% of importance variance**, held out on disjoint genes.
- **`p_decay` is another head's output.** Every alignment figure is internal
  accounting, not ground truth about which frame is used.
- **One bank, one seed** for the alignment and branch-agreement results.
- **The moving-boundary intervention is unavailable.** Testing the fill-geometry
  account causally requires a model trained at a different window extent.
  `build_tensor.py:47` hardcodes `ATG_LEFT, ATG_RIGHT = 900, 100` with no argparse
  exposure, and the `atg1000`/`atg2000` sweep checkpoints are a **different
  architecture** — 453,130 B against v6's 281,978 B. **⇒ If the deferred retrain
  varies the downstream extent across even two settings, this test comes free.**
  Recorded as a design requirement rather than a dead experiment.
- **The story has no ending.** We can say the scanner gates mostly on ORF length
  and that part of that signal is our own fill geometry. We cannot yet distinguish
  *"a weak initiation model"* from *"substantially reading our own encoding"* —
  the first is about the model, the second about us.

---

## The next measurement, pre-registered

**Tiled scrambling of the ATG window, measuring Δ`z_p`** — the capture head's own
logit (interpretability window). No routing confound, because `z_p` is
per-candidate and computed before the queue. No region calling, because tiles are
pre-specified and therefore need no null to establish they exist. Tile width set by
the encoder — receptive field ~42, bins of 1000/8 = 125 — bin width primary,
half-bin as the finer arm.

> **PREDICTION, REGISTERED BEFORE THE RUN (model window, derived from job 8899965).**
> Capture is **sparse, not weak**: its 10th and 25th percentiles sit at zero in
> floor units while its 75th and 90th (9.7e7, 1.25e9) *exceed* decay's (6.9e7,
> 1.6e8). If that sparsity reflects initiation biology, tiling should show **sharp
> concentration at the start-codon neighbourhood and near-zero response across most
> of the upstream 900.**
>
> Concentration at the start codon → a weak initiation model.
> Concentration at the **fill boundary** → the head is reading our encoding.
> Neither → the sparsity has some third source and this account is wrong.

**What it will not do:** rescue the motif question, say anything about decay, or —
if it confirms concentration at the start codon — produce a new finding rather than
a confirmation. It answers one thing.

---

## Errors made and corrected, kept because they shaped the result

1. **Position proposed as the route for C6.** Refuted — holding position
   *strengthened* the association (−0.582 against −0.460).
2. **A band-split prediction that came out backwards** because the split tested
   where ORF length still had range rather than the mechanism claimed for it.
3. **C11 written as a mechanism claim** when it is an accuracy comparison.
4. **`model.py` and `data_prep.py` cited for architecture** — both are the
   superseded v5 generation. C10 survived only because both builders implement the
   same clip; that was luck, not method.
5. **The capture-branch re-plan**, proposed on a floor comparison that was the
   wrong statistic and withdrawn when the signal-to-noise ratio was measured.
6. **Treating a mechanism as a nuisance, twice, by both windows** — routing, then
   ORF length. Pete caught both; the machinery caught neither. The length partials
   in particular condition away the fill mechanism *by construction*, since fill =
   length/2 deterministically below 200 nt, so C8's −0.009 supports "length is the
   route" and "we conditioned away the only route" equally.
