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

## THE BENCHMARK WAS WRONG, AND FIXING IT REVERSES THE HEADLINE

*Pete, 2026-08-02. C2/C3/C11 score on recovering the ANNOTATED reference CDS. For
an NMD substrate that is close to a **disjoint** concept from finding the
decay-causing ORF. **ATF4 is the proof:** the posterior flips 18× onto the 179-nt
uORF, puts 93% of the signal there, reproduces the textbook mechanism — and on the
old benchmark scores as a **miss**. A benchmark on which the right answer is wrong
is not a benchmark.*

Rebuilt on transcripts where the annotated CDS is **itself PTC-bearing** — its own
stop has a junction downstream — so recovery and decay-causation coincide. No
biotype column exists in our tables; the set is identified structurally. Job 8900407.

**Anchored on the GENCODE annotation:**

| stratum | n | prior | posterior |
|---|---|---|---|
| **PTC-bearing anchor, NMD** | 1,020 | **0.821** | **0.885** |
| non-PTC anchor, NMD | 435 | 0.710 | 0.462 |
| PTC-bearing anchor, control | 458 | 0.587 | 0.703 |
| non-PTC anchor, control | 1,022 | 0.884 | 0.604 |

> **⚠ CORRECTED. The first version of this table read 0.908 / 0.941 and was wrong.**
> `cand_is_gencode_start` is int8 with values `[-1, 0, 1]`, where **−1 is a sentinel
> meaning "no GENCODE CDS" — and `astype(bool)` returns True for −1.** The anchor
> therefore selected 40,024 candidates instead of 2,949 and 4,916 transcripts
> instead of 2,935, so most "GENCODE anchors" were an arbitrary sentinel candidate.
> Caught by the interpretability window, who gave an arithmetic ceiling I could not
> reconcile — only 16,655 transcripts have a GENCODE CDS at all — and an independent
> cross-check (2,949 equals the count of annotated bank transcripts). **I had told
> them their numbers were wrong. Mine were, by the mechanism they suspected.**
>
> **`cand_is_ref_cds` carries no sentinel** — values `[0, 1]` — so every claim
> anchored on it is unaffected, and the pool-reference rows are identical across
> both runs. The contamination was confined to the GENCODE arm.

**When recovery means decay-causation the model is right 89% of the time**, against
the 0.697 headline. That headline was averaging two populations the model handles
completely differently.

**And the posterior is a decay-seeking correction, visible in its sign.** It
*improves* recovery where the answer is decay-causing (+0.064) and *degrades* it
where it is not (−0.248 in NMD, −0.280 in control). The decay term pulls weight
toward junction-bearing candidates — right when that is the answer, wrong
otherwise. **This is the backwards-reasoning account showing up in the benchmark
itself**, which is the BETTER outcome registered before the run.

**The anchor choice was the largest lever in the measurement**: 0.753 on the pool
reference against 0.821 on the GENCODE annotation. Both were run because an
unexamined choice between them is this project's error class — and measured, the
two anchors agree on only **40.2%** of candidates. "The reference ORF" and "the
GENCODE CDS" are different objects in six cases out of ten.

**Population, CORRECTED.** My earlier figures of 40,914 and 25,982 came from the
same `astype(bool)` defect applied to the TSV, where `is_gencode_start` is float
with `NaN` — and **`NaN` also casts to True**. The true counts: **16,655**
transcripts have a GENCODE CDS at all, **15,526** have a candidate sitting at it,
and in the bank **2,949** do. Both of my population claims were wrong and the
interpretability window's were right.

### Flag-column audit — which columns are safe to cast

*Run after the retraction, across every flag column in the bank, because one
instance of a defect is not evidence that there is only one.*

| column | values | verdict |
|---|---|---|
| `valid` | native **bool** | safe — `astype(bool)` is an identity cast |
| `cand_is_ref_cds` | `[0, 1]` | safe |
| `cand_upstream_of_ref` | `[0, 1]` | safe |
| `cand_has_gencode_cds` | `[0, 1]` | safe — but it is a **transcript-level** flag broadcast to every candidate row, so counting it per-candidate inflates by the candidate count |
| `labels` | `[0, 1]` | safe |
| `obs` | `[-1, 0..3]` | **−1 sentinel**; every use in this project is `obs >= 0`, which is correct |
| **`cand_is_gencode_start`** | **`[-1, 0, 1]`** | **UNSAFE** — the retracted defect |
| `cand_overlaps_gencode_start` | `[-1, 0, 1]` | **UNSAFE** — never used here |
| `cand_upstream_of_gencode` | `[-1, 0, 1]` | **UNSAFE** — never used here |

**All three unsafe columns are the GENCODE family**, and only one was ever touched.
Every other result in this document is unaffected by the retraction: the gate, the
factor-alignment finding, branch agreement, the branch SNR verdict, Kozak-versus-length
and the ATF4 case study all stand.

*The lesson, recorded because it recurred twice in one hour:* **`astype(bool)` is
never the right cast on a flag column here.** These columns use `-1` (h5) and `NaN`
(TSV) as "not applicable", and both are truthy. Use `== 1`. The same defect produced
a wrong benchmark and two wrong population figures, and neither was caught by any
check I ran — it was caught by an arithmetic ceiling someone else supplied.

### ⇒ Repeated against GENCODE's own NMD biotype, and the circularity did not bite

*Pete asked whether we had used GENCODE-encoded NMD as the gold standard. We had
not — the target above is a structural proxy built with the EJC rule, which is
circular against a benchmark that partly tests whether the model uses that rule.*

GENCODE `transcript_type` is a curated call made independently of anything we
compute. Mapped from `gencode.v49.primary_assembly.annotation.gtf.gz`: **2,645 of
4,999 bank transcripts carry GENCODE IDs and all 2,645 matched**, giving **1,125
`nonsense_mediated_decay`** and **1,332 `protein_coding`**. Job 8900631.

| GENCODE biotype | n | prior | posterior | posterior − prior |
|---|---|---|---|---|
| **`nonsense_mediated_decay`** | 1,099 | 0.793 | **0.883** | **+0.090** |
| `protein_coding` | 1,285 | 0.844 | 0.591 | **−0.253** |

**The decay-seeking correction is confirmed with no EJC rule in the target
definition.** The posterior helps where the annotated frame is decay-causing and
hurts where it is not, exactly as registered.

**And the structural proxy was very good:** precision 0.950, **recall 1.000**,
agreement 0.976 against the curated biotype. All 1,099 NMD-biotype transcripts were
captured; the only errors are 58 `protein_coding` transcripts wrongly included. The
proxy result (0.821 / 0.885) and the gold-standard result (0.793 / 0.883) agree
closely, so the earlier measurement is **validated rather than replaced**.

*One asymmetry worth keeping:* the two definitions of "NMD substrate" are not the
same object. Among GENCODE NMD-biotype transcripts our expression-derived label
calls 916 NMD and 209 control; among `protein_coding` it calls 414 NMD. Curation
and differential expression disagree on roughly a fifth of cases, and neither is
wrong — they answer different questions.

> **⇒ C2, C3 and C11 are hereby scoped to MAIN-ORF RECOVERY.** "A length heuristic
> reproduces 97%" is a statement about finding the main ORF and says nothing about
> finding the ORF that causes decay. On that second question, which is the one that
> matters, the model scores 0.941 and no heuristic baseline has been measured
> against it.

*A defect in my own test, recorded:* the docstring registered a BETTER branch and
the code implemented only the ≥0.60 MEANINGFUL threshold, so the script printed the
coarser verdict. The registered rule governs; the implementation was less precise
than the pre-specification.

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

### What C10's magnitude does and does not license

It plausibly accounts for **both** `capture ~ ORF length` at +0.760 **and** the
longest-ORF heuristic reproducing 97% of selection accuracy — the encoding hands the
head most of that heuristic for free, before any sequence is read.

**It is NOT established that the head uses it** (interpretability window's own
caution, and it is the right one). The information is present and highly
discriminative; whether `p_capture` reads it is a separate question and this
measurement cannot answer it. **That distinction is the difference between a claim
about our encoding and a claim about the model**, and the two must not be merged.

*One precision note:* 47.1× is the **probability** ratio. The odds ratio is 52.8×.
Both are computed here; the document uses the probability ratio throughout.

**Instrument limit:** ISM sees only what is fragile to single substitutions, so a
feature both heads encode robustly would be invisible to both profiles and would
not register as agreement. This bounds detectable shared *reading*, not shared
representation.

---

## Findings about our instrument

| # | finding | job / source |
|---|---|---|
| **C10/C12** | The ATG window's downstream fill is clipped at the ORF midpoint (`build_tensor.py:271`, `limit_hi=mid`) and runs 100 nt past the AUG, so **fill = min(100, length/2)** — below 200 nt the fill boundary encodes ORF length exactly. Measured: the length association is **+0.442** below 200 nt and **+0.200** at or above. A contributor, not the whole account. | 8899353 |
| **C10 magnitude** | **The fill boundary is a 47× marker for "this is the real ORF", available from geometry with no sequence read.** Verified independently on `results_pool_v6/orf_pool.tsv`, 802,035 candidates over 42,043 transcripts: median candidate ORF length **81 nt**; **69.1%** of candidates fall below 200 nt where fill encodes length exactly — but only **4.5%** of reference-CDS candidates do, against **71.5%** of non-reference. So `P(reference \| fill saturated) = 11.1%` against `P(reference \| not saturated) = 0.24%`. **The encoding separates real from spurious candidates almost exactly along the axis the initiation head exists to compute.** | interpretability window; verified on the pool |
| **C9** | The reading-frame channels are written across the **entire** window including all 900 upstream UTR positions (`data_prep.py:207-211`). A periodic 3-cycle grid is supplied throughout the 5′UTR, so **prior observations of "ORF periodicity bleeding into the 5′UTR" are most likely the encoding rather than the sequence.** | code |
| — | **Selection mass confounds any two-branch comparison.** Both `vals` columns scale with mass, so a naive correlation between them measures mass co-scaling. Job 8899766 failed its own sanity check on exactly this: noise-vs-noise agreement +0.294, *higher* than the +0.266 among real positions. Fixed by stratifying on mass, never dividing it out. | 8899766 → 8899820 |
| — | **The capture branch's exclusion from the programme was correct**, and its real reason was never written down. Median `\|vals\|` is 6,861× smaller than decay's while its own floor is only 23× lower, so **capture's signal-to-noise is 296× worse**. 64% of live positions clear their own floor against decay's 99%. | 8899965 |

---

## The head fails where initiation biology says context decides

*Pete, 2026-08-02: "it is kind of interesting that NMD does not arise from these
many short ORFs." The pool is 69.1% ORFs under 200 nt, median 81. Background
biology: most produce no decay because ribosomes do not USE them — leaky scanning
past weak start codons, and reinitiation after a short uORF. Suppressing that mass
is exactly what `p_capture` exists to do.*

Predictions registered before the run; outcome **A**, length-driven (job 8900114):

| | accuracy | n |
|---|---|---|
| reference ORF **long** (≥200 nt) | 0.735 | 3,088 |
| reference ORF **short** (<200 nt) | **0.276** | 294 |
| NMD transcripts only, short reference | 0.294 | 153 |

    capture ~ kozak,  ORFs < 200 nt    +0.061
    capture ~ length, ORFs < 200 nt    +0.429      length beats kozak 7x

**The head does not read initiation context where biology says context decides.**
A strong Kozak rescues a short reference ORF only weakly — 0.311 against 0.193.

### ATF4: right answer, right reasons, wrong mechanism

Both ATF4 transcripts in the bank are NMD-labelled. For `ENST00000674920.3`:

    ranked by PRIOR      1055-nt main ORF   p_sel 0.4765   d 0.011   -> 0.0053
    ranked by POSTERIOR   179-nt uORF       p_sel 0.4742   d 0.203   -> 0.0964

**The prior picks the main ORF; the posterior flips to the uORF by 18×**, and 93%
of the NMD signal lands there. That is the textbook ATF4 mechanism, recovered.

The *reasons* are genuine — an upstream ORF terminating with a junction behind it
is why ATF4 is a substrate. **The mechanism is not initiation modelling.**

> **CORRECTED.** This section first said the uORF survives because it is *upstream*
> and stick-breaking hands it queue priority. **Measured, that explains 22.8% of
> cases.** Among short candidates the head's top-scoring one is the 5′-most only
> 22.8% of the time — but carries a downstream junction **76.3%** of the time, and
> `capture ~ p_decay` among short candidates is **+0.362**. The uORF carries its
> weight because it is **decay-relevant**, and the *initiation* head has learned to
> detect that from the start window alone.

### The head reasons backwards, and it is visible in the weights

*Pete's framing: the model is searching for a reason ATF4 decays. That cannot
happen at inference — the label is not available — but it is exactly what the loss
does during training, since `∂L/∂z_p_k` is scaled by `d_k`. The search happened in
gradient descent and is frozen into the weights.*

    among SHORT candidates, within transcript          (job 8900229 producer)
      capture ~ p_decay                       +0.362
      capture ~ downstream junction count     +0.100    (vs -0.46 in aggregate)
      top-capture short candidate has an EJC   76.3%
      top-capture short candidate is 5'-most   22.8%

**The aggregate sign was masking this.** `capture ~ junction count` is −0.46 over
all candidates, mediated by length; within the class where the head must actually
choose among uORFs it is **+0.100**, and it predicts `d` at +0.362.

**This sharpens the branch-agreement result rather than contradicting it.** The two
heads *do* read different bases (+0.093 within mass band) **and** they compute the
same thing among short candidates. Different windows, converged outputs — the
architecture separated the inputs and the loss re-converged them.

### Is it one detector? UNRESOLVED, and stopped by its own power floor

If both heads are junction detectors, the model should have no channel for NMD that
is not junction-dependent. Registered before the run: ONE-TRACK if the NMD-vs-control
AUC falls below 0.60 among transcripts with no junction-bearing candidate,
UNDERPOWERED below 50 such transcripts.

    has junction-bearing candidate   NMD n=2,438  median P(NMD) 0.662   AUC 0.887
                                 control n=2,357                 0.071
    NO junction-bearing candidate    NMD n=   49  median P(NMD) 0.081   AUC 0.787
                                 control n=  155                 0.033

**n = 49. The floor was 50. Under the registered rule nothing is drawn from this**,
though the eight-fold collapse in what the model would actually *call* points the
way Pete's hypothesis predicts.

*Cross-check passed:* `Σ p_select·d` against `sigmoid(base_logit)` agree at
r = 1.0000, max difference 3.35e-07 — so `Σ p·d` is the model's output and every
framing built on it holds.

*Powering it is not a re-read.* The five banks are five **seeds** over the same
4,999 transcripts, so pooling them adds model replicates and not transcripts.
Resolving this needs forward passes over the full 42,043-transcript pool.



*Caveat against overstating it:* `P(NMD)` is 0.107 and 0.104 for the two
transcripts. The model attributes correctly and predicts weakly.

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

**TWO MUTUALLY EXCLUSIVE PREDICTIONS, EACH DERIVED FROM AN INDEPENDENT MEASUREMENT,
BOTH REGISTERED BEFORE ANY TEST.** They cannot both be right, and the tiling
distinguishes them directly. This is an experiment rather than a characterisation.

> **PREDICTION A — model window, from job 8899965 (sparsity).** Capture is *sparse,
> not weak*: its 10th and 25th percentiles sit at zero in floor units while its 75th
> and 90th (9.7e7, 1.25e9) *exceed* decay's (6.9e7, 1.6e8). If that sparsity
> reflects initiation biology, tiling shows **sharp concentration at the
> start-codon neighbourhood** and near-zero response across most of the upstream
> 900. ⇒ *a weak initiation model.*
>
> **PREDICTION B — interpretability window, from the pool measurement (fill).** A
> 47× marker for the real ORF sits at the fill boundary, so a head optimising
> selection accuracy would look there. Tiling shows **concentration at the fill
> boundary**. ⇒ *the head is reading our encoding, and the finding is about us
> rather than about the model.*
>
> **Neither** ⇒ both accounts are wrong and the sparsity has a third source.

**What it will not do:** rescue the motif question, or say anything about decay. It
answers one thing — but that one thing is currently the story's missing ending, and
under two incompatible registered predictions it cannot come back uninformative.

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
