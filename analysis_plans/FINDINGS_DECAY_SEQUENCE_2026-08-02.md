# Two findings on what the decay branch reads from sequence

Written 2026-08-02 by the interpretability window. Both measured on the production ISM banks
(`results_ism_v6/bank_interp_s{100..500}.h5`, 4,999 transcripts, 11,062,149 valid positions per
bank, `chunk_rows` constant at 4,096). Both concern **`vals_decay`** — the change in the
transcript logit through the decay branch alone, with every capture probability held at its
unperturbed value. Neither concerns `vals` or `vals_capture`.

## SCOPE — every finding here is a MODEL claim, not a biology claim

*Required by the primary directive, and flagged as a live hazard 2026-08-02.*

**Everything below is measured on `vals_decay`, which is the model's own sensitivity.** A producer
that reads an ISM bank never loads a checkpoint, so it *looks* like a data analysis — but the bank is
cached model output, and every statistic computed on it is a statement about **what this trained
network is sensitive to**, one step removed. Not about transcript sequence composition.

So: "elevated positions sit at a keto base in a uridine-rich window" means **the positions this model
is most sensitive to** sit at a keto base in a uridine-rich window. It does *not* mean U-rich context
is a feature of NMD-relevant sequence. The biology-shaped reading is the more interesting one, which
is exactly why it is the easy drift.

**The standing proof that the two come apart is in this document.** Finding B: the +4 composition
bias after a stop codon is real and structured **in the data**, and absent from **what the model
reads**. Same positions, opposite answers, depending on which claim is being made.

*(Same shape as the A1 logit cache: the number came out of a `.npz`, the model was nowhere in the
call stack, and it read as data.)*

---

**Where the numbers come from.** Producers are `analysis_plans/analysis_ism_regions.py` and the
probes named per section. Jobs: `8886733` (k-mer, five banks), `8893173` (region-matched and the
GC arms), `8893305` (PWM), plus the composition probes below. The model window reproduced the
central results from an independent implementation; agreements are stated where they exist.

---

## Finding A — the decay branch is most sensitive at keto bases, in a uridine-rich, cytosine-poor window

> **SCOPE: this is a claim about the model, not about transcript biology.** Retitled
> 2026-08-02; it previously read "elevated positions sit at a keto base in a
> uridine-rich, cytosine-poor window," which makes *sequence* the subject and invites
> the reading that U-rich context matters for NMD. It does not say that. `vals_decay`
> **is** the trained network's own sensitivity, so every number below describes what
> this network responds to.
>
> The hazard is specific and was flagged by the results window: a claim built on
> **cached** model output classifies as biology under the obvious test — does the
> producer load weights? — because these probes read ISM banks rather than
> checkpoints. Same shape as the A1 logit cache, where the number came from a `.npz`
> and the model was nowhere in the call stack. The biology-shaped reading is the more
> interesting one and therefore the easier one to drift into.

### What is measured

Take the top 1% of positions **within each transcript** by `max_b |vals_decay|`, and read the
base composition at and around them.
(`analysis_plans/probe_elevated_composition_profile.py`)

    base composition at the elevated position, against all valid positions of the same transcripts

      background   A 0.251   C 0.250   G 0.261   T 0.239      G+C 0.511
      elevated     A 0.210   C 0.213   G 0.299   T 0.278      G+C 0.512

      keto  (G+T)  1.156x        amino (A+C)  0.845x        G+C  1.004x

      4,999 transcripts; 11,062,149 valid positions; 110,631 elevated at top 1%
      within transcript; seed 100; dead positions included; finite mask
      (np.isfinite(vals_decay).sum(1) == 3); pooled by count.
      Producer: analysis_plans/model_a2_enumeration.py, runlog beside it,
      Explorer job 8896445.

> **STRUCK 2026-08-02, on Pete's call.** This table previously read keto **1.148×**, amino 0.851×,
> with background A 0.253 C 0.245 G 0.257 T 0.245 — and cited
> `probe_elevated_composition_profile.py` as its producer. **That script cannot have produced those
> lines:** it prints a background row and a per-offset U/C/A+T/G+C table and nothing else, never an
> elevated A/C/G/T row and never keto or amino. Re-running its exact set definition on the same bank
> returns keto 1.158, not 1.148.
>
> The value is not sensitive to the choice of set. Six definitions spanning 600 against 4,999
> transcripts, ±60 edge trim against none, and dead-in against dead-out backgrounds all return
> **1.155–1.162**. The numbers above are the widest of those and are the ones to quote.

**The enrichment is keto versus amino and it is orthogonal to GC.** G+C is flat to four parts in a
thousand, so there is no GC bias at the sensitive position — and therefore no AU bias either, since
A+T and G+C are complementary.

### The flanks carry a different signature from the centre

    ratio to background, by distance from the elevated position

      offset      U        C       A+T      G+C
        0       1.178    0.850    1.002    0.998        <- keto, GC-neutral
       +/-2     1.25-1.32  0.89-0.92  1.12-1.14  0.87-0.89
       +/-4     1.28-1.41  0.81-0.84  1.15-1.18  0.83-0.85
       +/-6     1.19-1.32  0.76-0.84  1.13-1.20  0.81-0.87
       +/-8     1.15-1.18  0.81-0.89  1.06-1.11  0.89-0.94
       +/-20    1.06       0.92-1.00  1.04-1.06  0.95-0.96
       +/-40    1.02-1.07  0.96-0.98  1.00-1.03  0.97-1.00

**It is uridine-rich, not AU-rich.** Across the flanks U runs 1.18–1.41 while **A is flat at
0.98–1.09**. The A+T enrichment is carried almost entirely by U, so the canonical ARE reading
(AUUUA, A and U together) is not what this is.

### ~~It is not the GC-averaging window~~ — RETIRED 2026-08-02

> **The length-scale argument does not work, and it fails by two independent routes.** It previously
> read: channel 5 is a rolling GC fraction over ±25, so an encoding artifact would plateau to ±25 and
> show a shoulder at the window edge, whereas "the profile decays to baseline within about 8 bases and
> is gone by 20" — wrong length scale for the encoding, right length scale for a sequence feature.
>
> **Route 1 — it reads the wrong column of its own table.** The decay quoted is the **U** profile. The
> encoding channel is a **G+C** channel, and the G+C column of the table directly above runs 0.89–0.94
> at ±8, **0.95–0.96 at ±20** and 0.97–1.00 at ±40. The quantity that would carry an encoding artifact
> is still depleted at ±20 and only returns past ±40 — which is the scale of the channel, not a scale
> that excludes it. "Gone by 20" is false for G+C by the table the sentence cites.
>
> **Route 2 — the track's length scale does not discriminate at all.**
> `SEQUENCE_ENRICHMENT_APPROACH.md` §7.1: raw autocorrelation persists past lag 80, "which the ±25 GC
> window cannot explain either, so the length-scale argument was never doing the work we credited it
> with." Reached from autocorrelation; route 1 reached from composition; same conclusion.
>
> **What this does and does not touch.** The composition profile itself is unaffected — keto 1.156×,
> amino 0.845×, G+C flat — because those are measurements. What dies is the *inference* built on top
> of them. **We currently have no argument excluding the GC encoding channel as a contributor**, and
> the sentence that claimed one has been standing in for that argument.
>
> The nearest thing pointing the other way is that **G+C is flat at the elevated position itself**
> (1.004×), so positions are not selected by their own GC status. That is suggestive and it is not
> decisive: channel 5 is a ±25 *average*, so what would matter is the neighbourhood's GC, and the
> neighbourhood is GC-depleted out to ±20 — a signature the encoding could produce rather than one it
> rules out.
>
> **What would settle it is conditioning, not ablation.** §5.3 records that both window leaks found so
> far were invisible to ablation, because every channel is blank outside the filled region and a "turn
> off feature X" test silently retains the fill mask; both were found by conditioning. So the test is
> to stratify on channel 5's value and ask whether the composition profile survives within strata.
> Note that the one conditioning-style control we had here — the run-length result surviving with the
> GC channel held bitwise constant at 57% of its arity-matched level — was a control *on the retracted
> analysis* and does not transfer.

*Enumeration for the flank table above, per field 13:* it is the output of
`probe_elevated_composition_profile.py`, which reads `bank_interp_s100.h5` alone and loops over the
**first 600 transcripts in bank order** (not a sample), requiring ≥170 positions and trimming 60
positions from each end before selecting. Seed 100. That set is **not** the 4,999-transcript set the
centre table above is now measured over, and the two should not be read as one measurement.

### Controls run, and what each excludes

| control | result | excludes |
|---|---|---|
| **region-matched background** — elevated 3′UTR against *other* 3′UTR positions | 9 of the pooled top-10 5-mers survive inside `downstream`; the model window's independent version retains r = 0.77–0.81 | 3′UTR composition. The enrichment is **within** the 3′UTR, not because of it |
| **region anchor definition** — annotated stop vs max-`p_select` stop | same k-mers, same correlations; only split sizes move (model window, job 8893238) | the boundary choice being load-bearing |
| **across members** — k-mer enrichment vector, five independently trained seeds | mean pairwise r **0.7529**, range 0.7176–0.8111, over all 1,024 5-mers | one initialisation's private solution |
| **clustering underneath** — runs of ≥4 elevated positions | present at every threshold 0.2%–5%; disjoint gene arms give 914 confirmation against 951 discovery; survives GC-preserving scoring at 57% of its arity-matched level | scattered sensitivity; GC smoothing as the cause |
| **PWM, held out on disjoint genes** | r = 0.1316 held out against 0.1316 in-sample, 5.5M confirmation positions | overfitting. The sequence→importance relationship generalizes |

### What is NOT established

- **That this is a motif.** A single PWM at width 9 explains **1.73%** of `vals_decay` importance
  variance at best, and its column-mean profile (T > A > G > C) is identical in seven of nine
  columns — a flat preference with modest positional structure, not a consensus.
- **That any RBP is involved.** Nothing here has been compared to a binding database.
- **That the U-enrichment is a cause rather than a correlate** of the model's sensitivity.

### Corrections made reaching this, all of which moved the answer

Recorded because the final numbers are only interpretable with them.

- **The AU-rich reading was wrong.** GC is flat *at* the elevated position; the AU enrichment is in
  the **flanks**. Reporting composition only at the centre and describing the whole signal from it
  was the error, and Pete's question is what exposed it.
- **The GC confound never existed.** We inferred one because the k-mers *looked* AU-rich and neither
  window measured base composition at the selected positions — one line, and it settles it.
- **The GC-preserving control was itself GC-biased**, 0.682 against 0.501, because an A/T position
  can only be scored A↔T and a C/G position only C↔G. Its k-mer output is discarded, not fixed.
- **The elevation rule was wrong.** Fold-over-median selected 1.7% of positions on a short-transcript
  pilot and 10.7% on the real banks, and the random null *beat* the data. Replaced by a fixed
  top-fraction per transcript.
- **The k-mer depleted tail was truncated** by iterating the foreground, so k-mers never appearing
  under an elevated position were absent — exactly the most depleted end.
- **Directionality was quoted three times before it was right**: 62% of an attainable ceiling →
  ~15% above an analytic null that does not describe this noise → **20.5% above the measured
  in-sample null**, 22% of the range 0.387–0.75
  (`analysis_plans/probe_directionality_insample_null.py`). Each correction made it smaller.
  **And the quantity is now unsettled, so none of these is quotable.** Three magnitudes are in
  circulation across four documents — 20.5% and "22% of the range" here, ~21% in
  `SEQUENCE_ENRICHMENT_APPROACH.md` §6, the interpretability handoff and the modeling handoff. Only
  the figure here cites a producer. 21% may be a rounding of 20.5% rather than an independent
  measurement, and **nobody has checked which**, so it is not established even that this is a
  disagreement. A3 decides whether the quantity survives at all; until it runs, quote no number.

---

## Finding B — the decay branch does not read the stop codon it is anchored on

### What is measured

For each transcript's reference candidate (or its highest-mass candidate where no reference
exists), the percentile rank of each stop-codon base within that transcript's own `vals_decay`
distribution. 600 transcripts, stop codon verified T-first in every one.
(`analysis_plans/probe_stop_codon_control.py`)

    percentile rank within the transcript          in top 1%
      stop base 1 (U)      median 74.9   mean 67.9      2.2%
      stop base 2          median 77.9   mean 71.1      5.7%
      stop base 3          median 76.2   mean 69.3      3.5%
      control, +/-25 and 30 from the stop  74.8  68.6   2.5%

**The stop-codon bases are indistinguishable from control positions 25–30 bases away.** They sit at
the 75th percentile, which is what any position in the stop window does.

### The coordinate system is verified, so this is not a lookup error

The composition profile centered on the same anchor recovers the stop codon exactly:

      offset -2    T 1.000
      offset -1    A 0.465   G 0.535
      offset  0    A 0.793   G 0.207

That is UAA / UAG / UGA. The machinery finds the codon; the model does not respond to it.

### Why — and it is the same structural blindness as capture

**The candidate pool is enumerated before substitution and never recomputed.** Destroying the stop
codon in the sequence leaves the candidate's `orf_end`, window geometry and `n_downstream_ejc`
untouched — the model is never told the ORF now reads through.

And during training **every candidate had a stop codon at its anchor by construction**, so there was
no negative example and no gradient that could teach the model to check. That is precisely the
diagnosis already on record for the capture head and the start codon: *"every candidate has an AUG
by construction, so there is no negative example and no gradient that could teach the model to
check."* **It had not been tested for the decay branch. It holds there too.**

### Consequences

1. **A §5 claim in its own right.** The decay branch's premature-stop signal cannot be coming from
   reading the stop codon; it comes from the structural block (`n_downstream_ejc`) and from window
   geometry.
2. **It makes Finding A more interesting, not less.** The branch ignores the one sequence feature
   its scaffold guaranteed it would see, and responds instead to a U-rich context the pool
   construction did not hand it.
3. **It removes the only internal positive control.** The stop codon cannot validate an
   enrichment method here, because the model genuinely does not use it. **The SpliceAI port is
   the only known-answer test left.**
4. **It is an argument for the same pool fix as capture** — admit candidates whose stop codon is
   absent or disrupted, so the question becomes askable.

### Correction made

The first pass reported the 24th percentile — below average — from sampling `stop_last−3,−2,−1`
where the codon is at `stop_last−2,−1,0`. Correcting the off-by-one moved it to the 75th and
changed the reading from "actively suppressed" to "indistinguishable from its neighbourhood." The
conclusion is unchanged and better supported.
