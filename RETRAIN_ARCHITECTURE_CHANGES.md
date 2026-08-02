# Architecture and encoding changes to make at the next retrain

**READ THIS BEFORE RETRAINING OR RE-ARCHITECTING. It is the accumulated list of things
the current design does that we found by interpreting it, and that a retrain should fix
rather than reproduce.**

*Started 2026-08-02 by the interpretability window, at Pete's request, during the ORF
scanner investigation. Living document — add to it whenever interpretation turns up
something the architecture or the encoding did to us.*

Each item states **what it is**, **how we know**, and **what a retrain should do**. Items
are marked with how much confidence stands behind them, because some are measured and
some are architectural reasoning.

---

## 1. The bin-max representation discards multiplicity, spacing and order

**What it is.** `WindowEncoder` takes a maximum within each of B bins and hands a linear
layer `conv_channels × B` numbers — at `c32_b8`, **256 numbers to summarise a
1000-position window**. That records *which filters fired strongly and roughly where*. It
does not record how many times a filter fired, how far apart two occurrences were, or the
order of two different filters inside a bin.

**How we know.** `model_v6.py`, `WindowEncoder.forward` — `parts = tensor_split(h, n_bins)`
then `p.amax(dim=-1)` per part. Code read, not inferred.

**Why it matters, and this is reasoning rather than measurement.** A weak motif occurring
ten times is represented identically to one weak occurrence, because only the maximum
survives. If NMD-relevant sequence works through many weak, redundant, distributed
signals — which is what RBP-mediated regulation usually looks like — **this architecture
compresses it into exactly the diffuse compositional smear we spent a week chasing.** It
would explain the 1.73% PWM ceiling, the weak keto/U-rich signal and the absence of any
crisp consensus in one stroke rather than three.

**What a retrain should do.** Keep max-pooling if you like, but add a channel that
survives accumulation — mean or sum pooling alongside the max, or an attention pool.
Raise the bin count so position is retained more finely. The cheap version is to
concatenate max and mean per bin and let the linear layer choose.

---

## 2. Vocabulary is capped at `conv_channels`

**What it is.** 32 convolution channels per window encoder is a hard ceiling on how many
distinct sequence features can be learned, even allowing the second convolution to compose
the first's outputs.

**How we know.** `model_v6.py` constructor; checkpoint path `runs/interp_c32_b8_s100`.

**What a retrain should do.** Treat 32 as a hypothesis rather than a default and sweep it.
If the motif story is "many weak features," capacity is the first thing to test.

---

## 3. The ATG window's fill boundary leaks ORF length

**What it is.** The ATG window is filled with `limit_hi = mid` where
`mid = (atg_pos + stop_pos) // 2`, and the window runs 100 nt past the AUG. So
**downstream fill extent = min(100, ORF_length / 2)** — for any ORF shorter than 200 nt,
*where the fill stops encodes ORF length exactly*, with no sequence involved.

**How we know.** `data_prep.py:262-274` and the clip in `encode_window_v5:141-144`. Code
read. Measured consequence: capture correlates with ORF length at **+0.760**, and the
association weakens by more than half below/above the 200 nt boundary (job 8899353),
confirming fill is a contributor though not the whole account.

**How much it matters — measured 2026-08-02 after Pete asked whether it applies to enough
of the data to care about.** It does, and the shape is the damaging one:

      share of candidates with ORF length < 200 nt (fill encodes length)
        all candidates              69.1%   (553,940 of 802,035)
        REFERENCE CDS candidates     4.5%
        non-reference candidates    71.5%
      median candidate ORF length    81 nt

**Real ORFs are long and spurious ones are short, so the fill boundary separates them
almost perfectly.** A reference candidate has its fill pinned at 100; a spurious upstream
ORF has its fill boundary sitting at length/2, well inside the window. Working the odds:
`P(reference | fill saturated)` = **11.1%** against **0.23%** when it is not — **fill
saturation alone is a ~47× odds marker for "this is the real ORF,"** available from
geometry with no sequence read.

That is the discriminative axis the initiation head exists to compute, handed to it for
free by our encoding. It plausibly accounts for capture's +0.760 correlation with ORF
length, and for a one-line longest-ORF heuristic reproducing 97% of the model's selection
accuracy. **Not yet established that the head uses it** — that needs the tiled-perturbation
test, which would localise sensitivity to the fill boundary or away from it.

**Why it matters.** The initiation head is supposed to model where a ribosome starts. It
is reading a geometric artifact of our own encoding that happens to correlate with the
answer. **This is the third leak of this class** — `SEQUENCE_ENRICHMENT_APPROACH.md` §5.3
records two earlier ones and notes that none was visible to ablation.

**What a retrain should do.** Do not let fill extent carry information. Either pad every
window to a fixed extent regardless of ORF length, or clip at a boundary that is not a
function of the ORF's own coordinates, or supply a fill mask explicitly as its own channel
so the model cannot use it covertly.

---

## 4. The forward separation between the heads is given back by the loss

**What it is.** Three separate encoders, with a test asserting capture is invariant to the
stop window, and a comment saying that separation *"is what licenses reading `p_k` as
initiation rather than as 'which ORF is real'."* But `P(NMD) = Σ p_k · d_k` with the loss
on the product, so `∂L/∂p_k ∝ d_k` — **the gradient reaching the capture head is scaled by
the decay probability**, and under stick-breaking candidate *j*'s logit carries `d_k` for
every candidate downstream of it.

**How we know.** `model_v6.py:122-129` (three encoders), `:199-200` (the product),
`train_v6.py:358` (BCE on the transcript logit), `:194` (exclusive cumsum). Pete's
observation; verified independently by both windows.

**Why it matters.** The capture head cannot *observe* termination and is *trained to
predict it*. So `p_capture` cannot be read as initiation, which is the exact thing the
architectural separation was built to license. **The architecture buys a property that the
objective then spends.**

**What a retrain should do.** If `p_k` is to be readable as initiation, the training has to
protect that — an auxiliary loss on an initiation target, a stop-gradient on the decay
path into `p`, or a two-stage schedule. Otherwise drop the claim and describe the head as
what it is.

---

## 5. Reading-frame channels are supplied across the entire window, including the UTR

**What it is.** Channels 6–8 are codon phase relative to the candidate's own AUG, written
to **every filled position** — including all 900 upstream UTR positions.

**How we know.** `data_prep.py:207-211`, `genomic_positions = arange(w_start, w_end)`,
phase written throughout.

**Why it matters.** A periodic 3-cycle grid is handed to the model across sequence that has
no reading frame. Prior work reported ORF periodicity appearing in the 5′UTR; that is most
plausibly the encoding rather than the biology. Not purely artifact — the same grid enables
legitimate in-frame reasoning about upstream AUGs and stops — but **attribution goes to the
channel before it goes to the sequence.**

**What a retrain should do.** Decide deliberately. Either restrict phase to in-ORF
positions, or add a channel marking where phase is meaningful, so "in frame with the AUG"
and "upstream of the AUG entirely" are distinguishable.

---

## 6. Every candidate has a stop codon and a start codon by construction

**What it is.** The candidate pool is enumerated with an AUG at every start anchor and a
stop at every stop anchor. So during training there was **no negative example** for either
landmark and no gradient from which to learn to check them.

**How we know.** Measured for the decay branch — stop-codon bases sit at the 75th
percentile, indistinguishable from control positions 25–30 bases away, while the +4
composition bias is real and structured in the data. Already on record for the capture head
and the start codon.

**What a retrain should do.** Admit candidates whose stop codon is absent or disrupted, so
the question becomes askable. Until then, "the model does not read the stop codon" is a
statement about our pool rather than about the model.

---

## 7. Window geometry is hardcoded and cannot be swept

**What it is.** `ATG_LEFT, ATG_RIGHT = 900, 100` and `STOP_LEFT, STOP_RIGHT = 500, 500`
are module constants in `build_tensor.py:47-48`, not read from config, and the builder's
argparse exposes no window options.

**Why it matters.** Any experiment that needs to vary window geometry — including testing
whether item 3's leak is causal — requires editing source and rebuilding tensors. We
designed such a test today and had to abandon it for this reason.

**And the existing sweep checkpoints cannot substitute**, which was established by
parameter count rather than by naming: the sweep checkpoints are 453,130 B against v5
production's 453,242 B and the v6 interpretability checkpoint's **281,978 B**. They are a
different *architecture*, not a different window on the same model, so comparing across
them varies builder, candidate set and architecture at once. Two windows reached that
conclusion independently — one from the absent tensors and the argparse surface, one from
the constants — and **neither inference was evidence until someone weighed the files.**

**What a retrain should do.** Parameterise them, record them in the tensor attrs (which it
already does), and keep at least one alternative-geometry tensor so leak questions are
answerable without a rebuild.

---

## What is NOT on this list, and why

**Junction positions being supplied (channel 4) is not a defect.** It is a design choice
and a reasonable one. It only means "the model uses junction position" is not a finding —
we gave it that. Worth knowing when writing claims, not worth changing.

---

## How this document is meant to reach you

If you are reading this because a pointer sent you here, the pointer worked. The pointers
live in `03_train.py`, `train_v6.py`, `CLAUDE.md`, and the user's persistent memory index,
because a retrain can be started from any of those directions and a document that is only
findable from one of them will be missed from the other three.

**If you add an item, add it here and nowhere else.** A second copy is how this project
lost two days to one number that differed between two documents.
