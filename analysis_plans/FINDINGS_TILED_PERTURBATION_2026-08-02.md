# What the initiation head reads from its window

*Interpretability window, 2026-08-02. Row: `ROW_TILED_PERTURBATION_2026-08-02.md`, with both
predictions registered before either run. Producers `analysis_plans/interp_tiled_perturbation.py`,
jobs **8900209** (12 tiles) and **8900420** (40 tiles), runlogs committed alongside.*

## SCOPE — these are MODEL claims, with one INSTRUMENT claim

Everything here is measured on `z_p`, the initiation head's own per-candidate logit, under
perturbation of its input window. It is a statement about **what this trained head responds to**.

One finding is about **our encoding** rather than the model, and is marked.

---

## Finding A — a localised initiation signal, and it is the clean one

**Perturbing the 25 bases immediately 5′ of the start codon moves the head's logit more than
perturbing anything else in its 900-base upstream window**, and it does so at the **same position**
regardless of ORF length.

    tile immediately 5' of the AUG (offset -13), by ORF-length band

      <80        80-160     160-200     >=200
      0.4132     0.2825      0.2406      0.2748

    the approach to it, <80 band, 25 nt tiles

      offset   -88     -63     -38     -13
              0.0479  0.0489  0.1252  0.4132

**Why this one is trustworthy where the rest of this document is not: that tile is 99.6% filled in
every length band.** It is therefore not subject to the fill confound that blocks Finding B — there
is no band in which the perturbation is silently a no-op.

**And the position is where initiation biology puts it.** Kozak context occupies roughly −6 to +4
around the start codon. This is the first uncontaminated evidence in this thread that the head reads
genuine initiation context rather than a property of our encoding.

### The background it sits on is diffuse, not sparse

**The model window predicted, before the run, that capture sensitivity is *sparse*** — most positions
carrying nothing, a minority carrying everything — and therefore that adjacent 25 nt tiles should be
**highly variable**. Measured (job 8900420, `<80` band, ratio between adjacent tiles):

    far upstream  (-888 .. -113), 32 tiles   median 1.06   IQR [1.02, 1.13]   max 1.50
    near anchor   ( -88 ..  -13),  4 tiles   median 2.56   IQR [1.79, 2.93]   max 3.30

**The prediction fails across the bulk of the 5′UTR and holds next to the anchor.** Thirty-two
consecutive tiles differ by about 6% from their neighbours and never by more than 50%: that is a
smooth, diffuse background, not sparse structure. The sharp behaviour is confined to the last ~90
bases.

So the registered outcome is the third one: **a localised signal on a diffuse background.** The
12-tile run could not have seen this — at 125 nt a tile holding one critical base and a tile with
weak signal spread over 125 give the same median, which is the model window's point and it is
correct.

---

## Finding B — a second component that tracks ORF length, and it is NOT clean

**The head also responds downstream of the start codon at a position that moves with ORF length.**

    peak offset by band, and the fill boundary it would track

      band       n       median ORF   fill boundary   peak    peak is inside by
      <80      5,918          30            15         -13           28
      80-160   1,907         111            56         +12           44
      160-200    457         180            90         +37           53
      >=200    3,718         687           100         +87           13

**Prediction A — a peak at a fixed offset — is refuted.** The peak moves monotonically.

**Prediction B — the head reads the fill boundary — is NOT earned, and the fault is in my design.**
For short ORFs the downstream tiles are **unfilled**, and the self-test confirms an unfilled tile is
exactly a no-op, so the peak has nowhere to sit but near the anchor. **The measurement cannot
separate "reads the fill boundary" from "reads whatever is filled, and the filled region moves with
length."**

*What survives the mechanical explanation:* in the 80–160 band the peak sits at an interior tile with
filled tiles on both sides carrying less. That single band is the whole of the positive evidence, and
at 160–200 the peak (0.3153) and its neighbour (0.3113) differ by 1.3%, which is not an interior
peak.

*What was proposed and withdrawn:* the model window argued the 28–53 offsets are what B predicts,
since a boundary detector peaks about one receptive field inside the boundary. Checked with per-band
median ORF length rather than band midpoints, the offsets **rise** (28 → 44 → 53) where a fixed
receptive-field displacement predicts a constant. Withdrawn by them. **And the instrument cannot
settle it either way**: downstream tiles are 25 nt, already below the ~42-position receptive field,
so no tiling run can resolve constant from rising. That question is closed to this method.

---

## The two components together

| | position | across length bands | fill confound |
|---|---|---|---|
| **initiation-proximal** | fixed, −13 | present in all four | **none** — 99.6% filled everywhere |
| **length-tracking** | moves, +12 → +87 | shifts with ORF length | **unresolved** |

**Their relative size shifts with ORF length.** Fixed dominates for short ORFs (0.413 against 0.220);
length-tracking dominates for long (0.590 against 0.275). So the head does both, and which prevails
depends on the candidate — the "both" row of the row's decision table, which neither window argued
for in advance.

---

## INSTRUMENT claim — the far-upstream zeros are ours, not the model's

In the `>=200` band the three most-upstream tiles read exactly **0**. That is not a finding about
long ORFs: long ORFs start near the transcript 5′ end, so those windows are **unfilled** there.
Reported because a reader scanning the column would otherwise read it as a signal.

---

## What is NOT established

- **That the head reads the fill boundary.** Finding B is unearned and the design cannot earn it.
- **Anything about the decay head, or about motifs.** Different branch, different question.
- **That single-tile perturbation sees everything the head uses.** A degenerate feature spread across
  tiles would be under-detected; the multi-tile arm was not run.

## Bounds that travel with every claim here

- **Shuffled windows are off-manifold.** The model has never seen one. Shuffling preserves
  composition *and* the positional reading-frame channels, so it is closer to real input than
  substitution — but it is not real input.
- **Location cannot be resolved below ~42 positions**, the receptive field.
- **Sub-bin tiles saturate.** The encoder maxes within each of 8 bins, so perturbing part of a bin
  registers only if it held that bin's maximum.

## Enumeration (field 13)

12,000 candidates sampled from 796,584, seed 0, identical sample across both runs. Job 8900209: 12
tiles, 125 nt upstream and 25 nt downstream. Job 8900420: 40 tiles, 25 nt throughout. Checkpoint
`runs/interp_c32_b8_s100/best.pt`, `n_bins=8`, `permute_bins=False`, `conv_channels=32`,
`variant=interpretable`. Producer sha256 `2d829acb…` printed inside both jobs. Statistic: median
|Δ`z_p`| under within-tile shuffle, preserving fill mask, junction bits and composition — asserted on
1,115 real unfilled (window, tile) pairs before either model load.

## Corrections made reaching this

- **I proposed a finer re-run to settle constant-versus-rising offset. Withdrawn** — downstream tiles
  were already below the receptive field, so no tiling can settle it.
- **I attributed the window-edge value (0.5811) to convolution padding. Withdrawn** — `conv1` pads at
  both ends, and the upstream edge is the *lowest* cell in the table. A one-sided effect is not
  padding. That value is also filled only in the `>=200` band and is not comparable across bands.
- **The model window's constant-offset argument, withdrawn by them** after the per-band medians.
