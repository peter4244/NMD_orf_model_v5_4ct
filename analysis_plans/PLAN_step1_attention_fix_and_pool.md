# Plan for review — fix the attention export, then rebuild the pool locally

*Model-side window, 2026-08-01. For Pete's review before anything is changed.*

Two jobs, in this order. Nothing here touches the model architecture or trains anything.

---

# Job 1 — `infer_uorf_attention.py` assumes exactly 5 ORF slots

## What is actually wrong, read from the code rather than recalled

| where | what |
|---|---|
| `infer_uorf_attention.py:174` | the results dict is a literal: `"attn_0", "attn_1", "attn_2", "attn_3", "attn_4"` |
| `infer_uorf_attention.py:207` | `for k in range(5): results[f"attn_{k}"].append(attnL[j][k])` |
| `infer_uorf_attention.py:47` | the docstring names the five columns |
| `compute_uorf_attention_metrics_pathB.R:95` | `select(..., attn_0, attn_1, attn_2, attn_3, attn_4)` — an explicit five-column list |

**Two distinct failures, not one.** Above 5 slots it silently keeps the first five and drops the
rest. Below 5 slots it raises `IndexError`. Only exactly-5 works.

**How bad the silent case is, precisely.** The model's forward pass uses all K slots, so `prob` and
`logit` stay correct — it is only the attention export that truncates. But attention is a softmax
over all K, so at K = 50 the five exported numbers would sum to roughly a tenth rather than to 1,
and every downstream fraction computed from them would be wrong by that factor. Nothing about the
output would look malformed.

## What is already safe, checked rather than assumed

Two of the three R consumers are **already K-agnostic** — `compute_uorf_attention_metrics.R:89` and
`audit_uorf_attention.R:92,188` both use `starts_with("attn_")` followed by `pivot_longer`. They
will accept 50 columns without modification. Only `pathB.R` hardcodes the list.

## The fix

1. **Take K from the data, never from a literal.** The HDF5 already records it —
   `data_prep.py:978` writes `f.attrs["max_orfs"]`. Read that, cross-check it against the actual
   width of the attention tensor, and build the column list from it.
2. **Export all K columns**, `attn_0 … attn_{K-1}`. Wide format is kept deliberately: the two
   K-agnostic consumers keep working untouched, and at 41,765 rows × 50 columns the file is about
   20 MB. Long format would be tidier at K = 50 but would break every consumer for no gain.
3. **Change `pathB.R:95`** from the explicit five-column list to `starts_with("attn_")`, matching
   its two siblings.
4. **Assert the invariant that actually broke.** Attention is a softmax over unmasked slots, so it
   must sum to 1 per transcript. Check it on every row and fail loudly if not.

On point 4 — this is the kind of check I think is worth having, and it is a different kind from the
guard Pete told me to back out earlier. That one encoded a *judgement* (how many strata should
overlap) as a threshold I invented. This encodes an *invariant*: a softmax sums to one, always, with
no context in which it should not. It cannot be argued with and there is nothing to tune around.

## How I will know the fix is right

- Re-run at K = 5 against the published `results_4ct/uorf_attention_predictions.tsv` and require
  the output to match **bitwise**. If the fix changes any published number, it is wrong.
- Run a synthetic case at K = 3 and K = 8 to show both the old failure modes are gone — the first
  currently raises, the second currently truncates in silence.
- Re-run `compute_uorf_attention_metrics.R` on the K = 5 output and require its metrics TSV to match
  the existing one.

**Cost: under an hour. No new dependencies. Reversible.**

---

# Job 2 — rebuild the candidate ORF pool, locally

## What "the pool" is and what changes

Three changes, all already argued in `RETRAIN_PLAN_2026-08-01.md`:

| | now | after |
|---|---|---|
| minimum ORF length | 33 nt (`minimumLength = 9`, codons excluding start and stop) | **none** (`minimumLength = 0`) |
| how many kept | top **5** by annotation priority | **everything above an initiation floor** — mean ≈ 50 |
| slot order | annotation priority | **position in the transcript**, 5′ → 3′ |

The floor is the PWM score at the **1st percentile of real annotated start codons** (−2.705).
First percentile and not stricter, because a stricter floor would filter on initiation strength —
the very thing the model is supposed to discover — and remove the weak examples before it ever saw
them.

## What "locally" can and cannot produce

**Can:** the candidate table itself — one row per (transcript, ORF) with start, end, length, PWM
score, positional rank, and whether it is upstream of the annotated CDS. Plus all the validation
below. This is CPU work on data already on this machine.

**Cannot:** the training HDF5. At ~50 slots it is roughly 29 GB, against 8 GiB of RAM here. That
goes to the cluster, and I will ask before connecting.

## The problem this has to solve first, and it is the real risk

The existing scan is R — `ORFik::findORFs` via `05s_orfik_scan.R` — and ORFik is not installed here.
So the local rebuild is a **Python reimplementation of somebody else's ORF scanner**, and a
reimplementation that is subtly wrong would poison everything downstream while looking fine.

**The gate, before the new table is used for anything:** restrict my Python scan to ORFs of at least
33 nt and require it to reproduce `orf_features.tsv` — 1,540,674 ORFs over 42,043 transcripts —
**exactly**, on transcript, start, end and length. Same operation, same floor, so it must agree row
for row. If it does not agree at 100%, the scanner is wrong and nothing proceeds.

That gate is available because the old scan is a strict subset of the new one, and it is the whole
reason I would trust a local rebuild at all.

The PWM is the same one already validated against `Isopair::scoreKozakPWM` to 4.99e-13
(`kozak_pwm_rescore.py`), so initiation scoring is not a new risk.

## What comes out, and what I will check

1. **The gate above.** 100% or stop.
2. Pool size, against what I measured today: mean 54.6 candidates per transcript with no floor,
   50.1 after the floor, against 36.6 now.
3. **Coverage** of the population the rebuild exists for — upstream ORFs whose own stop has a
   junction more than 50 nt downstream. Should be 93.4%, against 10.5% today.
4. **The main ORF must still be in the pool.** With admission by initiation score rather than by
   annotation, the annotated CDS is no longer guaranteed a slot. If it drops out for some
   transcripts I need to know the rate before, not after — 51.5% of all positives are transcripts
   degraded through the main ORF.
5. Slot-count distribution, including the tail: the maximum today is 1,789 candidates on one
   transcript, and something has to decide what happens there.
6. What it will cost in HDF5 size and training time, from the actual counts rather than an estimate.

## What I am NOT doing in this job

Not changing `data_prep.py`, not building the HDF5, not touching the architecture, not training.
This produces a table and a set of numbers for review, and stops.

---

# Order and stopping points

1. Fix the attention export, verify bitwise against the published output. **Stop, report.**
2. Reimplement the scan, pass the exact-reproduction gate. **Stop — if the gate fails, nothing
   proceeds.**
3. Build the pool, produce the six checks above. **Stop, report, and get agreement on the floor and
   the cost before anything touches `data_prep.py`.**

## Open questions I would rather settle before starting than after

- **The floor.** 1st percentile of annotated starts costs ~50 slots and ~29 GB and buys 93.4%
  coverage; 5th percentile costs ~42 slots and buys 79.3%. I argue for the 1st on the
  don't-filter-on-your-discovery-target grounds, but it is the expensive call in the whole plan.
- **The tail.** A transcript with 1,789 candidates cannot have them all. Options: a hard cap with
  the overflow logged, or bucketed batching so long transcripts are simply slower. I lean to the
  second — a cap silently reintroduces exactly the truncation defect Job 1 exists to remove — but
  it complicates the data loader.
- **Wide versus long** for the attention export at K = 50. I have chosen wide to avoid breaking
  consumers; the interpretation window is the one that has to live with it.
