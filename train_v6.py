#!/usr/bin/env python
"""
train_v6.py — fit one configuration of the scanning-selection model.

Implements section 7 of analysis_plans/RETRAIN_PLAN_2026-08-01.md. One run per
invocation, so a sweep is a SLURM array rather than a loop in here.

  loss        binary cross-entropy on the transcript logit
  batching    bucketed by candidate count, bounded by PADDED candidates per
              batch rather than by transcript count
  stopping    val_clean AUC, patience 5, best checkpoint kept
  selection   nothing here reads the test split

RESUME IS NOT OPTIONAL. The gpu partition caps at 8 hours and a sweep is far
longer, so every epoch writes a checkpoint carrying the model, the optimiser, the
epoch counter, the best score, the patience counter and the RNG states of torch,
numpy and python. Restarting the same command picks up where it stopped. A run
that cannot resume is a run that silently restarts from epoch 0 when the queue
preempts it, and the only symptom is that it takes longer.

Usage:
    python train_v6.py --tensor results_tensor_v6 --out runs/interp_c32_b8_s100 \\
        --conv-channels 32 --n-bins 8 --variant interpretable --seed 100
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from model_v6 import ScanningNMDModel, count_parameters
from tensor_io import decode_windows

STRUCTURAL_COLS = ["n_downstream_ejc", "is_ref_cds", "is_sqanti_cds",
                   "frac_start", "frac_stop"]
INTERPRETABLE = [0]                 # n_downstream_ejc only
# is_sqanti_cds (index 2) is WITHHELD. It is TD2 calling the TARGET isoform's own
# CDS, and TD2 avoids ORFs that terminate at a premature stop -- which is exactly
# the ORF an NMD-susceptible isoform has. Every other TD2 exposure in v6 is absent
# or indirect; this one is direct and points the wrong way on the class we care
# about. is_ref_cds (index 1) stays: its anchor is by construction the dominant
# NON-NMD isoform, whose CDS does not terminate prematurely, so the bias mechanism
# is weakest exactly there -- and the interpretable variant never receives it.
PREDICTOR = [0, 1, 3, 4]            # n_downstream_ejc, is_ref_cds, frac_start, frac_stop


# ------------------------------------------------------------------ metrics
def auc_roc(y, s):
    """Rank-based AUC; no sklearn dependency on the cluster env."""
    y = np.asarray(y); s = np.asarray(s)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within ties, or a saturated model scores artificially well
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auprc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


# ------------------------------------------------------------------ batching
def make_batches(counts, max_padded, rng=None):
    """Group transcripts of similar candidate count.

    The bound is on PADDED candidates -- batch size times the widest transcript
    in it -- because that is what is allocated. Bounding transcript count instead
    lets one 565-candidate transcript decide the memory of the whole run.
    """
    order = np.argsort(counts, kind="stable")
    batches, cur, cur_max = [], [], 0
    for i in order:
        k = int(counts[i])
        nm = max(cur_max, k)
        if cur and (len(cur) + 1) * nm > max_padded:
            batches.append(np.array(cur)); cur, cur_max = [i], k
        else:
            cur.append(i); cur_max = nm
    if cur:
        batches.append(np.array(cur))
    if rng is not None:
        rng.shuffle(batches)
    return batches


class TensorSource:
    """Reads padded batches out of the ragged HDF5."""

    def __init__(self, path, struct_cols, blank_sequence=False,
                 blank_junctions=False, zero_structural=False, in_memory=True):
        self.path = str(path)
        self.struct_cols = struct_cols
        self.blank_sequence = blank_sequence
        self.blank_junctions = blank_junctions
        self.zero_structural = zero_structural
        self._f = None
        with h5py.File(self.path, "r") as f:
            self.offset = f["offset"][:]
            self.count = f["count"][:]
            self.labels = f["labels"][:].astype(np.float32)
            self.split = np.array([s.decode() for s in f["split"][:]])
            self.gene = np.array([s.decode() for s in f["gene_id"][:]])
            self.window = int(f.attrs["window"])
            self.atg_left = int(f.attrs["atg_left"])
            self.stop_left = int(f.attrs["stop_left"])
            self.orf_start = f["orf_start"][:]
            self.orf_end = f["orf_end"][:]
            self.pool_sha256 = f.attrs.get("pool_sha256", "")
            self.admission = f.attrs.get("admission", "{}")
            self.norm_mean = f["normalization/mean"][:]
            self.norm_std = f["normalization/std"][:]
            self.pool_complete = (f["pool_complete"][:]
                                  if "pool_complete" in f else None)
            self.structural = f["structural"][:]
            # THE CODES LIVE IN RAM. Profiled on the real split, the HDF5 read
            # was 67% of the batch against 29% for the decode: a batch is bucketed
            # by candidate count, so its rows are scattered across the file and
            # every lzf chunk they touch is decompressed again. Uncompressed the
            # whole array is 1.6 GB, so it simply fits -- which is only true
            # because a position is one byte rather than nine float16 channels.
            # Prefetching would not have fixed this; overlapping a 1.2 s load with
            # a 0.02 s compute saves the 0.02 s.
            self.codes = f["codes"][:] if in_memory else None
        if self.codes is not None:
            print(f"  codes resident in RAM: {self.codes.nbytes/1e9:.2f} GB")

    @property
    def f(self):
        if self._f is None:                      # opened lazily, per process
            self._f = h5py.File(self.path, "r")
        return self._f

    def indices(self, split):
        return np.flatnonzero(self.split == split)

    def batch(self, idxs, device):
        counts = self.count[idxs]
        K = int(counts.max())
        rows = np.concatenate([np.arange(self.offset[i], self.offset[i] + self.count[i])
                               for i in idxs])
        srt = np.argsort(rows, kind="stable")    # h5py needs increasing indices
        inv = np.argsort(srt, kind="stable")
        if self.codes is not None:
            codes = self.codes[rows]
            st = self.structural[rows][:, self.struct_cols]
        else:
            r = rows[srt]
            codes = self.f["codes"][r][inv]
            st = self.structural[rows][:, self.struct_cols]

        # The nine channels are rebuilt here rather than stored. tensor_io holds
        # the only definition of the mapping, and it is checked against the
        # nine-channel encoder rather than against a description of it.
        s0 = self.orf_start[rows]
        e0 = self.orf_end[rows]
        atg = decode_windows(codes[:, 0], s0, self.atg_left, s0)
        stop = decode_windows(codes[:, 1], e0 - 1, self.stop_left, s0)

        n = len(idxs)
        A = np.zeros((n, K, atg.shape[1], self.window), dtype=np.float32)
        S = np.zeros_like(A)
        U = np.zeros((n, K, len(self.struct_cols)), dtype=np.float32)
        M = np.zeros((n, K), dtype=bool)
        at = 0
        for j, c in enumerate(counts):
            c = int(c)
            A[j, :c] = atg[at:at + c]
            S[j, :c] = stop[at:at + c]
            U[j, :c] = st[at:at + c]
            M[j, :c] = True
            at += c

        if self.blank_sequence:
            # channels 0-3 AND 5: zeroing the bases alone leaves the rolling GC
            # fraction, which is composition, and composition accounted for every
            # 3'UTR motif this project has tested
            A[:, :, [0, 1, 2, 3, 5]] = 0.0
            S[:, :, [0, 1, 2, 3, 5]] = 0.0
        if self.blank_junctions:
            A[:, :, 4] = 0.0
            S[:, :, 4] = 0.0
        if self.zero_structural:
            # The block is kept at its declared width and set to zero, rather
            # than removed: the model requires n_structural >= 1, and `cols = []`
            # does NOT drop it -- `cols if cols else [0]` resolves to [0] and
            # feeds n_downstream_ejc straight back in. That silently made the
            # no-junction arm a junction-derived arm while its own checkpoint
            # recorded structural_columns=[].
            U[:] = 0.0

        t = lambda x, d=torch.float32: torch.as_tensor(x, dtype=d, device=device)
        return (t(A), t(S), t(U), torch.as_tensor(M, device=device),
                t(self.labels[idxs]))


# ------------------------------------------------------------------ resume
def save_checkpoint(path, model, opt, epoch, best, patience, history, rng):
    tmp = str(path) + ".tmp"
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                    epoch=epoch, best=best, patience=patience, history=history,
                    torch_rng=torch.get_rng_state(),
                    numpy_rng=rng["numpy"].bit_generator.state,
                    python_rng=random.getstate()), tmp)
    os.replace(tmp, path)        # atomic: a checkpoint is never half-written


def load_checkpoint(path, model, opt, rng):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    torch.set_rng_state(ck["torch_rng"])
    rng["numpy"].bit_generator.state = ck["numpy_rng"]
    random.setstate(ck["python_rng"])
    return ck["epoch"], ck["best"], ck["patience"], ck["history"]


# ------------------------------------------------------------------ the run
def evaluate(model, src, idxs, device, max_padded):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for b in make_batches(src.count[idxs], max_padded):
            sel = idxs[b]
            A, S, U, M, y = src.batch(sel, device)
            logit = model(A, S, U, M)
            ys.append(y.cpu().numpy()); ps.append(logit.cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    return auc_roc(y, p), auprc(y, p), len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="interpretable",
                    choices=["interpretable", "predictor"])
    ap.add_argument("--conv-channels", type=int, default=32)
    ap.add_argument("--n-bins", type=int, default=8)
    ap.add_argument("--permute-bins", action="store_true")
    ap.add_argument("--blank-sequence", action="store_true")
    ap.add_argument("--blank-junctions", action="store_true")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-padded", type=int, default=0,
                    help="0 = size it from conv_channels and --mem-budget-gb")
    ap.add_argument("--mem-budget-gb", type=float, default=10.0)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()

    # Peak activation memory is very close to linear in BOTH the padded batch
    # and the channel count: measured on a V100, 0.8 / 3.1 GB at 512 / 2048
    # candidates with 32 channels, and 2.9 / 11.4 at 128. A fixed batch therefore
    # OOMs at 128 channels while wasting the card at 16, and 8192 x 128 already
    # did OOM. Above ~1024 the epoch barely improves, so the cap is 2048.
    GB_PER_UNIT = 4.35e-5                       # GB per (padded candidate x channel)
    if args.max_padded <= 0:
        args.max_padded = int(np.clip(
            args.mem_budget_gb / (args.conv_channels * GB_PER_UNIT), 256, 2048))
        print(f"max_padded auto-sized to {args.max_padded} for "
              f"conv_channels={args.conv_channels} at a "
              f"{args.mem_budget_gb:.0f} GB budget")

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    ckpt = outdir / "checkpoint.pt"
    best_path = outdir / "best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed); random.seed(args.seed)
    rng = dict(numpy=np.random.default_rng(args.seed))

    cols = INTERPRETABLE if args.variant == "interpretable" else PREDICTOR
    # n_downstream_ejc is junction-derived, so the no-junction arm must lose it
    # as well as channel 4, or it is not a junction ablation at all.
    zero_struct = bool(args.blank_junctions)
    src = TensorSource(Path(args.tensor) / "nmd_tensor.h5",
                       cols,
                       blank_sequence=args.blank_sequence,
                       blank_junctions=args.blank_junctions,
                       zero_structural=zero_struct)

    model = ScanningNMDModel(conv_channels=args.conv_channels,
                             n_bins=args.n_bins,
                             n_structural=len(cols),
                             permute_bins=args.permute_bins).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    tr = src.indices("train"); va = src.indices("val")
    # Everything a downstream artifact needs to state its own population and
    # configuration, as first-class fields rather than buried in an arg string.
    provenance = dict(
        pool_sha256=src.pool_sha256,
        admission=json.loads(src.admission),
        tensor=str(Path(args.tensor) / "nmd_tensor.h5"),
        conv_channels=args.conv_channels, n_bins=args.n_bins,
        permute_bins=bool(args.permute_bins), variant=args.variant,
        blank_sequence=bool(args.blank_sequence),
        blank_junctions=bool(args.blank_junctions), seed=args.seed,
        structural_columns=([] if zero_struct
                            else [STRUCTURAL_COLS[c] for c in cols]),
        structural_width=len(cols),
        structural_zeroed=zero_struct,
        normalization=dict(columns=STRUCTURAL_COLS,
                           mean=src.norm_mean.tolist(),
                           std=src.norm_std.tolist()),
        split_sizes={s: int((src.split == s).sum())
                     for s in ["train", "val", "val_paralog",
                               "test", "test_paralog"]})
    print(f"device {device}   parameters {count_parameters(model):,}")
    print(f"variant {args.variant}   structural columns "
          f"{'(all zeroed)' if zero_struct else [STRUCTURAL_COLS[c] for c in cols]}"
          f"   block width {len(cols)}")
    print(f"train {len(tr):,}   val_clean {len(va):,}   "
          f"positives train {src.labels[tr].mean()*100:.1f}%", flush=True)

    start_epoch, best, patience, history = 0, -np.inf, 0, []
    if ckpt.exists():
        start_epoch, best, patience, history = load_checkpoint(ckpt, model, opt, rng)
        print(f"RESUMED from {ckpt} at epoch {start_epoch}, "
              f"best val AUC {best:.4f}, patience {patience}", flush=True)

    t0 = time.time()
    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        tot, nb = 0.0, 0
        for b in make_batches(src.count[tr], args.max_padded, rng["numpy"]):
            # An OOM on one batch of a 40-run array should cost that batch, not
            # the run. The offending batch is split and retried; if the halves
            # still do not fit, that is a real failure and it propagates.
            for attempt, chunks in enumerate(([b], np.array_split(b, 2),
                                              np.array_split(b, 4))):
                try:
                    for ch in chunks:
                        if not len(ch):
                            continue
                        A, S, U, M, y = src.batch(tr[ch], device)
                        opt.zero_grad(set_to_none=True)
                        loss = F.binary_cross_entropy_with_logits(
                            model(A, S, U, M), y)
                        loss.backward(); opt.step()
                        tot += loss.item(); nb += 1
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if attempt == 2:
                        raise
                    print(f"  OOM at {len(b)} transcripts, splitting", flush=True)
        vauc, vpr, n = evaluate(model, src, va, device, args.max_padded)
        history.append(dict(epoch=epoch, train_loss=tot / max(nb, 1),
                            val_auc=float(vauc), val_auprc=float(vpr),
                            seconds=time.time() - t0))
        improved = vauc > best
        if improved:
            best, patience = vauc, 0
            torch.save(dict(model=model.state_dict(), epoch=epoch,
                            val_auc=float(vauc), args=vars(args),
                            **provenance), best_path)
        else:
            patience += 1
        print(f"epoch {epoch:>3}  loss {tot/max(nb,1):.4f}  "
              f"val_clean AUC {vauc:.4f}  AUPRC {vpr:.4f}  "
              f"{'*' if improved else f'patience {patience}'}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        save_checkpoint(ckpt, model, opt, epoch + 1, best, patience, history, rng)
        if patience >= args.patience:
            print(f"stopping: no val_clean improvement in {args.patience} epochs")
            break

    (outdir / "history.json").write_text(json.dumps(
        dict(args=vars(args), best_val_auc=float(best),
             epochs_run=len(history), history=history), indent=2))
    print(f"\nbest val_clean AUC {best:.4f} over {len(history)} epochs, "
          f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
