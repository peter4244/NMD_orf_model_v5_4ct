#!/usr/bin/env python3
"""verify_determinism.py — prove the hardened set_seed actually makes a run repeatable.

WHY THIS EXISTS. utils.set_seed was hardened on 2026-07-27 (cudnn.deterministic, benchmark off,
use_deterministic_algorithms, CUBLAS_WORKSPACE_CONFIG) because the §5 retrain regenerates the
headline AUC/AUPRC and a number we cannot reproduce ourselves cannot be asked of a reader. But
setting the flags is not evidence that the run is deterministic: `use_deterministic_algorithms(True)`
raises when an op has no deterministic kernel, and whether THIS architecture hits such an op is an
empirical question about the ops it actually uses.

So this runs the real NMDOrfModel over synthetic, seeded batches — no HDF5 required, so it can be
run on the cluster in seconds BEFORE committing to a multi-hour training job — and checks two
things a passing seed alone does not establish:

  1. the model does a forward AND backward pass with deterministic algorithms enabled
     (this is what surfaces an unimplemented deterministic kernel, loudly and early)
  2. two independently seeded runs produce BIT-IDENTICAL weights and losses

Run it on the machine that will do the canonical training, not on a laptop: determinism is a
property of (hardware, library versions), and a CPU pass proves nothing about the GPU path.

    python verify_determinism.py            # uses config.yaml dims
    python verify_determinism.py --steps 10

Exit 0 = repeatable. Exit 1 = NOT repeatable; do not run the canonical training until it is.
"""
import argparse
import hashlib
import io
import sys

import torch
import torch.nn as nn

from model import build_model
from utils import load_config, set_seed


def model_config(cfg):
    """Build the model config EXACTLY as 03_train.py does (03_train.py:130-132).

    NMDOrfModel reads a FLAT dict -- config.get("n_orf_features", 4) -- while config.yaml nests
    those keys under `model:`. Passing the whole config therefore silently builds a 4-feature
    encoder against 5-feature data and dies on a shape mismatch. Constructing it the same way the
    trainer does is the point: a determinism harness that builds a different model is not
    checking the model that gets trained.
    """
    return {**cfg["model"],
            "window_size_atg": cfg["data"]["window_size_atg"],
            "window_size_stop": cfg["data"]["window_size_stop"]}


def synthetic_batch(cfg, batch=8, device="cpu"):
    """One batch shaped exactly like NMDDataset yields, from the CURRENT torch RNG.

    Drawn from the seeded global RNG on purpose: if seeding were ineffective the two runs would
    differ in their inputs as well as their kernels, and this check should catch either.
    """
    K = cfg["data"]["max_orfs_per_tx"]
    C = cfg["model"]["n_seq_channels"]
    F = cfg["model"]["n_orf_features"]
    wa = cfg["data"]["window_size_atg"]
    ws = cfg["data"]["window_size_stop"]
    atg = torch.randn(batch, K, C, wa, device=device)
    stop = torch.randn(batch, K, C, ws, device=device)
    feats = torch.randn(batch, K, F, device=device)
    mask = torch.ones(batch, K, dtype=torch.bool, device=device)
    mask[:, -1] = False                      # exercise the masked/aggregator path too
    y = torch.randint(0, 2, (batch, 1), device=device).float()
    return atg, stop, feats, mask, y


def one_run(cfg, steps, device, seed):
    set_seed(seed)
    model = build_model(model_config(cfg)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    crit = nn.BCEWithLogitsLoss()
    losses = []
    model.train()
    for _ in range(steps):
        atg, stop, feats, mask, y = synthetic_batch(cfg, device=device)
        opt.zero_grad()
        out = model(atg, stop, feats, mask)
        logits = out[0] if isinstance(out, tuple) else out
        loss = crit(logits, y)
        loss.backward()                      # the backward pass is where nondeterminism lives
        opt.step()
        losses.append(float(loss.item()))
    # Hash the full weight tensor set: a mean or a norm can collide, bytes cannot.
    # Serialised through torch, NOT through .numpy(): the tensor->numpy bridge needs a matching
    # NumPy ABI and raises "Numpy is not available" where it does not match, which would read as
    # a determinism failure when it is nothing of the kind.
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    buf = io.BytesIO()
    torch.save({k: state[k] for k in sorted(state)}, buf)
    return losses, hashlib.sha256(buf.getvalue()).hexdigest(), state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=None, help="defaults to config training.seed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["training"]["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== determinism check ===")
    print(f"  torch {torch.__version__} | device {device} | "
          f"cuda {torch.version.cuda if torch.cuda.is_available() else 'n/a'}")
    if device == "cpu":
        print("  WARNING: running on CPU. This does NOT validate the GPU path, and the canonical\n"
              "           training runs on GPU. Re-run this on the training machine.")
    print(f"  steps {args.steps} | seed {seed}\n")

    try:
        print("run 1:")
        l1, h1, s1 = one_run(cfg, args.steps, device, seed)
        print("run 2:")
        l2, h2, s2 = one_run(cfg, args.steps, device, seed)
    except RuntimeError as e:
        if "Numpy is not available" in str(e) or "_ARRAY_API" in str(e):
            # Environment, not determinism. Saying otherwise would be a false finding.
            print(f"\nENVIRONMENT ERROR (not a determinism result): {e}")
            print("torch and NumPy have mismatched ABIs here. Fix the environment and re-run;\n"
                  "this tells you nothing about determinism either way.")
            return 2
        # This is the expected failure when an op lacks a deterministic implementation. It is a
        # RESULT, not a crash: it tells us determinism cannot be had without changing the model
        # or relaxing the setting -- far better learned here than after an 8-hour job.
        print(f"\nFAILED: {e}\n")
        print("An operation has no deterministic implementation, or CUBLAS_WORKSPACE_CONFIG was\n"
              "set too late. Do NOT run the canonical training yet. Options: identify the op and\n"
              "substitute a deterministic equivalent, or record explicitly in REPRODUCTION.md that\n"
              "the metrics are reproducible only to within GPU nondeterminism.")
        return 1

    print(f"\n  run 1 losses: {[round(x, 8) for x in l1]}")
    print(f"  run 2 losses: {[round(x, 8) for x in l2]}")
    print(f"  run 1 weights sha256: {h1}")
    print(f"  run 2 weights sha256: {h2}")

    # Elementwise comparison as well as the hash: the hash proves the serialised bytes match,
    # torch.equal proves the tensors do. If they ever disagree, the harness is at fault.
    keys_match = set(s1) == set(s2)
    tensors_equal = keys_match and all(torch.equal(s1[k], s2[k]) for k in s1)
    if (h1 == h2) != tensors_equal:
        print("\n  WARNING: hash and elementwise comparison disagree — trust the elementwise one "
              "and treat this harness as suspect.")
    print(f"  weights elementwise identical: {tensors_equal}")

    same = (l1 == l2) and (h1 == h2) and tensors_equal
    print("\n=== RESULT ===")
    if same:
        print("PASS — two seeded runs are bit-identical in losses and weights.\n"
              "       Determinism holds on this hardware and library version. State it no more\n"
              "       strongly than that: a reader on different hardware may still differ.")
        return 0
    print("FAIL — two seeded runs DIVERGED.\n"
          f"       losses identical: {l1 == l2} | weights identical: {h1 == h2}\n"
          "       The seed is being set but the run is not reproducible. Do not publish metrics\n"
          "       from this configuration as canonical until this passes.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
