#!/usr/bin/env python3
"""test_local_smoke.py — run the whole train/evaluate path on synthetic data, on CPU, locally.

WHY THIS EXISTS. On 2026-07-29 the cluster's GPU queue was ~90 minutes deep while the work
itself took 8 minutes, so every mistake cost a 90-minute round trip to discover. Two defects
that shipped that day would have been caught here in seconds: `--split` was made required on
evaluate.py without updating the four drivers that call it (they exited 2), and `run_deepshap`
was called POSITIONALLY so inserting a parameter mid-signature silently rebound `atg_window`.
Neither is visible to py_compile, and neither needs a GPU, real data, or Slurm to find.

WHAT IT COVERS that unit tests do not: the INTEGRATION. Training writes
best_model_{tag}_seed{S}.pt; evaluate.py then has to find that exact file through
utils.resolve_checkpoint and score the split it was asked for. That handoff is where the
member-naming and split work can break, and it cannot be tested without actually running both.

WHAT IT DOES NOT COVER, stated so nobody mistakes a pass for more than it is: numbers. The
data is random, so AUC is ~0.5 and means nothing. This answers "does the pipeline run and do
the pieces agree on filenames and splits", not "are the results right". It also runs CPU-only,
so it says nothing about determinism or CUDA kernels -- verify_determinism.py on the training
GPU is the gate for that.

    conda run -n nmd_model_local python test_local_smoke.py
    conda run -n nmd_model_local python test_local_smoke.py --atg 500 --stop 500  # real config

Exit 0 = the path is intact. Exit 1 = it is broken, and you know in seconds.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent


def build_synthetic_h5(path, n_tx, k_orfs, n_chan, windows, seed=0):
    """An HDF5 with data_prep.py's exact layout (data_prep.py:666-727), just tiny.

    Faithful STRUCTURE, random CONTENT. If the layout here drifts from data_prep's, this test
    starts passing against a file the real code could not read -- so it is written against the
    create_dataset calls rather than from memory.
    """
    rng = np.random.default_rng(seed)
    # Splits must include every category the code can ask for, INCLUDING val_paralog --
    # otherwise split="val_clean" raises and the test cannot exercise it.
    cats = (["train"] * (n_tx - 24) + ["val"] * 8 + ["val_paralog"] * 4
            + ["test"] * 8 + ["test_paralog"] * 4)
    assert len(cats) == n_tx, (len(cats), n_tx)
    chrs = (["chr8"] * (n_tx - 24) + ["chr2"] * 12 + ["chr1"] * 12)

    with h5py.File(path, "w") as f:
        for w in windows:
            grp = f.create_group(f"w{w}")
            grp.create_dataset("atg_windows",
                               data=rng.standard_normal((n_tx, k_orfs, n_chan, w)).astype(np.float32))
            grp.create_dataset("stop_windows",
                               data=rng.standard_normal((n_tx, k_orfs, n_chan, w)).astype(np.float32))
        n_feat = 5
        f.create_dataset("orf_features",
                         data=rng.standard_normal((n_tx, k_orfs, n_feat)).astype(np.float32))
        mask = np.ones((n_tx, k_orfs), dtype=bool)
        mask[:, 3:] = rng.random((n_tx, 2)) > 0.5      # some transcripts have fewer ORFs
        mask[:, 0] = True                               # rank-0 always present
        f.create_dataset("orf_mask", data=mask)
        f.create_dataset("labels", data=(rng.random(n_tx) > 0.75).astype(np.float32))
        str_dt = h5py.string_dtype()
        f.create_dataset("chr", data=np.array(chrs, dtype="S"), dtype=str_dt)
        f.create_dataset("isoform_id",
                         data=np.array([f"SYN{i:05d}" for i in range(n_tx)], dtype="S"),
                         dtype=str_dt)
        f.create_dataset("split", data=np.array(cats, dtype="S"), dtype=str_dt)
        ng = f.create_group("normalization")
        ng.create_dataset("orf_feat_mean", data=np.zeros(n_feat, dtype=np.float32))
        ng.create_dataset("orf_feat_std", data=np.ones(n_feat, dtype=np.float32))
        f.attrs["window_sizes"] = json.dumps(list(windows))
        f.attrs["max_orfs"] = k_orfs
        f.attrs["n_seq_channels"] = n_chan
        f.attrs["build_complete"] = True  # NMDDataset requires it; the synthetic file must
        # honour the same contract as a real one or the smoke test tests a fiction.
        f.attrs["n_transcripts"] = n_tx
    return path


def write_config(path, h5_path, atg, stop, epochs, batch):
    """Mirror config.yaml's shape, including the `selected:` block paths_config requires."""
    base = yaml.safe_load((HERE / "config.yaml").read_text())
    base["data"]["hdf5_path"] = str(h5_path)
    base["data"]["window_size_atg"] = atg
    base["data"]["window_size_stop"] = stop
    base["selected"] = {"window_size_atg": atg, "window_size_stop": stop}
    base["training"]["epochs"] = epochs
    base["training"]["batch_size"] = batch
    base["training"]["patience"] = epochs
    base.setdefault("training", {})["mixed_precision"] = False   # CPU
    Path(path).write_text(yaml.safe_dump(base))
    return path


def run(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atg", type=int, default=100)
    ap.add_argument("--stop", type=int, default=100)
    ap.add_argument("--n-tx", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=900)
    ap.add_argument("--keep", action="store_true", help="do not delete the scratch dir")
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="nmd_smoke_"))
    results = tmp / "results_smoke"
    results.mkdir()
    tag = f"atg{a.atg}_stop{a.stop}"
    fails = []

    print(f"scratch: {tmp}")
    h5 = build_synthetic_h5(results / "nmd_orf_data.h5", a.n_tx, 5, 9, sorted({a.atg, a.stop}))
    cfg = write_config(tmp / "config_smoke.yaml", h5, a.atg, a.stop, a.epochs, 8)
    print(f"synthetic HDF5: {a.n_tx} transcripts, windows {sorted({a.atg, a.stop})}, tag {tag}\n")

    def check(label, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
        if not ok:
            fails.append(label)

    # 1. TRAIN, with an explicit --seed. The checkpoint must carry the seed in its name.
    rc, out, err = run([sys.executable, "03_train.py", "--config", str(cfg),
                        "--results-dir", str(results), "--atg-window", str(a.atg),
                        "--stop-window", str(a.stop), "--seed", str(a.seed)], HERE)
    check("03_train.py runs", rc == 0, (err.strip().splitlines() or ["ok"])[-1][:110])
    ckpt = results / f"best_model_{tag}_seed{a.seed}.pt"
    check("checkpoint carries the seed", ckpt.exists(), ckpt.name)
    check("training log carries the seed",
          (results / f"training_log_{tag}_seed{a.seed}.csv").exists())
    check("no seedless checkpoint was written", not (results / f"best_model_{tag}.pt").exists())

    # 2. evaluate.py REFUSES what it should. These are the B2 gates.
    rc, _, err = run([sys.executable, "evaluate.py", "--config", str(cfg),
                      "--results-dir", str(results), "--atg-window", str(a.atg),
                      "--stop-window", str(a.stop), "--member-seed", str(a.seed)], HERE)
    check("evaluate.py requires --split", rc != 0 and "--split" in err)
    rc, _, err = run([sys.executable, "evaluate.py", "--config", str(cfg),
                      "--results-dir", str(results), "--atg-window", str(a.atg),
                      "--stop-window", str(a.stop), "--member-seed", str(a.seed),
                      "--split", "test_clean"], HERE)
    check("test split refused without --final", rc != 0 and "--final" in err)

    # 3. THE INTEGRATION: evaluate must find the seeded checkpoint training just wrote.
    rc, out, err = run([sys.executable, "evaluate.py", "--config", str(cfg),
                        "--results-dir", str(results), "--atg-window", str(a.atg),
                        "--stop-window", str(a.stop), "--member-seed", str(a.seed),
                        "--split", "val"], HERE)
    check("evaluate.py --split val runs", rc == 0, (err.strip().splitlines() or ["ok"])[-1][:110])
    check("it loaded the SEEDED checkpoint", f"_seed{a.seed}.pt" in out,
          "resolve_checkpoint picked the member")
    m = results / f"metrics_{tag}_seed{a.seed}_val.json"
    check("outputs carry member AND split", m.exists(), m.name)
    if m.exists():
        d = json.loads(m.read_text())
        check("metrics record the split inline", d.get("split") == "val", str(d.get("split")))
        check("n_test absent for a val run", "n_test" not in d,
              "a val score must not read as a test score")

    # 4. val_clean works when val_paralog exists, and refuses when it does not.
    rc, out, err = run([sys.executable, "evaluate.py", "--config", str(cfg),
                        "--results-dir", str(results), "--atg-window", str(a.atg),
                        "--stop-window", str(a.stop), "--member-seed", str(a.seed),
                        "--split", "val_clean"], HERE)
    check("val_clean runs when val_paralog is present", rc == 0,
          (err.strip().splitlines() or ["ok"])[-1][:110])

    print(f"\n=== {'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    if a.keep:
        print(f"kept: {tmp}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
