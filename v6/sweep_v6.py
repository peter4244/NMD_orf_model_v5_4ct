#!/usr/bin/env python
"""
sweep_v6.py — enumerate the runs, then select one configuration from them.

Implements section 7 step 4, step 5 and section 8 step 1 of
analysis_plans/RETRAIN_PLAN_2026-08-01.md. Writes a manifest that a SLURM array
consumes one line at a time; each line is one (configuration, seed) and one call
to train_v6.py.

TWO PHASES, AND THE ORDER IS THE POINT.

  phase 1   the hyperparameter sweep, interpretable variant only. 8
            configurations x 5 seeds = 40 runs. Nothing here reads the test
            split.
  select    one configuration, on mean val_clean AUC over its 5 seeds.
  phase 2   the arms, all at the SELECTED configuration: predictor,
            sequence-blanked, no-junction. 3 x 5 = 15 runs.

Phase 2 cannot be enumerated before the selection exists, which is why this is
two manifests rather than one. Reporting test metrics for 40 configurations and
then quoting the best of them is how a selection effect gets published.

THE SWEEP IS A CROSS, NOT A GRID. conv_channels varies at B=8 and B varies at
conv_channels=32, so it establishes channel saturation AT B=8 and bin sensitivity
AT C=32, and nothing joint. The projection width is a function of both, so a
capacity ceiling is a property of the pair and this design cannot locate it. That
is a stated limit, not an oversight.

Usage:
    python sweep_v6.py --phase 1 --out sweep                # write manifest
    sbatch --array=1-40 sweep_v6.sbatch sweep/phase1.tsv
    python sweep_v6.py --collect sweep/phase1.tsv           # progress + results
    python sweep_v6.py --select  sweep/phase1.tsv           # pick a configuration
    python sweep_v6.py --phase 2 --out sweep                # arms at that config
"""

import argparse
import json
from pathlib import Path

SEEDS = [100, 200, 300, 400, 500]
FIELDS = ["name", "variant", "conv_channels", "n_bins", "permute_bins",
          "blank_sequence", "blank_junctions", "seed"]


def phase1_configs():
    """8 configurations: a cross through (conv_channels=32, n_bins=8), plus the
    permuted-bin control at that same point."""
    seen, out = set(), []

    def add(c, b, perm=False):
        key = (c, b, perm)
        if key in seen:
            return
        seen.add(key)
        tag = f"c{c}_b{b}" + ("_perm" if perm else "")
        out.append(dict(name=f"interp_{tag}", variant="interpretable",
                        conv_channels=c, n_bins=b, permute_bins=int(perm),
                        blank_sequence=0, blank_junctions=0))

    for c in (16, 32, 64, 128):          # channel arm, at B = 8
        add(c, 8)
    for b in (1, 8, 16, 32):             # bin arm, at C = 32
        add(32, b)
    add(32, 8, perm=True)                # the control for the bin arm
    return out


def phase2_configs(conv_channels, n_bins):
    """The arms of section 8, all at the selected configuration."""
    base = dict(conv_channels=conv_channels, n_bins=n_bins, permute_bins=0)
    tag = f"c{conv_channels}_b{n_bins}"
    return [
        dict(name=f"predictor_{tag}", variant="predictor",
             blank_sequence=0, blank_junctions=0, **base),
        dict(name=f"blankseq_{tag}", variant="interpretable",
             blank_sequence=1, blank_junctions=0, **base),
        dict(name=f"nojunc_{tag}", variant="interpretable",
             blank_sequence=0, blank_junctions=1, **base),
    ]


def write_manifest(configs, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(FIELDS)]
    for cfg in configs:
        for s in SEEDS:
            row = dict(cfg, seed=s)
            lines.append("\t".join(str(row[f]) for f in FIELDS))
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}  ({len(lines)-1} runs over {len(configs)} configurations)")
    print(f"  sbatch --array=1-{len(lines)-1} sweep_v6.sbatch {path}")


def read_manifest(path):
    rows = [l.split("\t") for l in Path(path).read_text().strip().split("\n")]
    hdr = rows[0]
    return [dict(zip(hdr, r)) for r in rows[1:]]


def collect(manifest, runroot):
    runs = read_manifest(manifest)
    done, missing = [], 0
    for r in runs:
        h = Path(runroot) / f"{r['name']}_s{r['seed']}" / "history.json"
        if not h.exists():
            missing += 1
            continue
        j = json.loads(h.read_text())
        done.append(dict(name=r["name"], seed=int(r["seed"]),
                         auc=j["best_val_auc"], epochs=j["epochs_run"],
                         seconds=j["history"][-1]["seconds"]))
    print(f"{len(done)} of {len(runs)} runs complete, {missing} outstanding\n")
    if not done:
        return {}
    by = {}
    for d in done:
        by.setdefault(d["name"], []).append(d)
    print(f"  {'configuration':<24} {'n':>3} {'mean val AUC':>13} "
          f"{'range':>18} {'mean epochs':>12} {'mean hours':>11}")
    print(f"  {'-'*24} {'-'*3} {'-'*13} {'-'*18} {'-'*12} {'-'*11}")
    for name in sorted(by, key=lambda k: -sum(x["auc"] for x in by[k]) / len(by[k])):
        g = by[name]
        a = [x["auc"] for x in g]
        print(f"  {name:<24} {len(g):>3} {sum(a)/len(a):>13.4f} "
              f"{f'[{min(a):.4f}, {max(a):.4f}]':>18} "
              f"{sum(x['epochs'] for x in g)/len(g):>12.1f} "
              f"{sum(x['seconds'] for x in g)/len(g)/3600:>11.2f}")
    return by


def select(manifest, runroot):
    by = collect(manifest, runroot)
    complete = {k: v for k, v in by.items() if len(v) == len(SEEDS)}
    if not complete:
        print("\nno configuration has all seeds finished; not selecting")
        return None
    # Highest mean val_clean AUC. On a tie, the smaller model: a sweep whose
    # point is whether extra capacity is used should not award it on a tie.
    def key(name):
        g = complete[name]
        c = int(name.split("_c")[1].split("_")[0])
        b = int(name.split("_b")[1].split("_")[0])
        return (-sum(x["auc"] for x in g) / len(g), c * b)
    best = sorted(complete, key=key)[0]
    if "perm" in best:
        print(f"\nWARNING: the permuted-bin CONTROL has the highest mean val AUC "
              f"({best}). That is a result about the sweep, not a configuration "
              f"to carry forward — it says the bins are not being used for "
              f"position. Selecting the best non-control instead.")
        best = sorted([k for k in complete if "perm" not in k], key=key)[0]
    c = int(best.split("_c")[1].split("_")[0])
    b = int(best.split("_b")[1].split("_")[0])
    g = complete[best]
    print(f"\nSELECTED {best}: conv_channels={c}, n_bins={b}, "
          f"mean val_clean AUC {sum(x['auc'] for x in g)/len(g):.4f} over "
          f"{len(g)} seeds")
    print("  test metrics are computed for this configuration only, after this point")
    (Path(runroot) / "selected.json").write_text(json.dumps(
        dict(name=best, conv_channels=c, n_bins=b,
             mean_val_auc=sum(x["auc"] for x in g) / len(g)), indent=2))
    return c, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, choices=[1, 2])
    ap.add_argument("--out", default="sweep")
    ap.add_argument("--runroot", default="runs")
    ap.add_argument("--collect")
    ap.add_argument("--select")
    args = ap.parse_args()

    if args.collect:
        collect(args.collect, args.runroot); return
    if args.select:
        select(args.select, args.runroot); return
    if args.phase == 1:
        write_manifest(phase1_configs(), Path(args.out) / "phase1.tsv")
    elif args.phase == 2:
        sel = Path(args.runroot) / "selected.json"
        if not sel.exists():
            raise SystemExit(f"{sel} not found — run --select on phase 1 first")
        s = json.loads(sel.read_text())
        print(f"phase 2 at the selected configuration: "
              f"conv_channels={s['conv_channels']}, n_bins={s['n_bins']}")
        write_manifest(phase2_configs(s["conv_channels"], s["n_bins"]),
                       Path(args.out) / "phase2.tsv")
    else:
        ap.error("give --phase, --collect or --select")


if __name__ == "__main__":
    main()
