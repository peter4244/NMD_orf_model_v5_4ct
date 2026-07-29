#!/usr/bin/env python3
"""Resolve the external input paths this repo does not produce.

WHY THIS IS NOT IN utils.py. utils imports torch; resolving a filesystem path should not
cost a deep-learning framework, and data_prep.py deliberately imports neither.

WHY IT EXISTS AT ALL. data_prep.py, export_rds.R, 09_export_polya.py and
relabel_tx_summary_4ct.R each hardcoded absolute /projects/talisman/... literals. That
pinned the model's ENTIRE isoform universe to one machine's copy of the legacy isopair
tree, and it was invisible: measured 2026-07-26, 0 of 24 isoforms that the rebuilt
structures.rds added appear in the model's tx_summary.tsv, so the published model was
trained and explained on that stale universe with nothing to say so.

Resolution order: environment variable > config.yaml `paths:`. Defaults in config.yaml are
the original Channing locations, so behaviour is unchanged wherever they exist.
"""
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent

PATH_ENV = {
    "isopair_cache": "NMD_ISOPAIR_CACHE",
    "isopair_data":  "NMD_ISOPAIR_DATA",
    "sqanti_class":  "NMD_SQANTI_CLASS",
    "sqanti_fasta":  "NMD_SQANTI_FASTA",
    "mashr_dir":     "NMD_MASHR_DIR",
}


def resolve_path(key, config_path=None, must_exist=False):
    """environment variable > config.yaml. `must_exist` is opt-in: a caller that only
    reports the configured location must not fail where the input is absent."""
    cfg_path = Path(config_path) if config_path else REPO / "config.yaml"
    env = PATH_ENV.get(key)
    val = os.environ.get(env) if env else None
    src = f"${env}" if val else str(cfg_path)
    if not val:
        with open(cfg_path) as fh:
            val = (yaml.safe_load(fh).get("paths") or {}).get(key)
    if not val:
        raise KeyError(f"no path configured for '{key}': "
                       f"set ${env} or add paths.{key} to {cfg_path}")
    p = Path(val).expanduser()
    if must_exist and not p.exists():
        raise FileNotFoundError(
            f"{key} does not exist:\n    {p}\n  (from {src})\n"
            f"  Set ${env} or edit paths.{key} in {cfg_path}.")
    return p


def load_config(config_path="config.yaml"):
    """Read a config file. Lives here, not in utils, because utils imports torch.

    Moved 2026-07-29: seven export scripts (09*, 10*) need only the selected window tag and
    have no other reason to load a deep-learning framework. Importing utils for it made them
    depend on torch at module level.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def selected_tag(config):
    """The tag of the SELECTED window configuration. One place, read by every consumer.

    WHY (2026-07-29, D-B3.6). "Which configuration did we choose?" had no authoritative
    answer. It was encoded as the literal `atg500_stop500` in eight `--tag` argparse defaults,
    nine SLURM `TAG=` lines and `audit_report.R`, and *not* in config.yaml -- whose
    `data.window_size_*` is the sweep's starting grid point (atg=100, stop=1000), a
    configuration matching no artifact anyone uses. Every real invocation passes
    --atg-window 500 --stop-window 500 explicitly (slurm_train_4ct.sh:17, slurm_train_dn.sh:30).

    Re-selecting the window config therefore meant a 217-occurrence edit across 69 files, where
    a single missed `--tag` default keeps reading the OLD tag's artifacts at exit 0, plausibly
    and silently. With the sweep about to be re-run on train+val a different winner is live --
    AUPRC already favours atg500_stop1000 over atg500_stop500 (0.8387 vs 0.8330) while AUC
    favours 500. This makes that a two-line change in config.

    RAISES if `selected:` is absent rather than falling back to the old literal. A default here
    would be the same always-succeeding fallback that let the vendored-copy and hybrid-universe
    defects pass at exit 0: an unstated selection must be loud.
    """
    sel = config.get("selected")
    if not sel or "window_size_atg" not in sel or "window_size_stop" not in sel:
        raise KeyError(
            "config has no `selected:` block naming the chosen window configuration. "
            "Add:\n  selected:\n    window_size_atg: 500\n    window_size_stop: 500\n"
            "Do NOT read data.window_size_* for this -- those are the sweep grid's starting "
            "point, not the selection.")
    return f"atg{sel['window_size_atg']}_stop{sel['window_size_stop']}"


if __name__ == "__main__":
    # Shell entry point so SLURM drivers can read the selected tag instead of restating it.
    # `TAG=$(python3 paths_config.py --selected-tag)` -- torch-free, so it costs milliseconds.
    import argparse
    _ap = argparse.ArgumentParser(description="Query the repo's path/selection config.")
    _ap.add_argument("--selected-tag", action="store_true",
                     help="print the selected window-configuration tag and exit")
    _ap.add_argument("--config", default="config.yaml")
    _a = _ap.parse_args()
    if _a.selected_tag:
        print(selected_tag(load_config(_a.config)))
