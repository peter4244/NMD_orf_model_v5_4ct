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
