#!/usr/bin/env python3
"""check_drivers.py — a deposit-native driver may not name a window, tag or results tree by literal.

WHY THIS EXISTS. On 2026-08-04 the deposit-native sweep re-selected the window configuration
from atg500_stop500 to atg1000_stop1000, and NINE driver scripts had to be repaired because
they named the old values directly:

    slurm_train_dn.sh, slurm_uorf_dn.sh, slurm_export_chain_dn.sh, slurm_eval_all_dn.sh,
    slurm_eval_final_dn.sh, and all four slurm_deepshap_*_dn.sh

plus two Python scripts resolving a config at the wrong moment (09_export_polya.py at argparse
parse time, infer_uorf_attention.py from a hardcoded config.yaml).

EVERY ONE WOULD HAVE PRODUCED NUMBERS RATHER THAN AN ERROR. A driver pinned to 500/500 after a
re-selection reads a checkpoint that exists, builds a model that runs, and writes a metrics
file that looks exactly like the right one. The DeepSHAP drivers were the worst case: they
would have computed the attributions section 5's interpretability rests on, for a model that is
not the selected one, into a directory segregated as untrustworthy.

AND THEY SURFACED ONLY BY LUCK. The sweep happened to change the selection. Had it confirmed
500/500 -- which was likely, since ten configurations were statistically inseparable -- all
nine would still be there, silently correct for the wrong reason. That is not a defect anyone
would have found by reading.

So the property "deposit-native drivers follow the selection" is enforced here rather than
maintained by hand. paths_config.py --selected-tag is the one source; a driver states the config
file it reads and derives everything else.

THE SCOPE IS DEPOSIT-NATIVE DRIVERS, AND THAT NARROWING IS THE POINT (2026-08-11). An earlier
form of this check ran over every slurm_*.sh and every *.py and reported 33 files. Almost none
were defects. The PUBLISHED chain -- slurm_train_4ct.sh, slurm_interpret_v5.sh,
slurm_kernel_shap.sh, slurm_export_subgroup_v5.sh, slurm_determinism.sh and the rest -- pins
500/500 CORRECTLY, because those drivers reproduce a fixed historical run whose selection was
500/500. Rewriting them to follow the current selection would be the actual error. And the *.py
hits were docstrings: usage examples and prose recording history, matched because the
comment-skip only understood lines beginning with # or //.

A checker that fires on things that are correct does not get obeyed; it gets bypassed, and then
it protects nothing. So the rule is stated where it is true: `slurm_*_dn.sh` and `submit_*_dn.sh`
must derive, because "deposit-native" MEANS tracking the current selection.

THE PYTHON RULE IS SEPARATE AND IS CHECKED BY PARSING, NOT BY REGEX. `load_config("config.yaml")`
inside a script defeats the caller's `--config`, which is how results_4ct_dn came to hold an
HDF5 built from Channing inputs. It is found with `ast`, so a docstring that mentions the call
cannot trip it and a call spread over two lines cannot hide from it.

THE EXEMPTIONS ARE THE INTERESTING PART. A literal is legitimate where it names a thing that is
NOT the selection -- a superseded run being deliberately reproduced, an archive path, a
comment recording history. Those are listed explicitly rather than pattern-matched, because an
exemption nobody can see is how the last set accumulated.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

# Whole files exempt, each with the reason stated.
EXEMPT_FILES = {
    "check_drivers.py":    "this file quotes the patterns it forbids",
    "paths_config.py":     "defines selected_tag; must name the config files to explain them",
    "config.yaml":         "IS the Channing config",
    "config_dn.yaml":      "IS the deposit-native config, and states the selection",
    "test_local_smoke.py": "builds a synthetic fixture at fixed sizes, not a run",
}

# Single lines exempt, keyed by (filename, exact stripped text). A literal here names something
# that is deliberately NOT the current selection, so following the selection would be wrong.
EXEMPT_LINES = {
    ("slurm_uorf_dn.sh",
     'if cmp -s "$CKPT" results_4ct/best_model_atg500_stop500.pt; then'):
        "compares the new checkpoint against the PUBLISHED one; naming it is the entire point",
    ("slurm_train_cpu_dn.sh", "OUT=results_4ct_dn_cpu"):
        "a separate CPU-only smoke tree, not the deprecated results_4ct_dn",
}

PATTERNS = [
    (re.compile(r"--atg-window\s+\d+"),       "window size stated as a literal"),
    (re.compile(r"--stop-window\s+\d+"),      "window size stated as a literal"),
    (re.compile(r"--atg\s+\d+\b"),            "window size stated as a literal"),
    (re.compile(r"--stop\s+\d+\b"),           "window size stated as a literal"),
    (re.compile(r"atg\d+_stop\d+"),           "member tag stated as a literal"),
    (re.compile(r"--results-dir\s+results_"), "results tree stated as a literal"),
]


def offending_lines(path: Path):
    out = []
    for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        stripped = line.strip()
        # Comments may DISCUSS the literals -- that is how the history is recorded.
        if stripped.startswith("#") or stripped.startswith("//"):
            continue
        if (path.name, stripped) in EXEMPT_LINES:
            continue
        for rx, why in PATTERNS:
            if rx.search(line):
                out.append((n, why, stripped[:100]))
                break
    return out


def hardcoded_config_calls(path: Path):
    """load_config("config.yaml") found by parsing, so docstrings cannot trip it."""
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name != "load_config":
            continue
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                    and arg.value.endswith("config.yaml"):
                out.append((node.lineno, "config.yaml read directly rather than named by the caller",
                            f"load_config({arg.value!r})"))
    return out


def main() -> int:
    drivers = sorted(
        p for p in list(REPO.glob("slurm_*_dn.sh")) + list(REPO.glob("submit_*_dn.sh"))
        if p.name not in EXEMPT_FILES and "superseded" not in str(p)
    )
    pyfiles = sorted(
        p for p in REPO.glob("*.py")
        if p.name not in EXEMPT_FILES and "superseded" not in str(p)
    )

    bad = {}
    for p in drivers:
        hits = offending_lines(p)
        if hits:
            bad[p.name] = hits
    for p in pyfiles:
        hits = hardcoded_config_calls(p)
        if hits:
            bad.setdefault(p.name, []).extend(hits)

    print(f"checked {len(drivers)} deposit-native driver(s) and {len(pyfiles)} python file(s)")
    if not bad:
        print("OK — no deposit-native driver names a window, tag or results tree by literal, "
              "and no script resolves config.yaml behind its caller")
        return 0

    print(f"\n{len(bad)} file(s) name something that should come from the config:\n")
    for name, hits in sorted(bad.items()):
        print(f"  {name}")
        for n, why, text in hits[:4]:
            print(f"    :{n}  {why}\n         {text}")
    print("\nDerive them instead:")
    print('    TAG=$($PY paths_config.py --selected-tag --config config_dn.yaml) || exit 1')
    print("    ATG=${TAG#atg}; ATG=${ATG%%_*}; STOP=${TAG##*stop}")
    print('    RESULTS_DIR="${RESULTS_DIR:-results_deposit_h5_2026-08-04}"')
    print("\nIf the literal is deliberate, add it to EXEMPT_LINES with the reason.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
