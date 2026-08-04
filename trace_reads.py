#!/usr/bin/env python3
"""Record every file a Python script opens. The Python sibling of tools/trace_reads.R.

WHY. The section 5 model producers are Python -- evaluate.py, 11_kernel_shap_branches.py,
06_export_deepshap_tsv.py, infer_uorf_attention.py -- and trace_reads.R cannot see them, so
until now every statement about what they read came from reading. D27 step 3 wants execution.

HOW, and why this is stronger than the R version. CPython's audit hooks (PEP 578) fire below
the library layer: `open`, `os.open`, h5py, np.load, pandas.read_csv, torch.load and any C
extension that goes through the CPython file APIs all raise `open`. That covers the read routes
this project has been bitten by -- a path built from a key variable, a read made through a
library function whose name is not a read verb -- without needing to know the library.

THE AUDIT HOOK IS BLIND TO HDF5, AND THE SELF-TEST CAUGHT IT -- measured 2026-07-29, not
reasoned about. h5py opens through the HDF5 C library, which never touches CPython's file API,
so NO `open` event fires: an isolated probe wrote a 2,088-byte .h5 and read it back and recorded
ZERO events. That is not a footnote here -- `results_4ct/nmd_orf_data.h5` is THE central input
of evaluate.py, 11_kernel_shap_branches.py and infer_uorf_attention.py, so an unpatched trace of
any of them would look perfectly clean while omitting the one file that matters. Exactly the C34
shape: an unwrapped entry point is indistinguishable from a script that reads nothing. Fixed by
wrapping h5py.File explicitly, the same way the R tracer wraps connection constructors rather
than trusting a generic layer.

WHAT IT STILL CANNOT SEE, stated plainly:
  - any OTHER C extension calling open(2) directly. h5py is patched because it was measured;
    the next one will not be. Grow the self-test whenever the pipeline reaches for a new
    I/O route.
  - reads inside a SUBPROCESS. `subprocess` raises its own audit event, so the command line is
    recorded verbatim -- same treatment as system2() in the R tracer -- but the files that
    child reads are its own business and are NOT in this trace.
Treat a surprisingly SHORT trace as suspect. That rule cost this project two deposit
dependencies and four retractions.

Usage:
    python3 tools/trace_reads.py --self-test
    python3 tools/trace_reads.py --run <script.py> [--out trace.tsv] [-- <script args>]
"""
import argparse, os, runpy, sys, sysconfig

_ROWS = []
_SEEN = set()

# Locations that are not provenance: the interpreter, site-packages, and the conda prefix.
# Dropped by LOCATION, not by name -- same rule as the R tracer, for the same reason.
_NOISE_PREFIXES = tuple(
    os.path.realpath(p) for p in {
        sys.prefix, sys.base_prefix,
        sysconfig.get_paths().get("purelib", ""),
        sysconfig.get_paths().get("platlib", ""),
        sysconfig.get_paths().get("stdlib", ""),
    } if p
)

_WRITE_MODES = ("w", "a", "x", "+")


def _is_noise(path: str) -> bool:
    if not path:
        return True
    rp = os.path.realpath(path)
    if rp.startswith(_NOISE_PREFIXES):
        return True
    return "/__pycache__/" in rp or rp.endswith((".pyc", ".so"))


def _record(kind: str, path: str, detail: str = "") -> None:
    key = (kind, path, detail)
    if key in _SEEN:
        return
    _SEEN.add(key)
    _ROWS.append((kind, path, detail))


def _hook(event: str, args) -> None:
    try:
        if event == "open":
            path, mode = args[0], args[1]
            if not isinstance(path, (str, bytes, os.PathLike)):
                return                      # an fd, not a path
            path = os.fspath(path)
            if isinstance(path, bytes):
                path = path.decode("utf-8", "replace")
            if _is_noise(path):
                return
            mode = mode if isinstance(mode, str) else ""
            kind = "write" if any(m in mode for m in _WRITE_MODES) else "read"
            _record(kind, os.path.realpath(path), mode)
        elif event in ("subprocess.Popen", "os.system"):
            # The command line, verbatim. What the CHILD reads is not in this trace.
            _record("shell", " ".join(str(a) for a in args if a is not None)[:400], "")
        elif event == "urllib.Request":
            _record("url", str(args[0]), "")
    except Exception:
        pass                                # a tracer must never break the traced run


def install_h5py_shim() -> bool:
    """Wrap h5py.File, because the audit hook cannot see the HDF5 C library (measured).

    Returns True if the shim was installed. Import is attempted rather than assumed: wrapping a
    package that is not in use would force-load it and change what the traced script does --
    the same rule tools/trace_reads.R applies to data.table and readr.
    """
    try:
        import h5py
    except ImportError:
        return False
    if getattr(h5py.File, "_nmd_traced", False):
        return True
    orig = h5py.File.__init__

    def traced_init(self, name, mode="r", *a, **kw):
        try:
            if isinstance(name, (str, bytes, os.PathLike)):
                p = os.fspath(name)
                if isinstance(p, bytes):
                    p = p.decode("utf-8", "replace")
                m = mode if isinstance(mode, str) else "r"
                _record("write" if any(x in m for x in _WRITE_MODES) else "read",
                        os.path.realpath(p), f"h5py:{m}")
        except Exception:
            pass
        return orig(self, name, mode, *a, **kw)

    traced_init._nmd_traced = True
    h5py.File.__init__ = traced_init
    h5py.File._nmd_traced = True
    return True


def dump(out):
    # A path touched by any WRITE is a write, even if it was also opened for reading --
    # the R tracer learned this when saveRDS's underlying gzfile() made every output look
    # like an input and inverted the dependency graph.
    written = {p for k, p, _ in _ROWS if k == "write"}
    rows = []
    for k, p, d in _ROWS:
        if k == "read" and p in written:
            k = "write"
        rows.append((k, p, d))
    order = {"read": 0, "write": 1, "shell": 2, "url": 3}
    rows = sorted(set(rows), key=lambda r: (order.get(r[0], 9), r[1]))
    lines = ["kind\tpath\tmode"] + [f"{k}\t{p}\t{d}" for k, p, d in rows]
    text = "\n".join(lines)
    if out:
        with open(out, "w") as fh:
            fh.write(text + "\n")
    print("\n=== traced accesses ===")
    print(text)
    return rows


def self_test():
    """Five probes with known answers. A probe recording 0 accesses is a BLIND SPOT."""
    import tempfile, json
    print("=== trace_reads.py self-test: which KINDS of read does this instrument see? ===\n")
    tmp = tempfile.mkdtemp()
    csv_p = os.path.join(tmp, "probe.csv")
    with open(csv_p, "w") as fh:
        fh.write("a,b\n1,2\n")
    sys.addaudithook(_hook)
    print("h5py shim installed:", install_h5py_shim(), "\n")

    probes = {}
    probes["builtin open"] = lambda: open(csv_p).read()
    def _np():
        import numpy as np
        p = os.path.join(tmp, "probe.npz"); np.savez(p, x=np.arange(3)); return np.load(p)["x"]
    probes["numpy .npz (C-level)"] = _np
    def _pd():
        import pandas as pd
        return pd.read_csv(csv_p)
    probes["pandas.read_csv"] = _pd
    def _h5():
        import h5py, numpy as np
        p = os.path.join(tmp, "probe.h5")
        with h5py.File(p, "w") as f: f["d"] = np.arange(3)
        with h5py.File(p, "r") as f: return f["d"][:]
    probes["h5py (HDF5 C library -- the risk)"] = _h5
    probes["subprocess"] = lambda: __import__("subprocess").run(
        ["head", "-1", csv_p], capture_output=True)

    for name, fn in probes.items():
        before = len(_ROWS)
        print(f"-- probe: {name}")
        try:
            fn()
        except ImportError as e:
            print(f"   [skipped: {e}]"); continue
        except Exception as e:
            print(f"   [error: {e}]")
        print(f"   recorded {len(_ROWS) - before} access(es)")
    dump(None)
    print("\nRead the table against the probe list: a probe with 0 recorded accesses is a")
    print("BLIND SPOT of this instrument and must be handled another way.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--out")
    known, rest = ap.parse_known_args()
    if known.self_test:
        return self_test()
    if not known.run:
        ap.error("one of --self-test or --run is required")
    if rest and rest[0] == "--":
        rest = rest[1:]
    sys.argv = [known.run] + rest
    sys.addaudithook(_hook)
    install_h5py_shim()
    sys.path.insert(0, os.path.dirname(os.path.abspath(known.run)))
    try:
        runpy.run_path(known.run, run_name="__main__")
    finally:
        dump(known.out)


if __name__ == "__main__":
    main()
