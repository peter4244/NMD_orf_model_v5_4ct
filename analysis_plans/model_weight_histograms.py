#!/usr/bin/env python
"""
model_weight_histograms.py — the distribution of every derived weight over the
transcript universe, split at the 200 nt reading frame boundary.

SCOPE: claims about the MODEL. Terms are in Stories/ORF_selection/GLOSSARY.md.

WHY. Every statement so far has been a correlation or a median. A median hides
shape: "median scan_p is 0.015 among short candidates and 0.850 among long ones"
is compatible with two tight modes, with two broad overlapping ones, or with one
smear whose halves happen to sit apart. Which of those it is decides whether
"the scanner sorts candidates into two groups" is a description or a metaphor.

INTERNAL FIGURE, not a publication figure. Plain matplotlib, no style module, no
validator gate. If any of these reach a manuscript they get rebuilt under the
figures workflow.

Outputs a PNG and the underlying bin counts as TSV, so the figure is checkable
without rerunning it.

Usage:
    python model_weight_histograms.py --out runlog.txt [--all]
"""
import argparse, sys, hashlib, time, os, pathlib
from pathlib import Path
import numpy as np, h5py, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(os.environ.get(
    "NMD_TOOLS", Path.home() / "claude_projects/nmd_lung_longread_2026/tools"))))
from tensor_io import decode_windows                      # noqa: E402
from model_v6 import ScanningNMDModel                     # noqa: E402
from claim_emit import emit                               # noqa: E402

ATG_LEFT, STOP_LEFT, BOUNDARY = 900, 500, 200


def in_repo(rel):
    p = (REPO_ROOT / rel).resolve()
    if not str(p).startswith(str(REPO_ROOT.resolve()) + "/"):
        raise SystemExit(f"REFUSED: {rel} resolves outside the repo, to {p}")
    if not p.exists():
        raise SystemExit(f"REFUSED: {rel} does not exist under {REPO_ROOT}")
    return p


TENSOR = in_repo("results_tensor_v6/nmd_tensor.h5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    log = open(a.out, "w")

    def P(*s):
        print(*s); print(*s, file=log); log.flush()

    ckpt = in_repo(f"results_interp_all/v6_checkpoints/b8_s{a.seed}.pt")
    # W288. One key per line, fixed order, labelled (observed). A library this producer
    # does not import reports `absent` so the key set is stable across producers.
    import platform as _pl, socket as _sk, sys as _sys
    P("=== environment (observed) ===")
    P(f"  python {_pl.python_version()}")
    for _m in ("numpy", "torch"):
        try:
            P(f"  {_m} {__import__(_m).__version__}")
        except Exception:
            P(f"  {_m} absent")
    P(f"  node {_sk.gethostname()}")
    P(f"  platform {_sys.platform}")

    P("=== provenance (sha256) ===")
    P(f"  {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}  {Path(__file__).name}")
    P(f"  {hashlib.sha256(ckpt.read_bytes()).hexdigest()}  {ckpt.name}")

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = ScanningNMDModel(n_seq_channels=9, conv_channels=ck["conv_channels"],
                             n_bins=ck["n_bins"], seq_embed_dim=32, n_structural=1)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    f = h5py.File(TENSOR, "r")
    off, cnt = f["offset"][:], f["count"][:]
    keep = np.arange(len(off)) if a.all else np.arange(min(4000, len(off)))
    pop = "all 41,765 transcripts" if a.all else f"first {len(keep):,} transcripts"
    P(f"\n=== POPULATION: {pop}, seed {a.seed} ===")

    SP, SS, SQ, PS, LN, RES = [], [], [], [], [], []
    t0 = time.time()
    for i, t in enumerate(keep):
        lo, hi = int(off[t]), int(off[t]) + int(cnt[t])
        K = hi - lo
        if K < 1:
            continue
        codes = f["codes"][lo:hi]
        os_, oe_ = f["orf_start"][lo:hi], f["orf_end"][lo:hi]
        st = f["structural"][lo:hi]
        atg = decode_windows(codes[:, 0, :], os_, ATG_LEFT, os_)
        stp = decode_windows(codes[:, 1, :], oe_, STOP_LEFT, os_)
        with torch.no_grad():
            _, pr = model(torch.from_numpy(atg)[None], torch.from_numpy(stp)[None],
                          torch.from_numpy(st[:, :1])[None],
                          torch.ones(1, K, dtype=torch.bool), return_parts=True)
        sp = pr["p"][0].numpy().astype(np.float64)
        sq = pr["d"][0].numpy().astype(np.float64)
        ss = pr["p_select"][0].numpy().astype(np.float64)
        pn = float(pr["p_nmd"][0])
        SP.append(sp); SQ.append(sq); SS.append(ss)
        PS.append(ss * sq / pn)
        LN.append((oe_ - os_ + 1).astype(np.float64))
        RES.append(float(pr["residual"][0]))
        if (i + 1) % 10000 == 0:
            P(f"    {i+1:,}/{len(keep):,}  {time.time()-t0:.0f}s")
    f.close()

    sp = np.concatenate(SP); sq = np.concatenate(SQ); ss = np.concatenate(SS)
    ps = np.concatenate(PS); ln = np.concatenate(LN); res = np.array(RES)
    short, long_ = ln < BOUNDARY, ln >= BOUNDARY
    P(f"  {len(sp):,} candidates over {len(res):,} transcripts, {time.time()-t0:.0f}s")

    panels = [("scan_p", sp, "per candidate"), ("scan_select", ss, "per candidate"),
              ("seq_p_total", sq, "per candidate"), ("p_select", ps, "per candidate"),
              ("scan_residual", res, "per transcript")]
    bins = np.linspace(0, 1, 61)
    fig, axes = plt.subplots(1, 5, figsize=(21, 3.6))
    tsv = ["quantity\tstratum\tbin_lo\tbin_hi\tcount"]
    for ax, (nm, v, scope) in zip(axes, panels):
        if scope == "per candidate":
            for m, lab, c in ((short, f"< {BOUNDARY} nt", "#4C72B0"),
                              (long_, f">= {BOUNDARY} nt", "#C44E52")):
                h, _ = np.histogram(v[m], bins=bins)
                ax.step(bins[:-1], h / max(h.sum(), 1), where="post", label=lab, color=c, lw=1.4)
                for j in range(len(h)):
                    tsv.append(f"{nm}\t{lab}\t{bins[j]:.4f}\t{bins[j+1]:.4f}\t{h[j]}")
            ax.legend(fontsize=7, frameon=False)
        else:
            h, _ = np.histogram(v, bins=bins)
            ax.step(bins[:-1], h / max(h.sum(), 1), where="post", color="#55A868", lw=1.4)
            for j in range(len(h)):
                tsv.append(f"{nm}\tall\t{bins[j]:.4f}\t{bins[j+1]:.4f}\t{h[j]}")
        ax.set_yscale("log"); ax.set_title(f"{nm}\n({scope})", fontsize=9)
        ax.set_xlabel("value", fontsize=8); ax.tick_params(labelsize=7)
    axes[0].set_ylabel("fraction (log)", fontsize=8)
    fig.suptitle(f"Distribution of each derived weight — {pop}, seed {a.seed}", fontsize=10)
    fig.tight_layout()
    png = REPO_ROOT / "analysis_plans" / "fig_model_weight_histograms.png"
    fig.savefig(png, dpi=150); plt.close(fig)
    (REPO_ROOT / "analysis_plans" / "fig_model_weight_histograms_bins.tsv").write_text("\n".join(tsv) + "\n")
    P(f"  wrote {png.name} and its bin counts")

    P(f"\n=== SHAPE, which a median cannot show ===")
    # D92/D94. CANDIDATE-unit split at BOUNDARY, so the U-TENSOR-ORF-* pair. The non-all
    # arm is a positional first-N slice nothing has filed over, so it declares itself
    # unregistered rather than borrowing an id it was not computed over.
    # KEYED ON THE EMIT LOOP'S LABELS, which are "short"/"long" (line ~171). The PLOTTING
    # loop above uses "< 200 nt"/">= 200 nt" for the same split; I keyed on those first and
    # it died with KeyError: 'short' after 30 minutes of compute. Two loops, two label
    # vocabularies, one split -- read the loop that calls this, not the one that looks like it.
    UNIV_C = ({"short": "U-TENSOR-ORF-SHORT", "long": "U-TENSOR-ORF-LONG"} if a.all
              else {"short": "UNREGISTERED:first-N-of-B positional slice, short",
                    "long": "UNREGISTERED:first-N-of-B positional slice, long"})

    def POPS_(lab, est):
        return (f"universe={UNIV_C[lab]}; restriction=none; "
                f"estimator={est}; params=boundary_nt={BOUNDARY},all={a.all},seed={a.seed}")
    for nm, v in (("scan_p", sp), ("scan_select", ss), ("seq_p_total", sq)):
        for m, lab in ((short, "short"), (long_, "long")):
            x = v[m]
            lo_f, hi_f = float((x < 0.05).mean()), float((x > 0.95).mean())
            P(f"  {nm:12s} {lab:6s}  median {np.median(x):.4f}   "
              f"below 0.05 {100*lo_f:5.1f}%   above 0.95 {100*hi_f:5.1f}%")
            emit("shape", f"{nm} below 0.05, {lab} candidates", lo_f, n=int(m.sum()),
                 population=POPS_(lab, f"share of {nm} below 0.05"))
            emit("shape", f"{nm} above 0.95, {lab} candidates", hi_f, n=int(m.sum()),
                 population=POPS_(lab, f"share of {nm} above 0.95"))
    P("\n=== exit: 0 ===")
    log.close()


if __name__ == "__main__":
    main()
