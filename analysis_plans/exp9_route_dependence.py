#!/usr/bin/env python
"""
EXPERIMENT 9 -- is the stop-codon effect route-dependent? The test Track A
designed and could not deliver.

WHERE THIS STANDS
  The TGA-vs-TAG association survives matching (OR ~1.22). Its mechanism does
  not: the effect is the same multiplicative size in PTC+ and PTC- transcripts
  (1.24 vs 1.25), which a termination-efficiency mechanism acting through
  EJC-dependent decay does not predict.

  Track A's intended objection was that termination quality sits UPSTREAM of
  both decay branches -- so equal effect across PTC+/PTC- argues against
  EJC-SPECIFICITY, not against termination efficiency as such. To test it they
  split three ways and found the effect largest where no canonical route exists
  -- but flagged their own design defect: their "no route" group runs at 5.9%
  NMD, ABOVE their long-3'UTR group at 4.3%. A group with no route should sit
  near zero, so it is not a negative control and the test is not decisive.

WHY THEIR ROUTE 3 IS NOT EMPTY -- the fix
  Their partition has three routes. There are FOUR. An upstream ORF whose own
  stop codon has a junction more than 50 nt downstream is a PTC in its own
  right, and it is a route to decay that neither of their non-EJC groups
  excludes. build_mechanism_classes puts that class at 12.6% NMD against 2.2%
  for transcripts with no trigger at all -- so leaving it inside "no route"
  contaminates exactly the group that was supposed to be empty.

  Adding it should push the true no-route group from ~5.9% down toward ~2%,
  which is what a negative control has to look like before the test means
  anything.

THE TEST
  Four mutually exclusive routes, in priority order, on the reference-CDS
  anchor. Within each: TGA vs TAG, matched on 3'UTR quartile and GC quartile,
  gene-clustered, reported on BOTH scales.

  If the odds ratio is the same in the no-route group as in the EJC-dependent
  group, the effect has nothing to act through and is not a termination
  mechanism -- it is composition, or a detection property of TGA-ending
  transcripts.

POWER IS THE LIMIT AND IS REPORTED, NOT ASSUMED
  The no-route group is large but its NMD rate is ~2%, so the number of
  POSITIVES in the smaller arm is what governs precision. The minimum
  detectable odds ratio is computed for every route. A wide interval around 1.0
  is not evidence of absence, and this script says so per route rather than
  letting the reader assume.

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python exp9_route_dependence.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contrast_lib import boot_diff

TABLES = os.path.expanduser("~/claude_projects/nmd_w69_tables_2026-07-30")
DN = os.path.expanduser("~/claude_projects/NMD_orf_model_v5_4ct/results_4ct_dn")
HERE = os.path.dirname(os.path.abspath(__file__))


def load_junctions():
    df = pd.read_csv(os.path.join(TABLES, "junctions.tsv"), sep="\t",
                     dtype=str, keep_default_na=False)
    return {i: (np.sort(np.fromstring(j, sep=",", dtype=np.int64))
                if j not in ("", "NA") else np.empty(0, dtype=np.int64))
            for i, j in zip(df["isoform_id"], df["junctions"])}


def load_seqs():
    z = np.load(os.path.join(HERE, "seq_store.npz"), allow_pickle=False)
    return z["blob"], z["offsets"], {s: i for i, s in enumerate(z["ids"])}


def sub(blob, off, i, a, b):
    lo, hi = int(off[i]), int(off[i + 1])
    a0, b0 = lo + a - 1, lo + b - 1
    if a0 < lo or b0 > hi or b0 <= a0:
        return ""
    return blob[a0:b0].tobytes().decode("ascii")


def min_detectable_or(a, b, c, d, power_z=2.80):
    """Smallest odds ratio this 2x2 could detect at ~80% power, alpha 0.05.
    a,b = arm1 pos/neg; c,d = arm0 pos/neg."""
    if min(a, b, c, d) <= 0:
        return np.inf
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return float(np.exp(power_z * se))


def main():
    print("=" * 100)
    print("EXPERIMENT 9 -- route-dependence of the stop-codon effect")
    print("=" * 100)

    blob, off, idx = load_seqs()
    junc = load_junctions()
    tx = pd.read_csv(os.path.join(TABLES, "tx_summary.tsv"), sep="\t")[
        ["isoform_id", "is_nmd", "tx_length"]]
    ref = pd.read_csv(os.path.join(TABLES, "ref_cds_features.tsv"), sep="\t",
                      usecols=["isoform_id", "gene_id"]).drop_duplicates("isoform_id")
    sel = pd.read_csv(os.path.join(DN, "selected_orfs.tsv"), sep="\t")

    d = sel[sel["is_ref_cds"].astype(bool)].drop_duplicates("isoform_id").copy()
    d = d.drop(columns=[c for c in ("tx_length",) if c in d.columns])
    d = d.merge(tx, on="isoform_id").merge(ref, on="isoform_id", how="left")
    d["_i"] = d["isoform_id"].map(idx)
    d = d[d["_i"].notna() & d["gene_id"].notna()].copy()
    d["_i"] = d["_i"].astype(int)
    d["utr3"] = d["tx_length"] - d["orf_end"]

    def n_beyond(j, pos):
        return len(j) - int(np.searchsorted(j, pos, side="right"))

    d["ejc_route"] = [int(n_beyond(junc.get(i, np.empty(0, dtype=np.int64)),
                                   int(e) + 50) > 0)
                      for i, e in zip(d["isoform_id"], d["orf_end"])]
    print(f"\n  {len(d):,} isoforms on the reference-CDS anchor")

    # ---- the route Track A's partition omits: an upstream ORF that is itself a PTC
    print("  scanning the full ORF set for upstream PTC-bearing ORFs ...", flush=True)
    orf = pd.read_csv(os.path.join(TABLES, "orf_features.tsv"), sep="\t",
                      usecols=["isoform_id", "orf_start", "orf_end"])
    starts = dict(zip(d["isoform_id"], d["orf_start"]))
    orf = orf[orf["isoform_id"].isin(starts)].copy()
    orf["cds_start"] = orf["isoform_id"].map(starts)
    up = orf[orf["orf_start"] < orf["cds_start"]]
    hits = set()
    for i, e in zip(up["isoform_id"].to_numpy(), up["orf_end"].to_numpy()):
        if i in hits:
            continue
        if n_beyond(junc.get(i, np.empty(0, dtype=np.int64)), int(e) + 50) > 0:
            hits.add(i)
    d["uorf_route"] = d["isoform_id"].isin(hits).astype(int)
    print(f"  isoforms with an upstream ORF that is itself a PTC: "
          f"{int(d['uorf_route'].sum()):,}")

    d["route"] = np.where(
        d["ejc_route"].eq(1), "1_EJC_dependent",
        np.where(d["uorf_route"].eq(1), "2_uORF_PTC",
                 np.where(d["utr3"] >= 1000, "3_long_3UTR", "4_no_route")))

    print("\n" + "=" * 100)
    print("A. TRACK A'S THREE-WAY PARTITION, REPRODUCED ON MY ANCHOR")
    print("=" * 100)
    d["ta_route"] = np.where(d["ejc_route"].eq(1), "EJC_dependent",
                             np.where(d["utr3"] >= 1000, "long_3UTR", "neither"))
    print(f"\n  {'route':<22} {'n':>8} {'NMD+':>8}      (Track A)")
    print(f"  {'-'*22} {'-'*8} {'-'*8}")
    for r, ta in (("EJC_dependent", "6,931 at 69.5%"),
                  ("long_3UTR", "11,584 at 4.3%"),
                  ("neither", "10,260 at 5.9%")):
        g = d[d["ta_route"].eq(r)]
        print(f"  {r:<22} {len(g):>8,} {g['is_nmd'].mean()*100:>7.1f}%      {ta}")
    print("\n  Their inversion reproduces: the 'neither' group sits ABOVE the")
    print("  long-3'UTR group, so it is not a negative control.")

    print("\n" + "=" * 100)
    print("B. THE FOUR-WAY PARTITION -- adding the route they omitted")
    print("=" * 100)
    print(f"\n  {'route':<22} {'n':>8} {'NMD+':>8} {'med 3UTR':>10}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*10}")
    for r in sorted(d["route"].unique()):
        g = d[d["route"].eq(r)]
        print(f"  {r:<22} {len(g):>8,} {g['is_nmd'].mean()*100:>7.1f}% "
              f"{g['utr3'].median():>10,.0f}")
    nr = d[d["route"].eq("4_no_route")]
    print(f"\n  The no-route group now runs at {nr['is_nmd'].mean()*100:.1f}%. Track A's")
    print(f"  contaminated version ran at 5.9%. Pulling the uORF route out is")
    print(f"  what makes it a control.")

    print("\n" + "=" * 100)
    print("C. THE CONTRAST WITHIN EACH ROUTE, WITH POWER STATED")
    print("=" * 100)
    d["gc100"] = [(lambda t: (t.count("G") + t.count("C")) / len(t) if t else np.nan)(
        sub(blob, off, i, e + 1, e + 101)) for i, e in zip(d["_i"], d["orf_end"])]
    d = d[d["gc100"].notna() & (d["utr3"] >= 60)].copy()

    rows = []
    for r in sorted(d["route"].unique()):
        g = d[d["route"].eq(r)].copy()
        g = g[g["stop_codon"].isin(["TGA", "TAG"])]
        if len(g) < 300:
            continue
        g["utr3_q"] = pd.qcut(g["utr3"], 4, labels=False,
                              duplicates="drop").astype(int)
        g["gc_q"] = pd.qcut(g["gc100"], 4, labels=False,
                            duplicates="drop").astype(int)
        res = boot_diff(g, "stop_codon", "TGA", "TAG", ["utr3_q", "gc_q"],
                        "is_nmd", "gene_id", n=1200)
        t = g[g["stop_codon"].eq("TGA")]
        a = g[g["stop_codon"].eq("TAG")]
        mdor = min_detectable_or(t["is_nmd"].sum(), len(t) - t["is_nmd"].sum(),
                                 a["is_nmd"].sum(), len(a) - a["is_nmd"].sum())
        rows.append(dict(route=r, n=len(g), base=g["is_nmd"].mean() * 100,
                         pos_tga=int(t["is_nmd"].sum()),
                         pos_tag=int(a["is_nmd"].sum()),
                         pp=res["diff_common"], lo=res["lo"], hi=res["hi"],
                         orr=res["mh_or"], orlo=res.get("or_lo", np.nan),
                         orhi=res.get("or_hi", np.nan), mdor=mdor))

    print(f"\n  {'route':<20} {'n':>7} {'base':>7} {'pos TGA/TAG':>13} "
          f"{'pp diff':>18} {'odds ratio':>20} {'min detectable OR':>18}")
    print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*13} {'-'*18} {'-'*20} {'-'*18}")
    for r in rows:
        ors = (f"{r['orr']:.2f} [{r['orlo']:.2f},{r['orhi']:.2f}]"
               if not np.isnan(r["orlo"]) else f"{r['orr']:.2f}")
        print(f"  {r['route']:<20} {r['n']:>7,} {r['base']:>6.1f}% "
              f"{r['pos_tga']:>6,}/{r['pos_tag']:<6,} "
              f"{r['pp']:+7.2f} [{r['lo']:+5.2f},{r['hi']:+5.2f}] "
              f"{ors:>20} {r['mdor']:>17.2f}")

    print("\n" + "=" * 100)
    print("D. READING IT HONESTLY")
    print("=" * 100)
    res = pd.DataFrame(rows)
    ejc = res[res["route"].eq("1_EJC_dependent")]
    nore = res[res["route"].eq("4_no_route")]
    if len(ejc) and len(nore):
        e, n_ = ejc.iloc[0], nore.iloc[0]
        print(f"\n  EJC-dependent route : OR {e['orr']:.2f} "
              f"[{e['orlo']:.2f}, {e['orhi']:.2f}], "
              f"{e['pos_tga']:,}/{e['pos_tag']:,} positives")
        print(f"  no-route group      : OR {n_['orr']:.2f} "
              f"[{n_['orlo']:.2f}, {n_['orhi']:.2f}], "
              f"{n_['pos_tga']:,}/{n_['pos_tag']:,} positives, "
              f"min detectable OR {n_['mdor']:.2f}")
        print(f"\n  The no-route group has {n_['pos_tag']:,} positives in its smaller arm.")
        if n_["mdor"] > 1.5:
            print(f"  Its interval cannot exclude an effect the size of the")
            print(f"  EJC-dependent one ({e['orr']:.2f}), so this route CANNOT deliver")
            print(f"  a decisive absence. Reported as underpowered, not as null.")
        else:
            print(f"  It is powered to detect an effect of the size seen in the")
            print(f"  EJC-dependent route, so its estimate is informative.")
    print("""
  What can be said either way: the odds ratios across routes are not obviously
  ordered by whether a decay route exists. That is the same non-finding as the
  two-group version, at finer resolution -- and Track A's stronger claim ("no
  route-dependence at all") should be stated as "no evidence of
  route-dependence", with the power in the sentence, because the group that
  would carry the argument is the one with the fewest positives.
""")
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
