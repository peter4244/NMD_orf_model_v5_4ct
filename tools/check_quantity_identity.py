#!/usr/bin/env python3
"""One quantity must have one value, and one population.

    python3 tools/check_quantity_identity.py [--strict]

WHY THIS IS THE MOST IMPORTANT CHECK IN THE FORWARD SYSTEM. "Two quantities with the same name,
computed over different sets" is the FIRST section of the analysis repo's CLAUDE.md — this
project's declared signature defect. Eleven errors across 2026-08-01/02 were all that one error.
None of them threw. Every one produced a plausible table.

And the forward system could not see it. `quantity` came through claim_emit, survived filing, and
was keyed on by nothing: check_d50.py matches on value alone, index.tsv has no quantity column,
and no tool compared two rows. Found by Harold on review, 2026-08-02, inside the scope D65 already
permits — this adds no field and names nothing specific to this project.

Two real cases it is calibrated against, both from documents rather than from filed rows, because
nothing was filed when they happened:

  the GC channel     68.2% in the section 5 claim list and INTERPRETABILITY_METHOD, 66.7% in the
                     interpretability handoff. No producer located for either.
  keto composition   1.16x and 1.148x. Not a rival measurement — 1.148x had NO producer, and the
                     script it was credited to computes neither keto nor amino. Struck as D52.

## WHAT IT FLAGS, and the rulings behind each

Pete, 2026-08-02:

  quantity matching     EXACT string match. Not normalised — a normalising key would decide by
                        itself that two differently-worded quantities are the same thing, which is
                        a judgement made inside an implementation and therefore invisible to
                        review.
  population differing  FLAG IT. Two rows asserting one quantity over different populations is
                        exactly the recovery-numbers shape: bank statistics quoted as population
                        statistics, each row internally correct. This is the flag most likely to
                        fire on something real, and it is not automatically an error — a legitimate
                        two-population comparison looks identical and must be declared.

  disagreement precision   the COARSER row's precision. Not ruled explicitly; taken as the default
                           because it is the convention already in force — the manuscript reports
                           2 dp, so a 3 dp shift is not a disagreement. Stated here rather than
                           buried, so it can be overridden.

## WHAT IT CANNOT DO

It compares FILED rows. Both motivating cases lived in prose and would still live in prose, so
this check and check_d50.py are two halves: that one finds numbers with no row behind them, this
one finds rows that disagree. Neither catches a single row that is simply wrong.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
QVIEW = REPO / "claim_records" / "quantities.tsv"


def dp(lit):
    """Decimals as WRITTEN, never recovered from the parsed float. float("819") is 819.0, whose
    string form contains a decimal point, so recovering precision after parsing gives every
    integer 1 dp instead of 0 — measured in the analysis repo before it was fixed."""
    s = str(lit).strip().replace(",", "")
    if "." not in s:
        return 0
    return len(s.split(".")[1].split("e")[0].split("E")[0])


def load():
    if not QVIEW.exists():
        sys.exit(f"{QVIEW.relative_to(REPO)} not found — run: "
                 "python3 tools/build_results_view.py  (nothing filed yet is also possible)")
    lines = [l for l in QVIEW.read_text().splitlines() if l.strip()]
    head = lines[0].split("\t")
    return [dict(zip(head, l.split("\t"))) for l in lines[1:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on any flag. Off by default for the same reason "
                         "check_d50.py reports rather than gates.")
    a = ap.parse_args()

    rows = [r for r in load() if r.get("state") != "RETRACTED"]
    groups = defaultdict(list)
    for r in rows:
        groups[r["quantity"]].append(r)

    value_flags, pop_flags = [], []
    for q, rs in sorted(groups.items()):
        if len(rs) < 2:
            continue
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                x, y = rs[i], rs[j]
                try:
                    vx = float(str(x["value"]).replace(",", ""))
                    vy = float(str(y["value"]).replace(",", ""))
                except ValueError:
                    continue
                d = min(dp(x["value"]), dp(y["value"]))          # the coarser row
                if round(vx, d) != round(vy, d):
                    value_flags.append((q, x, y, d))
                elif x["population"] != y["population"]:
                    pop_flags.append((q, x, y))

    print(f"\n  quantity identity — {len(rows)} filed value(s), "
          f"{sum(1 for v in groups.values() if len(v) > 1)} quantity name(s) measured more than once")

    print(f"\n  ONE QUANTITY, TWO VALUES   {len(value_flags)}")
    for q, x, y, d in value_flags[:20]:
        print(f"    {q!r}")
        print(f"      {x['r_id']}  {x['value']}  ({x['population'] or 'no population'})")
        print(f"      {y['r_id']}  {y['value']}  ({y['population'] or 'no population'})")
        print(f"      compared at {d} dp — the coarser of the two")

    print(f"\n  ONE QUANTITY, TWO POPULATIONS   {len(pop_flags)}")
    if pop_flags:
        print("    values agree; the sets do not. Declare the comparison or reconcile the sets —")
        print("    this is the shape of a bank statistic quoted as a population statistic.")
    for q, x, y in pop_flags[:20]:
        print(f"    {q!r}  {x['value']}")
        print(f"      {x['r_id']}  {x['population'] or 'no population'}")
        print(f"      {y['r_id']}  {y['population'] or 'no population'}")

    if not value_flags and not pop_flags:
        print("\n  nothing flagged. Note this compares FILED rows only — a number that lives")
        print("  in prose is check_d50.py's business, not this check's.")
    print()
    if a.strict and (value_flags or pop_flags):
        sys.exit(1)


if __name__ == "__main__":
    main()
