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

  disagreement precision   FULL precision. No tolerance.

REVERSED 2026-08-02 on Harold's second pass, and the reason I first gave was worse than the default
it justified. I took the coarser row's precision because it is "the convention already in force" —
but that convention belongs to the BACKWARD system, where one operand is a manuscript quote rounded
by a human. Here BOTH operands come from claim_emit, whose docstring forbids exactly that: the value
is "the number, as computed. Not rounded to the manuscript's precision: rounding here would hide a
disagreement that only shows in the third decimal."

So the tolerance protected a case the emit contract already prohibits, and its only reachable effect
was that a row filed at low precision could never disagree with anything — file 0.9 and it silently
absorbs 0.8834. The emit-contract violation was being rewarded by the check that should surface it.
A filed value with suspiciously few decimals is now its own flag instead.

WORTH WRITING DOWN, because the reasoning will be reapplied. Rounding the fine operand to the coarse
one is the SAME SHAPE as the defect in check_d50.py's matches(). What made it survivable here and
fatal there is the QUANTIFIER: matches() is existential over a haystack of hundreds, this is pairwise
over two rows already sharing an exact quantity string. "The convention already in force" was never
the reason — the quantifier is.

AND THE TWO FLAGS ARE INDEPENDENT, not an if/elif chain. A pair differing in BOTH value and
population previously reported only the value flag, so the reader was never shown that the sets
differ — which is precisely the recovery-numbers diagnosis, two values that disagree BECAUSE the
populations differ.

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
                # INDEPENDENT, not elif. Harold: an `elif` meant a pair differing in BOTH value
                # and population reported only the value flag, so the reader was never shown that
                # the populations differ -- which is precisely the recovery-numbers diagnosis, two
                # values that disagree BECAUSE the sets differ.
                if vx != vy:
                    value_flags.append((q, x, y, max(dp(x["value"]), dp(y["value"]))))
                if x["population"] != y["population"]:
                    pop_flags.append((q, x, y))

    print(f"\n  quantity identity — {len(rows)} filed value(s), "
          f"{sum(1 for v in groups.values() if len(v) > 1)} quantity name(s) measured more than once")

    print(f"\n  ONE QUANTITY, TWO VALUES   {len(value_flags)}")
    for q, x, y, d in value_flags[:20]:
        print(f"    {q!r}")
        print(f"      {x['r_id']}  {x['value']}  ({x['population'] or 'no population'})")
        print(f"      {y['r_id']}  {y['value']}  ({y['population'] or 'no population'})")
        print(f"      compared at full precision ({d} dp is the finer of the two)")

    lowp = [r for r in rows if r["value"] and dp(r["value"]) == 1]
    if lowp:
        print(f"\n  FILED AT ONE DECIMAL   {len(lowp)}")
        print("    claim_emit records the number as computed, not rounded. A single decimal is")
        print("    usually a hand-written row or a rounded one, and it cannot disagree with much.")
        for r in lowp[:10]:
            print(f"    {r['r_id']}  {r['quantity']} = {r['value']}")

    print(f"\n  ONE QUANTITY, TWO POPULATIONS   {len(pop_flags)}")
    if pop_flags:
        print("    one quantity measured over different sets. Declare the comparison or reconcile")
        print("    the sets — this is the shape of a bank statistic quoted as a population one.")
        print("    A pair listed HERE AND ABOVE is the worst case: the values disagree BECAUSE the")
        print("    populations do, which is the recovery-numbers diagnosis exactly.")
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
