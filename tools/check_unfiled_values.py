#!/usr/bin/env python3
"""A number was measured, and nobody filed it.

    python3 tools/check_unfiled_values.py --values-dir <dir> [--strict]

THE OTHER ORPHAN. The forward and backward systems hunt opposite orphans, and the spec names two
checks for the forward one:

    an R-row with no emitted value    -> the filing refusal in file_result.py. Built.
    a value emitted with no R-row     -> THIS. Specified, and not built until now.

Harold's finding on review, 2026-08-02: the second was missing, and it is the one that would have
caught the backlog W229 describes -- auditing a narrative turned up four numbers that each had a
job id and a runlog and no claim row, because the claims table had stalled and essentially every
afternoon measurement went unfiled. The marker vocabulary already exists for it:

    [unclaimed]                  nothing measured behind it
    [unclaimed -- job NNNNNNN]   measured, producer and runlog exist, never filed

Two markers rather than one, because one marker for both failures is noise, and a marker readers
learn to ignore is worse than none.

WHY IT LOOKS AT THE VALUES FILES AND NOT AT PROSE. This is the mirror of check_d50.py. That one
starts from a document and asks whether a filed result stands behind the number. This one starts
from what a producer actually emitted and asks whether anyone wrote it down. A measurement that
was run, emitted, and never filed is invisible to the other direction entirely -- no document
mentions it, so no scan reaches it.

D66(b) keys claim_values by run_id, so a directory of per-run files is the expected input rather
than one growing table. `emitted_at` is not used for anything here: a file mtime is not a commit
time and an emit timestamp is not evidence of filing.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RECORDS = REPO / "claim_records"


def filed_keys():
    """Every (quantity, value) a filed row carries, with the row it came from.

    Keyed on the pair rather than on value alone. Two producers legitimately emit the same number
    for different quantities, and treating a value as filed because some unrelated quantity shares
    it is the false-resolution failure that made check_d50.py's `resolved` count meaningless."""
    out = {}
    if not RECORDS.exists():
        return out
    for p in sorted(RECORDS.glob("R*.json")):
        d = json.loads(p.read_text())
        for v in d.get("values", []):
            out.setdefault((str(v.get("quantity", "")).strip(),
                            str(v.get("value", "")).strip()), []).append(d["id"])
    return out


REQUIRED = ("quantity", "value", "population", "producer_file", "producer_line", "run_id")


def emitted(values_dir):
    """Read every claim_values file in the directory. A file that cannot be parsed OR whose header
    is not claim_emit's is REPORTED, not skipped.

    THE HEADER CHECK IS HERE BECAUSE ITS ABSENCE WAS MEASURED. Written without it, a file of
    binary garbage decoded without raising, produced a one-element header, contributed ZERO rows,
    and was counted as clean -- while this docstring already claimed it would be reported. That is
    the failure this project logs as 'an instrument that errors reports nothing, not zero', and it
    was live inside the guard against it. An empty contribution is indistinguishable from a file
    with no unfiled values, which is the direction that reads as good news."""
    rows, bad = [], []
    files = sorted(Path(values_dir).glob("*.tsv"))
    for f in files:
        try:
            lines = [l for l in f.read_text().splitlines() if l.strip()]
        except Exception as e:                     # noqa: BLE001 — reporting, not handling
            bad.append(f"{f.name}: unreadable ({e})")
            continue
        if not lines:
            bad.append(f"{f.name}: empty")
            continue
        head = lines[0].split("\t")
        missing = [c for c in REQUIRED if c not in head]
        if missing:
            bad.append(f"{f.name}: not claim_emit output — missing {', '.join(missing)}")
            continue
        for ln in lines[1:]:
            r = dict(zip(head, ln.split("\t")))
            r["_file"] = f.name
            rows.append(r)
    return files, rows, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values-dir", required=True,
                    help="directory of per-run claim_values TSVs (D66b keys them by run_id)")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    d = Path(a.values_dir)
    if not d.is_dir():
        sys.exit(f"not a directory: {d}")

    files, rows, bad = emitted(d)
    filed = filed_keys()

    unfiled = []
    for r in rows:
        key = (str(r.get("quantity", "")).strip(), str(r.get("value", "")).strip())
        if key not in filed:
            unfiled.append(r)

    print(f"\n  unfiled measurements — {len(files)} values file(s) in {d.name}, "
          f"{len(rows)} emitted value(s)")
    print(f"  {len(filed)} (quantity, value) pair(s) filed across "
          f"{len(list(RECORDS.glob('R*.json'))) if RECORDS.exists() else 0} result row(s)")
    if bad:
        print(f"\n  {len(bad)} values file(s) COULD NOT BE READ — reported, not skipped:")
        for b in bad:
            print(f"    {b}")

    print(f"\n  UNFILED {len(unfiled)}")
    if unfiled:
        print("    measured, with a producer and a run behind it, and no result row:")
        for r in unfiled[:25]:
            job = str(r.get("run_id", "")).strip()
            mark = f"[unclaimed -- job {job}]" if job else "[unclaimed]"
            print(f"    {mark}  {r.get('quantity','?')} = {r.get('value','?')}"
                  f"   ({r.get('population','no population')})")
            print(f"        {r.get('producer_file','?')}:{r.get('producer_line','?')}"
                  f"   in {r['_file']}")
        if len(unfiled) > 25:
            print(f"    … and {len(unfiled) - 25} more")
    else:
        print("    nothing emitted is unfiled.")
    print()
    if a.strict and unfiled:
        sys.exit(1)


if __name__ == "__main__":
    main()
