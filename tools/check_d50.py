#!/usr/bin/env python3
"""D50's gate: a number stated in a shared document must have a filed result behind it.

    python3 tools/check_d50.py [--repo <path>] [--strict]

REPORTING ONLY BY DEFAULT, ON PURPOSE. Pete, 2026-08-02. Early in a rollout almost every number
is unresolved, and a checker that fails on arrival trains everyone to ignore it —
check_claim_values.py makes exactly this argument about its own --strict, and C50 records the
cost: a check that fires on correct behaviour gets muted, and a muted check no longer catches
the violation it exists for. So this prints and exits 0. --strict exists and is wired to no
hook. Turn it on when the exemption backlog is declared, not before.

WHAT IT IS ENFORCING. D50: a result enters the ledger the moment it is written into a document
another window reads. That trigger was chosen because it is OBSERVABLE — a number appearing in
a tracked document — where "important enough to record" is a judgement call. Until this ran,
D50 was a prose rule, and W256 records that every prose rule written on 2026-08-02 was broken
on 2026-08-02, usually by whoever wrote it, often within the hour.

WHY THE DETECTOR IS IMPORTED AND NOT COPIED. check.py already carries STATE_NUM and
strip_exempt, and strip_exempt is the part that took the work: it drops code spans, link
targets and dates so the scan does not cry wolf. Copying it would create the restatement
failure this project keeps paying for — two copies, drifting. It is imported from the analysis
repo, the same cross-repo route the analysis_plans producers already use for claim_emit.

THE ONE IT WOULD HAVE CAUGHT. 1.148x sat in a governing document credited to a producer that
never computes keto or amino at all. This resolves numbers against EMITTED values, not against
a credit line in prose, so it fails on the day the number is written rather than after three
windows spend two days reconciling it.

WHAT IT CANNOT CATCH, stated so it is not oversold. It catches ABSENCE, not error. A number
that is filed but filed WRONG — a bank statistic carrying a population field that says
"population" — resolves cleanly here. That is the story gate's job, and no linter catches a
wrong question.
"""
import argparse
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:                       # scope can still be passed with --scope
    yaml = None

TRACK_A = Path.home() / "claude_projects" / "nmd_lung_longread_2026"
sys.path.insert(0, str(TRACK_A / "tools"))
from check import STATE_NUM, strip_exempt                       # noqa: E402

MODEL_REPO = Path(__file__).resolve().parent.parent
RECORDS = MODEL_REPO / "claim_records"
EXEMPTIONS = RECORDS / "exemptions.tsv"

# The exponent is part of the number. Without it "1e-65" reads as 1 and -65, which made every
# p-value in the analysis repo's ledger compare as nonsense until it was measured and fixed.
NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?")


def literals(line):
    """Full numeric literals on a line, with the decimal count of each as written.

    STATE_NUM finds the LINE — it reports '883' for '0.883' because of its bare-multi-digit
    branch — so it is used to decide whether a line states a result, and the literal is then
    recovered here. Precision comes from the literal, never from the parsed float:
    float("819") is 819.0, whose string form contains a decimal point, so recovering dp after
    parsing makes every published integer compare at 1 dp instead of 0."""
    out = []
    for m in NUM.finditer(line):
        s = m.group(0).replace(",", "")
        try:
            v = float(s)
        except ValueError:
            continue
        dp = len(s.split(".")[1].split("e")[0].split("E")[0]) if "." in s else 0
        out.append((m.group(0), v, dp))
    return out


def emitted_values():
    """Every value any filed R-row carries."""
    vals = []
    for p in sorted(RECORDS.glob("R*.json")):
        d = json.loads(p.read_text())
        for v in d.get("values", []):
            try:
                vals.append((float(str(v.get("value", "")).replace(",", "")), d["id"]))
            except ValueError:
                continue
    return vals


def claim_values():
    """Values the backward ledger already holds, so a manuscript number is not reported as an
    unfiled forward result. Absence of the file is reported, not treated as zero — an
    instrument that errors reports nothing, not zero (C62's shape)."""
    f = TRACK_A / "docs" / "claim_status.tsv"
    if not f.exists():
        print(f"  NOTE: {f} not found — manuscript claims will read as unresolved")
        return []
    lines = f.read_text().splitlines()
    head = lines[0].split("\t")
    i = head.index("values") if "values" in head else None
    if i is None:
        return []
    out = []
    for ln in lines[1:]:
        cells = ln.split("\t")
        if len(cells) > i:
            for _, v, _ in literals(cells[i]):
                out.append((v, cells[0]))
    return out


def exemptions():
    """Declared as DATA beside the claim, never as a list inside the checker (P16, and C65 is
    the cautionary case for a judgement stored where a rebuild can revert it).

    W121 established the classes and the split that makes --strict reachable at all:
    thresholds, approximations, English intensifiers, selection criteria and inequality bounds
    are numbers no producer computes. Two buckets, and ONLY 'nobody has said why' gates."""
    if not EXEMPTIONS.exists():
        return []
    out = []
    for ln in EXEMPTIONS.read_text().splitlines()[1:]:
        if not ln.strip() or ln.startswith("#"):
            continue
        c = ln.split("\t")
        if len(c) >= 3:
            try:
                out.append((float(c[0].replace(",", "")), c[1], c[2]))
            except ValueError:
                continue
    return out


def matches(v, dp, pool):
    """Does a document number correspond to a filed one?

    REWRITTEN 2026-08-02 on Harold's measurement. The previous rule rounded the POOL value to
    the DOCUMENT's precision, so a coarse document number matched any fine pool value near it.
    Measured over 48 documents against a 241-value pool: bare integers resolved at 56.5%
    (2,884 of 5,103) against 8.0% (84 of 1,056) for literals with three or more decimals, and
    2,884 of the 3,357 total resolutions -- 86% -- were bare integers. Line numbers, page
    counts and chromosome numbers were "resolving" against unrelated claim values, and the rate
    climbs toward 100% as the pool grows.

    The split, and why it is not simply "ignore low precision":

      the document literal has decimals   round both to the COARSER precision. This is the
                                          rounding tolerance that has to exist -- a manuscript
                                          reporting 0.88 against a producer's 0.8834 is a
                                          match, not a disagreement.
      the document literal is an integer  require EXACT equality against an integer pool value.
                                          No rounding window at all. An n of 10,520 still
                                          matches; a line number 593 no longer matches 0.883.

    Harold's alternative -- match at the FINER precision -- was rejected because it would break
    the legitimate case above: 0.88 against 0.8834 at 4 dp is not a match, and quoting a rounded
    number is normal in prose.
    """
    if dp == 0:
        return any(float(x).is_integer() and int(x) == int(v) for x, _ in pool)
    return any(round(x, dp) == round(v, dp) for x, _ in pool)


def in_scope(path, globs):
    """Depth-aware, copied in behaviour from check.py's function of the same name: Python's
    fnmatch lets `*` match `/`, so a bare `*.md` scope swallows every nested README. A pattern
    matches only paths at its own depth."""
    return any(g.count("/") == path.count("/") and fnmatch.fnmatch(path, g) for g in globs)


def scope_for(repo, override):
    """Which documents count as 'a document another window reads'.

    NOT INVENTED HERE. Harold's steer, accepted: take the globs from artifacts.yml rather than
    building a second scoping mechanism, because a second mechanism is the restatement failure
    this project keeps paying for.

    AND FAIL CLOSED WHEN THERE IS NO DECLARATION. The analysis repo declares `scope` in
    config/artifacts.yml. This repo does not, and I am deliberately NOT guessing which of its 48
    tracked documents are current -- BUGFIX_STOP_CODON_2026-03-31.md sits at depth 0 and matches
    the analysis repo's `*.md` glob, so borrowing that scope would not have excluded it anyway.
    Deciding it here would be resolving an ambiguity inside an implementation, which is invisible
    to review and is the origin of most errors on this project. The owning windows declare it,
    with --scope or by adding a declaration.
    """
    if override:
        return [g.strip() for g in override.split(",") if g.strip()], "--scope"
    cfg = repo / "config" / "artifacts.yml"
    if cfg.exists() and yaml is not None:
        d = yaml.safe_load(cfg.read_text())
        if isinstance(d, dict) and d.get("scope"):
            return list(d["scope"]), str(cfg.relative_to(repo))
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(MODEL_REPO))
    ap.add_argument("--scope", default="",
                    help="comma-separated globs, e.g. 'analysis_plans/*.md'. Required for a repo "
                         "with no config/artifacts.yml scope declaration.")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()

    globs, src = scope_for(repo, a.scope)
    if globs is None:
        sys.exit(
            f"NO SCOPE DECLARED for {repo.name}.\n"
            "D50 concerns documents another window reads NOW, not every document ever tracked —\n"
            "scanning this repo's history reported 4,498 unresolved numbers from files like\n"
            "BUGFIX_STOP_CODON_2026-03-31.md. Which documents are current is the owning windows'\n"
            "call, not this tool's.\n\n"
            "  declare it:  config/artifacts.yml  ->  scope: [...]      (as the analysis repo does)\n"
            "  or pass:     --scope 'analysis_plans/*.md,*.md'")

    r = subprocess.run("git ls-files '*.md'", shell=True, capture_output=True, text=True, cwd=repo)
    if r.returncode != 0:
        sys.exit(f"git ls-files failed in {repo}:\n{r.stderr.strip()}")
    docs = [d for d in r.stdout.split() if d and in_scope(d, globs)]

    pool = emitted_values() + claim_values()
    exempt = exemptions()
    resolved = exempted = 0
    unresolved = []
    # Broken out by precision so `resolved` cannot be quoted without its composition. When 86%
    # of resolutions were bare integers, the single number read as coverage and was noise.
    by_dp = {"integer": [0, 0], "1-2 dp": [0, 0], ">=3 dp": [0, 0]}

    for rel in docs:
        for i, raw in enumerate((repo / rel).read_text(errors="replace").splitlines(), 1):
            line = strip_exempt(raw)
            if not STATE_NUM.search(line):
                continue
            for lit, v, dp in literals(line):
                b = by_dp["integer" if dp == 0 else ("1-2 dp" if dp <= 2 else ">=3 dp")]
                b[1] += 1
                if matches(v, dp, pool):
                    resolved += 1
                    b[0] += 1
                elif any(round(x, dp) == round(v, dp) and (sc in ("*", rel))
                         for x, sc, _ in exempt):
                    exempted += 1
                else:
                    unresolved.append((rel, i, lit))

    print(f"\n  D50 gate — {repo.name}, {len(docs)} document(s) in scope "
          f"({', '.join(globs)} from {src})")
    print(f"  {len(pool)} value(s) in the resolution pool "
          f"({len(emitted_values())} emitted, {len(claim_values())} from the claim ledger)")
    if not EXEMPTIONS.exists():
        print(f"  no exemption table yet ({EXEMPTIONS.relative_to(MODEL_REPO)}) — every "
              "threshold and approximation will read as unresolved")
    print(f"\n  resolved   {resolved}    <- NOT a coverage figure. See the split.")
    for k in ("integer", "1-2 dp", ">=3 dp"):
        r, n = by_dp[k]
        if n:
            print(f"      {k:9} {r:5}/{n:<5} = {100 * r / n:4.1f}%")
    print("      the integer row is not evidence of anything. Exact integer matching removed")
    print("      the rounding window (56.5% -> 49.3% when measured), but small integers collide")
    print("      by nature: a chromosome number matches a count, a year matches an n. Value")
    print("      matching without context cannot separate them. Quote UNRESOLVED, not resolved.")
    print(f"  exempt     {exempted}")
    print(f"  UNRESOLVED {len(unresolved)}")
    if unresolved:
        print("\n  a number stated in a shared document with no filed result behind it:")
        for rel, i, lit in unresolved[:25]:
            print(f"    {rel}:{i}  {lit}")
        if len(unresolved) > 25:
            print(f"    … and {len(unresolved) - 25} more")
    print("\n  REPORTING ONLY — this exits 0 by design. See the docstring.\n")
    if a.strict and unresolved:
        sys.exit(1)


if __name__ == "__main__":
    main()
