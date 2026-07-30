#!/usr/bin/env python3
"""lint_doc_citations.py — keep METHODS.md's `file.py:LINE` citations pointing at what they claim.

WHY THIS EXISTS. METHODS.md cites code by line number ~36 times. Line numbers move whenever
anything above them is edited -- including a comment-only edit -- and nothing checked them, so
they rotted silently and a reader following one landed on an unrelated statement.

Measured twice:
  * 3f96643, the commit that rewrote METHODS "corrected against an independent review", added
    +6 lines to model.py and +15 to 03_train.py and broke SIX of its own citations. Two in the
    same paragraph WERE updated -- the author fixed what they were editing and left the
    neighbours -- so this is not carelessness that more care would fix.
  * 420a264, mine, added ~91 lines to data_prep.py and broke two more while I was writing up
    the first six.

The ledger's G2 views have exactly this failure mode and are covered by tools/check.py. METHODS
had no equivalent. This is it. Rules in code, not in docs.

HOW IT WORKS. A lockfile records what each citation POINTED AT when it was last verified -- the
stripped text of the target line, not its number. On --check, a citation is stale when the text at
its line no longer matches. Because the lock stores content, this can also FIND where the anchor
moved to, so fixing is mechanical rather than a manual hunt.

    python3 lint_doc_citations.py            # verify; exit 1 if any citation is stale
    python3 lint_doc_citations.py --fix      # rewrite the docs to the anchors' new lines
    python3 lint_doc_citations.py --update   # re-lock after an INTENTIONAL re-pointing

--update is the deliberate act: it accepts whatever the citations currently say. Use it when a
citation is repointed on purpose, never to clear a red run you have not read.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS = ["METHODS.md", "README.md", "CLAUDE.md"]
LOCK = ROOT / "doc_citations.lock.tsv"

# `file.ext:123`, `file.ext:123-456`, and the COMMA form `file.ext:101,103` -- one token citing
# several lines. The comma form was a blind spot in the first version: the regex matched only the
# leading number, so `model.py:101,103` checked 101 and left 103 unverified, and --fix rewrote the
# start to produce the nonsense `model.py:105,103`. A linter with a blind spot is the failure it
# exists to catch, so the whole numeric spec is parsed and every component is an anchor.
#
# Bare `:212` continuations are still NOT matched -- they carry no filename and cannot be resolved
# mechanically. Write them as `file.py:212`.
CITE = re.compile(r'([A-Za-z0-9_/]+\.(?:py|R|Rmd|sh|yaml))[:](\d+(?:\s*-\s*\d+)?(?:,\d+)*)')


def parse_spec(spec):
    """'101,103' -> [(101, None), (103, None)];  '41-46' -> [(41, 46)]."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.append((int(a), int(b.strip())))
        else:
            out.append((int(part), None))
    return out


def render_spec(parts):
    return ",".join(f"{a}-{b}" if b is not None else f"{a}" for a, b in parts)


def read_lines(rel: str):
    p = ROOT / rel
    if not p.exists():
        return None
    return p.read_text(errors="replace").split("\n")


def citations():
    """Every (doc, doc_line, raw, file, start, end) in the documentation set."""
    for doc in DOCS:
        lines = read_lines(doc)
        if lines is None:
            continue
        for i, line in enumerate(lines, 1):
            for m in CITE.finditer(line):
                for start, end in parse_spec(m.group(2)):
                    yield doc, i, f"{m.group(1)}:{start}", m.group(1), start, end


def anchor_at(src, n):
    """The stripped text of line n, or None if out of range."""
    if src is None or n < 1 or n > len(src):
        return None
    return src[n - 1].strip()


def find_anchor(src, text, exclude=None):
    """Lines whose stripped text equals `text`. Blank text is not a usable fingerprint."""
    if not text:
        return []
    return [i for i, ln in enumerate(src, 1) if ln.strip() == text and i != exclude]


def load_lock():
    if not LOCK.exists():
        return {}
    out = {}
    for row in LOCK.read_text().split("\n")[1:]:
        if not row.strip():
            continue
        parts = row.split("\t")
        if len(parts) >= 4:
            out[(parts[0], parts[1], parts[2])] = parts[3]
    return out


def save_lock(entries):
    body = ["doc\tcite\tfile\tanchor_text"]
    for (doc, cite, f), text in sorted(entries.items()):
        body.append(f"{doc}\t{cite}\t{f}\t{text}")
    LOCK.write_text("\n".join(body) + "\n")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--update", action="store_true",
                   help="re-lock: accept the citations as they stand (deliberate re-pointing)")
    g.add_argument("--fix", action="store_true",
                   help="rewrite the docs so each citation points at its locked anchor's new line")
    args = ap.parse_args()

    lock = load_lock()
    srcs = {}
    stale, unlockable, moved, missing = [], [], [], []
    new_lock = {}

    for doc, dline, raw, f, start, end in citations():
        if f not in srcs:
            srcs[f] = read_lines(f)
        src = srcs[f]
        key = (doc, raw, f)

        if src is None:
            missing.append((doc, dline, raw, f"{f} is not in this repo"))
            continue

        cur = anchor_at(src, start)
        if cur is None:
            missing.append((doc, dline, raw, f"line {start} > {f} has {len(src)} lines"))
            continue
        if end is not None and (end < start or end > len(src)):
            missing.append((doc, dline, raw, f"range end {end} invalid for {f}"))
            continue

        if args.update:
            if not cur:
                unlockable.append((doc, dline, raw, "points at a BLANK line -- cite a real one"))
            new_lock[key] = cur
            continue

        locked = lock.get(key)
        if locked is None:
            # Not yet locked. Record it, but refuse to bless a blank target: a citation pointing
            # at whitespace is already wrong, and locking it would make the wrongness permanent.
            if not cur:
                unlockable.append((doc, dline, raw, "points at a BLANK line -- cite a real one"))
            else:
                new_lock[key] = cur
            continue

        if cur == locked:
            new_lock[key] = cur
            continue

        # Stale. Try to say where the anchor went, so the fix is mechanical.
        found = find_anchor(src, locked, exclude=start)
        stale.append((doc, dline, raw, f, start, locked, cur, found))
        if len(found) == 1:
            moved.append((doc, raw, f, start, found[0]))
        new_lock[key] = locked  # keep the old fingerprint until it is genuinely repointed

    if args.update:
        save_lock(new_lock)
        print(f"locked {len(new_lock)} citations across {len(DOCS)} docs -> {LOCK.name}")
        for doc, dline, raw, why in unlockable:
            print(f"  WARN  {doc}:{dline}  {raw}  {why}")
        return 1 if unlockable else 0

    if args.fix:
        if not moved:
            print("nothing to fix: no citation has a single unambiguous new location")
            return 0

        # BY BYTE OFFSET, RIGHT TO LEFT -- NOT BY str.replace (2026-07-30).
        #
        # The first version replaced citation strings sequentially and was wrong two ways, both
        # caught by planting a comment-only shift in model.py:
        #   1. COLLISION. Fixing `model.py:101` -> `model.py:105` put the literal text
        #      "model.py:105" into the document; the next citation to fix WAS `model.py:105`, so
        #      str.replace hit the line just rewritten. 35 citations became 34.
        #   2. RANGE ENDS WERE LEFT BEHIND. `model.py:209-214` became `model.py:213-214`, an
        #      inverted-ish range whose end still pointed at the pre-shift line.
        # Rewriting by offset makes each edit independent of every other, and the end moves by
        # the same delta as the start -- which is right because a block shifts as a unit.
        by_doc = {}
        for doc, raw, f, old, new in moved:
            by_doc.setdefault(doc, []).append((raw, f, old, new))

        n_fixed = 0
        for doc, items in by_doc.items():
            p = ROOT / doc
            text = p.read_text()
            targets = {(f, old): new for _, f, old, new in items}
            edits = []
            for m in CITE.finditer(text):
                f = m.group(1)
                parts = parse_spec(m.group(2))
                if not any((f, a) in targets for a, _ in parts):
                    continue
                # EVERY component of the spec is shifted independently. A range end moves by the
                # same delta as its own start, since a block shifts as a unit.
                newparts = []
                for a, b in parts:
                    if (f, a) in targets:
                        na = targets[(f, a)]
                        newparts.append((na, b + (na - a) if b is not None else None))
                    else:
                        newparts.append((a, b))
                repl = f"{f}:{render_spec(newparts)}"
                edits.append((m.start(), m.end(), m.group(0), repl))
            for s, e, old_txt, repl in reversed(edits):
                text = text[:s] + repl + text[e:]
            for _, _, old_txt, repl in edits:
                print(f"  {doc}: {old_txt} -> {repl}")
                n_fixed += 1
            p.write_text(text)

        print(f"\nrewrote {n_fixed} citation(s). Re-run without --fix to verify, then --update.")
        print("NOTE range ENDS were shifted by the same delta as their start, on the assumption "
              "the block moved as a unit. Check any range that spans an edited region.")
        return 0

    ok = len(new_lock) - len(stale)
    print(f"=== doc citation check ===\n  {ok}/{len(new_lock)} citations resolve to their anchor")

    for doc, dline, raw, why in missing:
        print(f"  BROKEN  {doc}:{dline}  {raw}  --  {why}")
    for doc, dline, raw, why in unlockable:
        print(f"  WARN    {doc}:{dline}  {raw}  --  {why}")
    for doc, dline, raw, f, start, locked, cur, found in stale:
        print(f"  STALE   {doc}:{dline}  {raw}")
        print(f"            expected: {locked[:88]}")
        print(f"            found   : {cur[:88] or '(blank line)'}")
        if len(found) == 1:
            print(f"            anchor moved to {f}:{found[0]} -- run --fix")
        elif found:
            print(f"            anchor appears at {f}:{found} -- ambiguous, fix by hand")
        else:
            print(f"            anchor text no longer present in {f} -- the code changed, not "
                  f"just its line number. Re-read the citation before repointing it.")

    if not stale and not missing and not unlockable:
        if not lock:
            save_lock(new_lock)
            print(f"  (no lockfile existed; wrote {LOCK.name} with {len(new_lock)} entries)")
        return 0
    print(f"\n  {len(stale)} stale, {len(missing)} broken, {len(unlockable)} unlockable")
    return 1


if __name__ == "__main__":
    sys.exit(main())
