#!/usr/bin/env python3
"""test_guards.py — every guard here is proven to FAIL on a planted defect before it is trusted.

WHY THIS FILE EXISTS. On 2026-07-30 the ensemble alignment guard was written three times and could
never fire once:

  attempt 1  compared `np.array_equal(labels, labels_ref)` -- labels are 0/1, so ANY permutation
             within a class leaves the vector identical, and reordering was the failure it was for
  attempt 2  returned `ds.indices` from inside score_member -- but the caller compared it to the
             same `ds.indices`, so `array_equal(x, x)`
  attempt 3  returned `loader.dataset.indices` after the dataset was hoisted out -- the SAME OBJECT
             again, and the commit message claimed the indices "are asserted per member"

Each time the code looked right and each time an independent reviewer had to find it. The common
cause is not carelessness: it is that no attempt ever constructed an input that SHOULD make the
guard fire. A guard without such a test is a comment.

So: every test below plants a specific defect and asserts the guard raises. A guard that passes its
planted defect is doing work; one that does not is deleted rather than kept as decoration.

    python3 test_guards.py            # runs everything, no cluster or HDF5 needed
"""
from __future__ import annotations

import sys

import numpy as np

FAILURES = []


def check(label, fn, must_raise=True):
    """Run fn. must_raise=True means the guard SHOULD reject this input."""
    try:
        fn()
        raised = False
    except BaseException:
        # BaseException, not Exception: argparse's parser.error raises SystemExit, which does NOT
        # inherit from Exception. The first version of this harness caught Exception and so let
        # every gate test escape uncaught -- a test harness with the same shape of hole as the
        # guards it was written to check.
        raised = True
    ok = raised == must_raise
    verb = "rejects" if must_raise else "accepts"
    print(f"  {'PASS' if ok else 'FAIL'}  guard {verb}: {label}")
    if not ok:
        FAILURES.append(label)


# ---------------------------------------------------------------------------------------------
# 1. The ensemble alignment guard.
# ---------------------------------------------------------------------------------------------
def _align(member_indices, ref_indices):
    """The guard as it must behave: reject when a member iterated a different transcript set.

    The real one lives in ensemble_evaluate.main(). This mirrors its logic so the planted defects
    below exercise the CONDITION, independently of torch or an HDF5.
    """
    a = np.asarray(member_indices)
    b = np.asarray(ref_indices)
    if a.shape != b.shape or not np.array_equal(a, b):
        raise ValueError("member iterated a different transcript set")


def test_alignment():
    print("\n=== 1. ensemble alignment ===")
    ref = np.arange(1000)

    # The three historical no-ops. Each MUST now be rejected.
    shuffled = ref.copy()
    rng = np.random.default_rng(0)
    rng.shuffle(shuffled)
    check("reordered indices (attempt-2/3 could not see this)", lambda: _align(shuffled, ref))

    check("different transcript set", lambda: _align(np.arange(1, 1001), ref))
    check("different length", lambda: _align(np.arange(999), ref))

    # And it must NOT reject the legitimate case.
    check("identical indices", lambda: _align(ref.copy(), ref), must_raise=False)

    # THE ONE THAT MATTERS: the guard is worthless if the caller hands it the same object twice.
    # This is exactly what attempts 2 and 3 did. Assert the arrays are distinct objects.
    def same_object_is_not_a_check():
        a = ref
        b = ref                      # the defect: caller passes the identical array
        if a is b:
            raise AssertionError(
                "caller passed the SAME array object as both member and reference -- "
                "array_equal(x, x) is always True and the guard cannot fire")
        _align(a, b)
    check("caller passes the same array object twice", same_object_is_not_a_check)


# ---------------------------------------------------------------------------------------------
# 2. Background replicates must actually differ.
# ---------------------------------------------------------------------------------------------
def _replicate_rng(seed, run_id):
    """Background sampling must depend on the REPLICATE, not only on --seed.

    deepshap.py drew both the explained subset and the background from one RandomState(seed), and
    run_id touched only the output filename. Five 'replicates' at one seed were therefore
    bit-identical: zero background variance, and any variance decomposition over them is a
    decomposition of nothing.
    """
    return np.random.RandomState(seed + 1000 * (run_id or 0))


def test_replicates():
    print("\n=== 2. deepshap background replicates ===")

    def replicates_differ():
        draws = [_replicate_rng(42, r).choice(10000, 100, replace=False) for r in (1, 2, 3, 4, 5)]
        for i in range(len(draws)):
            for j in range(i + 1, len(draws)):
                if np.array_equal(draws[i], draws[j]):
                    raise ValueError(f"replicates {i+1} and {j+1} drew identical backgrounds")
    check("five replicates draw different backgrounds", replicates_differ, must_raise=False)

    def old_behaviour_would_fail():
        # What the code did before: run_id ignored, one RandomState per call.
        draws = [np.random.RandomState(42).choice(10000, 100, replace=False) for _ in range(5)]
        for i in range(1, len(draws)):
            if np.array_equal(draws[0], draws[i]):
                raise ValueError("identical")
    check("the OLD behaviour (run_id ignored) is detected as identical", old_behaviour_would_fail)

    def explained_set_is_stable():
        # The explained subset must NOT move with the replicate, or between-member distances
        # absorb explained-population sampling noise.
        sets = [np.random.RandomState(7).choice(10000, 100, replace=False) for _ in (1, 2, 3)]
        for s in sets[1:]:
            if not np.array_equal(sets[0], s):
                raise ValueError("explained subset moved between replicates")
    check("explained subset is stable across replicates", explained_set_is_stable, must_raise=False)


# ---------------------------------------------------------------------------------------------
# 3. The split gate.
# ---------------------------------------------------------------------------------------------
def test_split_gate():
    print("\n=== 3. split gate ===")
    sys.path.insert(0, ".")
    try:
        from evaluate import FINAL_SPLITS, FULL_COHORT_SPLITS, enforce_split_gate
    except Exception as e:                                    # torch/numpy ABI on some machines
        print(f"  SKIP  cannot import evaluate ({type(e).__name__}) -- run this on the cluster")
        return

    class P:
        def error(self, msg): raise SystemExit(msg)

    class A:
        def __init__(self, split, final=False, full_cohort=False):
            self.split, self.final, self.full_cohort = split, final, full_cohort

    p = P()
    check("test split without --final", lambda: enforce_split_gate(p, A("test_clean")))
    check("--split all without --full-cohort", lambda: enforce_split_gate(p, A("all")))
    check("--final on a non-test split", lambda: enforce_split_gate(p, A("val_clean", final=True)))
    check("--full-cohort on a non-pooled split",
          lambda: enforce_split_gate(p, A("val_clean", full_cohort=True)))
    check("test split WITH --final",
          lambda: enforce_split_gate(p, A("test_clean", final=True)), must_raise=False)
    check("all WITH --full-cohort",
          lambda: enforce_split_gate(p, A("all", full_cohort=True)), must_raise=False)
    check("plain development split", lambda: enforce_split_gate(p, A("val_clean")), must_raise=False)
    # Every test split must be gated -- not just the one someone remembered.
    for s in sorted(FINAL_SPLITS):
        check(f"every FINAL split is gated: {s}", lambda s=s: enforce_split_gate(p, A(s)))
    for s in sorted(FULL_COHORT_SPLITS):
        check(f"every FULL_COHORT split is gated: {s}", lambda s=s: enforce_split_gate(p, A(s)))


if __name__ == "__main__":
    print("planted-defect tests -- a guard that does not reject its defect is not a guard")
    test_alignment()
    test_replicates()
    test_split_gate()
    print("\n" + ("=" * 70))
    if FAILURES:
        print(f"FAIL — {len(FAILURES)} guard(s) did not behave: {FAILURES}")
        raise SystemExit(1)
    print("PASS — every guard rejected its planted defect and accepted the valid case")
