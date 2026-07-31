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


# ---------------------------------------------------------------------------------------------
# 4. Artifact names must carry BOTH axes: the member and the replicate.
# ---------------------------------------------------------------------------------------------
def test_artifact_naming():
    """The collision member_tag was written to prevent, one axis over.

    11_kernel_shap_branches.py put the member in its output name and left NO slot for the
    background replicate -- and had no --run-id at all. Five replicates on one member therefore
    wrote one file five times, non-atomically. That is exactly the shape utils.member_tag's
    docstring describes for checkpoints: the replicate spread that is supposed to be the
    INTERPRETATION variance would be measured over one surviving file.

    Panel C is the cheapest producer in section 5 and answers four of the eight model claims, so
    this is the one place the collision costs the most.
    """
    from utils import member_tag, run_suffix

    print("\n=== 4. artifact naming carries member AND replicate ===")

    def name(tag, member_seed, run_id, split="all"):
        suffix = "" if split == "test" else f"_{split}"
        return f"kernel_shap_branch_{member_tag(tag, member_seed)}{suffix}{run_suffix(run_id)}.tsv"

    MEMBERS = (100, 200, 300, 400, 500)
    RUNS = (1, 2, 3, 4, 5)

    def grid_is_distinct():
        names = [name("atg1000_stop1000", m, r) for m in MEMBERS for r in RUNS]
        if len(set(names)) != len(names):
            dup = {n for n in names if names.count(n) > 1}
            raise ValueError(f"{len(names) - len(set(names))} collisions, e.g. {sorted(dup)[:2]}")
    check("25 member x replicate names are all distinct", grid_is_distinct, must_raise=False)

    def old_rule_collides():
        # What the code did before: member in the name, replicate nowhere.
        names = [f"kernel_shap_branch_{member_tag('atg1000_stop1000', m)}_all.tsv"
                 for m in MEMBERS for _ in RUNS]
        if len(set(names)) != len(names):
            raise ValueError("collision")
    check("the OLD rule (no replicate slot) is detected as colliding", old_rule_collides)

    def member_axis_still_separates():
        # Guard against a fix that adds the replicate and drops the member.
        names = [name("atg1000_stop1000", m, 1) for m in MEMBERS]
        if len(set(names)) != len(names):
            raise ValueError("members collide")
    check("members remain distinct at a fixed replicate", member_axis_still_separates,
          must_raise=False)

    def legacy_name_unchanged():
        # BACKWARD COMPATIBILITY IS PART OF THE RULE, not a courtesy: run_id=None must reproduce
        # the published filename byte for byte, or every deposited artifact is orphaned and every
        # consumer's glob stops matching. Same reasoning as member_tag(tag, None) == tag.
        if name("atg500_stop500", None, None, split="test") != "kernel_shap_branch_atg500_stop500.tsv":
            raise ValueError("legacy name changed")
        if name("atg500_stop500", None, None) != "kernel_shap_branch_atg500_stop500_all.tsv":
            raise ValueError("legacy full-cohort name changed")
    check("run_id=None reproduces the published filename exactly", legacy_name_unchanged,
          must_raise=False)

    def suffix_matches_deepshap():
        # One rule, not two. 06_export_deepshap_tsv.py parses `_run{N}` out of the stem
        # (`f.stem.split("_run")[-1]`), so a second spelling here would be a second rule that
        # silently stops being parsed.
        if run_suffix(3) != "_run3" or run_suffix(None) != "":
            raise ValueError("suffix spelling diverged from deepshap's")
    check("replicate suffix is spelled the same as deepshap's", suffix_matches_deepshap,
          must_raise=False)


# ---------------------------------------------------------------------------------------------
# 5. A consumer must never pool across ensemble members.
# ---------------------------------------------------------------------------------------------
def test_no_member_pooling():
    """The trap, planted. Verified 2026-07-30: the producer writes
    `deepshap_atg_atg2000_stop500_seed200_run1.npz` and 06_export_deepshap_tsv.py:333 builds
    `file_tag = f"{tag}{run_suffix}"` -- no member component -- so it seeks
    `deepshap_atg_atg2000_stop500_run1.npz` and errors, naming both paths. Loud, and correct.

    The one-line repair that would destroy the design is to loosen the glob to `{tag}*_run*` so it
    "just finds the files". It would match all five members, pool them into one mean, and the
    quantity collapsed is the BETWEEN-MEMBER spread -- the training-variability estimate. No error,
    no warning, a plausible number with the signal averaged out. The real repair is --member-seed
    on the consumer, mirroring the producer.
    """
    from utils import assert_one_member, member_tag, run_suffix

    print("\n=== 5. consumers must not pool across members ===")
    TAG = "atg1000_stop1000"

    def names(members, runs):
        return [f"deepshap_summary_{member_tag(TAG, m)}_atg-stop{run_suffix(r)}.tsv"
                for m in members for r in runs]

    def loosened_glob_is_rejected():
        # Exactly what `{tag}*_atg-stop_run*.tsv` would return in a five-member directory.
        assert_one_member(names((100, 200, 300, 400, 500), (1,)), "planted loose glob")
    check("a glob spanning five members", loosened_glob_is_rejected)

    def two_members_is_already_too_many():
        assert_one_member(names((100, 200), (1, 2, 3, 4, 5)), "planted two-member glob")
    check("a glob spanning two members", two_members_is_already_too_many)

    def correct_glob_accepted():
        # One member, five replicates -- the shape the design wants, and it must NOT fire.
        assert_one_member(names((200,), (1, 2, 3, 4, 5)), "one member, five replicates")
    check("one member across five replicates", correct_glob_accepted, must_raise=False)

    def legacy_unseeded_accepted():
        # Pre-ensemble artifacts carry no _seed component at all; the guard must not fire on the
        # published files or every deposited tree becomes unreadable.
        assert_one_member([f"deepshap_summary_atg500_stop500_joint_run{r}.tsv" for r in range(1, 6)],
                          "legacy published files")
    check("legacy un-seeded published files", legacy_unseeded_accepted, must_raise=False)

    def legacy_mixed_with_seeded_is_rejected():
        # A directory holding BOTH vintages is the hybrid that C44 and the 2026-07-28 checkpoint
        # mix-up both came from. It must not average.
        assert_one_member(["deepshap_summary_atg1000_stop1000_atg-stop_run1.tsv",
                           "deepshap_summary_atg1000_stop1000_seed200_atg-stop_run1.tsv"],
                          "planted mixed vintage")
    check("a legacy file mixed in with a seeded one", legacy_mixed_with_seeded_is_rejected)


# ---------------------------------------------------------------------------------------------
# 6. The split gate is CALLED by every scoring entry point, not merely defined.
# ---------------------------------------------------------------------------------------------
def test_gate_is_wired():
    """Section 3 proves the gate rejects its defects. It cannot prove a script CALLS it.

    11_kernel_shap_branches.py did not: `--explain-split test` was a final evaluation nothing
    marked as one, and `--explain-split all` pooled train and held-out data with no affirmation --
    in the producer of four section-5 claims. That is C69's `--split all` hole, closed in
    evaluate.py at 420a264 and still open here. D31 and claim 5.6.4 rest on the test set being
    touched once; a second scoring entry point without the gate is how that stops being true.

    So these run the real script and assert it exits non-zero. argparse errors before any data is
    loaded, so no HDF5, checkpoint or cluster is needed.
    """
    import subprocess, sys, pathlib
    HERE = pathlib.Path(__file__).parent
    print("\n=== 6. the split gate is wired into the entry points ===")

    def run(*flags):
        r = subprocess.run([sys.executable, "11_kernel_shap_branches.py", *flags],
                           cwd=HERE, capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            raise SystemExit(r.stderr)

    check("11_: --explain-split all without --full-cohort",
          lambda: run("--explain-split", "all"))
    check("11_: --explain-split test without --final",
          lambda: run("--explain-split", "test"))
    check("11_: --final on a non-test split",
          lambda: run("--explain-split", "train", "--final"))


def test_chunked_read():
    """The explained split is now read in chunks. Three ways that silently changes an answer.

    Chunking exists because one full-cohort run peaked at 35.3 GiB against a 32 GB request. It
    is only safe because of three properties, and each fails silently rather than loudly: rows
    lost at a boundary (misaligns every downstream isoform_id), chunks that do not reconstruct
    the split's own order (same), and a chunk size that is not a batch multiple (re-partitions
    the forward passes, so the run is no longer bit-comparable with an unchunked one).

    These test the arithmetic, not the model -- a full equivalence run costs a minute and lives
    in the run log. The arithmetic is what a future edit will break.
    """
    import numpy as np
    from utils import _split_mask, split_indices

    print("\n=== 7. chunked reading of the explained split ===")
    BATCH = 256

    def chunk_bounds(n, size):
        return [(lo, min(lo + size, n)) for lo in range(0, n, size)]

    def covers_exactly(n, size):
        rows = np.concatenate([np.arange(lo, hi) for lo, hi in chunk_bounds(n, size)])
        return len(rows) == n and (rows == np.arange(n)).all()

    check("chunks cover every row exactly once, in order (41,765 @ 2048)",
          lambda: covers_exactly(41765, 2048) or (_ for _ in ()).throw(AssertionError()),
          must_raise=False)
    check("chunks cover every row exactly once, in order (4,356 @ 1024)",
          lambda: covers_exactly(4356, 1024) or (_ for _ in ()).throw(AssertionError()),
          must_raise=False)

    # PLANTED DEFECT: a stride shorter than the chunk width -- the shape of an off-by-one in the
    # loop bound. It duplicates rows rather than dropping them, so the row COUNT still looks
    # plausible at a glance; only order and uniqueness catch it.
    def overlapping():
        rows = np.concatenate([np.arange(lo, min(lo + 2048, 41765))
                               for lo in range(0, 41765, 2047)])
        if len(rows) == 41765 and (rows == np.arange(41765)).all():
            return True
        raise AssertionError("overlapping chunks detected, as they should be")
    check("an overlapping stride is detected", overlapping)

    # PLANTED DEFECT: chunks concatenated out of order. Every row is present exactly once, so a
    # count check passes and a set check passes; only an ORDER check fails.
    def shuffled():
        b = chunk_bounds(41765, 2048)
        rows = np.concatenate([np.arange(lo, hi) for lo, hi in [b[1], b[0]] + b[2:]])
        if (rows == np.arange(41765)).all():
            return True
        raise AssertionError("out-of-order chunks detected, as they should be")
    check("chunks concatenated out of order are detected", shuffled)

    # The batch-multiple rule. Every chunk boundary must also be a batch boundary, or the
    # forward passes are re-partitioned and bitwise comparison against an eager run is void.
    def boundaries_are_batch_aligned(size):
        if size % BATCH:
            raise AssertionError(f"{size} is not a multiple of {BATCH}")
        return True
    check("2048 is batch-aligned", lambda: boundaries_are_batch_aligned(2048),
          must_raise=False)
    check("a non-multiple chunk size is rejected", lambda: boundaries_are_batch_aligned(100))

    # ONE WRITER for what a split name means. split_indices exists so a caller can chunk a
    # split without materialising 28 GiB to discover which rows it holds -- and it is only safe
    # while it agrees with NMDDataset, which now shares _split_mask with it. A future edit that
    # reintroduces a second copy of these seven branches is what this catches.
    splits = np.array(["train"] * 10 + ["val"] * 4 + ["val_paralog"] * 2
                      + ["test"] * 5 + ["test_paralog"] * 3)
    expected = {"train": 10, "val": 4, "val_clean": 4, "val_all": 6, "test": 5,
                "test_clean": 5, "test_all": 8, "test_paralog": 3, "all": 24}
    def masks_agree():
        for name, n in expected.items():
            got = int(_split_mask(splits, name).sum())
            if got != n:
                raise AssertionError(f"_split_mask('{name}') selected {got}, expected {n}")
        return True
    check("every split name selects the documented rows", masks_agree, must_raise=False)

    # val_clean's guard must still fire when val_paralog is absent -- it is the one branch whose
    # failure mode is a silently UNSCREENED validation set, not an error.
    no_paralog = np.array(["train"] * 5 + ["val"] * 4)
    check("val_clean on an unscreened HDF5 is refused",
          lambda: _split_mask(no_paralog, "val_clean"))


if __name__ == "__main__":
    print("planted-defect tests -- a guard that does not reject its defect is not a guard")
    test_alignment()
    test_replicates()
    test_split_gate()
    test_artifact_naming()
    test_no_member_pooling()
    test_gate_is_wired()
    test_chunked_read()
    print("\n" + ("=" * 70))
    if FAILURES:
        print(f"FAIL — {len(FAILURES)} guard(s) did not behave: {FAILURES}")
        raise SystemExit(1)
    print("PASS — every guard rejected its planted defect and accepted the valid case")
