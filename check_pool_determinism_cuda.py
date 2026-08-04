#!/usr/bin/env python3
"""check_pool_determinism_cuda.py — Job B of W16's scope. Does the pooling substitution
actually buy determinism ON THE BACKEND THAT MATTERS?

WHY THIS EXISTS RATHER THAN A CALL TO verify_pool_equivalence(). That function (model.py:265)
is the model repo's own check and it is CPU-ONLY: line 284 is `x = torch.randn(n, channels, L)`
with no device argument and no device parameter to pass. The substitution it validates exists
BECAUSE adaptive_max_pool2d_backward_cuda has no deterministic kernel (C69) — so the check runs
on the one backend where the problem cannot occur, while DeepSHAP, whose backward passes are
what section 5's SHAP values come from, runs on the one where it can. Parameterising the real
function is the right long-term fix and belongs to the model window; this script measures
without editing their file, and calls theirs too so the two are compared rather than swapped.

WHAT A PASS HERE LICENSES, precisely. That on this GPU, `x.max(dim=-1).values` and
`AdaptiveMaxPool1d(1)` agree bitwise forward, route gradients to the same element, and that the
max path is repeatable under torch.use_deterministic_algorithms(True) where the pooling path is
not. It does NOT license any claim about deepshap.py reproducing — that is Job A, and this
script exists so Job A's outcome is explained in advance rather than mysterious.

Run on a GPU node. Exit 0 = the substitution holds on CUDA. Exit 1 = it does not, and Job A's
comparison should be read in that light rather than as evidence about deepshap.py.
"""

import sys

import torch
import torch.nn as nn

LENGTHS = (1, 2, 7, 100, 125, 500)   # same set as model.py:265 — degenerate and non-multiple
N, CHANNELS, SEED = 64, 32, 0


def equivalence_on(device):
    """Forward bitwise equality and identical gradient routing, on one device.

    Mirrors model.py's assertions exactly, including comparing gradients as nonzero MASKS rather
    than by value — a tie broken differently is the failure being looked for, and comparing
    values would average it away.
    """
    torch.manual_seed(SEED)
    pool = nn.AdaptiveMaxPool1d(1).to(device)
    for L in LENGTHS:
        x = torch.randn(N, CHANNELS, L, device=device)

        a = pool(x).squeeze(-1)
        b = x.max(dim=-1).values
        if not torch.equal(a, b):
            print(f"  FAIL forward at length {L}: max abs diff "
                  f"{(a - b).abs().max().item():.3e}")
            return False

        xa = x.clone().requires_grad_(True)
        pool(xa).squeeze(-1).sum().backward()
        xb = x.clone().requires_grad_(True)
        xb.max(dim=-1).values.sum().backward()
        if not torch.equal(xa.grad != 0, xb.grad != 0):
            print(f"  FAIL gradient routing at length {L}")
            return False
    print(f"  ok  forward bitwise + gradient routing agree on {device}, "
          f"lengths {LENGTHS}")
    return True


def repeatable_under_deterministic_mode(device):
    """The load-bearing question: under deterministic mode, is max repeatable where pool is not?

    Two halves, and BOTH matter. If the pooling path does not fail here, the substitution is not
    buying anything on this torch/CUDA build and the code's stated reason is stale. If the max
    path is not bitwise repeatable, the substitution is not sufficient either.
    """
    torch.use_deterministic_algorithms(True)
    try:
        pool_failed = False
        torch.manual_seed(SEED)
        x = torch.randn(N, CHANNELS, 500, device=device)

        try:
            xp = x.clone().requires_grad_(True)
            nn.AdaptiveMaxPool1d(1).to(device)(xp).squeeze(-1).sum().backward()
        except RuntimeError as e:
            pool_failed = True
            print(f"  ok  pooling backward refuses under deterministic mode: "
                  f"{str(e).splitlines()[0][:110]}")
        if not pool_failed:
            print("  NOTE pooling backward did NOT raise under deterministic mode on this build "
                  "— the substitution may no longer be load-bearing here; not a failure of the "
                  "substitution, but the code's stated reason should be re-read against this.")

        grads = []
        for _ in range(2):
            xm = x.clone().requires_grad_(True)
            xm.max(dim=-1).values.sum().backward()
            grads.append(xm.grad.clone())
        if not torch.equal(grads[0], grads[1]):
            print("  FAIL max backward is NOT bitwise repeatable under deterministic mode")
            return False
        print("  ok  max backward is bitwise repeatable across two runs")
        return True
    finally:
        torch.use_deterministic_algorithms(False)


def main():
    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("NO GPU VISIBLE — this script proves nothing on CPU, which is the entire point "
              "of its existence. Run it on a GPU node.")
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}\n")

    ok = True

    print("[1] the model repo's own check, unmodified (CPU) — for comparison, not evidence")
    try:
        sys.path.insert(0, ".")
        from model import verify_pool_equivalence
        verify_pool_equivalence()
        print("  ok  verify_pool_equivalence() passes as written")
    except Exception as e:                                    # noqa: BLE001 — report, do not mask
        ok = False
        print(f"  FAIL verify_pool_equivalence() raised: {e}")

    print("\n[2] the same assertions on CUDA — the path DeepSHAP actually runs")
    ok &= equivalence_on("cuda")

    print("\n[3] repeatability under torch.use_deterministic_algorithms(True), on CUDA")
    ok &= repeatable_under_deterministic_mode("cuda")

    print("\n" + ("PASS — the substitution holds on CUDA" if ok else "FAIL — see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
