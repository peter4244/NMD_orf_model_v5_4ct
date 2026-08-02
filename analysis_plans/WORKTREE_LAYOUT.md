# Two worktrees, one repository

Set up 2026-08-01 on Pete's call, after three of the interpretability window's files landed under
the model window's commit messages in a single day.

## The problem it solves

Both windows worked in one checkout. Either window running `git add -A` or `git commit -a` stages
whatever the *other* window has in flight, so the other's work is committed under a message that
does not describe it. It happened three times on 2026-08-01 — `probe_bank_floor_chunk_invariance.py`
into `04629e2`, `probe_decay_head_beyond_ejc.py` into `4702c50`, `analysis_ism_regions.py` into
`7971931` — and once the anchor fix was split across two commits with half inside someone else's.

So far that has only cost provenance, which matters here because the project's discipline is that
every value traces to its producer. The reason to fix it now is the failure it has *not* caused
yet: with both windows editing the same file, the same race silently reverts an edit. The file
still runs, the tests still pass, and nobody notices.

## The layout

| | path | branch | holds |
|---|---|---|---|
| model window | `NMD_orf_model_v5_4ct` | `master` | the data — tensor, checkpoints, pool, ISM outputs |
| interpretability | `NMD_orf_model_v5_4ct_interp` | `interp` | tracked files only, data by symlink |

The model window did not move, deliberately: it has cluster jobs running and sync paths that name
the main checkout, and all 1.5 GB of untracked data lives there.

## The data symlinks, and why they are necessary

**`git worktree` duplicates tracked files only.** Tracked content here is 913 KiB; the data the
analysis reads is 1.5 GB and none of it is in git. Without links the worktree looks complete and
every script fails on a missing path.

Fully untracked directories are linked whole:

    results_tensor_v6  results_interp_all  results_4ct  results_4ct_sweep

`results_ism_v6` and `results_pool_v6` contain *tracked* provenance JSONs, so linking the directory
would shadow them and git would report them deleted. Their untracked data files are linked
individually instead:

    results_ism_v6/{discovery_confirmation_split,gencode_candidate_flags,ism_subset}.tsv
    results_pool_v6/{orf_pool,orf_pool_record}.tsv

**Both windows therefore read and write the same data.** The worktree separates *code and index*,
not results. Two builds writing one shard directory would still collide.

## The exclude entry

The four directory symlinks are listed in `.git/info/exclude`. They cannot be covered by
`.gitignore`: its patterns end in `/`, which matches a directory and not a symlink to one. That
file is **shared across worktrees** — git has no per-worktree exclude — but the entries are
redundant in the main checkout, where the real directories are already matched by `.gitignore`
lines 20–28. Verified rather than assumed.

## Working with it

    git -C NMD_orf_model_v5_4ct_interp merge master     # take the other window's work
    git -C NMD_orf_model_v5_4ct merge interp            # publish this window's

**Merge often, in both directions.** The cost of this arrangement is that neither window sees the
other's new files until a merge, and on 2026-08-01 each window repeatedly used the other's code
within minutes of it being written. Frequent merges keep that; infrequent merges trade one failure
mode for a worse one.

`analysis_plans/analysis_ism_regions.py` and `qc_ism_banks.py` both consume the banks and are the
likeliest conflict. They are owned by the interpretability and model windows respectively; edit
your own and merge rather than reaching across.
