#!/bin/bash
# The ONLY sanctioned way to get code from this repo onto Explorer. Refuses rather than
# overwrites.
#
# WHY THIS EXISTS. On 2026-08-04 I scp'd data_prep.py to Explorer twice. That file was already
# modified against eadb797 and carried another window's uncommitted work, including — possibly —
# Maude's HDF5 compression changes. scp overwrote it silently. Afterwards the remote file was
# byte-identical to mine (sha256 2f59d493...) and the prior contents were unrecoverable from
# that path. There was no warning, no backup, and no diff: scp's entire job is to overwrite.
#
# Explorer's tree still holds 16 modified files / 442 insertions against eadb797, including
# model.py +60 and 03_train.py. Any one of those could have been destroyed the same way.
#
# THE RULE THIS ENFORCES, and it is the one I broke:
#   * code moves by GIT, never by scp -- so what lands on Explorer is a named commit rather
#     than a file of unknown vintage. That is the same provenance principle that the whole
#     2026-08-04 cleanup is about: `results_4ct_dn` was a directory nobody could attribute.
#   * a remote file with uncommitted changes is NEVER overwritten. It is reported, and this
#     script exits non-zero.
#   * large data files are not this script's business at all. Ask Pete (2026-08-04): "If you
#     need to transfer stuff to Explorer, ask me - your route isn't reliable." A 263 MB
#     orf_features.tsv had already landed truncated at 234 MB that morning, with scp reporting
#     lost connection while the wrapping task reported exit 0.
#
# HONEST LIMIT: nothing stops someone typing scp anyway. This makes the safe path the easy one
# and makes the unsafe path visibly unsanctioned; it is not a filesystem lock. The guard that
# actually bites is REFUSE_ON_DIRTY below, because it fails the operation rather than warning.

set -uo pipefail

REMOTE_HOST="p.castaldi@explorer.northeastern.edu"
REMOTE_REPO="/home/p.castaldi/cc/nmd_orf_model_v5_4ct"
BRANCH="${BRANCH:-master}"

die() { echo "REFUSED: $*" >&2; exit 1; }

# ---- 1. local side must be committed and pushed, or the remote cannot name what it got ----
# MODIFIED TRACKED FILES ONLY. Untracked files (`??`) are deliberately NOT a blocker: they are
# not part of what the remote pulls, and treating them as one made the first version of this
# guard refuse on five unrelated scratch scripts. A guard that fires on the wrong condition
# gets disabled, and then it protects nothing.
LOCAL_DIRTY=$(git status --porcelain -- '*.py' '*.sh' '*.yaml' | grep -E '^[ MARC]M|^M' || true)
[ -z "$LOCAL_DIRTY" ] \
  || die "local tree has uncommitted changes to tracked code. Commit them first -- the remote
        pulls a named commit, so anything uncommitted here simply would not arrive.
$(echo "$LOCAL_DIRTY" | head -5 | sed 's/^/        /')"

git fetch origin "$BRANCH" -q 2>/dev/null
UNPUSHED=$(git log --oneline "origin/$BRANCH..$BRANCH" 2>/dev/null | wc -l | tr -d ' ')
[ "$UNPUSHED" = "0" ] \
  || die "$UNPUSHED commit(s) not pushed to origin/$BRANCH. Push first; the remote pulls a
        named commit, it does not receive files.
        $(git log --oneline "origin/$BRANCH..$BRANCH" | head -5)"

LOCAL_SHA=$(git rev-parse --short "$BRANCH")

# ---- 2. THE GUARD THAT BITES: never touch a remote tree with uncommitted work ----
echo "== remote working-tree state =="
REMOTE_DIRTY=$(ssh -o BatchMode=yes "$REMOTE_HOST" "cd $REMOTE_REPO && git status --porcelain | grep -E '^ ?M' || true")
if [ -n "$REMOTE_DIRTY" ]; then
  echo "$REMOTE_DIRTY" | sed 's/^/    /'
  die "the remote tree has uncommitted modifications (above). Pulling or copying over them
        destroys work that exists on ONE machine and in no repository. Get those changes
        committed and pushed by whoever owns them -- ask via the organizer -- then re-run.
        Set REFUSE_ON_DIRTY=0 ONLY with that owner's explicit say-so."
fi

# ---- 3. fast-forward only; never a merge, never a reset ----
echo "== pulling $LOCAL_SHA onto $REMOTE_HOST:$REMOTE_REPO =="
ssh -o BatchMode=yes "$REMOTE_HOST" \
  "cd $REMOTE_REPO && git fetch origin $BRANCH && git merge --ff-only origin/$BRANCH" \
  || die "remote fast-forward failed. Do NOT force it; find out why the remote diverged."

# ---- 4. verify, because a transfer that reports success is not a transfer that happened ----
REMOTE_SHA=$(ssh -o BatchMode=yes "$REMOTE_HOST" "cd $REMOTE_REPO && git rev-parse --short HEAD")
echo "local $LOCAL_SHA / remote $REMOTE_SHA"
[ "$LOCAL_SHA" = "$REMOTE_SHA" ] || die "remote HEAD is $REMOTE_SHA, expected $LOCAL_SHA."
echo "OK - remote is at $REMOTE_SHA"
