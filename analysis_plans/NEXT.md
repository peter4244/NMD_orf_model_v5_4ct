# NEXT — interpretability window

*Overwrite this file. Do not date it, do not copy it forward, do not add a section.
Permanent facts live in `CLAUDE.md`; what is true lives in `NARRATIVE_HOW_THE_MODEL_DECIDES.md`.*

**On arrival:** `git merge master && git diff --name-only HEAD...master` — nothing is running,
worktrees clean as of 2026-08-02 evening.

**Do first — extend the initiation-context finding.** Perturbing the 25 bases immediately 5′
of the start codon moves the initiation head's logit more than anything else in its 900-base
window, **at the same position in all four ORF-length bands**, in a tile 99.6% filled
throughout. It is the only result of 2026-08-02 that got *stronger* under scrutiny; everything
else chased that day was the architecture describing itself.
*`FINDINGS_TILED_PERTURBATION_2026-08-02.md`, jobs 8900209 / 8900420.*

**The open question is position versus content.** The head responds at −13; that it reads
Kozak *content* is not established. Separating them is the work — and design around the ~42
position receptive field rather than discovering it, since a Kozak motif is ~10 nt.

**Needs an Explorer login** — checkpoint and tensor are both remote. Ask Pete.

**Next after that:** C15's 0.31 NMD/control routing gap — but expect a confound, not a
mechanism; three claims reversed under conditioning on 2026-08-02.
