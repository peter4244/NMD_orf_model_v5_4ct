# superseded/

Files retained for provenance but **not** part of the canonical v2 pipeline. Nothing
here is required to train the model, run interpretability, or produce the manuscript's
Figure 5 / SF37–43 exports. Kept (not deleted) so the development record is recoverable.

## QC session logs (5-step report verification, April 2026)

`step1_findings.md` … `step5_findings.md`, `REPORT_AUDIT.md`, `report_review_log_4ct.md`
— dated outputs of the sequential 5-step verification protocol applied to
`orf_model_report_v5.Rmd` at specific historical commits. The re-runnable verifier
itself (`audit_report.R`) stays in the repo root; these are its point-in-time findings.

## Superseded SLURM wrappers

Redundant/older cluster job wrappers whose function is fully covered by the canonical
wrappers kept in the repo root:

| Archived | Superseded by |
|---|---|
| `slurm_train_v5.sh`, `slurm_train_v5_full.sh`, `slurm_train_v5_remaining.sh` | `slurm_train_4ct.sh` (best config) + `slurm_train_4ct_sweep.sh` / `_b2.sh` (window grid) |
| `slurm_deepshap_v5.sh` | old sampling (n-explain 2000, 100 background); replaced by `slurm_deepshap_joint.sh` (500 background) |
| `slurm_deepshap_seq_500bg.sh` | sequence-branch subset of `slurm_deepshap_joint.sh` |
| `slurm_deepshap_joint_orf1_4.sh`, `slurm_deepshap_joint_orf2_4_channing.sh` | per-ORF runs covered by `submit_orf.sh` (parametrized `--orf-index`); the `_channing` copy only changed the cluster working directory |

Verified before archiving: no exporter (`06`/`08`/`11`) or the report reads any output
unique to these wrappers.
