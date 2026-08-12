# Methods

**The methods for this model are the paper's Supplemental Methods, section "Deep Learning Model".**
This file is a pointer and states no methods of its own.

This repository used to carry a parallel 793-line methods document. It described the model in more
detail than the paper did, and it drifted: it labeled the superseded 500 nt window configuration as
the published one, carried the six-cell-type lineage the manuscript does not use, and reported a
dataset of 39,938 isoforms where the deposited model trains and scores on 41,776. Two documents
describing one model is how that happens, so there is now one.

The Supplemental Methods cover the split assignment, priority ORF selection and K=5, the 1000 nt
start and stop windows and all nine channels, the architecture and aggregation, the training
hyperparameters, the twelve-configuration window sweep and how 1000/1000 was selected from among the
statistically equivalent leaders, the DeepSHAP and KernelSHAP procedures, and the benchmark against
prior NMD predictors.

For what this repository contains and how to run it, see [`README.md`](README.md) — the model's
configuration and measured performance, the dataset and its splits, the build order, and how the
inputs are regenerated.

Before changing the architecture or retraining, read
[`RETRAIN_ARCHITECTURE_CHANGES.md`](RETRAIN_ARCHITECTURE_CHANGES.md), which is a record of
design consequences found by interpretation work and is not superseded by anything in the paper.

The prior document remains in git history if a derivation is ever needed:

```
git log --follow -- METHODS.md
git show <commit>^:METHODS.md
```
