# Canonical HPC QA launcher tree

This directory is the source of truth for every EgoLife QA Slurm launcher and
worker. Make and review HPC QA changes here first.

The project-level `hpc/qa` directory is only a deployment mirror for cluster
layouts that submit from the project root. Synchronization is one-way:

```text
egolife_two_user_qa/multi-user/hpc/qa  ->  hpc/qa
```

Never use the project-level mirror to overwrite this directory. Project-level
helpers under `hpc/shared` remain runtime dependencies and are maintained
separately from the canonical QA launcher tree.
