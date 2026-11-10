# R120 — checkpoint-only native/R114 selection

**RESEARCH_ONLY. Not production-admitted. R93 remains HOLD.**

R118-E's already-consumed results suggest that R114 harms the unsigned checkpoints
while helping some signed-checkpoint overlap configurations. This motivated
**one preregistered, zero-new-parameter candidate**:

- `STATE8_SIGNED8`: run frozen `R114` correction (unchanged).
- `STATE8`, `STATE8_TF1`: run frozen `NATIVE` HCA (unchanged).

This is **not** a dynamic, teacher-dependent or overlap-dependent router. It is
one static controller choice at engine initialization, using an authenticated
checkpoint identity. `policy.create_engine` uses the R119 default exact guarded
kernel with that frozen controller choice. No new C kernels or HCA weights.

`PROTOCOL_R120.json` must be committed before any fresh-seed evaluation, with
source identities pinned. The `qualify.py` runner reuses R118-E's exact numeric
execution for the two existing controls and derives the selected candidate from
the matching full 20-step trajectories (not from endpoint arithmetic), which is
valid because the policy never changes within a trajectory. It changes only
fresh evaluation seeds and structure seeds, both checked disjoint from history.

To evaluate after commit, from repository root, with dependencies installed:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m research.yang_mills_hca.r120_checkpoint_policy.qualify --out /dev/shm/r120_result.json
```

Existing R118-E is **discovery**, not a qualification holdout. R120 is a fresh
seed/structure replication with the same topology classes, not a proof on unseen
topologies or of improved physical/semantic accuracy. No teacher state is read
by controller selection. Any single successful result is not clearance of R93.

Original mathematical Yang–Mills provenance remains [creelie/4Y-M](https://github.com/creelie/4Y-M), cited by `../PROVENANCE.md`.
