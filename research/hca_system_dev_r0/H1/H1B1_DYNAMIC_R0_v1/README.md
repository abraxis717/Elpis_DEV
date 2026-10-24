<!-- Publication adaptation: local host paths redacted; source SHA-256 in manifest. -->
# HCA SYSTEM DEV R0 — H1B1 source-conditioned dynamic task contract (v1)

**Purpose:** create, mechanically audit and baseline a narrowly defined dynamic, typed n-ary source-conditioning task in **TRAIN and TUNE only**. This is exploratory instrument development, not scientific adjudication, not native B3 training, and not qualification.

## Single Linux/Konsole command

After placing `HCA_H1B1_DYNAMIC_R0_v1.zip` in `[LOCAL_HOST_PATH_REDACTED]`, execute:

```bash
unzip -n -q [LOCAL_HOST_PATH_REDACTED] -d [LOCAL_HOST_PATH_REDACTED] && python3 [LOCAL_HOST_PATH_REDACTED]
```

Persistent package source/evidence live under `[LOCAL_HOST_PATH_REDACTED]`. `[LOCAL_HOST_PATH_REDACTED]` receives only the transient handoff ZIP. No `set -e`, no `set -u`, no `pipefail`, no external dependency, no model provider and no network access. Python 3.9+ standard library only. The child runner exits nonzero on HOLD without terminating Konsole.

## Input authority (fail closed)

* Canonical Git HEAD `2c7588733180d1ab2e8e978cf277c30f9e4ab371`, zero tracked changes.
* Closed H1A `RESULT.json` SHA-256 `379c7c7663b1f28e208f1a7eb580b420699f639e14f910d879076e88c01e1acc`.
* H1B0 read-only census report SHA-256 `2d78a8ffb7fadc187b34131d85886590bf75a2de33450801d470be5db4f3fbb8`.
* H1A generator, auditor, baseline, and mutation files **hashed against hashes reported by H1B0**. The runner inventories definitions and packages only these four specific files inside a host-local review ZIP for later inspection. Matching hashes **do not constitute an independent semantic code review**.

The run refuses to rewrite historical H1A or H1B0. It writes a unique new H1B1 run under `runs/`. For each run, stdout includes mutation outcomes, baseline counts, important SHA-256s, and clear disposition. `H1A_SOURCE_REVIEW_LOCAL_ONLY.zip` is a host-local, optional user-uploadable source capsule; it is not transmitted automatically and must be reviewed before experimental authority is frozen.

## Exact scope and exclusions

The source/evidence are instrument prototypes; TUNE labels are deliberately generated, consumed, and published locally for adversarial inspection, **not blind**. There is no sealed DEV generation in this package. No production HCA integration, model training, formal qualification, native ABI modification, admission authority, or semantic claim. H1A HOLD remains unchanged.

## Results and interpretation

TRAIN: 8 disjoint worlds, 256 probes in 128 pairs. TUNE: 4 disjoint worlds, 128 probes in 64 pairs. Each pair type contributes 4 pairs/world. Probe identities are opaque, separate from generator rule labels. Per-world support examples expose enough visible evidence to identify the hidden rule for a properly informed symbolic competitor.

The exact support-conditioned symbolic algorithm has full access to the same visible support, graph roles, source events, and query data as a future B3. Perfect accuracy is expected by design and, if observed, **does not demonstrate HCA utility**. H1B2 should test whether an honestly incremental method improves *total online cost at parity quality* against recomputation/B0I and a capacity-matched typed factor-node learner; if no such opportunity remains, maintain scientific HOLD or independently version a new task. No accuracy-only superiority claims.

## Local verification

For package test adequacy (synthetic fixtures only):

```bash
python3 [LOCAL_HOST_PATH_REDACTED] --self-test
```

See `PROTOCOL.md` for precisely defined task, design risks, and next-phase acceptance criteria.
