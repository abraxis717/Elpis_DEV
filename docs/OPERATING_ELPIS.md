# Operating Elpis (developer deployment)

This guide installs, runs, stops and restarts the canonical Elpis runtime on one machine. What it runs is real:
the native ECS (K1 + typed memory, one ELPISEC1 composite state), RuntimeCore (execution authority, managed
transactions, durability, fail-stop), continuity (the durable identity register) and the K1 Recovery R0 checkpoint
store, driven only through `elpis.runtime.Runtime`. What it does **not** run: there is no text interface (no
qualified ECS<->DSV4 codec map exists: `ECS_CODEC_UNQUALIFIED`), no recurrent memory learning, no contextual dynamics
law. QUERY and LEARN take numbers and return numbers; typed structural evidence is stored, never interpreted.

## Requirements

Linux x86_64 (sealed memfd admission, `openat2`), a C11 compiler, CMake >= 3.20, a Rust toolchain (cargo),
`pkg-config`, `libsqlite3-dev`, `libpcre2-dev`, Python 3.11 or 3.12. The Python package has **no** runtime
dependencies. After installation nothing needs the network.

## Build and install (the one path)

```sh
git clone https://github.com/abraxis717/Elpis_DEV && cd Elpis_DEV
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
python3 -m venv .venv && .venv/bin/python -m pip install .
```

The entry point is `python -m elpis.runtime.cli` (this development distribution deliberately claims no `elpis`
console command; its release authority is `NONE_DEFERRED`). Below, `elpis` stands for
`.venv/bin/python -m elpis.runtime.cli --home "$HOME/elpis-home"`.

## Deploy

```sh
elpis init --native-build "$PWD/build" --source-commit "$(git rev-parse HEAD)"
elpis create --seed 0          # --dim 6 --width 36 --max-rows 64 --memory-dim 64 --banks 2 --slot-capacity 64 ...
elpis status
```

`init` copies the deployable libraries out of the build into `home/lib`, pins them (size and SHA-256) in
`home/native-catalog.json`, admits them once (ABI checks included) and only then writes `home/elpis.json`, whose
`catalog_sha256` is the deployment's trust root. Every later start verifies the catalog against that pin and admits
every library from sealed bytes; the build tree is never used again (the acceptance run deletes it). A changed library,
catalog, pin or ABI refuses every start (`DEPLOYMENT_PIN`, `RUNTIME_UNPINNED`, `DEPLOYMENT_ABI`), never re-pins.

`create` makes the one lineage: it provisions the checkpoint store for the composite envelope, creates the composite
from deterministic initial weights (`elpis.runtime.deployment.initial_weights`: a SHA-256 function of the seed, not a
model) and anchors it durably. The home must be an absolute path that is not a symlink, owned by you and not
writable by group or others. `init` refuses a non-empty home and `create` an existing lineage: nothing is overwritten.

## Operate

Every command prints JSON (one object per operation) and exits `0` when every operation succeeded or was refused
exactly as expected, `3` when an operation was refused unexpectedly, `4` when the deployment itself was refused at
start, `2` on unusable input.

| command | what it does |
|---|---|
| `status` | deployment, pinned library digests, ABI, lineage, recovery disposition, live vs durable identity, footprint, fuel ceiling, capability and nonclaim manifests |
| `query --x '[[...6 numbers...], ...]'` | QUERY: `f_W(x)` of the live K1 state (read-only) |
| `learn --x .. --y .. --schedule '[[rows, steps], ...]' --rate R --grant TEXT [--publish]` | LEARN under an explicit authority (the grant): the qualified K1 law, natively; volatile unless `--publish` |
| `evidence --items JSON --grant TEXT --sources 1,2 [--publish]` | typed HACF evidence recorded by an observe-only turn: no K1 learning |
| `lookup --relation hyperedge:1 --target-role incidence:3 --cues JSON` | exact LOOKUP by net evidence |
| `slot --key HEX` | one memory slot and the hyperedge it was observed from |
| `publish` | recoverable publication of the live state |
| `candidates` | the recovery disposition (an unresolved candidate, if any) |
| `reconcile --adopt ID` / `--discard ID` | operator reconciliation of exactly that candidate |
| `session --script FILE` (or `-`) | many operations in one process |

**Volatile and durable.** A LEARN or an observation commits *volatile*: it lives in this process and is lost at exit
unless published (`durability: "VOLATILE"` says so). `publish` makes the complete composite durable: the checkpoint
slot is written, synced and read back verified, then continuity names it (`ACKNOWLEDGED`). Every start resumes
exactly the last acknowledged state. If a process dies inside a publication after the slot sync, the next start
reports `CANDIDATE_UNRESOLVED`: cognitive operations are refused (`LINEAGE_CANDIDATE_UNRESOLVED`) until you run
`reconcile --adopt` (take the captured state) or `--discard` (keep the authorized one). Nothing is adopted implicitly.

**Session scripts** are a JSON array (or JSON lines) of operations: `status`, `live`, `counters`, `query`, `learn`,
`evidence`, `lookup`, `slot`, `publish`, `candidates`, `reconcile`, `context_axiom`, `context_relation`,
`context_view`. Any operation may declare `"expect": "CODE"`: it then passes only when refused with exactly that code.
`learn` and `query` take `"consume_context": true` to record the coordinate of the session's current context view
(the view never conditions ECS). See `docs/examples/` (`demo-session.json`, `refusals.json`, `after-restart.json`).

**Evidence items** are `{relation, participants: [{node, role, ordinal?}], provenance, authority, polarity, tick}`:
`relation` as `"hyperedge:N"`, roles as `"incidence:N"` (the HACF namespaces), `node` and `provenance` as 64 hex
characters, `authority` the asserting source (structure admits 1..3), `polarity` your explicit `ASSERT`, `NEGATE` or
`UNCERTAIN` (HACF defines none, so none is inferred), `tick` the memory's logical clock (it must not go backwards).
Structure computes and validates every identity. LOOKUP answers only while the relation's net evidence (asserts minus
negates) is positive: two sources that disagree are reported, never resolved.

**Context** (`context_*`) is the session-local WARM lane: bounded, immutable, versioned views of admitted references
and candidate relations. It is never persisted, never waited on by a HOT operation and never an input to ECS.

**Limits.** The fuel ceiling (`status.budget_ceiling`) bounds every operation before it starts (experiences, rows,
steps, ECS work units, query rows, observations per turn); the memory's slot capacity bounds the relations a lineage
holds (a turn beyond it is refused whole: `ECS_REFUSED` / memory `CAPACITY`). The deployment keeps no logs or
history; its durable state is exactly continuity and the two checkpoint slots.

## Smoke and recovery test

```sh
.venv/bin/python -m elpis.runtime.acceptance --native-build "$PWD/build" --work "$PWD/acceptance" \
    --source-commit "$(git rev-parse HEAD)" --report acceptance.json
```

It deploys twice into `--work` (the production build, and the testing RuntimeCore build for crash injection), runs
the installed CLI as separate processes and checks 42 laws (pinned identity; QUERY, LEARN, evidence, LOOKUP, context
isolation, publication, restart, a second publication; SIGKILL at registered durability boundaries with the operator
reconciliation; and the refusals: no codec, QUERY-to-LEARN, unauthorized and unadmitted observations, invalid source,
memory capability, stale context, wrong ABI, wrong envelope, pin drift, budget, COLD on HOT, incompatible lineage,
existing destinations, unresolved candidate). It exits 0 only when all pass; the JSON report names each check.

## Release bundle

```sh
.venv/bin/python -m elpis.runtime.release --repo "$PWD" --native-build "$PWD/build" --out "$PWD/dist"
```

writes `dist/elpis-dev-<version>-<commit>/` with `MANIFEST.json` (source commit and tree, library digests,
toolchain, ABI and format declarations, dependencies, capability and nonclaim manifests, the commands),
`source.tar.gz`, `native-build/`, this guide, the examples and `SHA256SUMS`. Nothing is published.

## Capabilities and nonclaims

Working and mechanically verified: numeric QUERY and LEARN (the K1 law is qualified only in its frozen regime);
typed evidence (observe-only, ABI v5); exact LOOKUP; recoverable publication; restart; operator reconciliation;
pinned native admission; HOT residency refusals. Formally qualified: see `docs/qualification/README.md` (statuses).
Unavailable: text (`ECS_CODEC_UNQUALIFIED`). Not enabled: memory LEARN (unqualified), SLOW consolidation (not
provided). Research only: HCA and every laboratory under `research/` and `DEV/`. Not claimed: power-loss durability,
language understanding, reasoning, general intelligence, autonomous evolution, hierarchical superiority. An
observation is evidence storage, never truth or authority; pins authenticate bytes, not authors.
