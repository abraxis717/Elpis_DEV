# Operating Elpis (developer deployment)

This guide builds, installs, runs, stops and restarts the canonical Elpis runtime on one machine through its native
production entry point, the operator host `elpis` (`docs/NATIVE_HOST.md`). What it runs is real:
- the native ECS composite (K1 and typed memory, one ELPISEC1 state);
- RuntimeCore (execution authority, managed transactions, durability, fail-stop);
- the ECS_C authority (authenticated durable records with a protected rollback witness);
- the K1 Recovery R0 checkpoint store.

No interpreter runs. What it does **not** run:
- no text interface: no qualified ECS<->DSV4 codec map exists (`ECS_CODEC_UNQUALIFIED`);
- no recurrent memory learning;
- no contextual dynamics law.

QUERY and LEARN take numbers and return numbers; typed structural evidence is stored, never interpreted.

## Requirements

- **Platform:** Linux x86_64, with sealed memfd admission and `openat2`; Landlock ABI v5 for `--contained`.
- **Toolchain:** a C11 compiler, CMake >= 3.20 and a Rust toolchain (cargo), plus `pkg-config`, `libsqlite3-dev`
  and `libpcre2-dev`.
- **Python:** 3.11 or 3.12, only for the test suites and the offline reference interface below; nothing on the
  production path needs it.

After installation nothing needs the network.

## Build and install

```sh
git clone https://github.com/abraxis717/Elpis_DEV && cd Elpis_DEV
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
cmake --install build --prefix "$HOME/elpis" --component runtime        # bin/elpis, lib/elpis/ (seven libraries)
```

`--component testing` adds `bin/elpis_testing` and the testing RuntimeCore build. They exist for the crash suites
only, and run only a testing deployment. The release authority is `NONE_DEFERRED`: this is a developer installation,
not a published release.

Below, `elpis` stands for `$HOME/elpis/bin/elpis --home "$HOME/elpis-home"`.

## Deploy

```sh
elpis init --libraries "$HOME/elpis/lib/elpis" --source-commit "$(git rev-parse HEAD)"   # or --native-build "$PWD/build"
elpis create --seed 0      # --dim 6 --width 36 --max-rows 64 --memory-dim 64 --banks 2 --slot-capacity 64 ...
elpis status
```

**`init`:**
1. Copies the seven libraries into `home/lib` and pins them (size and SHA-256) in `home/catalog`.
2. Admits them once, ABI checks included.
3. Provisions the ECS_C authority: `home/state`, and `home/authority` unless `--authority-dir` names another
   protected directory.
4. Writes `home/host.cfg` last. It pins the catalog and the host binary itself.

Every later start does the following, and never uses the library source again:
- verifies the host binary and the catalog against their pins;
- checks every library's dependencies before loading it;
- admits each from a sealed copy.

A changed library, catalog, pin, ABI or host binary refuses every start (`DEPLOYMENT_PIN`, `DEPLOYMENT_ABI`,
`DEPLOYMENT_CONFIG`); nothing is ever re-pinned.

**`create`** makes the one lineage:
1. Writes `home/lineage.cfg` once.
2. Provisions the checkpoint store for the composite envelope.
3. Creates the composite from deterministic initial weights (a SHA-256 function of the seed, not a model:
   `elpis.deployment.initial-w.v1`, bit-identical to the Python law).
4. Anchors it durably.

The home must be an absolute path that is not a symlink, owned by you and not writable by group or others. `init`
refuses a non-empty home and `create` an existing lineage: nothing is overwritten. After `create` the home is a fixed
set of 17 files; every later write rewrites a fixed slot in place.

## Operate

Every operation prints one JSON object on one line.

| Exit | Meaning |
|---|---|
| `0` | every operation succeeded or was refused exactly as expected |
| `3` | an operation was refused, or an expectation was unmet |
| `4` | the deployment itself was refused at start |
| `2` | unusable input; nothing ran |

| command | what it does |
|---|---|
| `status` | deployment, pins, ABI, lineage, the authority, the recovery disposition, live vs durable identity, footprint, fuel ceiling, capability and non-claim manifests |
| `query --x '[[...6 numbers...], ...]'` | QUERY: `f_W(x)` of the live K1 state (read-only) |
| `learn --x .. --y .. --schedule '[[rows, steps], ...]' --rate R --grant TEXT [--publish]` | LEARN under an explicit grant and rate: the qualified K1 law, natively; volatile unless `--publish` |
| `source-found --label TEXT --source-authority N --scope-label TEXT [--per-turn N]` | found an ACTIVE evidence source in the ECS_C authority (an operator event) |
| `source-lifecycle --source-authority N --to ACTIVE\|DORMANT\|TERMINATED`, `source-version`, `source-reclaim` | the source's lifecycle, its version and scope (compare-and-set), the reclamation of a terminated row |
| `evidence --items JSON --grant TEXT --sources 1,2 [--publish]` | typed HACF evidence in one observe-only turn (no K1 learning), admitted only from the authority's ACTIVE sources within their per-turn bounds |
| `lookup --relation hyperedge:1 --target-role incidence:3 --cues JSON` | exact LOOKUP by net evidence |
| `slot --key HEX`, `live`, `counters`, `candidates` | inspection |
| `publish` | recoverable publication; anything but ACKNOWLEDGED is refused with its ticket |
| `reconcile --adopt ID` / `--discard ID` | operator reconciliation of exactly that checkpoint candidate |
| `authority`, `authority-reconcile --kind K --digest D` | the authority's report and entities; exactly the reconciliation it names |
| `text` | refused: `ECS_CODEC_UNQUALIFIED` |
| `session --script FILE` (or `-`) | many operations in one process |

`elpis --home DIR --contained <command>` runs any of them except `init` and `create` with the host restricted to its
deployment's six fixed slots (docs/NATIVE_HOST.md, "Containment").

**Volatile and durable.**
- A LEARN or an observation commits *volatile*: it lives in this process and is lost at exit unless published
  (`durability: "VOLATILE"` says so).
- `publish` makes the complete composite durable: the checkpoint slot is written, synced and read back verified, then
  one authenticated ECS_C record and its rollback witness name it (`ACKNOWLEDGED`). Every start resumes exactly the
  last acknowledged state.
- **Process death mid-publication** is resolved by the next start. If the process died after the checkpoint slot sync
  and before the record, the start reports `CANDIDATE_UNRESOLVED`: cognition is refused until you `reconcile
  --adopt` (take the captured state) or `--discard` (keep the authorized one). Nothing is adopted implicitly.
- **Authority attacks.** Copied-back authority records, a corrupt newest record, an unauthenticated substitute or a
  missing witness refuse the runtime (`RUNTIME_AUTHORITY`). `authority` names the one reconciliation the authority
  accepts, and only `authority-reconcile` with exactly that kind and digest reopens it.

**Session scripts** are a JSON array (or JSON lines) of operations named as above (`source_found`, `authority_reconcile`,
... with underscores). Any operation may declare `"expect": "CODE"`: it then passes only when refused with exactly that
code. `open` and `close` restart the lineage inside one process.

**Evidence items** have this form:

```
{relation, participants: [{node, role, ordinal?}], provenance, authority, polarity, tick}
```

- `relation` as `"hyperedge:N"`, roles as `"incidence:N"` (the HACF namespaces);
- `node` and `provenance` as 64 hex characters;
- `authority` the asserting source: structure admits 0..3, and it must be founded and ACTIVE in the authority;
- `polarity` your explicit `ASSERT`, `NEGATE` or `UNCERTAIN`: HACF defines none, so none is inferred;
- `tick` the memory's logical clock; it must not go backwards.

Structure computes and validates every identity. LOOKUP answers only while the relation's net evidence (asserts minus
negates) is positive: two sources that disagree are reported, never resolved.

**Limits.**
- **Fuel.** The fuel ceiling (`status.budget_ceiling`) bounds every operation before it starts: experiences, rows,
  steps, ECS work units, query rows, observations per turn.
- **Memory.** The slot capacity bounds the relations a lineage holds; a turn beyond it is refused whole.
- **Authority.** The authority's entity capacity (`init --capacity`, default 16) bounds its rows; founding past it is
  refused before any write.
- **Persistence.** The deployment keeps no logs or history.

## Smoke and acceptance

```sh
ctest --test-dir build -R native_host_smoke                    # init, create, a session, a resumption: no interpreter
cmake -DHOST="$HOME/elpis/bin/elpis" -DLIBRARIES="$HOME/elpis/lib/elpis" -DSCRATCH="$PWD/smoke" \
      -P native/runtime/host/tests/host_smoke.cmake            # the same, against the installed product
ELPIS_NATIVE_HOST_PREFIX="$HOME/elpis" python -m pytest tests/integration/test_native_host.py --lane all
```

The last command runs the full native suite against the installed binaries. That needs the testing component, and the
build tree for the differential against the Python reference. The suite covers:
- the process closure;
- the pins and every drift refusal;
- the crash matrix (SIGKILL at every durable step, every failure plan);
- the authority attacks and their reconciliations;
- containment;
- a fixed file set over long runs;
- equality with the offline Python reference.

CI runs it against an installed prefix, offline, in the Installed system acceptance job.

## The offline reference interface (Python)

`python -m elpis.runtime.cli` (with `elpis.runtime.deployment`, `elpis.runtime.acceptance` and
`elpis.runtime.release`) is the **offline reference interface**. It is not the production path. It operates the same
composite lineage over the standalone continuity backend, through the Python facade over RuntimeCore, and exists for:
- differential tests (the native host must answer exactly as it does);
- the K1 Recovery R0 qualification instruments, which are bound to their own commits;
- research.

Its acceptance run (`python -m elpis.runtime.acceptance --native-build build --work DIR --report acceptance.json`)
and its release bundle (`python -m elpis.runtime.release`) keep their documented behavior for that purpose.
Standalone continuity, and with it this interface's durable backend, is retired once the ECS_C successor is
qualified (docs/NATIVE_SPLIT_CENSUS.md, sections 9 and 11).

## Capabilities and nonclaims

- **Working and mechanically verified:**
  - numeric QUERY and LEARN (the K1 law is qualified only in its frozen regime);
  - typed evidence (observe-only), admitted by the authority's sources;
  - exact LOOKUP;
  - recoverable publication, restart, and checkpoint and authority reconciliation;
  - operator source events;
  - pinned native admission and containment.
- **Formally qualified:** see `docs/qualification/README.md` (statuses).
- **Unavailable:** text (`ECS_CODEC_UNQUALIFIED`).
- **Not enabled:** memory LEARN (unqualified); SLOW consolidation (not provided).
- **Research only:** HCA, and every laboratory under `research/` and `DEV/`.
- **Not claimed:** power-loss durability, language understanding, reasoning, general intelligence, autonomous
  evolution, hierarchical superiority.

An observation is evidence storage, never truth or authority. Pins authenticate bytes, not authors; the deployment
key authenticates records, not the operator.
