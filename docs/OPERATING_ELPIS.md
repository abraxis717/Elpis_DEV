# Operating Elpis (developer deployment)

This guide builds, installs, runs, stops and restarts the canonical Elpis runtime on one machine, through its native
production entry point, the operator host `elpis` (`docs/NATIVE_HOST.md`). What it runs is real:
- the native ECS composite: K1 and typed memory, one ELPISEC1 state;
- RuntimeCore: execution authority, managed transactions and fail-stop;
- the volatile ECS_C session: entities, sources, the cognition lineage, the evolution authority.

Everything Elpis computes, learns or records lives in memory, and nothing is written to storage
(`docs/ELPIS_MISSION.md`, Amendment 3). A session ends with `close` or with the process, and a restart is a new
session at the lineage's initial state.

No interpreter runs. What it does **not** run:
- no text interface: no qualified ECS<->DSV4 codec map exists (`ECS_CODEC_UNQUALIFIED`);
- no recurrent memory learning;
- no contextual dynamics law.

QUERY and LEARN take numbers and return numbers. Typed structural evidence is stored, never interpreted.

## Requirements

- **Platform:** Linux x86_64, with sealed memfd admission, `openat2` and Landlock ABI v5 or later. Containment is
  mandatory: a kernel without Landlock v5 is refused.
- **Toolchain:** a C11 compiler, CMake >= 3.20 and a Rust toolchain (cargo), plus `pkg-config`, `libsqlite3-dev`
  and `libpcre2-dev`.
- **Python:** 3.11 or 3.12, only for the test suites and the offline reference interface below. Nothing on the
  production path needs it.
- **The host's side** (`docs/VOLATILE_RUNTIME_CENSUS.md`, section 6):
  - a read-only mount for the installed libraries;
  - swap off, or a memory-lock limit the host's memory fits in (the host locks its memory or refuses);
  - standard output and error on a pipe or a terminal.

After installation nothing needs the network.

## Build and install

```sh
git clone https://github.com/abraxis717/Elpis_DEV && cd Elpis_DEV
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
cmake --install build --prefix "$HOME/elpis" --component runtime        # bin/elpis, lib/elpis/ (six libraries)
```

The release authority is `NONE_DEFERRED`: this is a developer installation, not a published release. The installed
directory is only ever read. Mount it read-only if you can.

## Start: the catalog pin

There is no `init`, no home and nothing to create. Measure the installation once, and keep the pin it prints:

```sh
L="$HOME/elpis/lib/elpis"
"$HOME/elpis/bin/elpis" --libraries "$L" catalog | tee catalog.json   # the catalog and its SHA-256
PIN=...                                                                 # catalog_sha256 from catalog.json
```

`catalog` measures the host binary and every library from sealed, read-only copies. Every other command requires that
pin. A changed library or host binary refuses every start with `INSTALL_PIN`; nothing is ever re-pinned implicitly.
Run the host as an ordinary user: a privileged process is refused (`CONTAINMENT_REFUSED`).

Below, `elpis` stands for `$HOME/elpis/bin/elpis --libraries "$L" --catalog-sha256 "$PIN"`.

```sh
elpis status
elpis session --script docs/examples/demo-session.json
elpis session --script docs/examples/after-restart.json     # a new process: the initial state, nothing resumed
```

**The lineage** is not a file. Every session starts from a deterministic initial composite, built in memory from the
session options:
- `--dim 6 --width 36 --max-rows 64 --memory-dim 64 --banks 2 --slot-capacity 64 --codebook-capacity 128 --seed 0`
  (the defaults);
- initial weights that are a SHA-256 function of the seed, not a model (`elpis.deployment.initial-w.v1`,
  bit-identical to the Python law).

The same options always give the same initial identity. `--capacity N` (default 16) bounds the session's entities.
`--nonce-seed HEX` makes sessions reproducible across processes, for differential runs.

## Operate

Every operation prints one JSON object on one line.

| Exit | Meaning |
|---|---|
| `0` | every operation succeeded or was refused exactly as expected |
| `3` | an operation was refused, or an expectation was unmet |
| `4` | the host could not start: containment, the swap policy, admission or the pin refused; nothing ran |
| `2` | unusable input, or a retired command; nothing ran |

| command | what it does |
|---|---|
| `status` | the installation and its pins, `persistence: NONE`, the ABI, the containment established, the session, the live state, the footprint, the fuel ceiling, the capability and non-claim manifests |
| `query --x '[[...6 numbers...], ...]'` | QUERY: `f_W(x)` of the live K1 state (read-only) |
| `learn --x .. --y .. --schedule '[[rows, steps], ...]' --rate R --grant TEXT` | LEARN under an explicit grant and rate: the qualified K1 law, natively, volatile |
| `source-found --label TEXT --source-authority N --scope-label TEXT [--per-turn N]` | found an ACTIVE evidence source in the session (an operator event) |
| `source-lifecycle --source-authority N --to ACTIVE\|DORMANT\|TERMINATED`, `source-version`, `source-reclaim` | the source's lifecycle; its version and scope (compare-and-set); the reclamation of a terminated row |
| `evidence --items JSON --grant TEXT --sources 1,2` | typed HACF evidence in one observe-only turn (no K1 learning), admitted only from the session's ACTIVE sources, within their per-turn bounds |
| `lookup --relation hyperedge:1 --target-role incidence:3 --cues JSON` | exact LOOKUP by net evidence |
| `slot --key HEX`, `live`, `counters`, `authority` | inspection |
| `open`, `close` | start a session; end it, discarding everything it learned or recorded |
| `text` | refused: `ECS_CODEC_UNQUALIFIED` |
| `session --script FILE` (or `-`) | many operations in one process |

A one-shot command is its own process, so it runs in a fresh session. Founding a source and then using it needs one
`session` script.

**Volatile, by design.**
- A LEARN or an observation commits *volatile* (`durability: "VOLATILE"`). It lasts until the session ends.
- `close` reports what it discards (`discarded`), and the next operation starts a new session at the initial
  identity. A new process does the same.
- Nothing is published, checkpointed, recovered or resumed. To keep a result, keep the operations that produced it:
  a script replays them exactly.
- The retired v7 commands are refused with `OPERATION_RETIRED` (exit 2): `init`, `create`, `publish`, `reconcile`,
  `candidates`, `authority-reconcile`, `--home`, `--contained` and `--publish`.

**Session scripts** are a JSON array (or JSON lines) of operations named as above, with underscores
(`source_found`, ...). Any operation may declare `"expect": "CODE"`: it then passes only when refused with exactly
that code. `docs/examples` holds three:
- `demo-session.json`: found a source, QUERY, LEARN, evidence and LOOKUP, then `close`, and the new session's LOOKUP
  finds nothing;
- `after-restart.json`: the initial state in a new process;
- `refusals.json`: the boundary's refusals, each expected.

**Evidence items** have this form:

```
{relation, participants: [{node, role, ordinal?}], provenance, authority, polarity, tick}
```

- `relation` as `"hyperedge:N"`, roles as `"incidence:N"` (the HACF namespaces);
- `node` and `provenance` as 64 hex characters;
- `authority` the asserting source: structure admits 0..3, and it must be founded and ACTIVE in the session;
- `polarity` your explicit `ASSERT`, `NEGATE` or `UNCERTAIN`: HACF defines none, so none is inferred;
- `tick` the memory's logical clock; it must not go backwards.

Structure computes and validates every identity. LOOKUP answers only while the relation's net evidence (asserts minus
negates) is positive: two sources that disagree are reported, never resolved.

**Limits.**
- **Fuel.** The fuel ceiling (`status.budget_ceiling`) bounds every operation before it starts: experiences, rows,
  steps, ECS work units, query rows and observations per turn.
- **Memory.** The slot capacity bounds the relations a session holds; a turn beyond it is refused whole.
- **Entities.** `--capacity` bounds the session's rows; founding past it is refused before anything changes.
- **Nothing on disk.** The host keeps no logs, history, cache or temporary file, and its containment grants no
  write permission to any file.

## Stop and restart

Stopping is ending the process: `close`, end of script, or a signal. There is nothing to flush, and a killed process
leaves nothing behind. Restarting is starting it again, which opens a new session at the lineage's initial identity.

## Smoke and acceptance

```sh
ctest --test-dir build -R native_host_smoke                    # a session, close, a new process: no interpreter
cmake -DHOST="$HOME/elpis/bin/elpis" -DTESTING_HOST="$PWD/build/native/runtime/elpis_testing" \
      -DLIBRARIES="$HOME/elpis/lib/elpis" -DSCRATCH="$PWD/smoke" \
      -P native/runtime/host/tests/host_smoke.cmake            # the same, against the installed product
ELPIS_NATIVE_HOST_PREFIX="$HOME/elpis" python -m pytest tests/integration/test_native_host.py --lane all
```

The last command runs the full native suite against the installed binaries. It needs the build tree too, for the
testing host and for the differential against the Python reference. The suite covers:
- the process closure: no writable descriptor or shared mapping;
- the pins and every drift refusal;
- sessions, `close` and new processes resuming nothing;
- the operator events;
- every retired command;
- containment, including the testing host's deliberate write attempts, all refused;
- a killed process leaving nothing;
- equality with the offline Python reference.

CI runs it against an installed prefix, offline, in the Installed system acceptance job. The composed zero-write
qualification is `research/volatile_runtime_v1`.

## The offline reference interface (Python)

`python -B -m elpis.runtime.cli` is the **offline reference interface**. It is not the production path. It operates
the same composite lineage in the same volatile session, through the Python facade over RuntimeCore, and exists for:
- differential tests: the native host must answer exactly as it does;
- research.

Its acceptance run (`python -B -m elpis.runtime.acceptance --native-build build --work DIR --report
acceptance.json`) and its developer bundle (`python -B -m elpis.runtime.release`) are development tooling (census:
DEVELOPMENT_ONLY). They write only where you point them (the report, the harness's own tamper copies, the bundle),
and the acceptance run checks that no Elpis process writes anything.

The CLI is gated like the host, and refuses to start (`ENVIRONMENT_REFUSED`) unless:
- bytecode writing is off from startup (`-B` or `PYTHONDONTWRITEBYTECODE=1`);
- standard output and error are not storage;
- core dumps can be disabled;
- with swap active, its memory can be locked.

## Capabilities and nonclaims

- **Working and mechanically verified:**
  - numeric QUERY and LEARN (the K1 law is qualified only in its frozen regime);
  - typed evidence (observe-only), admitted by the session's sources;
  - exact LOOKUP;
  - operator source events;
  - pinned native admission and mandatory containment.
- **Formally qualified:** see `docs/qualification/README.md` (statuses).
- **Unavailable:** text (`ECS_CODEC_UNQUALIFIED`).
- **Retired:** recoverable publication, restart recovery and reconciliation (`OPERATION_RETIRED`).
- **Not enabled:** memory LEARN (unqualified); SLOW consolidation (not provided).
- **Research only:** HCA, and every laboratory under `research/` and `DEV/`.
- **Not claimed:** persistence or crash survival of a session, language understanding, reasoning, general
  intelligence, autonomous evolution, hierarchical superiority. Nothing is claimed about what host services outside
  Elpis's process-control boundary write (`docs/NONCLAIMS.md`).

An observation is evidence storage, never truth or authority. Pins authenticate bytes, not authors.
