# R118 — scalable sparse AWR over the frozen equivariant HCA

**Independent Elpis research. RESEARCH_ONLY. Not production-admitted.**
`R93_GATE=ENERGY_NORMALIZED_DENOMINATOR_HOLD` is unchanged; nothing here promotes
the operator or claims that local transition accuracy is solved.

This component is an executable, bounded, sparse relational substrate that hosts
1–50,000 independently addressable HCA state instances over a registry of up to
millions of relations, and invokes the **authentic frozen 88-parameter HCA**
(six R77/R78 checkpoints, reconstructed from the published repository and
verified by hash) over a strictly budgeted active set.

| Phase | Disposition | Evidence |
|---|---|---|
| R118-A authentic operator reconstruction | **PASS** — 39/39 sources authenticated; six checkpoint hashes; R77 evaluation replay 576/576 rows bitwise; R78 archive replays byte-identically from a clean checkout | `evidence/r118a.json` |
| R118-B sparse substrate correctness | **PASS** — native engine equals the intentionally simple reference bitwise on every graph family; capacities 1…50,000; budgets refuse before work | test suite (`tests/`) |
| R118-C geometry and global recurrence | **PASS** — historical rollouts replay bitwise through the substrate; native ≤ 3.6e-16; end-to-end gauge covariance with overlapping windows ≤ 2.7e-15; no hidden channels | `evidence/r118c.json` |
| R118-D performance | **HOT_PATH_QUALIFIED** for the registered primary workload (two runs of the frozen protocol), **with a disclosed regime exception** (input-free deeply decayed states on the exact path) | `evidence/r118d_run1.json`, `evidence/r118d_run2.json`, `evidence/r118d_subnormal_diagnostic.json` |
| R118-E transfer characterization | **DESCRIPTIVE ONLY** — composition changes magnitude fidelity, not direction; no new instability observed; R93 unaffected | `evidence/r118e.json` |

Full measurements: [RESULTS.md](RESULTS.md). Design and proofs: [ARCHITECTURE.md](ARCHITECTURE.md).

## Derived mapping (from the frozen source, not assumed)

| Concept | Historical object | Carries state |
|---|---|---|
| global participant | member label inside relation rows; anchors one frame `U[r,p]` per incidence | **no** |
| relation | one row of `graph`; one HCA node with `(A, B) ∈ R⁴ × R⁴` in its own gauge frame | yes, 8 coordinates |
| local operator role | slot `d ∈ {0,1,2,3}` of the `[batch, 4, 4]` tensor (historical source at 0, readout at 3) | no |
| HCA state instance | one batch row: 4 relation slots × 2 channels × 4 = **32 coordinates**; the R104/R114 controller acts on these jointly | yes (view of its relations) |
| link | (src relation, dst relation, shared participant `p`), operator `U[dst,p] U[src,p]ᵀ` | derived per window |

Evidence: `portable_mechanics.link_builder` iterates over relation rows;
`r62_source.graph_step_split` hard-codes `range(4)`; `README_R78.txt`: *"The HCA's
native state is (A:R4,B:R4) across four relation nodes."* The historical "ten
participants" are therefore stateless frame anchors, and the operator is a
four-relation-node operator, not a ten-node one. A participant is never one
32-coordinate HCA state.

## Layout

| File | Role |
|---|---|
| `provenance.py` | stdlib-only authentication: pinned manifest, ledger, bounded canonical archive reads, checkpoint hashes, volatile staging |
| `historical_replay.py` | the historical `run.sh` protocol rebuilt from the published loose files (unmodified `verify_r78.py`, `spot_r78.py`, `r78.py run` in `/dev/shm`) |
| `frozen_oracle.py` | in-process frozen operator from authenticated in-memory bytes (torch) |
| `contract.py` | instance contract, 32-coordinate layout, controller policy (delegates R114 to `../reference.py`) |
| `reference_adapter.py` | **exact reference adapter**: historical `graph_step_split` over substrate windows |
| `registry.py` | immutable sparse registry: 64-bit IDs, CSR incidence, reverse index, lossless frames |
| `awr_state.py` | addressable instances, bounded materialized relation-state pool, O(bindings) reverse index |
| `engine.py` | hot path, budgets, refusals, transactional commit, backends `native` / `oracle` |
| `csrc/awr_kernel.c` | C11 kernel (built at runtime into volatile scratch; no binary is committed) |
| `simple_reference.py` | intentionally simple pure-Python reference of the full semantics |
| `fixtures.py`, `gauge.py` | historical import, synthetic graphs, neighborhood extraction; substrate-wide gauge actions |
| `qualify_a.py`, `qualify_c.py`, `bench.py`, `diag_subnormal.py`, `characterize_e.py` | phase runners producing `evidence/*.json` |
| `PROTOCOL_R118D.json` | performance protocol, committed before any latency measurement |

## Running (from the repository root)

PyTorch is required only for the frozen oracle (A, C, E and the oracle baseline);
the native hot path needs numpy and a C compiler. Use an isolated environment,
for example a venv with `pip install --index-url https://download.pytorch.org/whl/cpu torch`
(measured with torch 2.15.0+cpu; no historical torch version is pinned in the
frozen sources — see RESULTS.md).

```bash
python -B -m unittest discover -s research/yang_mills_hca/r118_scalable_awr/tests -t .
```

```bash
python -B -m research.yang_mills_hca.r118_scalable_awr.qualify_a --out /tmp/r118a.json
```

```bash
python -B -m research.yang_mills_hca.r118_scalable_awr.qualify_c --out /tmp/r118c.json
```

```bash
python -B -m research.yang_mills_hca.r118_scalable_awr.bench --all --out /tmp/r118d.json
```

These tests are deliberately outside the repository CI lanes (`testpaths = ["tests"]`).
Nothing writes under the repository; all execution scratch is in `/dev/shm`.

## Attribution

The Yang–Mills mathematical context is the upstream research
[creelie/4Y-M](https://github.com/creelie/4Y-M), *Mass Gap of Four-Dimensional
Lattice Yang–Mills Theory for Every Compact Simple Group*, by Deep Bhattacharjee,
Ushashi Bhattacharya, James Montgomery Lockwood and Tristan Romanov (pinned
v4.21.0, MIT; see [../PROVENANCE.md](../PROVENANCE.md)). No upstream code is used
here. The upstream authors did not develop, endorse or qualify the Elpis HCA,
this substrate or any measurement reported here.
