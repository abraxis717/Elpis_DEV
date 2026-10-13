# Elpis ECS / experimental model-free memory (DEV)

**Status: isolated native research prototype. Not production, cognitive writer, published continuity, or scientific qualification.**

This directory contains a pure C11, **zero-model** memory experiment. There is no pretrained model, checkpoint, neural runtime, safetensors or pickle code. It does not reach into Elpis's existing canonical runtime.

## Components

- `bounded_lattice.[ch]` — the D31-B fixed-capacity linear-scan reference.
- `indexed_lattice.[ch]` — the D31-C indexed fixed-capacity gated-delta arena, with a 9-cell normalized raised-cosine footprint. The numerical state and write rule are *not trained*.
- `model_free_memory.[ch]` — new deterministic address + feature derivation from explicit `(context, entity, role)` identifiers. It replaces the *interface* that previously depended on learned projections, **not their semantic or generalization capability**. Values remain 64 caller-supplied FP32 numbers.
- `tests/` — model-free deterministic smoke and D31-B↔D31-C differential mechanics; the latter asserts no heap operations after construction via GNU/Linux linker wrappers.

## Build and test

```sh
cmake -S native/ecs_memory_experimental -B /tmp/elpis-ecs-memory-build -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/elpis-ecs-memory-build
ctest --test-dir /tmp/elpis-ecs-memory-build --output-on-failure
```

This standalone project has **no Torch/Python/model dependencies**. Build output is deliberately outside the source tree.

## Safety and physical limits

- Capacity and address radius are operator-chosen at creation. Minimum 9 cells, maximum 8192. Two FP32 heads of 32×32 state per cell; ~8 KiB per cell, **plus** the direct-index grid and metadata. `ecs_mf_allocated_bytes` reports arena + adapter resident bytes (not allocator overhead or call-stack staging).
- No heap operations after creation; reads do not admit cells. The write plan refuses an impossible full 9-cell update before mutating state (REFUSE) or deterministically evicts (LRU).
- Coordinate index lookup is O(1) within fixed bounds, but free-slot/LRU victim selection remains O(capacity). Single writer; no concurrent readers during writes. Write atomicity is in-process only, **not crash consistency**.
- Hashed deterministic addressing is not collision-free, cryptographic, learned, semantic, or an N-ary HACF topology. Bounded collisions and gated-delta updates can destroy older associations; test efficacy rather than assume it.
- No serialization, checkpoint publishing, recovery authority, operator-broker, external-file reading or self-growth interface is present. Only a future authorized integration may use existing Elpis checkpoint authority.

## Provenance and evidence boundary

The bounded reference and indexed math are source-identical to **local D31-B/D31-C** frozen research handoffs. D31-C mechanics on Ouroboros reached `PASS_D31C_FIXED_DIRECT_INDEX_DEV`; D31-Q closed as rehearsed DESIGN, with no prospective science qualification. The model-free adapter is *new* and only smoke-qualified here.

The mathematical donor is [`eelbaz/mLattice`](https://github.com/eelbaz/mLattice) pinned at `873dfcab5c9fc29a5d974f74c2a1ab6511c026e1` (MIT); its learned weights and neural model are **not included**. The MIT notice is in `licenses/MULATTICE_MIT.txt`.

See [`docs/research/ECS_MEMORY_MODEL_FREE_DEV.md`](../../docs/research/ECS_MEMORY_MODEL_FREE_DEV.md) for continuation boundaries for Claude or a later integrator.
