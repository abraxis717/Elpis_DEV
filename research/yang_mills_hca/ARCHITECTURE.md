# Operator and large-scale AWR architecture

## Frozen experimental HCA (implemented historically)

The frozen R77/R78 HCA is an 88-parameter recurrent local geometric operator with two 4×4 state channels per local node, directed links and SO(2)×SO(2)-type gauge transport. Historical tests checked equivariance of jointly transformed states and links on the evaluated graphs. There are six frozen checkpoints: two seeds (`54007`, `54031`) by three variants (`STATE8`, `STATE8_SIGNED8`, `STATE8_TF1`). The test fixture comprises **ten participants and four relations of four members each**. These are *fixture dimensions*, not the intended global-memory capacity.

For a same-gauge-frame local prediction `p=F(s,A)`, let `d=p-s`. For each local node separately:

```text
radial     = s * dot(s,d) / max(dot(s,s), epsilon)
transverse = d - radial
p_control  = p + alpha*radial + beta*transverse
```

The frozen R104 control is `(alpha,beta)=(0.50,0.50)`. The frozen R107 control is `(0.70,0.45)`. The R114 checkpoint-conditioned shadow policy selects R107 for `STATE8_SIGNED8`, and selects R104 for the other checkpoint variants. This is **metadata gating**, not a trained router, and does not need teacher state during inference. Its **controller** requires zero new trainable parameters; the underlying HCA still has 88 parameters.

The post-hoc teacher-norm-matched quotient error is only a diagnostic oracle. Do not use teacher norms during ordinary inference.

## Large sparse AWR (NOT IMPLEMENTED / NOT QUALIFIED)

> **R118 update:** a research implementation of this substrate now exists in [`r118_scalable_awr/`](r118_scalable_awr/ARCHITECTURE.md). It derives that the frozen operator's nodes are the four *relations* of a window (participants are stateless frame anchors), keeps the 4-slot local arity, and composes overlapping windows with order-independent reconciliation. It is engineering-qualified for its registered workload only; it is not production integration and does not resolve R93.

AWR's target is a large addressable relational substrate, potentially millions of relationships, not ten permanently materialized slots. The intended representation is a bounded-arity **sparse hypergraph** with global participant IDs, relation IDs and an incidence index. Its logical storage should scale with `O(P + R + M)` for P participants, R relations and M total incidence memberships, before indexes, metadata and dynamic state overhead; it must never materialize a dense global `P×P` adjacency matrix.

A shared, fixed-parameter local HCA would be repeatedly applied to selected bounded neighborhoods. This separates global addressability from a **strictly budgeted active working set** in volatile RAM/VRAM. Immutable H-grams/relations on NVMe are read-only; inference must never modify checkpoint tensors or NAND storage. Active identity, membership counts, relation arity, scheduler work and per-turn state must be bounded with explicit refusal/eviction semantics.

Unqualified engineering seams: sparse gather and frame transport; overlapping-neighborhood scatter/reconciliation; conflict/degree control; bounded materialization; dynamic scheduling; end-to-end gauge covariance; native AWR/GDN/ERSA integration; large-scale memory and performance benchmarks. Shared 88-parameter weights do **not** establish scalable inference or storage capacity. It is insufficient to tile a ten-node operator without resolving these seams.

## Existing Elpis HCA names

The historical ECS_C native typed-hypergraph HCA is a distinct salience comparator, not this trained gauge-equivariant HCA. Do not change `native/ECS_C/hca`, canonical `src/elpis`, the runtime ABI or any independent historical evidence merely to align names.

**Binding gate:** `ENERGY_NORMALIZED_DENOMINATOR_HOLD` from R93 remains unresolved. Shadow control results do not authorize runtime integration.
