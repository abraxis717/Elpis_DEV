"""R118-B volatile AWR state: addressable HCA state instances over a sparse registry.

* An **HCA state instance** has a stable 64-bit ID, a dense handle with a
  generation counter, and binds up to four distinct relations into slots 0..3
  (the frozen operator's local operator roles). Its 32-coordinate state is the
  view ``(A[slot], B[slot])`` of its bound relations' rows (contract.py layout).
* A **relation** carries volatile state (A[4], B[4]) only while at least one
  instance binds it (materialization). Rows come from a bounded pool; the last
  release discards the row. Unbound relations exist only in the registry.
* Instances that share a relation share that relation's state; instances with
  disjoint relation sets are fully isolated (tests check both).

All capacities are explicit and preallocated; every violation raises
``StateError`` with a stable code *before* any mutation (transactional).
Nothing here touches the filesystem; weights are not referenced at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contract import SLOT_COORDS, SLOTS

U64_MAX = (1 << 64) - 1


class StateError(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(code + (":" + detail if detail else ""))
        self.code = code


@dataclass(frozen=True)
class StateCapacity:
    max_instances: int = 50_000
    max_materialized_relations: int = 200_000
    max_bindings_per_relation: int = 64   # overlap degree of one relation (native limit 1024)

    def __post_init__(self):
        if not 1 <= self.max_instances <= (1 << 24):
            raise StateError("INSTANCE_CAPACITY_RANGE")
        if not 1 <= self.max_materialized_relations <= (1 << 26):
            raise StateError("ROW_CAPACITY_RANGE")
        if not 1 <= self.max_bindings_per_relation <= 1024:
            raise StateError("BINDING_CAPACITY_RANGE")


class AWRState:
    def __init__(self, registry, capacity: StateCapacity = StateCapacity()):
        self.registry = registry
        self.capacity = capacity
        n_i, n_r, k = capacity.max_instances, capacity.max_materialized_relations, capacity.max_bindings_per_relation
        self.inst_id = np.zeros(n_i, dtype=np.uint64)
        self.inst_live = np.zeros(n_i, dtype=np.uint8)
        self.inst_gen = np.zeros(n_i, dtype=np.uint32)
        self.slot_rel = np.full((n_i, SLOTS), -1, dtype=np.int32)
        self.slot_row = np.full((n_i, SLOTS), -1, dtype=np.int32)
        self.inst_links = np.zeros(n_i, dtype=np.int32)
        self.inst_memberships = np.zeros(n_i, dtype=np.int32)
        self.inst_bound = np.zeros(n_i, dtype=np.int8)
        self._free_handles = list(range(n_i - 1, -1, -1))
        self._handle_of: dict[int, int] = {}
        self.pool = np.zeros((n_r, SLOT_COORDS), dtype=np.float64)
        self.row_rel = np.full(n_r, -1, dtype=np.int32)
        self.row_ref = np.zeros(n_r, dtype=np.int32)
        self.row_bind = np.full((n_r, k), -1, dtype=np.int32)  # handle * 4 + slot
        self.rel_row = np.full(registry.num_relations, -1, dtype=np.int32)
        self._free_rows = list(range(n_r - 1, -1, -1))
        self.mark = np.full(n_r, -1, dtype=np.int32)  # native compaction scratch; all -1 between steps
        self.mutation_epoch = 0

    # -- instance lifecycle --------------------------------------------------
    @property
    def live_count(self) -> int:
        return len(self._handle_of)

    def create_instance(self, instance_id: int, relation_ids) -> tuple[int, int]:
        """One instance; ``relation_ids`` is 4 relation IDs or None (empty slot)."""
        if len(relation_ids) != SLOTS:
            raise StateError("SLOT_COUNT")
        mask = np.array([[r is not None for r in relation_ids]])
        rel = np.array([[0 if r is None else int(r) for r in relation_ids]], dtype=np.uint64)
        h, g = self.create_instances(np.array([int(instance_id)], dtype=np.uint64), rel, mask)
        return int(h[0]), int(g[0])

    def window_link_counts(self, slot_rel: np.ndarray) -> np.ndarray:
        """Directed link count per window, vectorized: sum_p c_p (c_p - 1), where c_p is the
        number of the window's relations containing participant p (ordered slot pairs
        sharing p). Cross-checked against Registry.window_links in the tests."""
        reg = self.registry
        n = slot_rel.shape[0]
        win, slot = np.nonzero(slot_rel >= 0)
        rels = slot_rel[win, slot].astype(np.int64)
        lo, hi = reg.rel_offsets[rels], reg.rel_offsets[rels + 1]
        lens = hi - lo
        total = int(lens.sum())
        if total == 0:
            return np.zeros(n, dtype=np.int64)
        idx = np.repeat(lo - np.cumsum(np.concatenate(([0], lens[:-1]))), lens) + np.arange(total)
        w_of = np.repeat(win.astype(np.int64), lens)
        keys = w_of * np.int64(max(reg.num_participants, 1)) + reg.members[idx].astype(np.int64)
        uk, counts = np.unique(keys, return_counts=True)
        per = counts.astype(np.int64) * (counts.astype(np.int64) - 1)
        return np.bincount(uk // np.int64(max(reg.num_participants, 1)), weights=per, minlength=n).astype(np.int64)

    def create_instances(self, instance_ids, relation_ids, bound_mask) -> tuple[np.ndarray, np.ndarray]:
        """All-or-nothing bulk creation. ``relation_ids``: uint64 [n,4]; ``bound_mask``: bool [n,4]."""
        ids = np.asarray(instance_ids, dtype=np.uint64).reshape(-1)
        rid = np.asarray(relation_ids, dtype=np.uint64).reshape(-1, SLOTS)
        mask = np.asarray(bound_mask, dtype=bool).reshape(-1, SLOTS)
        n = ids.shape[0]
        if rid.shape[0] != n or mask.shape[0] != n:
            raise StateError("SHAPE")
        if np.unique(ids).shape[0] != n or any(int(i) in self._handle_of for i in ids.tolist()):
            raise StateError("DUPLICATE_INSTANCE_ID")
        if n > len(self._free_handles):
            raise StateError("INSTANCE_CAPACITY", str(self.capacity.max_instances))
        if n and not mask.any(axis=1).all():
            raise StateError("EMPTY_WINDOW")
        slot_rel = np.full((n, SLOTS), -1, dtype=np.int32)
        try:
            slot_rel[mask] = self.registry.relation_index(rid[mask])
        except ValueError as exc:
            raise StateError("UNKNOWN_RELATION", str(exc)) from None
        for d in range(SLOTS):
            for e in range(d + 1, SLOTS):
                if np.any((slot_rel[:, d] >= 0) & (slot_rel[:, d] == slot_rel[:, e])):
                    raise StateError("DUPLICATE_RELATION_IN_WINDOW")
        bound_rel = slot_rel[mask]
        uniq, cnt = np.unique(bound_rel, return_counts=True)
        rows_now = self.rel_row[uniq]
        refs_now = np.where(rows_now >= 0, self.row_ref[np.maximum(rows_now, 0)], 0)
        if np.any(refs_now + cnt > self.capacity.max_bindings_per_relation):
            raise StateError("BINDING_CAPACITY")
        new_rel = uniq[rows_now < 0]
        if new_rel.size > len(self._free_rows):
            raise StateError("STATE_ROW_CAPACITY", str(self.capacity.max_materialized_relations))
        links = self.window_link_counts(slot_rel)
        member_counts = np.zeros((n, SLOTS), dtype=np.int64)
        member_counts[mask] = self.registry.arity(bound_rel)
        # -- all checks passed: mutate --
        handles = np.array([self._free_handles.pop() for _ in range(n)], dtype=np.int32)
        self.inst_gen[handles] = (self.inst_gen[handles].astype(np.uint64) + 1).astype(np.uint32)
        self.inst_id[handles] = ids
        self.inst_live[handles] = 1
        self.slot_rel[handles] = slot_rel
        self.inst_links[handles] = links.astype(np.int32)
        self.inst_memberships[handles] = member_counts.sum(axis=1).astype(np.int32)
        self.inst_bound[handles] = mask.sum(axis=1).astype(np.int8)
        new_rows = np.array([self._free_rows.pop() for _ in range(new_rel.size)], dtype=np.int32)
        self.rel_row[new_rel] = new_rows
        self.row_rel[new_rows] = new_rel
        self.pool[new_rows] = 0.0
        slot_row = np.full((n, SLOTS), -1, dtype=np.int32)
        slot_row[mask] = self.rel_row[bound_rel]
        self.slot_row[handles] = slot_row
        win, slot = np.nonzero(mask)
        b_rows = slot_row[win, slot]
        codes = handles[win].astype(np.int64) * SLOTS + slot
        order = np.argsort(b_rows, kind="stable")
        srows = b_rows[order]
        start = np.searchsorted(srows, srows, side="left")
        rank = np.arange(srows.size) - start
        self.row_bind[srows, self.row_ref[srows] + rank] = codes[order].astype(np.int32)
        np.add.at(self.row_ref, b_rows, 1)
        for i, h in zip(ids.tolist(), handles.tolist()):
            self._handle_of[i] = h
        self.mutation_epoch += 1
        return handles, self.inst_gen[handles].copy()

    def release_instance(self, handle: int, generation: int) -> None:
        h = int(handle)
        self._check_one(h, generation)
        for d in range(SLOTS):
            row = int(self.slot_row[h, d])
            if row < 0:
                continue
            k = int(self.row_ref[row])
            entries = self.row_bind[row, :k]
            pos = int(np.nonzero(entries == h * SLOTS + d)[0][0])
            entries[pos] = entries[k - 1]
            self.row_bind[row, k - 1] = -1
            self.row_ref[row] = k - 1
            if k - 1 == 0:
                r = int(self.row_rel[row])
                self.rel_row[r] = -1
                self.row_rel[row] = -1
                self.pool[row] = 0.0
                self._free_rows.append(row)
        del self._handle_of[int(self.inst_id[h])]
        self.inst_live[h] = 0
        self.slot_rel[h] = -1
        self.slot_row[h] = -1
        self.inst_links[h] = 0
        self.inst_memberships[h] = 0
        self.inst_bound[h] = 0
        self.inst_id[h] = 0
        self._free_handles.append(h)
        self.mutation_epoch += 1

    # -- addressing ------------------------------------------------------------
    def resolve(self, instance_ids) -> tuple[np.ndarray, np.ndarray]:
        ids = [int(i) for i in np.asarray(instance_ids, dtype=np.uint64).reshape(-1)]
        missing = [i for i in ids if i not in self._handle_of]
        if missing:
            raise StateError("UNKNOWN_INSTANCE", f"{len(missing)} unknown")
        h = np.fromiter((self._handle_of[i] for i in ids), dtype=np.int32, count=len(ids))
        return h, self.inst_gen[h].copy()

    def _check_one(self, h: int, generation: int) -> None:
        if not 0 <= h < self.capacity.max_instances or not self.inst_live[h]:
            raise StateError("DEAD_HANDLE", str(h))
        if int(self.inst_gen[h]) != int(generation):
            raise StateError("STALE_HANDLE", str(h))

    def check_handles(self, handles: np.ndarray, gens: np.ndarray) -> None:
        h = np.asarray(handles)
        if h.size and (h.min() < 0 or h.max() >= self.capacity.max_instances):
            raise StateError("HANDLE_RANGE")
        if not np.all(self.inst_live[h] == 1):
            raise StateError("DEAD_HANDLE")
        if not np.array_equal(self.inst_gen[h], np.asarray(gens, dtype=np.uint32)):
            raise StateError("STALE_HANDLE")

    # -- volatile state I/O (explicit external input, never teacher state) ---------
    def set_relation_state(self, relation_id: int, ab8) -> None:
        r = int(self.registry.relation_index([relation_id])[0])
        row = int(self.rel_row[r])
        if row < 0:
            raise StateError("RELATION_NOT_MATERIALIZED", str(relation_id))
        v = np.asarray(ab8, dtype=np.float64).reshape(-1)
        if v.shape != (SLOT_COORDS,) or not np.all(np.isfinite(v)):
            raise StateError("STATE_VALUE")
        self.pool[row] = v
        self.mutation_epoch += 1

    def relation_state(self, relation_id: int):
        r = int(self.registry.relation_index([relation_id])[0])
        row = int(self.rel_row[r])
        return None if row < 0 else self.pool[row].copy()

    def instance_state32(self, handle: int, generation: int) -> np.ndarray:
        h = int(handle)
        self._check_one(h, generation)
        out = np.zeros(32, dtype=np.float64)
        for d in range(SLOTS):
            row = int(self.slot_row[h, d])
            if row >= 0:
                out[d * 4:(d + 1) * 4] = self.pool[row, :4]
                out[16 + d * 4:16 + (d + 1) * 4] = self.pool[row, 4:]
        return out

    def instances_binding(self, relation_index: int) -> np.ndarray:
        row = int(self.rel_row[relation_index])
        if row < 0:
            return np.zeros(0, dtype=np.int32)
        return (self.row_bind[row, :self.row_ref[row]] // SLOTS).astype(np.int32)

    def nbytes(self) -> dict:
        names = ("inst_id", "inst_live", "inst_gen", "slot_rel", "slot_row", "inst_links", "inst_memberships",
                 "inst_bound", "pool", "row_rel", "row_ref", "row_bind", "rel_row", "mark")
        out = {n: int(getattr(self, n).nbytes) for n in names}
        out["total"] = sum(out.values())
        return out
