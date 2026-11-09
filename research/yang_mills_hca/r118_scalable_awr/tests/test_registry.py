"""R118-B registry: identity, CSR incidence, reverse index, contracts, canonical reindexing."""
from __future__ import annotations

import unittest

import numpy as np

from ..fixtures import extract_window, synthetic_registry
from ..registry import RegistryError, RegistryLimits, build_registry, rotation_blocks


def tiny():
    pids = [30, 10, 20, 40]
    rids = [500, 100]
    offsets = [0, 3, 5]
    mids = [40, 10, 20, 20, 30]
    frames = rotation_blocks(np.linspace(-1, 1, 5), np.linspace(2, -2, 5))
    return pids, rids, offsets, mids, frames


class RegistryContracts(unittest.TestCase):
    def test_canonical_layout(self):
        reg = build_registry(*tiny())
        self.assertEqual(reg.participant_ids.tolist(), [10, 20, 30, 40])
        self.assertEqual(reg.relation_ids.tolist(), [100, 500])
        # relation 100 = input row 1 (members 20, 30); relation 500 = input row 0 (40, 10, 20)
        self.assertEqual(reg.members_of(0).tolist(), [1, 2])
        self.assertEqual(reg.members_of(1).tolist(), [0, 1, 3])
        # frames travel with their incidence through both sorts
        f = rotation_blocks(np.linspace(-1, 1, 5), np.linspace(2, -2, 5))
        np.testing.assert_array_equal(reg.frames[reg.rel_offsets[1]:reg.rel_offsets[2]], f[[1, 2, 0]])
        np.testing.assert_array_equal(reg.frames[0:2], f[[3, 4]])
        self.assertFalse(reg.members.flags.writeable)
        self.assertFalse(reg.frames.flags.writeable)

    def test_lookups_and_refusals(self):
        reg = build_registry(*tiny())
        self.assertEqual(reg.relation_index([500, 100]).tolist(), [1, 0])
        self.assertEqual(reg.participant_index([40]).tolist(), [3])
        with self.assertRaises(RegistryError) as cm:
            reg.relation_index([999])
        self.assertEqual(cm.exception.code, "UNKNOWN_RELATION")
        with self.assertRaises(RegistryError):
            reg.participant_index([11])

    def test_build_refusals(self):
        p, r, o, m, f = tiny()
        cases = {
            "DUPLICATE_PARTICIPANT": (p + [10], r, o, m, f),
            "DUPLICATE_RELATION": (p, [500, 500], o, m, f),
            "UNKNOWN_MEMBER": (p, r, o, [40, 10, 20, 20, 99], f),
            "DUPLICATE_MEMBER_IN_RELATION": (p, r, o, [40, 10, 10, 20, 30], f),
            "ARITY_CONTRACT": (p, r, [0, 0, 5], m, f),
            "OFFSETS": (p, r, [0, 3, 4], m, f),
        }
        for code, args in cases.items():
            with self.assertRaises(RegistryError, msg=code) as cm:
                build_registry(*args)
            self.assertEqual(cm.exception.code, code)
        with self.assertRaises(RegistryError) as cm:
            build_registry(p, r, o, m, f, RegistryLimits(max_arity=2))
        self.assertEqual(cm.exception.code, "ARITY_CONTRACT")
        with self.assertRaises(RegistryError) as cm:
            build_registry(p, r, o, m, f, RegistryLimits(max_participant_degree=1))
        self.assertEqual(cm.exception.code, "PARTICIPANT_DEGREE_CONTRACT")

    def test_frame_contract(self):
        p, r, o, m, f = tiny()
        bad = f.copy()
        bad[2] *= 1.01  # not orthonormal
        with self.assertRaises(RegistryError) as cm:
            build_registry(p, r, o, m, bad)
        self.assertEqual(cm.exception.code, "FRAME_NOT_SO2")
        refl = f.copy()
        refl[1, 4:8] = [1.0, 0.0, 0.0, -1.0]  # reflection: orthonormal but det -1
        with self.assertRaises(RegistryError):
            build_registry(p, r, o, m, refl)
        nan = f.copy()
        nan[0, 0] = np.nan
        with self.assertRaises(RegistryError) as cm:
            build_registry(p, r, o, m, nan)
        self.assertEqual(cm.exception.code, "FRAME_NONFINITE")

    def test_reverse_index_and_storage(self):
        reg = synthetic_registry(400, 3000, seed=11, arity=(1, 8), zipf=1.1)
        P, R, M = reg.num_participants, reg.num_relations, reg.num_incidences
        # every incidence appears exactly once in the reverse index, grouped by participant
        self.assertEqual(sorted(reg.part_incidences.tolist()), list(range(M)))
        for p in range(P):
            inc = reg.incidences_of_participant(p)
            self.assertTrue(np.all(reg.members[inc] == p))
            self.assertTrue(np.all(np.diff(reg.inc_relation[inc]) > 0))
        for r in range(R):
            self.assertTrue(np.all(np.diff(reg.members_of(r)) > 0))
            self.assertTrue(np.all(reg.inc_relation[reg.rel_offsets[r]:reg.rel_offsets[r + 1]] == r))
        nb = reg.nbytes()
        expected = 8 * P + 8 * R + 8 * (R + 1) + 4 * M + 64 * M + 4 * M + 8 * (P + 1) + 4 * M
        self.assertEqual(nb["total"], expected)  # O(P + R + M), no dense P x P or R x R array

    def test_reindexing_is_canonical(self):
        rng = np.random.default_rng(5)
        p, r, o, m, f = tiny()
        base = build_registry(p, r, o, m, f)
        # permute participant list, relation order and member order within relations
        rel_rows = [(r[i], list(zip(m[o[i]:o[i + 1]], f[o[i]:o[i + 1]]))) for i in range(len(r))]
        rng.shuffle(rel_rows)
        rids2, off2, mids2, fr2 = [], [0], [], []
        for rid, mem in rel_rows:
            mem = [mem[j] for j in rng.permutation(len(mem))]
            rids2.append(rid)
            mids2 += [x for x, _ in mem]
            fr2 += [y for _, y in mem]
            off2.append(len(mids2))
        other = build_registry(list(rng.permutation(p)), rids2, off2, mids2, np.array(fr2))
        for name in ("participant_ids", "relation_ids", "rel_offsets", "members", "frames", "inc_relation",
                     "part_offsets", "part_incidences"):
            np.testing.assert_array_equal(getattr(base, name), getattr(other, name), err_msg=name)

    def test_window_extraction_is_bounded_and_deterministic(self):
        reg = synthetic_registry(300, 2000, seed=3, arity=(2, 5), zipf=1.0)
        for s in (0, 17, 1999):
            w1, w2 = extract_window(reg, s), extract_window(reg, s)
            self.assertEqual(w1, w2)
            self.assertEqual(w1[0], s)
            self.assertLessEqual(len(w1), 4)
            self.assertEqual(len(set(w1)), len(w1))
            for r in w1[1:]:
                self.assertTrue(np.intersect1d(reg.members_of(s), reg.members_of(r)).size > 0)
        with self.assertRaises(RegistryError) as cm:
            extract_window(reg, 0, max_incidences=1)
        self.assertEqual(cm.exception.code, "WINDOW_EXPANSION_BUDGET")


if __name__ == "__main__":
    unittest.main()
