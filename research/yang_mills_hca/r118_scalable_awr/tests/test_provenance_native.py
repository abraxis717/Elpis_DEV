"""R118-A provenance (stdlib) and native-kernel arithmetic contracts (no torch)."""
from __future__ import annotations

import json
import math
import os
import random
import unittest

import numpy as np

from .. import native
from .. import provenance as P


class Provenance(unittest.TestCase):
    def test_manifest_ledger_and_pins(self):
        rep = P.verify_manifest()
        self.assertEqual(rep["files_verified"], P.MANIFEST_FILE_COUNT)
        self.assertEqual(rep["ledger_entries_verified"], 26)
        recs = P.checkpoint_records()
        self.assertEqual(sorted(recs), list(P.CHECKPOINTS))
        for key, rec in recs.items():
            self.assertEqual(rec["sha256"], P.CHECKPOINT_SHA256[key])
            mutated = [row[:] for row in rec["weights"]]
            mutated[3][7] = math.nextafter(mutated[3][7], math.inf)  # one ULP
            self.assertNotEqual(P.checkpoint_hash(mutated, rec["bias"]), rec["sha256"])

    def test_sealed_archives(self):
        r77, sha77 = P.read_sealed_result(P.R77_ARCHIVE)
        self.assertEqual(sha77, P.R77_RESULT_JSON_SHA256)
        self.assertEqual(len(r77["rows"]), 576)
        r78, _ = P.read_sealed_result(P.R78_ARCHIVE)
        self.assertEqual(r78["parent_r77_report_sha256"], P.R77_ARCHIVE_SHA256)
        self.assertEqual(r78["numerics"]["max_r77_native_parity"], 0.0)
        with self.assertRaises(P.ProvenanceError):
            json.loads(b'{"x": NaN}', parse_constant=P._reject_constant)

    def test_staging_is_exact_and_volatile(self):
        with P.volatile_scratch() as d:
            staged = P.stage_snapshot(d)
            self.assertEqual(len(staged), 28)
            for name, digest in staged.items():
                self.assertEqual(P.sha256_file(d / name), digest)
            path = str(d)
        self.assertFalse(os.path.exists(path))
        P.verify_manifest()  # frozen tree still authentic after staging


class NativeKernel(unittest.TestCase):
    def test_build_info_and_fsum_port_is_bitwise_math_fsum(self):
        k = native.build()
        self.assertEqual(k.info["abi_version"], native.ABI_VERSION)
        self.assertIn("-ffp-contract=off", k.info["cflags"])
        rng = random.Random(118)
        for t in range(5000):
            n = rng.randint(0, 32)
            v = [rng.uniform(-4, 4) * rng.uniform(-4, 4) * 10 ** rng.randint(-12, 12) for _ in range(n)]
            if t % 3 == 0 and n >= 2:
                v[1] = -v[0]
            a = np.ascontiguousarray(v, dtype=np.float64)
            got = k.lib.awr_fsum_probe(a.ctypes.data, len(v))
            want = math.fsum(v)
            self.assertTrue(got == want and math.copysign(1, got) == math.copysign(1, want), (v, got, want))


class FlushSubnormals(unittest.TestCase):
    def test_ftz_is_scoped_to_the_call_and_default_is_exact(self):
        from .. import engine as E
        from ..awr_state import AWRState, StateCapacity
        from ..fixtures import synthetic_registry
        from ._util import compare_with_simple, random_windows

        k = native.build()
        reg = synthetic_registry(80, 300, 3, arity=(2, 5))
        st = AWRState(reg, StateCapacity(max_instances=40, max_materialized_relations=160, max_bindings_per_relation=64))
        rel, mask = random_windows(reg, 40, 2)
        h, _ = st.create_instances(np.arange(40, dtype=np.uint64), rel, mask)
        rows = np.nonzero(st.row_ref > 0)[0]
        st.pool[rows] = np.random.default_rng(1).normal(0, 1e-150, size=(rows.size, 8))  # deep-underflow regime
        before = k.lib.awr_mxcsr_probe()
        ftz = E.Engine(st, "54007/STATE8", controller="R104", flush_subnormals=True)
        self.assertEqual(ftz.step(h, passes=3).disposition, "EXECUTED")
        self.assertEqual(k.lib.awr_mxcsr_probe(), before)          # caller's FP environment untouched
        self.assertEqual(before & 0x8040, 0)
        with self.assertRaises(ValueError):
            E.Engine(st, "54007/STATE8", backend="oracle", flush_subnormals=True)
        res, diff, eq, n = compare_with_simple(E.Engine(st, "54007/STATE8", controller="R104"), h, passes=3)
        self.assertEqual((res.disposition, eq), ("EXECUTED", n))  # default (no FTZ) stays bitwise exact


if __name__ == "__main__":
    unittest.main()
