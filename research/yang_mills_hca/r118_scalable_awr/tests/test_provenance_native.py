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


if __name__ == "__main__":
    unittest.main()
