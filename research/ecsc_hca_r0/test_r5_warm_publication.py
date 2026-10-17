"""R5 research-only mechanics: isolated fail-closed WARM publication and readout."""
from dataclasses import replace
from unittest import TestCase
from unittest.mock import patch

from elpis.runtime.context import ContextLimits, ContextCoordinate, successor

from .baselines import Refusal
from .r5_warm_publication import WarmExactPublication
from .test_r4_hacf_sources import HACFSourceBoundaryTests
from .r2_common import semantic
from .task import predict


class WarmPublicationTests(TestCase):
    def setUp(self):
        self.f = HACFSourceBoundaryTests(methodName="test_original_bytes_reverified_in_place_and_normalized_digest_bound")
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.p = WarmExactPublication()

    def publish(self, view=None, *, task=None, authority=None):
        f = self.f
        return self.p.refresh(f.root, f.manifest, f.view if view is None else view,
                              f.authority if authority is None else authority,
                              f.bindings, f.task if task is None else task)

    def test_first_publication_exact_matches_B0_and_readout_is_coordinate_bound(self):
        out = self.publish()
        self.assertIs(out, self.p.peek(self.f.view.coordinate))
        self.assertEqual(semantic(out.result), semantic(predict(self.p._state.verified.graph, self.f.task, "B0")))
        self.assertEqual(out.mode, "EXPLICIT_COLD_OPEN")
        self.assertIsNone(self.p.peek(ContextCoordinate(999, "0" * 64)))

    def test_same_view_reverification_then_nonblocking_peek(self):
        one = self.publish()
        two = self.publish()
        self.assertEqual(two.mode, "SAME_PUBLISHED_VIEW")
        self.assertEqual(one.evidence_digest, two.evidence_digest)
        # Foreground peek must never call a materializer, even if the WARM
        # operation would be stalled or fail.
        with patch("research.ecsc_hca_r0.r5_warm_publication.open_verified_hacf", side_effect=AssertionError), \
             patch("research.ecsc_hca_r0.r5_warm_publication.advance_verified_hacf", side_effect=AssertionError):
            self.assertIs(self.p.peek(self.f.view.coordinate), two)

    def test_immediate_successor_publishes_only_complete_new_coordinate(self):
        first = self.publish()
        second_view = successor(self.f.view, (), ContextLimits(), through=2, dropped=0)
        out = self.publish(second_view)
        self.assertEqual(out.mode, "VERIFIED_IMMEDIATE_SUCCESSOR")
        self.assertIsNone(self.p.peek(self.f.view.coordinate))
        self.assertIs(self.p.peek(second_view.coordinate), out)
        self.assertEqual(first.coordinate, self.f.view.coordinate)

    def test_detected_stale_source_quarantines_readout_and_restoration_reverifies(self):
        first = self.publish()
        self.f.path.write_bytes(b"changed bytes")
        with self.assertRaises(Refusal) as caught:
            self.publish()
        self.assertEqual(caught.exception.code, "SOURCE_INTEGRITY")
        self.assertIsNone(self.p.peek(self.f.view.coordinate))
        self.assertEqual(first.coordinate, self.f.view.coordinate)
        self.f.path.write_bytes(self.f.data)
        restored = self.publish()
        self.assertEqual(restored.evidence_digest, first.evidence_digest)
        self.assertIs(self.p.peek(self.f.view.coordinate), restored)

    def test_skipped_versions_refuse_and_recovery_is_exact(self):
        self.publish()
        immediate = successor(self.f.view, (), ContextLimits(), through=2, dropped=0)
        skipped = successor(immediate, (), ContextLimits(), through=3, dropped=0)
        with self.assertRaises(Refusal) as caught:
            self.publish(skipped)
        self.assertEqual(caught.exception.code, "PREDECESSOR")
        self.assertIsNone(self.p.peek(self.f.view.coordinate))
        self.assertEqual(self.publish(immediate).coordinate, immediate.coordinate)

    def test_authority_switch_refuses_without_old_readout_leakage(self):
        self.publish()
        invalid = replace(self.f.authority, epoch="TEST_ONLY_ALIEN")
        with self.assertRaises(Refusal) as caught:
            self.publish(authority=invalid)
        self.assertEqual(caught.exception.code, "AUTHORITY_CONTINUITY")
        self.assertIsNone(self.p.peek(self.f.view.coordinate))
        self.assertEqual(self.publish().coordinate, self.f.view.coordinate)

    def test_scope_switch_does_not_reuse_old_publication(self):
        self.publish()
        new_task = replace(self.f.task, scope="other-project")
        with self.assertRaises(Refusal) as caught:
            self.publish(task=new_task)
        self.assertEqual(caught.exception.code, "SCOPE_TRANSITION")
        self.assertIsNone(self.p.peek(self.f.view.coordinate))

    def test_two_threads_cannot_publish_same_old_version_out_of_order(self):
        from concurrent.futures import ThreadPoolExecutor
        self.publish()
        immediate = successor(self.f.view, (), ContextLimits(), through=2, dropped=0)
        skipped = successor(immediate, (), ContextLimits(), through=3, dropped=0)
        # The WARM writer mutex must serialize publication. Either order
        # leaves at most one exact view available and never a torn readout.
        def attempt(v):
            try:
                return self.publish(v)
            except Refusal:
                return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(attempt, (immediate, skipped)))
        accepted = [x for x in outcomes if x is not None]
        self.assertGreaterEqual(len(accepted), 1)
        self.assertIsNotNone(self.p.peek(skipped.coordinate) if len(accepted) == 2
                             else self.p.peek(immediate.coordinate))


if __name__ == "__main__":
    from unittest import main
    main()
