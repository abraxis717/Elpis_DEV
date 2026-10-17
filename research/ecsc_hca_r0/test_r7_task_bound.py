"""R7 task-specific readout: no view-only answer aliasing."""
from dataclasses import replace
from unittest import TestCase
from unittest.mock import patch

from elpis.runtime.context import ContextCoordinate

from .baselines import Refusal
from .r7_task_bound import TaskBoundPublication
from .test_r4_hacf_sources import HACFSourceBoundaryTests


class TaskBoundTests(TestCase):
    def setUp(self):
        self.f = HACFSourceBoundaryTests(
            methodName="test_original_bytes_reverified_in_place_and_normalized_digest_bound")
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.p = TaskBoundPublication()

    def refresh(self, task=None, view=None, authority=None):
        f = self.f
        return self.p.refresh(f.root, f.manifest, f.view if view is None else view,
                              f.authority if authority is None else authority,
                              f.bindings, f.task if task is None else task)

    def test_original_task_can_read_exact_published_result(self):
        first = self.refresh()
        self.assertIs(first, self.p.peek(self.f.view.coordinate, self.f.task))
        self.assertEqual(first.result.query, self.f.task)

    def test_another_task_on_the_same_view_cannot_get_old_answers(self):
        first = self.refresh()
        other_task = replace(self.f.task, output_kinds=("different",))
        self.assertIsNone(self.p.peek(self.f.view.coordinate, other_task))
        second = self.refresh(other_task)
        self.assertIsNone(self.p.peek(self.f.view.coordinate, self.f.task))
        self.assertIs(self.p.peek(self.f.view.coordinate, other_task), second)
        self.assertNotEqual(first.result.predictions, second.result.predictions)
        self.assertEqual(second.result.predictions, ())

    def test_task_budget_changes_invalidate_readout_key(self):
        self.refresh()
        other = replace(self.f.task, readout_budget=0)
        self.assertIsNone(self.p.peek(self.f.view.coordinate, other))
        self.assertEqual(self.refresh(other).result.predictions, ())
        self.assertIsNone(self.p.peek(self.f.view.coordinate, self.f.task))

    def test_wrong_view_or_input_types_refused_without_raising(self):
        self.refresh()
        self.assertIsNone(self.p.peek(ContextCoordinate(999, "0" * 64), self.f.task))
        self.assertIsNone(self.p.peek(self.f.view.coordinate, object()))
        self.assertIsNone(self.p.peek("not-coordinate", self.f.task))

    def test_peek_does_not_read_hacf_or_recompute(self):
        issued = self.refresh()
        with patch("research.ecsc_hca_r0.r5_warm_publication.open_verified_hacf",
                   side_effect=AssertionError("unexpected WARM call")), \
             patch("research.ecsc_hca_r0.r5_warm_publication.advance_verified_hacf",
                   side_effect=AssertionError("unexpected WARM call")):
            self.assertIs(self.p.peek(self.f.view.coordinate, self.f.task), issued)

    def test_detected_source_mutation_revokes_old_result(self):
        self.refresh()
        self.f.path.write_bytes(b"changed fact")
        with self.assertRaises(Refusal):
            self.refresh()
        self.assertIsNone(self.p.peek(self.f.view.coordinate, self.f.task))
        self.f.path.write_bytes(self.f.data)
        self.assertIs(self.refresh(), self.p.peek(self.f.view.coordinate, self.f.task))

    def test_scope_refusal_revokes_old_result(self):
        self.refresh()
        with self.assertRaises(Refusal):
            self.refresh(replace(self.f.task, scope="different-project"))
        self.assertIsNone(self.p.peek(self.f.view.coordinate, self.f.task))

    def test_bad_authority_revokes_old_result(self):
        self.refresh()
        with self.assertRaises(Refusal):
            self.refresh(authority=replace(self.f.authority, epoch="unrelated"))
        self.assertIsNone(self.p.peek(self.f.view.coordinate, self.f.task))

    def test_old_view_is_unavailable_after_valid_successor(self):
        from elpis.runtime.context import ContextLimits, successor
        old = self.refresh()
        new_view = successor(self.f.view, (), ContextLimits(), through=2, dropped=0)
        result = self.refresh(view=new_view)
        self.assertIsNone(self.p.peek(old.coordinate, self.f.task))
        self.assertIs(self.p.peek(new_view.coordinate, self.f.task), result)

    def test_refresh_does_not_bypass_hacf_reverification(self):
        self.refresh()
        self.f.path.write_bytes(b"different but still a file")
        with self.assertRaises(Refusal):
            self.refresh()
        self.assertIsNone(self.p.peek(self.f.view.coordinate, self.f.task))
