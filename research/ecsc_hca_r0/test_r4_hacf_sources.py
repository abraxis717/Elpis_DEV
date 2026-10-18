"""R4 DEV research-only HACF source-identity bridge tests. No canonical writes."""
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from elpis.runtime.context import (
    ContextAxiom, ContextLimits, ContextReference, EMPTY_VIEW, Observation, successor,
)
from elpis.structure.retrieval.objects import (
    ChunkClaim, CorpusManifest, chunk_identity, normalize,
)

from .baselines import AxiomGrant, Authority, Refusal, SourceGrant
from .r4_hacf_sources import (
    HACFBinding, advance_verified_hacf, open_verified_hacf, verified_source_checks,
)
from .task import Task


def sha(value):
    return sha256(value).hexdigest()


class HACFSourceBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = b"source fact  \n"
        self.path = self.root / "original.txt"
        self.path.write_bytes(self.data)
        self.doc_digest = sha(self.data)
        self.profile = sha(b"R4 test chunker profile; no native chunker assertion")
        self.norm_digest = sha(normalize(self.data))
        self.claim = ChunkClaim(chunk_identity(self.doc_digest, 0, 0, len(self.data),
                                               self.norm_digest, self.profile),
                                self.doc_digest, 0, 0, len(self.data))
        document = {"digest": self.doc_digest, "size_bytes": len(self.data),
                    "chunk_count": 1, "chunk_profile_digest": self.profile,
                    "media_type": "text/plain"}
        raw = json.dumps({"documents": [document]}, sort_keys=True, separators=(",", ":")).encode()
        self.manifest = CorpusManifest.verified(raw, expected_digest=sha(raw))
        self.ref = ContextReference(self.claim.chunk_digest, self.manifest.digest,
                                    sha(b"R4 synthetic proposal; NOT admission proof"))
        self.axiom = ContextAxiom(sha(b"R4 example axiom"), (self.ref,))
        self.view = successor(EMPTY_VIEW, (Observation(axiom=self.axiom),), ContextLimits(),
                              through=1, dropped=0)
        self.authority = Authority((AxiomGrant(self.axiom, "project", "source"),),
                                   (SourceGrant(self.ref, "project", self.norm_digest),),
                                   (), (), epoch="TEST_ONLY_R4")
        self.bindings = (HACFBinding("project", self.ref, self.claim, "original.txt"),)
        self.task = Task("project", (self.axiom.key,), (), ("source",))

    def checks(self, **kw):
        return verified_source_checks(self.root, self.manifest, self.view,
                                      self.authority, self.bindings, **kw)

    def refusal(self, code, fn):
        with self.assertRaises(Refusal) as caught:
            fn()
        self.assertEqual(caught.exception.code, code)

    def open(self):
        return open_verified_hacf(self.root, self.manifest, self.view,
                                  self.authority, self.bindings, self.task)

    def test_original_bytes_reverified_in_place_and_normalized_digest_bound(self):
        (check,) = self.checks()
        self.assertEqual(check.reference, self.ref)
        self.assertEqual(check.observed_digest, self.norm_digest)
        state, result, meta = self.open()
        self.assertEqual(meta["lifetime"], "EXPLICIT_COLD_OPEN")
        self.assertEqual(result.coordinate, self.view.coordinate)
        self.assertEqual(tuple(p.hit.grant.axiom.key for p in result.predictions),
                         (self.axiom.key,))
        self.assertTrue(state.verified.graph.axioms[self.axiom.key].source_valid)
        self.assertEqual(state.verified.graph.axioms[self.axiom.key].source_state, "VALID")
        self.assertEqual(state.checks, (check,))

    def test_changed_existing_file_fails_closed_without_mutating_state(self):
        state, result, _ = self.open()
        before = repr(state)
        self.path.write_bytes(b"altered fact \n")
        self.refusal("SOURCE_INTEGRITY", self.checks)
        self.refusal("SOURCE_INTEGRITY", lambda: advance_verified_hacf(
            state, self.root, self.manifest, self.view, self.authority,
            self.bindings, self.task))
        self.assertEqual(before, repr(state))
        self.assertEqual(result.coordinate, self.view.coordinate)

    def test_missing_and_extra_reference_bindings_refuse(self):
        self.refusal("SOURCE_BINDING", lambda: verified_source_checks(
            self.root, self.manifest, self.view, self.authority, ()))
        self.refusal("SOURCE_BINDING", lambda: verified_source_checks(
            self.root, self.manifest, self.view, self.authority,
            self.bindings + self.bindings))
        self.refusal("SOURCE_BINDING", lambda: verified_source_checks(
            self.root, self.manifest, self.view, self.authority,
            (replace(self.bindings[0], scope="unrelated"),)))

    def test_corpus_and_chunk_coordinate_refuse_forgery(self):
        self.refusal("SOURCE_COORDINATE", lambda: verified_source_checks(
            self.root, self.manifest, self.view, self.authority,
            (replace(self.bindings[0], claim=replace(self.claim,
                     chunk_digest=sha(b"forged"))),)))
        changed_view_ref = replace(self.ref, corpus_digest=sha(b"other manifest"))
        changed_axiom = replace(self.axiom, references=(changed_view_ref,))
        v = successor(self.view, (Observation(axiom=changed_axiom),), ContextLimits(),
                      through=2, dropped=0)
        grants = replace(self.authority, axioms=(AxiomGrant(changed_axiom, "project", "source"),))
        self.refusal("SOURCE_COORDINATE", lambda: verified_source_checks(
            self.root, self.manifest, v, grants, self.bindings))

    def test_independently_granted_text_digest_must_match(self):
        modified = replace(self.authority, sources=(SourceGrant(self.ref, "project", sha(b"lie")),))
        self.refusal("SOURCE_ATTESTATION", lambda: verified_source_checks(
            self.root, self.manifest, self.view, modified, self.bindings))

    def test_escaped_path_and_symlink_refused_by_canonical_resolver(self):
        self.refusal("SOURCE_INTEGRITY", lambda: verified_source_checks(
            self.root, self.manifest, self.view, self.authority,
            (replace(self.bindings[0], source_path="../original.txt"),)))
        (self.root / "shortcut.txt").symlink_to(self.path)
        self.refusal("SOURCE_INTEGRITY", lambda: verified_source_checks(
            self.root, self.manifest, self.view, self.authority,
            (replace(self.bindings[0], source_path="shortcut.txt"),)))

    def test_whole_document_budget_is_preflighted(self):
        self.refusal("SOURCE_BUDGET", lambda: self.checks(
            max_document_bytes=4, max_total_document_bytes=4))

    def test_successor_remains_incremental_and_old_state_unchanged(self):
        state, initial, _ = self.open()
        old = repr(state)
        successor_view = successor(self.view, (), ContextLimits(), through=2, dropped=0)
        other, result, meta = advance_verified_hacf(
            state, self.root, self.manifest, successor_view, self.authority,
            self.bindings, self.task)
        self.assertEqual(meta["lifetime"], "VERIFIED_IMMEDIATE_SUCCESSOR")
        self.assertEqual(result.coordinate, successor_view.coordinate)
        self.assertEqual(repr(state), old)
        self.assertEqual(initial.coordinate, self.view.coordinate)

    def test_verification_is_not_proof_of_semantic_admission(self):
        state, _, _ = self.open()
        self.assertEqual(state.verified.graph.view.relations, ())
        self.assertEqual(self.axiom.status, "PROVISIONAL")


if __name__ == "__main__":
    unittest.main()
