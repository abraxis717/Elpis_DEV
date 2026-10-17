"""R11 research regression: source-backed candidate graph, no qualification data."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from elpis.runtime.context import CANDIDATE
from .r9_external_intake import wire, digest
from .r10_blind_retrieval import verify_public, R10Error
from .test_r10_blind_retrieval import fixture
from .r11_candidate_graph import (source_ranges, derived_manifest, scoped_graph,
                                  run_one, LIMITS, VIEW_LIMITS, demand)
from .r4_hacf_sources import verified_source_checks
from .baselines import Refusal


class CandidateGraphTests(unittest.TestCase):
    def setUp(self):
        self.dev,_=fixture()
        self.raw=wire(self.dev)
        self.spans,self.n=source_ranges(self.raw,self.dev)
        self.manifest=derived_manifest(self.raw,self.n)
        self.sample=self.dev['samples'][0]
        self.query=self.sample['questions'][0]

    def test_exact_turn_json_source_ranges(self):
        self.assertEqual(self.n,32)
        self.assertEqual(len(self.spans),32)
        for sample in self.dev['samples']:
            for block in sample['sessions']:
                for turn in block['turns']:
                    _,a,b=self.spans[sample['sample_id'],turn['dialog_id']]
                    self.assertEqual(self.raw[a:b], __import__('json').dumps(turn['text'],ensure_ascii=True).encode())

    def test_changed_turn_text_refuses_old_source(self):
        other=deepcopy(self.dev)
        other['samples'][0]['sessions'][0]['turns'][0]['text']='forged'
        with self.assertRaisesRegex(R10Error,'R11_NOT_R9_RELEASE_BYTES'):
            source_ranges(self.raw,other)

    def test_manifest_bound_to_release_bytes(self):
        self.assertEqual(self.manifest.document(digest(self.raw)).size_bytes,len(self.raw))
        self.assertEqual(self.manifest.document(digest(self.raw)).chunk_count,self.n)
        self.assertNotEqual(derived_manifest(self.raw+b'x',self.n).digest,self.manifest.digest)

    def test_all_edges_candidate_and_within_session(self):
        view,auth,bindings,task,lookup,nodes,edges=scoped_graph(
            self.sample,self.spans,self.raw,self.manifest,self.query['question'])
        self.assertGreaterEqual(nodes,8)
        self.assertLessEqual(nodes,24)
        self.assertLessEqual(edges,32)
        self.assertEqual(len(bindings),nodes)
        self.assertEqual(len(view.axioms),nodes)
        self.assertEqual(len(view.relations),edges)
        self.assertTrue(all(r.status==CANDIDATE for r in view.relations))
        self.assertTrue(all(s.weight<1000 for s in auth.schemas))
        self.assertEqual(len(task.seeds),len(set(task.seeds)))
        self.assertLessEqual(len(task.seeds),8)
        view.verify(VIEW_LIMITS)

    def test_category_does_not_enter_graph(self):
        original=scoped_graph(self.sample,self.spans,self.raw,self.manifest,self.query['question'])
        other=deepcopy(self.sample);other['questions'][0]['category']=5
        modified=scoped_graph(other,self.spans,self.raw,self.manifest,other['questions'][0]['question'])
        self.assertEqual(original,modified)

    def test_dev_samples_never_mix_turn_authority(self):
        a=scoped_graph(self.dev['samples'][0],self.spans,self.raw,self.manifest,
                       self.dev['samples'][0]['questions'][0]['question'])
        b=scoped_graph(self.dev['samples'][1],self.spans,self.raw,self.manifest,
                       self.dev['samples'][1]['questions'][0]['question'])
        self.assertNotEqual(a[0].digest,b[0].digest)
        self.assertNotEqual(a[3].scope,b[3].scope)
        self.assertNotEqual(a[4],b[4])

    def test_real_hacf_bytes_verified_and_b0_produces_eight(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'DEV_INPUT.json').write_bytes(self.raw)
            predicted,work=run_one(self.sample,self.spans,self.raw,self.manifest,
                                   self.query,root)
            self.assertEqual(len(predicted),8)
            self.assertEqual(len(set(predicted)),8)
            self.assertGreater(work['source_chunks_verified'],0)
            self.assertGreater(work['admission_work'],0)
            self.assertGreater(work['readout_work'],0)

    def test_stale_source_cannot_produce_readout(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'DEV_INPUT.json').write_bytes(self.raw+b'\n')
            with self.assertRaises(Exception):
                run_one(self.sample,self.spans,self.raw,self.manifest,self.query,root)

    def test_symlink_source_fails_closed(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'original').write_bytes(self.raw)
            (root/'DEV_INPUT.json').symlink_to('original')
            with self.assertRaises(Exception):
                run_one(self.sample,self.spans,self.raw,self.manifest,self.query,root)

    def test_label_leak_refused_at_public_boundary(self):
        other=deepcopy(self.dev)
        other['samples'][0]['questions'][0]['evidence']=['D1:2']
        with self.assertRaises(R10Error): verify_public(other)


if __name__ == '__main__':unittest.main()
