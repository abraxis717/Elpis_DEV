"""R10 label-blind method, integrity and evidence-score unit tests."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from .r10_blind_retrieval import (
    R10Error, SCHEMA, K, lexical_ranking, indexed_turns, produce_sample,
    verify_public, predictions_from_public, evaluate, predict, score,
)
from .r9_external_intake import DEV_IDS, SCHEMA as R9_SCHEMA, wire, digest, IntakeError


def fixture():
    samples=[]; gold=[]
    for sid in DEV_IDS:
        t1=[{'dialog_id':f'D1:{i+1}','speaker':'Alice',
             'text':('the rosemary garden next to the museum' if i==1 else f'message number {i} about tea')}
             for i in range(8)]
        t2=[{'dialog_id':f'D2:{i+1}','speaker':'Bob','text':f'message {i} about running'} for i in range(8)]
        questions=[]
        for i in range(152):
            qid=f'{sid}/q{i:04d}'
            questions.append({'id':qid,'question':'Which garden is next to the museum?', 'category':1 if i<150 else 5})
            gold.append({'id':qid,'sample_id':sid,'category':1 if i<150 else 5,
                         'answer_present':i<150,'answer':'rosemary' if i<150 else None,
                         'adversarial_answer':None,
                         'evidence_original':['D1:2'] if i%2 else ['D1:2','INVALID'],
                         'evidence_unresolved':[] if i%2 else ['INVALID']})
        samples.append({'sample_id':sid,'speakers':['Alice','Bob'],
                        'sessions':[{'session':'session_1','date_time':'Monday', 'turns':t1},
                                    {'session':'session_2','date_time':'Tuesday', 'turns':t2}],
                        'questions':questions})
    return {'schema':R9_SCHEMA+'.public-dev','status':'DEV_ONLY','samples':samples}, \
           {'schema':R9_SCHEMA+'.gold','status':'SCORER_ONLY','annotations':gold}


class R10Tests(unittest.TestCase):
    def setUp(self):
        self.dev,self.gold=fixture()

    def test_all_304_predicted_without_labels(self):
        p=predictions_from_public(self.dev,'0'*64)
        self.assertEqual(len(p['questions']),304)
        self.assertTrue(all(len(q['methods'][m])==K for q in p['questions'] for m in p['methods']))

    def test_category_never_affects_predictions(self):
        a=predictions_from_public(self.dev,'0'*64)['questions']
        b=deepcopy(self.dev)
        for s in b['samples']:
            for q in s['questions']:q['category']=5 if q['category']==1 else 1
        self.assertEqual(a,predictions_from_public(b,'0'*64)['questions'])

    def test_label_injection_refused(self):
        s=deepcopy(self.dev)
        s['samples'][0]['questions'][0]['answer']='leaked gold'
        with self.assertRaisesRegex(R10Error,'PUBLIC_QUESTION_KEYS'):
            predictions_from_public(s,'0'*64)
        s=deepcopy(self.dev)
        s['samples'][0]['sessions'][0]['turns'][0]['evidence']='D1:1'
        with self.assertRaisesRegex(R10Error,'PUBLIC_TURN_KEYS'):
            predictions_from_public(s,'0'*64)

    def test_no_cross_session_adjacent_edges(self):
        turns, adj=indexed_turns(self.dev['samples'][0])
        self.assertNotIn(8,adj[7]);self.assertNotIn(7,adj[8])

    def test_no_cross_sample_prediction(self):
        p=predictions_from_public(self.dev,'0'*64)
        for s in self.dev['samples']:
            legal={t['dialog_id'] for block in s['sessions'] for t in block['turns']}
            for q in p['questions']:
                if q['id'].startswith(s['sample_id']+'/'):
                    for arr in q['methods'].values():self.assertTrue(set(arr)<=legal)

    def test_unresolved_evidence_is_counted_not_repaired(self):
        p=predictions_from_public(self.dev,'0'*64)
        report=evaluate(p,self.gold,self.dev)
        stats=report['methods']['LEXICAL_TOP8']['answer_present']
        self.assertEqual(stats['queries'],300)
        self.assertEqual(stats['unresolved_evidence_references'],150)
        self.assertEqual(stats['evidence_references_raw'],450)
        self.assertEqual(stats['matched_references'],300)
        self.assertEqual(stats['all_hit_queries'],150)
        self.assertEqual(report['methods']['LEXICAL_TOP8']['missing_answer']['queries'],4)

    def test_forged_turn_is_refused(self):
        p=predictions_from_public(self.dev,'0'*64)
        p['questions'][0]['methods']['LEXICAL_TOP8'][0]='QUAL-ONLY:TURN'
        with self.assertRaisesRegex(R10Error,'PREDICTION_INVALID_TURN'):
            evaluate(p,self.gold,self.dev)

    def test_missing_gold_refused(self):
        p=predictions_from_public(self.dev,'0'*64)
        self.gold['annotations'].pop()
        with self.assertRaisesRegex(R10Error,'DEV_GOLD_IDS'):
            evaluate(p,self.gold,self.dev)

    def test_sample_reorder_refused(self):
        self.dev['samples'].reverse()
        with self.assertRaisesRegex(R10Error,'PUBLIC_IDS_ORDER'):
            verify_public(self.dev)

    def test_predictor_needs_no_raw_source_or_gold(self):
        # Local subprocess entry point's pure predictor has no source/gold open.
        # An attested receipt is constructed for this isolated unit fixture only.
        with TemporaryDirectory() as temp:
            d=Path(temp)/'research/ecsc_hca_r0/external/locomo_r0';d.mkdir(parents=True)
            wire_public=wire(self.dev)
            (d/'DEV_INPUT.json').write_bytes(wire_public)
            receipt={'schema':SCHEMA+'.attestation','status':'R9_BYTES_RECONSTRUCTED',
                     'source_sha256':'79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4',
                     'source_git_blob_sha1':'d95b872480b413d935821fdc3c84f8a8f5f29e73',
                     'dev_input_sha256':digest(wire_public),'scorer_gold_sha256':'a'*64,
                     'r9_audit_sha256':'b'*64,'sample_ids':list(DEV_IDS),'qual_count':8}
            (d/'R10_INPUT_ATTESTATION.json').write_bytes(wire(receipt))
            # Both these objects are ABSENT; predictor cannot read either.
            self.assertFalse((d/'SCORER_GOLD.json').exists())
            self.assertFalse((d/'locomo10.json').exists())
            predict(temp)
            self.assertTrue((d/'R10_DEV_PREDICTIONS.json').is_file())
            predict(temp) # idempotence
            changed=json.loads((d/'R10_DEV_PREDICTIONS.json').read_text())
            changed['questions'][0]['methods']['LEXICAL_TOP8'][0]='FAKE'
            (d/'R10_DEV_PREDICTIONS.json').write_bytes(wire(changed))
            with self.assertRaisesRegex(IntakeError,'EXISTING_OUTPUT_CONFLICT'):
                predict(temp)

if __name__=='__main__':unittest.main()
