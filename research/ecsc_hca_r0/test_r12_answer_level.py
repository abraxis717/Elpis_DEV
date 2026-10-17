"""R12 local contracts, including label blinding and journal corruption."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import r12_answer_level as r
from .r9_external_intake import wire,digest


def fixtures():
    samples=[]; gold=[]; r10=[];r11=[]; total=0
    for sid,n in [('conv-26',199),('conv-30',105)]:
        turns=[{'dialog_id':f'D1:{i+1}','speaker':'Alice','text':f'Text {i}'} for i in range(10)]
        qs=[]
        for i in range(n):
            qid=f'{sid}/q{i:04d}';qs.append({'id':qid,'question':'Where is my book?','category':1})
            ids=[t['dialog_id'] for t in turns]
            r10.append({'id':qid,'methods':{'LEXICAL_TOP8':ids[:8],
                                         'LEXICAL4_ADJACENT4':ids[1:9]}})
            r11.append({'id':qid,'turns':ids[2:10]})
            gold.append({'id':qid,'sample_id':sid,'answer_present':total<235,
                         'answer':'Beside the table' if total<235 else None,
                         'evidence_original':['D1:1'],'evidence_unresolved':[],
                         'category':1,'adversarial_answer':None})
            total+=1
        samples.append({'sample_id':sid,'speakers':['Alice','Bob'],
          'sessions':[{'session':'session_1','date_time':'2026-01-01 10:00','turns':turns}],
          'questions':qs})
    dev={'schema':'x','status':'DEV_ONLY','samples':samples}
    ten={'questions':r10};eleven={'questions':r11}
    return dev,ten,eleven,gold


class R12Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev,cls.r10,cls.r11,cls.gold=fixtures()
        cls.pk=r.make_packets(cls.dev,'fake',cls.r10,cls.r11)

    def test_complete_four_way_release_no_gold(self):
        self.assertEqual(len(self.pk['packets']),1216)
        self.assertEqual(set(x['method'] for x in self.pk['packets'][:4]),set(r.METHODS))
        self.assertNotIn('Beside the table',json.dumps(self.pk))
        self.assertNotIn('answer_present',json.dumps(self.pk))
        self.assertNotIn('SCORER_GOLD',json.dumps(self.pk))

    def test_equal_question_and_input_structure(self):
        for i in range(0,len(self.pk['packets']),4):
            group=self.pk['packets'][i:i+4]
            self.assertEqual(len(set(p['id'] for p in group)),1)
            self.assertEqual(len(set(p['question'] for p in group)),1)
            self.assertEqual({p['method']:len(p['turns']) for p in group},
                {'LEXICAL_TOP8':8,'LEXICAL4_ADJACENT4':8,'ECSC_B0_CANDIDATE_RADIUS2_TOP8':8,'NO_MEMORY':0})
            self.assertTrue(all(len(set(x['turn_id'] for x in p['turns']))==len(p['turns']) for p in group))

    def test_source_turns_in_transcript_order(self):
        row=next(x for x in self.pk['packets'][:4] if x['method']=='LEXICAL4_ADJACENT4')
        self.assertEqual(row['turns'][0]['turn_id'],'D1:2')
        self.assertEqual(row['turns'][-1]['turn_id'],'D1:9')

    def test_answer_schema_strict(self):
        self.assertTrue(r.parse_answer('{"answer":"yes","citations":["D1:1"]}',{'D1:1'})['valid'])
        self.assertTrue(r.parse_answer('{"answer":null,"citations":[]}',set())['valid'])
        for raw in ('not json','{"answer":"yes"}',
                    '{"answer":"yes","citations":[]}',
                    '{"answer":null,"citations":["D1:1"]}',
                    '{"answer":"yes","citations":["D1:2"]}',
                    '{"answer":"yes","citations":["D1:1","D1:1"]}'):
            self.assertFalse(r.parse_answer(raw,{'D1:1'})['valid'],raw)

    def test_no_remote_or_redirect_endpoint(self):
        r.check_local('http://127.0.0.1:8081/v1/chat/completions')
        for url in ('https://127.0.0.1:8081/v1/chat/completions',
                    'http://127.0.0.2:8081/v1/chat/completions',
                    'http://evil.com/v1/chat/completions',
                    'http://127.0.0.1:8081/v1/chat/completions?a=1',
                    'http://127.0.0.1:8081/internal'):
            with self.assertRaises(Exception):r.check_local(url)

    def test_exact_match_is_not_semantic_match(self):
        self.assertEqual(r.normalized('Beside, THE table!'),r.normalized('beside the table'))
        self.assertNotEqual(r.normalized('beside the table'),r.normalized('next to the table'))

    def test_scorer_accurate_denominators_and_invalids(self):
        rows=[]
        for i,p in enumerate(self.pk['packets']):
            g=self.gold[i//4]
            raw=('not-json' if i==0 else
                 '{"answer":"Beside the table","citations":["D1:1"]}'
                 if g['answer_present'] and p['method']=='LEXICAL_TOP8' else
                 '{"answer":null,"citations":[]}')
            rows.append({'completion_text':raw,'elapsed_ns':1000})
        report=r.evaluate(self.pk,rows,{'annotations':self.gold})
        invalid_method=self.pk['packets'][0]['method']
        self.assertEqual(report['methods'][invalid_method]['format_invalid'],1)
        self.assertEqual(report['methods']['LEXICAL_TOP8']['exact_answer'],235-int(invalid_method=='LEXICAL_TOP8'))
        self.assertEqual(report['methods']['NO_MEMORY']['exact_answer'],0)
        self.assertEqual(report['methods']['NO_MEMORY']['missing_abstain'],69)
        self.assertEqual(report['methods']['NO_MEMORY']['n'],304)

    def test_journal_tamper_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            packet=self.pk['packets'][0]
            body={'index':0,'id':packet['id'],'method':packet['method'],
                  'completion_text':'{"answer":null,"citations":[]}',
                  'elapsed_ns':100,'previous_sha256':r.ZERO}
            record=dict(body,record_sha256=digest(wire(body)))
            path=d/'R12_DEV_INFERENCE_JOURNAL.jsonl'
            path.write_text(json.dumps(record)+'\n')
            self.assertEqual(len(r.journal_entries(d,self.pk['packets'])),1)
            path.write_text(path.read_text().replace('"elapsed_ns": 100','"elapsed_ns": 101'))
            with self.assertRaises(Exception):r.journal_entries(d,self.pk['packets'])

    def test_torn_journal_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'R12_DEV_INFERENCE_JOURNAL.jsonl').write_text('{}')
            with self.assertRaises(Exception):r.journal_entries(Path(tmp),self.pk['packets'])

    def test_missing_qual_samples(self):
        self.assertEqual([x['id'] for x in self.pk['packets'] if x['id'].startswith('conv-41')],[])


if __name__=='__main__':unittest.main()

# Import-time tests are registered above; this class is discovered by unittest.
class R12Loopback(unittest.TestCase):
    def test_mock_loopback_inference_and_exact_resume(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from threading import Thread
        from tempfile import TemporaryDirectory
        class Handler(BaseHTTPRequestHandler):
            calls=0
            def do_POST(self):
                Handler.calls+=1
                data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                assert data['model']=='test-local'
                raw=json.dumps({'choices':[{'message':{'content':'{"answer":null,"citations":[]}'}}]}).encode()
                self.send_response(200);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(raw)))
                self.end_headers();self.wfile.write(raw)
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with TemporaryDirectory() as base:
                d=Path(base)/'research/ecsc_hca_r0/external/locomo_r0'
                d.mkdir(parents=True)
                prepared={'packets':R12Contracts.pk['packets'][:3]}
                with patch.object(r,'load_prepared',return_value=(d,prepared,{})):
                    endpoint=f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
                    r.infer(base,endpoint,'test-local',2)
                    self.assertEqual(len(r.journal_entries(d,prepared['packets'])),2)
                    r.infer(base,endpoint,'test-local',2)
                    self.assertEqual(len(r.journal_entries(d,prepared['packets'])),3)
                    self.assertEqual(Handler.calls,3)
        finally:
            server.shutdown();server.server_close();thread.join(timeout=2)

class R12Preparation(unittest.TestCase):
    def test_preparation_is_idempotent_and_needs_no_gold(self):
        with tempfile.TemporaryDirectory() as base:
            d=Path(base)
            dev,ten,eleven,gold=fixtures()
            with patch.object(r,'prior_inputs',return_value=(d,dev,'fake',ten,eleven)):
                r.prepare(base)
                first=(d/'R12_DEV_ANSWER_PACKETS.json').read_bytes()
                r.prepare(base)
                self.assertEqual((d/'R12_DEV_ANSWER_PACKETS.json').read_bytes(),first)
                root,packets,pre=r.load_prepared(base)
                self.assertEqual(root,d)
                self.assertEqual(len(packets['packets']),1216)
                self.assertEqual(pre['n_queries'],304)
                self.assertFalse((d/'SCORER_GOLD.json').exists())

    def test_incomplete_inference_cannot_touch_scorer(self):
        with tempfile.TemporaryDirectory() as base:
            d=Path(base)
            prepared={'packets':R12Contracts.pk['packets']}
            with patch.object(r,'load_prepared',return_value=(d,prepared,{})):
                with self.assertRaisesRegex(Exception,'PREDICTIONS_INCOMPLETE'):
                    r.score(base)
            self.assertFalse((d/'R12_DEV_ANSWER_SCORE.json').exists())
