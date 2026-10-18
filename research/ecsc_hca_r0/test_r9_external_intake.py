"""R9 intake mechanics only; synthetic unit fixtures NEVER scientific examples."""
import json
import tempfile
import unittest
from pathlib import Path

import r9_external_intake as intake


def fixture():
    return [{
        'sample_id':'conv-26',
        'conversation':{
            'speaker_a':'A','speaker_b':'B','session_1_date_time':'2024-01-01 10:00',
            'session_1':[{'speaker':'A','dia_id':'D1:1','text':'The lantern is green.'},
                         {'speaker':'B','dia_id':'D1:2','text':'Acknowledged.'}],
        },
        'qa':[
            {'question':'What color was the lantern?', 'category':1,
             'answer':'green','evidence':['D1:1']},
            {'question':'Who won?', 'category':5,
             'adversarial_answer':'An invented answer','evidence':['D1:2']},
        ]}]


def data(x):
    return json.dumps(x,ensure_ascii=True).encode()


class ExternalIntakeTests(unittest.TestCase):
    def test_gold_split_and_missing_answer(self):
        public,gold,audit=intake.convert(data(fixture()),enforce_release=False)
        qa=public['samples'][0]['questions']
        self.assertEqual(len(qa),2)
        for x in qa:
            self.assertEqual(set(x),{'id','question','category'})
        self.assertNotIn('green',json.dumps(qa))
        self.assertNotIn('An invented answer',json.dumps(public))
        self.assertEqual(gold['annotations'][0]['answer'],'green')
        self.assertFalse(gold['annotations'][1]['answer_present'])
        self.assertEqual(audit['missing_answer_count'],1)

    def test_unresolved_evidence_preserved_not_repaired(self):
        f=fixture();f[0]['qa'][0]['evidence']=['D1:1; D1:2']
        _,gold,audit=intake.convert(data(f),enforce_release=False)
        self.assertEqual(audit['unresolved_evidence_reference_count'],1)
        self.assertEqual(gold['annotations'][0]['evidence_original'],['D1:1; D1:2'])
        self.assertEqual(gold['annotations'][0]['evidence_unresolved'],['D1:1; D1:2'])

    def test_non_abstention_missing_answer_is_anomaly(self):
        f=fixture();f[0]['qa'][0].pop('answer')
        _,_,audit=intake.convert(data(f),enforce_release=False)
        self.assertIn('MISSING_NON_ABSTENTION_ANSWER',[a['type'] for a in audit['anomalies']])

    def test_duplicate_dialogue_refused(self):
        f=fixture();f[0]['conversation']['session_1'][1]['dia_id']='D1:1'
        with self.assertRaisesRegex(intake.IntakeError,'DUPLICATE_DIALOG_ID'):
            intake.convert(data(f),enforce_release=False)

    def test_released_blob_pin_rejects_tamper(self):
        with self.assertRaisesRegex(intake.IntakeError,'SOURCE_PIN_MISMATCH'):
            intake.convert(data(fixture()),enforce_release=True)

    def test_write_idempotency_conflict_and_no_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.json'
            self.assertTrue(intake.safe_write(p,b'ABC'))
            self.assertFalse(intake.safe_write(p,b'ABC'))
            with self.assertRaisesRegex(intake.IntakeError,'EXISTING_OUTPUT_CONFLICT'):
                intake.safe_write(p,b'CHANGED')
            s=Path(td)/'link.json'
            s.symlink_to(p)
            with self.assertRaisesRegex(intake.IntakeError,'EXISTING_OUTPUT_CONFLICT'):
                intake.safe_write(s,b'ABC')

    def test_git_blob_identity(self):
        data=b'abc'
        self.assertEqual(intake.gitblob(data),__import__('hashlib').sha1(b'blob 3\0abc').hexdigest())
        self.assertNotEqual(intake.gitblob(b'abc'),intake.gitblob(b'abd'))

    def test_public_has_no_gold_for_large_sample(self):
        f=fixture();f[0]['qa'] *= 40
        _,private,audit=intake.convert(data(f),enforce_release=False)
        self.assertEqual(len(private['annotations']),80)
        self.assertEqual(audit['question_count'],80)
        self.assertEqual(audit['missing_answer_count'],40)

    def test_pinned_local_corruption_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'locomo10.json';p.write_bytes(data(fixture()))
            with self.assertRaisesRegex(intake.IntakeError,'SOURCE_PIN_MISMATCH'):
                intake.get_pinned_source(p)


if __name__=='__main__':
    unittest.main()
