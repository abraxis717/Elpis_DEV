"""Pure accounting regressions for frozen DEV R10/R11 paired evidence diagnostics."""
from __future__ import annotations
from copy import deepcopy
import unittest

from .r11_paired_audit import paired_evaluation, AuditError, METHODS
from .r9_external_intake import DEV_IDS


def fixture():
    dev={'status':'DEV_ONLY','samples':[]};gold={'annotations':[]};r10={'top_k':8,'methods':list(METHODS[:2]),'questions':[]};r11={'top_k':8,'method':METHODS[2],'questions':[]}
    for sid in DEV_IDS:
        sample={'sample_id':sid,'sessions':[{'turns':[{'dialog_id':f'D1:{i}'} for i in range(1,12)]}], 'questions':[]}
        for qi in range(152):
            qid=sid+'/q'+str(qi).zfill(4)
            sample['questions'].append({'id':qid,'category':(qi%5)+1})
            gold['annotations'].append({'id':qid,'sample_id':sid,'category':(qi%5)+1,
                'answer_present':len(gold['annotations'])<235,
                'evidence_original': ['D1:1'] if len(gold['annotations']) < 235 else [],
                'evidence_unresolved':[]})
            r10['questions'].append({'id':qid,'methods':{
                METHODS[0]:[f'D1:{i}' for i in range(1,9)],
                METHODS[1]:[f'D1:{i}' for i in range(2,10)]}})
            r11['questions'].append({'id':qid,'turns':[f'D1:{i}' for i in range(3,11)]})
        dev['samples'].append(sample)
    # Exact gold denominator = 310 (235 answered + 75 second references).
    for x in gold['annotations'][:75]:x['evidence_original'].append('D1:2')
    return dev,gold,r10,r11


class PairAuditTests(unittest.TestCase):
    def test_expected_pairwise_losses_and_denominators(self):
        report=paired_evaluation(*fixture())
        self.assertEqual(report['methods'][METHODS[0]]['answer_present']['gold_references'],310)
        self.assertEqual(report['methods'][METHODS[0]]['answer_present']['matched_references'],310)
        self.assertEqual(report['methods'][METHODS[2]]['answer_present']['matched_references'],0)
        key=METHODS[2]+'__VS__'+METHODS[0]
        self.assertEqual(report['paired_all_questions'][key]['answer_present']['gold_refs_lost'],310)
        self.assertEqual(report['paired_all_questions'][key]['answer_present']['all_losses'],235)
    def test_invalid_source_turn_refused(self):
        d,g,a,b=fixture();b['questions'][0]['turns'][0]='D99:99'
        with self.assertRaisesRegex(AuditError,'INVALID_SELECTION'):paired_evaluation(d,g,a,b)
    def test_duplicate_selection_refused(self):
        d,g,a,b=fixture();b['questions'][0]['turns'][1]=b['questions'][0]['turns'][0]
        with self.assertRaisesRegex(AuditError,'INVALID_SELECTION'):paired_evaluation(d,g,a,b)
    def test_gold_anomaly_alteration_refused(self):
        d,g,a,b=fixture();g['annotations'][0]['evidence_unresolved']=['D1:2']
        with self.assertRaisesRegex(AuditError,'EVIDENCE_ANOMALY_DRIFT'):paired_evaluation(d,g,a,b)
    def test_gold_category_alteration_refused(self):
        d,g,a,b=fixture();g['annotations'][0]['category']=5
        with self.assertRaisesRegex(AuditError,'GOLD_METADATA'):paired_evaluation(d,g,a,b)
    def test_duplicate_question_ids_refused(self):
        d,g,a,b=fixture();b['questions'][1]['id']=b['questions'][0]['id']
        with self.assertRaisesRegex(AuditError,'R11_ORDER'):paired_evaluation(d,g,a,b)
    def test_heldout_b0_win_is_counted(self):
        d,g,a,b=fixture()
        g['annotations'][0]['evidence_original']=['D1:10','D1:2']
        report=paired_evaluation(d,g,a,b)
        name=METHODS[2]+'__VS__'+METHODS[0]
        pair=report['paired_all_questions'][name]['answer_present']
        self.assertEqual(pair['gold_refs_gained'],1)
        self.assertEqual(pair['gold_refs_lost'],309)

    def test_unresolved_reference_is_explicit_miss(self):
        d,g,a,b=fixture()
        g['annotations'][0]['evidence_original']=['D99:100','D1:2']
        g['annotations'][0]['evidence_unresolved']=['D99:100']
        report=paired_evaluation(d,g,a,b)
        self.assertEqual(report['methods'][METHODS[0]]['answer_present']['unresolved_references'],1)
        self.assertEqual(report['methods'][METHODS[2]]['answer_present']['unresolved_references'],1)

    def test_qual_label_excluded(self):
        d,g,a,b=fixture();g['annotations'].append({'id':'conv-41/q0000','sample_id':'conv-41'})
        report=paired_evaluation(d,g,a,b)
        self.assertEqual(len(report['question_accounting']),304)

if __name__=='__main__':unittest.main()
