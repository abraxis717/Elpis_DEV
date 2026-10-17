"""Focussed integrity checks; do not inspect withheld gold or assert scientific outcome thresholds."""
import unittest
from . import r16_scope_learning as s
from . import r17_raw_scope as r
from . import r17_scope_ablation_audit as a
from .r9_external_intake import wire

class ScopeAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train,_,_=s.fixtures()
        cls.model=r.fit_raw(cls.train)
        cls.model.update({'training_sha256':a.r.TRAIN_PIN,'eval_public_sha256':r.INPUT_PINS[r.PUBLIC_FILE]})
        cls.public,_=r.fixtures()
        cls.preds=[]
        for row in cls.public:
            try:
                p=r.raw_predict(cls.model,row)
                cls.preds.append({'id':row['id'],'raw_status':'ANSWER','raw_answer':int(p>=.5),'raw_p':round(p,9)})
            except r.Hold:
                cls.preds.append({'id':row['id'],'raw_status':'REFUSED','raw_answer':None,'raw_p':None})

    def test_model_byte_identity(self):
        self.assertEqual(r.sha(wire(self.model)),a.MODEL_SHA)

    def test_original_public_pin(self):
        self.assertEqual(r.sha(wire({'schema':r.SCHEMA+'.eval-public','cases':self.public})),
                         r.INPUT_PINS[r.PUBLIC_FILE])

    def test_no_gold_needed_for_audit(self):
        self.assertTrue(all('label' not in row for row in self.public))
        data=a.analyze(self.public,self.preds,self.model)
        self.assertEqual(data['evaluated_answerable']+data['original_refusals_excluded_from_ablation'],384)

    def test_mutations_apply_only_to_text(self):
        case={'id':'x','text':'not false that (false that (Mira owns the amber key)).','query':{'name':'Mira','item':'amber key'}}
        for mutation in a.MODIFICATIONS.values():
            new={**case,'text':mutation(case['text'])}
            self.assertEqual(new['query'],case['query'])
            self.assertNotEqual(new['text'],case['text'])

    def test_modified_accounting(self):
        report=a.analyze(self.public,self.preds,self.model)
        for v in report['controls'].values():
            self.assertEqual(v['modified'],v['same_answer']+v['changed_answer']+v['refused'])

    def test_wrong_id_refused(self):
        bad=[dict(x) for x in self.preds]
        bad[0]['id']='forged'
        with self.assertRaises(r.Hold):a.analyze(self.public,bad,self.model)

    def test_drifted_original_prediction_refused(self):
        bad=[dict(x) for x in self.preds]
        first=next(x for x in bad if x['raw_status']=='ANSWER')
        first['raw_answer']=1-first['raw_answer']
        with self.assertRaises(r.Hold):a.analyze(self.public,bad,self.model)

if __name__=='__main__':unittest.main()
