"""R19 DEV guard tests: do not inspect evaluation labels for performance selection."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
from . import r19_fixed_root as m

class R19Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train,cls.public,cls.gold=m.fixtures()

    def test_counts_pins_and_no_gold_public(self):
        self.assertEqual((len(self.train),len(self.public),len(self.gold)),(640,544,544))
        for name,kind,records in ((m.TRAIN,'train',self.train),(m.PUBLIC,'public',self.public),(m.GOLD,'gold',self.gold)):
            self.assertEqual(m.sha(m.wire({'schema':m.SCHEMA+'.'+kind,'records':records})),m.PINS[name])
        self.assertTrue(all('label' not in item and 'refuse' not in item for item in self.public))

    def test_holdout_and_training_text_nonoverlap(self):
        self.assertFalse({x['text'] for x in self.train}&{x['text'] for x in self.public})
        self.assertEqual(len(set(x['id'] for x in self.train+self.public)),1184)

    def test_fixed_root_pairs_are_opposite_label_and_same_tokens(self):
        for train,n,deep in ((True,128,False),(False,128,False),(False,64,True)):
            for i in range(n):
                (left,a),(right,b)=m.discriminative_pair(i,train,deep)
                self.assertNotEqual(a,b)
                self.assertEqual(m.r.bare_tokens(left['text']),m.r.bare_tokens(right['text']))
                self.assertNotEqual(left['text'],right['text'])
                for row in (left,right):self.assertIsNotNone(m.r.parse(row['text']))

    def test_root_only_pair_ceiling(self):
        for i in range(128):
            sides=m.discriminative_pair(i,False)
            roots=[]
            for row,y in sides:
                t=m.r.parse(row['text']);parity=0
                while t[0]=='not':t=t[1];parity+=1
                roots.append((int(t[0]=='or') ^ (parity%2),y))
            self.assertEqual(roots[0][0],roots[1][0])
            self.assertEqual(sum(p==y for p,y in roots),1)

    def test_training_balances_formal_truth_table(self):
        prim=[x for x in self.train if x['group']=='primitive_generalization']
        self.assertEqual(len(prim),192)
        labels=[x['label'] for x in prim]
        self.assertEqual(sum(labels),16*(1+1+4))  # two leaves, two NOTs, eight binary combinations
        self.assertTrue(240<=sum(x['label'] for x in self.train)<=400)

    def test_compound_trainable_truth_tables(self):
        model=m.r.fit(self.train)
        table={k:m.r.sigmoid(val)>=.5 for k,val in model['theta'].items()}
        correct={'leaf.owns':True,'leaf.does not own':False,'not.0':True,'not.1':False,
                 'and.0':False,'and.1':False,'and.2':False,'and.3':True,
                 'or.0':False,'or.1':True,'or.2':True,'or.3':True}
        self.assertEqual(table,correct)

    def test_training_exact_determinism(self):
        self.assertEqual(m.r.fit(self.train),m.r.fit(self.train))
        self.assertEqual(m.train_raw(self.train),m.train_raw(self.train))

    def test_raw_token_path_does_not_call_parser(self):
        with patch.object(m.r,'parse',side_effect=AssertionError('parser used')):
            model=m.train_raw(self.train)
            p=m.predict_raw(model,self.public[0]['text'])
            self.assertTrue(0<p<1)

    def test_unknown_and_malformed_refuse_structured(self):
        invalid=[x for x in self.public if x['group'].endswith('refusal')]
        self.assertEqual(len(invalid),64)
        for item in invalid:
            with self.assertRaises(m.r.Hold):m.r.parse(item['text'])

    def test_predictor_runs_without_gold_file(self):
        with TemporaryDirectory() as temp:
            root=m.root_for(temp);root.mkdir(parents=True)
            for name,kind,records in ((m.TRAIN,'train',self.train),(m.PUBLIC,'public',self.public)):
                (root/name).write_bytes(m.wire({'schema':m.SCHEMA+'.'+kind,'records':records}))
            with patch.object(m,'source_integrity',return_value=None):
                m.infer(temp)
            self.assertFalse((root/m.GOLD).exists())
            self.assertTrue((root/m.MODEL).exists())
            self.assertTrue((root/m.PRED).exists())

    def test_byte_identical_existing_files_and_conflicts(self):
        with TemporaryDirectory() as tmp:
            f=Path(tmp)/'file'
            m.r.write_once(f,b'abc');m.r.write_once(f,b'abc')
            with self.assertRaises(m.r.Hold):m.r.write_once(f,b'abd')
            self.assertEqual(f.read_bytes(),b'abc')

    def test_infer_is_idempotent_without_gold(self):
        with TemporaryDirectory() as tmp:
            root=m.root_for(tmp);root.mkdir(parents=True)
            for name,kind,records in ((m.TRAIN,'train',self.train),(m.PUBLIC,'public',self.public)):
                (root/name).write_bytes(m.wire({'schema':m.SCHEMA+'.'+kind,'records':records}))
            with patch.object(m,'source_integrity',return_value=None):
                m.infer(tmp)
                initial=[m.sha((root/n).read_bytes()) for n in (m.MODEL,m.PRED)]
                m.infer(tmp)
                self.assertEqual(initial,[m.sha((root/n).read_bytes()) for n in (m.MODEL,m.PRED)])

if __name__=='__main__':unittest.main()
