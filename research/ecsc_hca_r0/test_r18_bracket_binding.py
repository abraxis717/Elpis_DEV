"""DEV mechanics tests; deliberately do not open/score frozen evaluation gold outcomes."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from . import r18_bracket_binding as r

class PairedBracketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train,cls.public,cls.gold=r.fixtures()

    def test_frozen_fixture_hashes(self):
        for name,kind,rows in ((r.TRAIN_FILE,'.train',self.train),(r.PUBLIC_FILE,'.public',self.public),(r.GOLD_FILE,'.gold',self.gold)):
            self.assertEqual(r.sha(r.wire({'schema':r.SCHEMA+kind,'records':rows})),r.FIXTURE_PINS[name])

    def test_no_gold_in_public(self):
        self.assertTrue(all('label' not in row and 'refuse' not in row for row in self.public))
        self.assertEqual(len(self.public),448)
        self.assertEqual(len(self.train),640)
        self.assertFalse({row['text'] for row in self.train}&{row['text'] for row in self.public})

    def test_label_changing_pairs_match_words_and_order(self):
        for group,n in (('flip',r.TRAIN_PAIRS),('flip',r.EVAL_PAIRS),('deep',r.DEEP_PAIRS)):
            for i in range(n):
                (a,la),(b,lb)=r.pair(group,i,group=='flip' and n==r.TRAIN_PAIRS)
                self.assertEqual(r.bare_tokens(a['text']),r.bare_tokens(b['text']))
                self.assertNotEqual(la,lb)
                self.assertNotEqual(a['text'],b['text'])

    def test_group_neutral_pairs(self):
        for i in range(r.NEUTRAL_PAIRS):
            (a,la),(b,lb)=r.pair('neutral',i,True)
            self.assertEqual(r.bare_tokens(a['text']),r.bare_tokens(b['text']))
            self.assertEqual(la,lb)

    def test_parentheses_change_parsed_structure(self):
        (a,la),(b,lb)=r.pair('flip',9,False)
        self.assertNotEqual(r.parse(a['text']),r.parse(b['text']))
        self.assertNotEqual(la,lb)

    def test_unknown_operator_and_malformed_refuse(self):
        examples=[x for x in self.public if x['group'].endswith('refusal')]
        self.assertEqual(len(examples),64)
        for item in examples:
            with self.assertRaises(r.Hold):r.parse(item['text'])

    def test_no_unparenthesized_binary_reinterpretation(self):
        for text in ('Mira owns the amber key and Jonah owns the violet folder.',
                     '(Mira owns the amber key and Jonah owns the violet folder and Theo owns the gold lantern).',
                     'not Mira owns the amber key.',
                     '((Mira owns the amber key and Jonah owns the violet folder).'):
            with self.assertRaises(r.Hold):r.parse(text)

    def test_gradient_finite_difference(self):
        tree=r.parse('(Mira owns the amber key or Jonah does not own the violet folder).')
        theta={k:.2+(.01*i) for i,k in enumerate(r.PARAMS)}
        p,tape=r.forward(theta,tree);grad={k:0. for k in theta}
        r.backward(tape,1.,grad)
        for k in theta:
            delta=1e-5
            left=dict(theta);right=dict(theta)
            left[k]-=delta;right[k]+=delta
            numerical=(r.forward(right,tree)[0]-r.forward(left,tree)[0])/(2*delta)
            self.assertAlmostEqual(grad[k],numerical,places=6,msg=k)

    def test_train_is_deterministic(self):
        a=r.fit(self.train)
        self.assertEqual(a,r.fit(self.train))
        self.assertEqual(a['architecture'],'SUPPLIED_PARENTHESIS_PARSER_LEARNED_BOOLEAN_GATES')
        self.assertTrue(set(a['theta'])==set(r.PARAMS))

    def test_predictor_does_not_open_gold(self):
        # Use a temp root with no gold whatsoever; inference must still complete.
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/r.EVIDENCE_DIR;root.mkdir(parents=True)
            for name,data in ((r.TRAIN_FILE,self.train),(r.PUBLIC_FILE,self.public)):
                suffix='.train' if name==r.TRAIN_FILE else '.public'
                (root/name).write_bytes(r.wire({'schema':r.SCHEMA+suffix,'records':data}))
            old=r.provenance
            r.provenance=lambda _:None
            try:r.infer(tmp)
            finally:r.provenance=old
            self.assertFalse((root/r.GOLD_FILE).exists())
            self.assertTrue((root/r.PRED_FILE).is_file())

    def test_conflict_refused(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'x';p.write_bytes(b'abc')
            with self.assertRaises(r.Hold):r.write_once(p,b'abd')
            self.assertEqual(p.read_bytes(),b'abc')

    def test_strict_public_hash(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'x';p.write_bytes(b'{"schema":"x"}')
            with self.assertRaises(r.Hold):r.read_json(p,'x',r.FIXTURE_PINS[r.PUBLIC_FILE])

if __name__=='__main__':unittest.main()
