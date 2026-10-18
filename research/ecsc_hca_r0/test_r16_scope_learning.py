"""Direct R16 regression tests: no canonical Elpis mutation, no real model server."""
import unittest
from .r16_scope_learning import (GROUPS, EVAL_SIZE, TRAIN_SIZE, Hold, OPS,
    fixtures,fit_composition,fit_lexical,parse_query,predict,oracle,render,leaf,
    fixture_one,forward,LR,EPOCHS,lexical_probability)

class R16ScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train,cls.public,cls.gold=fixtures()
        cls.model=fit_composition(cls.train)
        cls.lexical=fit_lexical(cls.train)

    def test_cardinality_and_labels(self):
        self.assertEqual((len(self.train),len(self.public),len(self.gold)),
                         (TRAIN_SIZE,EVAL_SIZE,EVAL_SIZE))
        self.assertEqual(len(set(x['id'] for x in self.public)),EVAL_SIZE)

    def test_no_overlap_train_eval_text(self):
        self.assertFalse({x['text'] for x in self.train}&{x['text'] for x in self.public})

    def test_operator_effects_are_model_parameters(self):
        self.assertEqual(self.model['epochs'],EPOCHS)
        self.assertEqual(self.model['step'],LR)
        # The learned matrices, not the parser, determine polarity effects.
        self.assertGreater(self.model['theta']['op']['false that'][0],1)
        self.assertLess(self.model['theta']['op']['false that'][1],-1)

    def test_query_bound_not_distractor(self):
        a={'text':'false that (Mira owns the amber key). Meanwhile, Jonah owns the red ledger.',
           'query':{'name':'Mira','item':'amber key'}}
        b={'text':a['text'],'query':{'name':'Jonah','item':'red ledger'}}
        self.assertNotEqual(parse_query(a),parse_query(b))
        self.assertNotEqual(int(predict(self.model,a)>=.5),int(predict(self.model,b)>=.5))

    def test_unknown_operator_fails_closed(self):
        a={'text':'it is disputed that (Mira owns the amber key).',
           'query':{'name':'Mira','item':'amber key'}}
        with self.assertRaises(Hold):parse_query(a)

    def test_ambiguous_query_refused(self):
        a={'text':'Mira owns the amber key. Meanwhile, false that (Mira owns the amber key).',
           'query':{'name':'Mira','item':'amber key'}}
        with self.assertRaises(Hold):parse_query(a)

    def test_missing_target_refused(self):
        a={'text':'Mira owns the amber key.',
           'query':{'name':'Jonah','item':'amber key'}}
        with self.assertRaises(Hold):parse_query(a)

    def test_unbalanced_parentheses_refused(self):
        a={'text':'false that ((Mira owns the amber key).',
           'query':{'name':'Mira','item':'amber key'}}
        with self.assertRaises(Hold):parse_query(a)

    def test_unknown_syntax_group_refuses(self):
        rows=[x for x in self.public if x['group']=='unknown_operator_refusal']
        self.assertEqual(len(rows),32)
        for row in rows:
            with self.assertRaises(Hold):predict(self.model,row)

    def test_training_replay_exact(self):
        self.assertEqual(self.model,fit_composition(self.train))

    def test_heldout_scores_are_not_selection_thresholds(self):
        # Deliberately no assertion on held-out accuracies; failures must be recorded.
        for case,gold in zip(self.public,self.gold):
            self.assertEqual(case['id'],gold['id'])
            if gold['expected_refusal']:
                self.assertIsNone(gold['label'])
            else:
                self.assertIn(gold['label'],(0,1))
                self.assertGreaterEqual(lexical_probability(self.lexical,case['text']),0)

if __name__=='__main__':unittest.main()
