"""R17 mechanics tests. Never score withheld R17 labels here."""
import re
import unittest
from . import r17_raw_scope as r
from .r16_scope_learning import fixtures as old_fixtures
from .r9_external_intake import wire

class R17UnparsedSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train,_,_=old_fixtures()
        cls.model=r.fit_raw(cls.train)
        cls.public,_=r.fixtures()

    def test_public_fixture_pin_and_shape(self):
        self.assertEqual(len(self.public),r.EVAL_COUNT)
        self.assertEqual(len(set(x['id'] for x in self.public)),r.EVAL_COUNT)
        self.assertEqual(r.sha(wire({'schema':r.SCHEMA+'.eval-public','cases':self.public})),r.INPUT_PINS[r.PUBLIC_FILE])

    def test_train_data_not_eval_gold(self):
        self.assertEqual(len(self.train),r.TRAIN_COUNT)
        self.assertTrue(all('label' not in x for x in self.public))
        self.assertFalse(set(x['text'] for x in self.train) & set(x['text'] for x in self.public))

    def test_raw_clauses_no_parsed_operators(self):
        case={'text':'false that (Mira does not own the amber key). Meanwhile, Jonah owns the red ledger.',
              'query':{'name':'Mira','item':'amber key'}}
        words=r.choose_unparsed_tokens(case)
        self.assertIn('false',words)
        self.assertIn('not',words)
        self.assertIn('@name',words)
        self.assertIn('@item',words)
        self.assertNotIn('operators',words)
        self.assertFalse(any(isinstance(w,dict) for w in words))

    def test_query_changes_sentence_without_tree_extraction(self):
        case={'text':'false that (Mira owns the amber key). Meanwhile, Jonah does not own the red ledger.',
              'query':{'name':'Mira','item':'amber key'}}
        left=r.choose_unparsed_tokens(case)
        case['query']={'name':'Jonah','item':'red ledger'}
        right=r.choose_unparsed_tokens(case)
        self.assertNotEqual(left,right)

    def test_ambiguous_targets_refused(self):
        case={'text':'Mira owns the amber key. Meanwhile, Mira does not own the amber key.',
              'query':{'name':'Mira','item':'amber key'}}
        with self.assertRaises(r.Hold):r.choose_unparsed_tokens(case)

    def test_missing_query_refused(self):
        case={'text':'Mira owns the amber key.',
              'query':{'name':'Jonah','item':'red ledger'}}
        with self.assertRaises(r.Hold):r.choose_unparsed_tokens(case)

    def test_unknown_vocabulary_refused(self):
        case={'text':'it is disputed that (Mira owns the amber key).',
              'query':{'name':'Mira','item':'amber key'}}
        with self.assertRaises(r.Hold):r.raw_predict(self.model,case)

    def test_unbalanced_scope_refused(self):
        case={'text':'false that ((Mira owns the amber key).',
              'query':{'name':'Mira','item':'amber key'}}
        with self.assertRaises(r.Hold):r.choose_unparsed_tokens(case)

    def test_correct_identity_mask_new_names(self):
        a={'text':'false that (Ava owns the blue ribbon).',
           'query':{'name':'Ava','item':'blue ribbon'}}
        b={'text':'false that (Zara owns the gold lantern).',
           'query':{'name':'Zara','item':'gold lantern'}}
        self.assertEqual(r.choose_unparsed_tokens(a),r.choose_unparsed_tokens(b))

    def test_counterfactual_cases_same_context_except_target_polarity(self):
        cases=[x for x in self.public if x['group']=='paired_polarity_counterfactual']
        self.assertEqual(len(cases),64)
        for a,b in zip(cases[::2],cases[1::2]):
            self.assertEqual(a['query'],b['query'])
            self.assertNotEqual(a['text'],b['text'])
            wa=r.choose_unparsed_tokens(a)
            wb=r.choose_unparsed_tokens(b)
            self.assertNotEqual(wa,wb)
            neg=['does','not','own']
            def standardize(words):
                for j in range(len(words)-2):
                    if words[j:j+3]==neg: return words[:j]+['owns']+words[j+3:]
                return words
            self.assertEqual(standardize(wa),standardize(wb))

    def test_train_algorithm_deterministic(self):
        self.assertEqual(self.model,r.fit_raw(self.train))

    def test_model_has_no_prescribed_operator_label(self):
        self.assertEqual(self.model['architecture'],'QUERY_BOUND_UNPARSED_LEXICAL_TRANSITION')
        self.assertFalse({'true that','false that','not false that'} & set(self.model['theta']['tokens']))
        self.assertGreater(len(self.model['theta']['tokens']),5)

    def test_public_no_new_model_writes_in_tests(self):
        # Hold outcome assertions until the workstation scorer: no threshold tuning on EVAL.
        self.assertEqual(sum(n for _,n in r.GROUPS),384)

if __name__=='__main__':unittest.main()
