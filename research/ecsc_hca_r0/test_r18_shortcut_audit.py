"""Focused tests for read-only R18 shortcut audit. No canonical imports."""
import unittest
from . import r18_shortcut_audit as a
from . import r18_bracket_binding as r

class RootShortcutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train,cls.public,cls.gold=r.fixtures()
        cls.model=r.fit(cls.train)

    def test_root_rule_is_deliberately_leaf_blind(self):
        texts=['(Mira owns the amber key and Jonah owns the violet folder).',
               '(Mira does not own the amber key and Jonah owns the violet folder).']
        self.assertEqual([a.top_rule(t) for t in texts],[0,0])

    def test_root_rule_changes_on_outer_connective(self):
        self.assertEqual(a.top_rule('(Mira owns the amber key or Jonah owns the violet folder).'),1)
        self.assertEqual(a.top_rule('(Mira owns the amber key and Jonah owns the violet folder).'),0)

    def test_root_rule_tracks_not_parity(self):
        text='(Mira owns the amber key or Jonah owns the violet folder)'
        self.assertEqual(a.top_rule('not ('+text+').'),0)
        self.assertEqual(a.top_rule('not (not ('+text+')).'),1)

    def test_root_rule_refuses_bad_syntax(self):
        with self.assertRaises(r.Hold):a.top_rule('(Mira owns the amber key xor Jonah owns the violet folder).')

    def test_frozen_dev_prediction_equivalence(self):
        predictions=[]
        for row in self.public:
            try:p=r.predict(self.model,row);predictions.append(int(p>=.5))
            except r.Hold:continue
            self.assertEqual(a.top_rule(row['text']),predictions[-1],row['id'])
        self.assertEqual(len(predictions),384)

    def test_primitive_table_matches_explicit_oracle(self):
        table=a.truth_probe(self.model)
        self.assertEqual(len(table),10)
        self.assertEqual(table['leaf.1']['expected'],1)
        self.assertEqual(table['leaf.1']['prediction'],0)
        self.assertEqual(table['or.00']['expected'],0)
        self.assertEqual(table['or.00']['prediction'],1)

    def test_inner_grouping_pair_is_lexically_matched(self):
        pairs=a.held_root_pairs(self.model)
        self.assertEqual(len(pairs),16)
        self.assertTrue(all(p['expected'][0]!=p['expected'][1] for p in pairs))
        self.assertTrue(all(p['same_predicted'] for p in pairs))
        self.assertEqual(sum(p['both_correct'] for p in pairs),0)

    def test_no_source_mutation_by_computations(self):
        text=r.render(('and',r.leaf('Mira','amber key',1),r.leaf('Jonah','violet folder',0)))+'.'
        old=r.sha(text.encode())
        a.top_rule(text);a.truth_probe(self.model)
        self.assertEqual(r.sha(text.encode()),old)

if __name__=='__main__':unittest.main()
