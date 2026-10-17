"""Research R14: deterministic learning, label isolation, negative scope probes."""
import math
import unittest
from .r14_native_projector import (
    TRAIN_ITEMS, TRAIN_NAMES, DEV_ITEMS, DEV_NAMES, MODEL_ID, Hold,
    fixture_records, feature_vector, fit, probability, read_exact, SCHEMA,
)

class NativeProjectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.holdout=fixture_records()
        cls.model=fit(cls.train)

    def test_disjoint_train_and_holdout(self):
        self.assertEqual(len(self.train),288)
        self.assertEqual(len(self.holdout),288)
        self.assertEqual(len({r['id'] for r in self.train+self.holdout}),576)
        self.assertFalse({r['text'] for r in self.train}&{r['text'] for r in self.holdout})
        self.assertTrue(set(TRAIN_NAMES).isdisjoint(DEV_NAMES))
        self.assertTrue(set(TRAIN_ITEMS).isdisjoint(DEV_ITEMS))

    def test_label_balance_and_frozen_groups(self):
        groups={g:[r for r in self.holdout if r['group']==g]
                for g in {r['group'] for r in self.holdout}}
        self.assertEqual({k:len(v) for k,v in groups.items()},
                         {'unseen_entities_seen_templates':128,
                          'new_templates_known_negation':64,
                          'new_negation_vocabulary':64,
                          'negation_scope_shift':32})
        for grp in groups.values():self.assertEqual(sum(r['label'] for r in grp)*2,len(grp))

    def test_learned_not_hard_coded(self):
        # A polarity cue may be learned, not imported from a sentiment/polarity lexicon.
        self.assertEqual(self.model['model_id'],MODEL_ID)
        self.assertGreater(self.model['weights'].get('w:not',0),-10)
        self.assertTrue(all(math.isfinite(x) for x in self.model['weights'].values()))
        self.assertGreater(len(self.model['weights']),1)

    def test_deterministic_training(self):
        self.assertEqual(fit(self.train),self.model)
        self.assertEqual(fit(list(reversed(self.train))),self.model)

    def test_feature_bounding(self):
        self.assertRaises(Hold,feature_vector,'')
        self.assertRaises(Hold,feature_vector,'x'*513)
        self.assertRaises(Hold,feature_vector,'word '*100)

    def test_r13_content_pair_is_discriminated(self):
        pos='The amber key belongs to Mira.'
        neg='The amber key does not belong to Mira.'
        self.assertNotIn(pos,{x['text'] for x in self.train})
        self.assertNotIn(neg,{x['text'] for x in self.train})
        self.assertGreater(probability(self.model,pos),.5)
        self.assertLess(probability(self.model,neg),.5)

    def test_known_template_entity_generalization_smoke(self):
        group=[r for r in self.holdout if r['group']=='unseen_entities_seen_templates']
        self.assertEqual(sum(int(probability(self.model,r['text'])>=.5)==r['label'] for r in group),128)

    def test_scope_shift_evaluation_kept_without_pass_threshold(self):
        # Intentionally do not assert a high or low accuracy: scientific reporting
        # must record whichever outcome occurs without post-hoc test tuning.
        group=[r for r in self.holdout if r['group']=='negation_scope_shift']
        self.assertEqual(len(group),32)
        self.assertEqual(sum(r['label'] for r in group),16)
        self.assertTrue(all(0.0 < probability(self.model,r['text']) < 1.0 for r in group))

if __name__=='__main__':unittest.main()
