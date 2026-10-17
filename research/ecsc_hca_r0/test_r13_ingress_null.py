"""R13 test-only semantic ingress diagnostic; no language model, no gold."""
import unittest
from .r13_ingress_null import PROBE_TEXT, NativeRefusal, alter_one_turn, inspected_sample


def fixture():
    return {'samples': [{'sample_id':'s1', 'sessions': [{'session':'session_1',
       'turns':[{'dialog_id':'D1:1','speaker':'A','text':'original'},
                {'dialog_id':'D1:2','speaker':'B','text':'untouched'}]}]}]}


class ContentIngressTests(unittest.TestCase):
    def test_replaces_only_target(self):
        source = fixture()
        edited = alter_one_turn(source, 's1','D1:1','changed')
        self.assertEqual(inspected_sample(edited,'s1')['sessions'][0]['turns'][0]['text'],'changed')
        self.assertEqual(inspected_sample(edited,'s1')['sessions'][0]['turns'][1]['text'],'untouched')
        self.assertEqual(inspected_sample(source,'s1')['sessions'][0]['turns'][0]['text'],'original')
    def test_missing_center_refused(self):
        with self.assertRaises(NativeRefusal): alter_one_turn(fixture(),'s1','D1:404','changed')
    def test_missing_sample_refused(self):
        with self.assertRaises(NativeRefusal): inspected_sample(fixture(),'s404')
    def test_duplicate_center_refused(self):
        obj=fixture()
        obj['samples'][0]['sessions'][0]['turns'][1]['dialog_id']='D1:1'
        with self.assertRaises(NativeRefusal):alter_one_turn(obj,'s1','D1:1','changed')
    def test_contrast_contains_explicit_negation(self):
        self.assertIn('does not belong', PROBE_TEXT['NEGATED'])
        self.assertNotIn('does not belong', PROBE_TEXT['AFFIRMATIVE'])
    def test_no_mutable_frozen_gold_access(self):
        import inspect
        from . import r13_ingress_null as r13
        program=inspect.getsource(r13.run)+inspect.getsource(r13.compare_one)
        self.assertNotIn('SCORER_GOLD.json',program)
        self.assertNotIn('QUAL_INPUT',program)
        self.assertNotIn('/v1/chat/completions', program)

if __name__=='__main__': unittest.main()
