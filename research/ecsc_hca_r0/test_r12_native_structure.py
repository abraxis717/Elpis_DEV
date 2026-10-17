"""R12N DEV tests: structural relation admission != semantic truth."""
import unittest
from .r12_native_structure import (NativeRefusal,sequence_index,verify_order_witness,
    selected_centers,source_graph,NEXT,PREVIOUS)


def sample():
    def turns(ids):
        return [{'dialog_id':i,'speaker':'A','text':'text '+i} for i in ids]
    return {'sample_id':'dev-fixture','sessions':[
        {'session':'session_1','date_time':'2025','turns':turns(('D1:1','D1:2','D1:3','D1:4','D1:5'))},
        {'session':'session_2','date_time':'2025','turns':turns(('D2:1','D2:2','D2:3','D2:4','D2:5'))}]}

class NativeOrderTests(unittest.TestCase):
    def test_forward_witness(self):
        self.assertTrue(verify_order_witness(sample(),'D1:2','D1:3','NEXT'))
    def test_reverse_witness(self):
        self.assertTrue(verify_order_witness(sample(),'D1:3','D1:2','PREVIOUS'))
    def test_opposite_direction_refused(self):
        with self.assertRaises(NativeRefusal):verify_order_witness(sample(),'D1:2','D1:3','PREVIOUS')
    def test_skipped_turn_refused(self):
        with self.assertRaises(NativeRefusal):verify_order_witness(sample(),'D1:1','D1:3','NEXT')
    def test_cross_session_refused(self):
        with self.assertRaises(NativeRefusal):verify_order_witness(sample(),'D1:5','D2:1','NEXT')
    def test_missing_turn_refused(self):
        with self.assertRaises(NativeRefusal):verify_order_witness(sample(),'D1:1','D1:99','NEXT')
    def test_invalid_relation_type_refused(self):
        with self.assertRaises(NativeRefusal):verify_order_witness(sample(),'D1:1','D1:2','TEMPORALLY_BEFORE')
    def test_duplicate_ids_refused(self):
        a=sample();a['sessions'][1]['turns'][0]['dialog_id']='D1:1'
        with self.assertRaises(NativeRefusal):sequence_index(a)
    def test_anchor_selection_deterministic(self):
        a=sample()
        self.assertEqual(selected_centers(a,2),selected_centers(a,2))
        self.assertEqual(len(selected_centers(a,2)),2)
        for previous,center,nxt in selected_centers(a,2):
            self.assertTrue(verify_order_witness(a,center,nxt,'NEXT'))
            self.assertTrue(verify_order_witness(a,center,previous,'PREVIOUS'))
    def test_source_api_does_not_accept_gold(self):
        import inspect
        from . import r12_native_structure as ns
        self.assertNotIn('gold',inspect.getsource(ns.run).lower())
        self.assertNotIn('answer_model',inspect.getsource(ns.run))

if __name__=='__main__':unittest.main()
