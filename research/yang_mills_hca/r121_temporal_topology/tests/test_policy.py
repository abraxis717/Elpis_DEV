"""Pre-measurement policy and topology-diagnostic tests. No model execution."""
import unittest
from research.yang_mills_hca.r121_temporal_topology.policy import controller_at
from research.yang_mills_hca.r121_temporal_topology.qualify import signature

class TestStaticPolicy(unittest.TestCase):
    def test_signed_delayed(self):
        ck='54007/STATE8_SIGNED8'
        self.assertEqual(controller_at(ck,'DELAYED8',8),'NATIVE')
        self.assertEqual(controller_at(ck,'DELAYED8',9),'R114')
        self.assertEqual(controller_at(ck,'SIGNED_ONLY',1),'R114')
    def test_unsigned_never_corrects(self):
        for ck in ('54031/STATE8','54007/STATE8_TF1'):
            for h in (1,8,9,20):
                self.assertEqual(controller_at(ck,'DELAYED8',h),'NATIVE')
    def test_unknown_rejected(self):
        with self.assertRaises(ValueError):controller_at('54007/UNKNOWN','DELAYED8',9)
        with self.assertRaises(ValueError):controller_at('54007/STATE8_SIGNED8','DELAYED8',21)
    def test_relation_permutation_invariant(self):
        g=[[3,4,5,7],[0,1,2,6],[2,4,6,9],[0,1,3,8]]
        self.assertEqual(signature(g),signature(g[::-1]))
        self.assertNotEqual(signature(g),signature([[0,1,2,3]]*4))

if __name__=='__main__':unittest.main()
