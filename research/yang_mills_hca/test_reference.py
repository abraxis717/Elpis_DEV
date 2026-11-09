"""Tiny algebra sanity checks, not scientific HCA requalification."""
import unittest
from research.yang_mills_hca.reference import gains,correct_node

class ControllerTest(unittest.TestCase):
    def test_policy(self):
        self.assertEqual(gains('54007/STATE8_SIGNED8'),(.70,.45))
        self.assertEqual(gains('54031/STATE8_TF1'),(.50,.50))
        with self.assertRaises(ValueError): gains('bogus')
    def test_radial(self):
        x=(1.,)+(0.,)*31; p=(2.,)+(0.,)*31
        self.assertAlmostEqual(correct_node(x,p,'54007/STATE8_SIGNED8')[0],2.7)
    def test_transverse(self):
        x=(1.,)+(0.,)*31; p=(1.,2.)+(0.,)*30
        self.assertAlmostEqual(correct_node(x,p,'54007/STATE8_SIGNED8')[1],2.9)
    def test_shape(self):
        with self.assertRaises(ValueError): correct_node((0.,), (0.,), '54007/STATE8')
