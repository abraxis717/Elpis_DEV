"""R15 regression: a learned hypothesis stays source-bound and non-authoritative."""
import unittest

from elpis.runtime.context import (
    ADMITTED,CANDIDATE,PROVISIONAL,ContextAxiom,ContextReference,ContextRelation,
    EMPTY_VIEW,Observation,successor,
)
from .baselines import Authority,AxiomGrant,Baselines,Refusal,SourceGrant,SourceCheck
from .r11_candidate_graph import h
from .r12_native_structure import LIMITS,VIEW_LIMITS,NativeRefusal
from .r14_native_projector import fit,fixture_records
from .r15_candidate_ingress import (
    MODEL_PIN,AFFIRM_CANDIDATE,NEGATE_CANDIDATE,add_candidate,
    candidate_readout,learned_proposal,
)


class CandidateIngressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train,_=fixture_records()
        cls.model=fit(train)

    def foundation(self):
        scope='dev-r15-test'
        ref=ContextReference(h('r15.fixture.chunk'),h('r15.fixture.corpus'),h('r15.fixture.provenance'))
        anchor=ContextAxiom(h('r15.fixture.anchor'),(ref,),PROVISIONAL)
        view=successor(EMPTY_VIEW,(Observation(axiom=anchor),),VIEW_LIMITS,
                       through=1,dropped=0)
        source_digest=h('r15.fixture.bytes')
        auth=Authority((AxiomGrant(anchor,scope,'turn'),),
                       (SourceGrant(ref,scope,source_digest),),(),(),epoch='R15_TEST_ONLY')
        return scope,anchor,view,auth,(SourceCheck(ref,scope,source_digest),)

    def graph(self,label='The amber key belongs to Mira.'):
        scope,anchor,view,auth,checks=self.foundation()
        prop=learned_proposal(self.model,MODEL_PIN,label,anchor.references[0].chunk_digest)
        newview,newauth,hyp,rel=add_candidate(view,auth,{'center':anchor},scope,'center',prop)
        graph=Baselines.build(newview,newauth,checks,LIMITS)
        return graph,scope,anchor,hyp,rel,prop,newview,newauth,checks

    def test_affirmative_proposal_is_candidate(self):
        _,_,_,hyp,rel,prop,_,_,_=self.graph()
        self.assertEqual(prop['polarity'],1)
        self.assertEqual(prop['type'],AFFIRM_CANDIDATE)
        self.assertEqual(rel.status,CANDIDATE)
        self.assertEqual(hyp.status,PROVISIONAL)

    def test_negative_proposal_is_candidate(self):
        _,_,_,hyp,rel,prop,_,_,_=self.graph('The amber key does not belong to Mira.')
        self.assertEqual(prop['polarity'],0)
        self.assertEqual(prop['type'],NEGATE_CANDIDATE)
        self.assertEqual(rel.status,CANDIDATE)
        self.assertEqual(hyp.status,PROVISIONAL)

    def test_candidate_readout_and_admitted_refusal(self):
        graph,scope,anchor,hyp,rel,prop,*_=self.graph()
        visible,_=candidate_readout(graph,scope,anchor,hyp,prop['type'])
        inadmissible,_=candidate_readout(graph,scope,anchor,hyp,prop['type'],ADMITTED)
        self.assertTrue(visible)
        self.assertFalse(inadmissible)

    def test_opposite_relation_does_not_match(self):
        graph,scope,anchor,hyp,rel,prop,*_=self.graph()
        self.assertFalse(candidate_readout(graph,scope,anchor,hyp,NEGATE_CANDIDATE)[0])

    def test_original_structure_is_unchanged(self):
        scope,anchor,view,auth,checks=self.foundation()
        prop=learned_proposal(self.model,MODEL_PIN,'The amber key belongs to Mira.',
                              anchor.references[0].chunk_digest)
        newview,newauth,hyp,rel=add_candidate(view,auth,{'center':anchor},scope,'center',prop)
        self.assertEqual(newview.predecessor,view.digest)
        self.assertEqual(view.axioms,(anchor,))
        self.assertEqual(view.relations,())
        self.assertEqual(newview.axioms,(hyp,anchor) if hyp.key<anchor.key else (anchor,hyp))
        self.assertEqual(len(newview.relations),1)

    def test_authority_mismatch_fails(self):
        graph,scope,anchor,hyp,rel,prop,newview,newauth,checks=self.graph()
        forged=ContextRelation(rel.relation_type,rel.members,rel.provenance_digest,ADMITTED)
        bad=Authority(newauth.axioms,newauth.sources,(forged,),newauth.schemas,
                      epoch=newauth.epoch)
        with self.assertRaises(Refusal):
            Baselines.build(newview,bad,checks,LIMITS)

    def test_unknown_source_check_not_accepted(self):
        graph,scope,anchor,hyp,rel,prop,newview,newauth,checks=self.graph()
        bad=(SourceCheck(checks[0].reference,checks[0].scope,h('different bytes')),)
        stale=Baselines.build(newview,newauth,bad,LIMITS)
        with self.assertRaises(Refusal):
            candidate_readout(stale,scope,anchor,hyp,prop['type'])

    def test_provenance_changes_with_source_identity(self):
        p1=learned_proposal(self.model,MODEL_PIN,'The amber key belongs to Mira.',h('source-one'))
        p2=learned_proposal(self.model,MODEL_PIN,'The amber key belongs to Mira.',h('source-two'))
        self.assertNotEqual(p1['claim_key'],p2['claim_key'])
        self.assertNotEqual(p1['model_provenance_digest'],p2['model_provenance_digest'])
        self.assertEqual(p1['polarity'],p2['polarity'])

    def test_model_pin_and_invalid_identity_fail(self):
        with self.assertRaises(NativeRefusal):
            learned_proposal(self.model,'0'*64,'The amber key belongs to Mira.',h('src'))
        with self.assertRaises(NativeRefusal):
            learned_proposal(self.model,MODEL_PIN,'The amber key belongs to Mira.','ABC')

    def test_deterministic_candidate(self):
        p=learned_proposal(self.model,MODEL_PIN,'The amber key belongs to Mira.',h('src'))
        self.assertEqual(p,learned_proposal(self.model,MODEL_PIN,
                                            'The amber key belongs to Mira.',h('src')))


if __name__=='__main__':unittest.main()
