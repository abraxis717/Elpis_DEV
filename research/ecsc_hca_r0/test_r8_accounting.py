"""Unit checks for separate, post-hoc R8 ledger logic; no Elpis imports."""
import copy
import unittest

from .r8_accounting import accounting

METHODS = ('INDEX_ONLY', 'ADMITTED_ONLY', 'B0', 'B0I', 'B2S')
TARGETS = (5,5,0,5,5,1,5,1,5,5,5,1,5,1,5,5,5,5,5,1)
EVENTS = ('cold','identical_task','seed_removed','seed_restored',
          'same_policy_successor','source_stale','source_valid_again',
          'source_unknown','source_reverified','disconnected_distractor_removed',
          'disconnected_distractor_restored','optional_axiom_and_input_edges_removed',
          'optional_axiom_and_edges_restored','incident_relations_removed_member_retained',
          'incident_relations_restored','optional_queue_drop_counter_binding_retained',
          'partial_asynchronous_schedule','readout_omission','work_budget_refusal','other_scope')


def fixture():
    profiles=[]
    summaries={m:{'answered':0,'refusals':{},'tp':0,'fp':0,'fn':0,'logical_work':0} for m in METHODS}
    for arity in (4,8):
        for reverse in (False,True):
            rows=[]
            for name,n in zip(EVENTS,TARGETS):
                methods={}
                for m in METHODS:
                    refusal = ('WORK_LIMIT' if name == 'work_budget_refusal'
                               else 'SCOPE_TRANSITION' if name == 'other_scope' and m=='B2S'
                               else None)
                    if refusal:
                        methods[m]={'refusal':refusal,'score':None}
                        refusals=summaries[m]['refusals']
                        refusals[refusal]=refusals.get(refusal,0)+1
                    else:
                        tp = (0 if m == 'INDEX_ONLY' or name=='readout_omission'
                              else n if m != 'ADMITTED_ONLY' else (n if name=='cold' else 0))
                        s={'tp':tp,'fp':0,'fn':n-tp,'invalid':0,
                           'logical_work':1,'predicted_keys':['x']*tp}
                        methods[m]={'refusal':None,'score':s}
                        summary=summaries[m]
                        summary['answered']+=1
                        for key in ('tp','fp','fn'):
                            summary[key]+=s[key]
                        summary['logical_work']+=1
                rows.append({'event':name,'withheld_count':n,'methods':methods})
            profiles.append({'arity':arity,'reverse':reverse,'events':rows})
    return {'schema':'elpis.ecsc.r8.dev-epistemic-falsification.v1',
            'profiles':profiles,'summary':summaries}


class R8PostHocAccountingTests(unittest.TestCase):
    def test_known_denominator_and_scope_refusals(self):
        output=accounting(fixture())
        self.assertEqual(output['events'],80)
        self.assertEqual(output['expected_targets'],300)
        s=output['posthoc_accounting']
        self.assertEqual(s['B0']['tp'],260)
        self.assertEqual(s['B0']['fn_all_events'],40)
        self.assertEqual(s['B2S']['tp'],256)
        self.assertEqual(s['B2S']['fn_from_refused_events'],24)
        self.assertEqual(s['B2S']['fn_all_events'],44)
        self.assertEqual(s['B2S']['answered_events'],72)

    def test_wrong_withheld_denominator_refused(self):
        data=fixture()
        data['profiles'][0]['events'][0]['withheld_count']=0
        with self.assertRaisesRegex(ValueError,'PER_EVENT_TARGET_ACCOUNTING'):
            accounting(data)

    def test_forged_existing_summary_refused(self):
        data=fixture()
        data['summary']['B2S']['tp']+=1
        with self.assertRaisesRegex(ValueError,'FROZEN_SUMMARY_MISMATCH'):
            accounting(data)

    def test_prediction_mismatch_refused(self):
        data=fixture()
        data['profiles'][0]['events'][0]['methods']['B0I']['score']['predicted_keys'][0]='other'
        with self.assertRaisesRegex(ValueError,'B0I_PREDICTION_DRIFT'):
            accounting(data)


if __name__=='__main__':
    unittest.main()
