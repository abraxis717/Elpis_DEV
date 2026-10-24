"""Mutation-adequacy tests. A failed expected rejection is a HARD test failure.
No catch-all AssertionError swallowing. All mutants are kept out of evidence inputs.
"""
from copy import deepcopy
import checker
import oracle

def run(train,gold):
    # keep a separate mutated gold map where appropriate; never alter originals.
    cases=train['worlds'][0]['cases']
    tests={}
    def candidate(name,change):
        d=deepcopy(train);g=deepcopy(gold)
        change(d,g)
        try:
            checker.audit(d,g,'TRAIN')
        except checker.ContractError:
            tests[name]='REJECTED'
        else:
            raise AssertionError('MUTANT ACCEPTED: '+name)
    def mutate_gold(d,g):
        c=d['worlds'][0]['cases'][0];g[c['case_id']][0][c['queries'][0]['id']] ^= 1
    candidate('gold_flip',mutate_gold)
    candidate('public_label_leak',lambda d,g:d['worlds'][0]['cases'][0].update({'gold':1}))
    candidate('case_duplicate',lambda d,g:d['worlds'][0]['cases'][1].update(case_id=d['worlds'][0]['cases'][0]['case_id']))
    candidate('pair_drop',lambda d,g:d['worlds'][0]['cases'].pop())
    candidate('query_drift',lambda d,g:d['worlds'][0]['cases'][0]['queries'][0].update(to='illegitimate_query_drift'))
    candidate('event_step_collision',lambda d,g:d['worlds'][0]['cases'][0]['events'][1].update(step=0))
    candidate('factor_duplicate',lambda d,g:d['worlds'][0]['cases'][0]['factors'][1].update(id=d['worlds'][0]['cases'][0]['factors'][0]['id']))
    candidate('unknown_status',lambda d,g:d['worlds'][0]['cases'][0]['events'][0].update(status='BANANA'))
    candidate('source_forgery',lambda d,g:d['worlds'][0]['cases'][0]['events'][0].update(source='nonexistent_source'))
    candidate('missing_gold',lambda d,g:g.pop(d['worlds'][0]['cases'][0]['case_id']))
    candidate('role_binding_change',lambda d,g:d['worlds'][0]['cases'][0]['factors'][1]['roles'].update({'context':'spoofed_context'}))
    candidate('invalid_epoch',lambda d,g:d['worlds'][0]['cases'][0]['factors'][0].update(version=999))
    candidate('wrong_split',lambda d,g:d.update(split='SEALED'))
    candidate('duplicate_query_identity',lambda d,g:d['worlds'][0]['cases'][0]['queries'][1].update(id=d['worlds'][0]['cases'][0]['queries'][0]['id']))
    candidate('pair_target_change',lambda d,g:d['worlds'][0]['cases'][0].update(target_query_id='nonsense_id'))
    candidate('extra_gold',lambda d,g:g.update({'injected':'label'}))
    # Mutate permitted causal/nuisance inputs AND recompute gold; rejection
    # must come from the structural pair contract, not stale-label detection.
    def case_kind(d,kind):
        return next(s for s in d['worlds'][0]['cases'] if s['kind']==kind)
    def rescore(g,s):g[s['case_id']]=oracle.all_prefix_labels(s)
    def source_nuisance(d,g):
        s=case_kind(d,'source_swap')
        s['events'][6]['status']='UNKNOWN';rescore(g,s)
    def event_multiset(d,g):
        s=case_kind(d,'event_order')
        s['events'][6]['status']='UNKNOWN';rescore(g,s)
    def epoch_histogram(d,g):
        s=case_kind(d,'epoch_rebind')
        s['events'][7]['epoch']=1;rescore(g,s)
    def structural_marginal(d,g):
        s=case_kind(d,'structural')
        s['factors'][20]['roles']['to']='new_valid_opaque_node';rescore(g,s)
    candidate('source_swap_nuisance_rescored',source_nuisance)
    candidate('event_order_multiset_rescored',event_multiset)
    candidate('epoch_histogram_rescored',epoch_histogram)
    candidate('structural_marginal_rescored',structural_marginal)
    print('PASS_H1C0_MUTANT_ADEQUACY='+str(len(tests))+'/'+str(len(tests)),flush=True)
    return tests
