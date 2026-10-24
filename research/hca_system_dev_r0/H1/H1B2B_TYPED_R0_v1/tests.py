"""Contract and mutant adequacy tests; rejects forbidden input and real semantic drifts.
No catch-all around assertions; failures in tests remain failures.
"""
import copy
from factor_graph import vectorize_case,source_snapshot,prepare_batch,create_model,train_fixed,predict,metrics

class ExpectedFailure(Exception):pass

def rejects(fn,expected=(ValueError,KeyError,TypeError)):
    try:fn()
    except expected:return True
    raise ExpectedFailure('forbidden mutation silently accepted')


def run_mutants(contract,oracle):
    import torch
    world=contract.build_split('TRAIN',0x123456ab,worlds=1,pairs_per_type=1)['worlds'][0]
    case=world['probes'][0];support=[s['label'] for s in world['support']]
    names=[]
    p=copy.deepcopy(case);p['label']=0
    assert rejects(lambda:vectorize_case(p,support));names.append('public_target_label_injection')
    p=copy.deepcopy(case);p['hidden_rule']='A/C'
    assert rejects(lambda:vectorize_case(p,support));names.append('hidden_rule_injection')
    assert rejects(lambda:vectorize_case(case,[support[0]]));names.append('missing_support')
    assert rejects(lambda:vectorize_case(case,[0,2]));names.append('illegal_support_value')
    p=copy.deepcopy(case);p['events'][0]['kind']='FUTURE'
    assert rejects(lambda:vectorize_case(p,support));names.append('invalid_event_kind')
    p=copy.deepcopy(case);p['events'][0]['status']='FUTURE'
    assert rejects(lambda:vectorize_case(p,support));names.append('invalid_status')
    p=copy.deepcopy(case);p['events'][0]['epoch']=-1
    assert rejects(lambda:vectorize_case(p,support));names.append('negative_epoch')
    p=copy.deepcopy(case);p['events'][0]['ordinal']=p['events'][1]['ordinal'];p['events'][0]['tick']=p['events'][1]['tick']
    assert rejects(lambda:vectorize_case(p,support));names.append('colliding_event_times')
    p=copy.deepcopy(case);p['factors'].append(copy.deepcopy(p['factors'][0]))
    assert rejects(lambda:vectorize_case(p,support));names.append('duplicate_factor')
    p=copy.deepcopy(case);p['factors'][0]['rel']='FOREIGN'
    assert rejects(lambda:vectorize_case(p,support));names.append('unknown_relation')
    # Do not let named mutant tests pass by swallowing their own assertions.
    before=vectorize_case(case,support)
    after=vectorize_case(case,support,blind_source=True)
    assert before['factor_x']!=after['factor_x'] or before['node_x']!=after['node_x']
    names.append('source_mask_changes_representation')
    assert vectorize_case(case,support,swap_roles=True)['links']!=before['links']
    names.append('role_swap_changes_representation')
    assert vectorize_case(case,support,erase_factors=True)['factor_x']==[]
    names.append('factor_erasure_changes_representation')
    # Gold is supplied only to scoring, never public graph vectorization.
    train=contract.build_split('TRAIN',0x351126,worlds=1,pairs_per_type=1)
    gold={p['case_id']:oracle.solve(p,oracle.identify_rule(w['support'],contract.RULES))
          for w in train['worlds'] for p in w['probes']}
    b=prepare_batch(train['worlds'],gold)
    assert len(b['ids'])==8 and set(b['y'].tolist())=={0.,1.}
    model=create_model(hidden=12,rounds=2)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(model(b),b['y'])
    loss.backward()
    assert model.to_factor.grad is not None and model.to_node.grad is not None
    assert model.to_factor.grad.norm().item()>0 and model.to_node.grad.norm().item()>0
    names.append('bipartite_role_weight_gradient_nonzero')
    assert sum(v.numel() for v in model.parameters())>0
    names.append('actually_trainable_nonempty_model')
    assert len(names)==15
    print('PASS_H1B2B_SEMANTIC_AND_MUTATION_SUITE='+str(len(names))+'/'+str(len(names)),flush=True)
    for name in names:print('H1B2B_TEST='+name+' PASS',flush=True)
    return names
