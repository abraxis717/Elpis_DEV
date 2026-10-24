"""Adversarial contract tests; deliberate breakage MUST cause actual rejections."""
import copy
import engines


def fixture():
    return {'query':{'from':'a','to':'c','scope':'T'},
            'factors':[{'fid':'x','rel':'A','roles':{'from':'a','to':'b','scope':'T'},'source':'s0','source_version':0},
                       {'fid':'y','rel':'C','roles':{'from':'b','to':'c','scope':'T'},'source':'s1','source_version':0}],
            'events':[{'tick':0,'ordinal':0,'source':'s0','status':'VALID','epoch':0,'kind':'SOURCE'},
                      {'tick':0,'ordinal':1,'source':'s1','status':'VALID','epoch':0,'kind':'SOURCE'},
                      {'tick':1,'ordinal':0,'source':'s0','status':'STALE','epoch':0,'kind':'SOURCE'},
                      {'tick':2,'ordinal':0,'source':'s0','status':'VALID','epoch':0,'kind':'SOURCE'}]}


def expected_fails(label,fn,kind):
    try:fn()
    except kind:
        print('MUTANT_REJECTED='+label,flush=True)
    else:raise AssertionError('MUTANT_ACCEPTED='+label)


def test_all():
    cases=[]
    mutants=[]
    def case(name,scene,rule,trajectory):
        actual,_,_,_,_=engines.assert_parity(scene,rule)
        if actual!=trajectory:raise AssertionError('expected dynamics missing: '+name+' '+repr(actual))
        cases.append(name)
    s=fixture(); rule=('A','C')
    case('stale_then_valid',s,rule,[0,0,1,0,1])
    u=copy.deepcopy(s);u['events'][-1]['status']='UNKNOWN'
    case('unknown_cannot_recover',u,rule,[0,0,1,0,0])
    u=copy.deepcopy(s);u['events'][-1]['epoch']=1
    case('epoch_advance_invalidates_old',u,rule,[0,0,1,0,0])
    u=copy.deepcopy(s);u['factors'][0]['source_version']=1
    case('future_binding_not_early',u,rule,[0,0,0,0,0])
    # Above scene's epoch stays 0 so future binding never matches.
    u=copy.deepcopy(s);u['factors'][1]['roles']['scope']='OTHER'
    case('wrong_scope_cannot_compose',u,rule,[0,0,0,0,0])
    u=copy.deepcopy(s);u['factors'][1]['roles']['from']='not-b'
    case('wrong_binding_cannot_compose',u,rule,[0,0,0,0,0])
    u=copy.deepcopy(s);u['factors'][1]['rel']='D'
    case('wrong_relation_cannot_compose',u,rule,[0,0,0,0,0])
    u=copy.deepcopy(s);u['factors'][1]['source']='s0'
    case('source_changes_symmetric',u,rule,[0,1,1,0,1])
    u=copy.deepcopy(s)
    u['factors'][1]['roles']['to']='a'
    u['factors'].append({'fid':'z','rel':'A','roles':{'from':'a','to':'b','scope':'T'},'source':'s1','source_version':0})
    case('unsupported_cycle_not_truth',u,rule,[0,0,0,0,0])
    u=copy.deepcopy(s);u['factors'][0]['source_version']=1;u['events'][-1]['epoch']=1
    case('future_epoch_activates_only_after_rebind',u,rule,[0,0,0,0,1])
    u=copy.deepcopy(s);u['factors'][1]['fid']=u['factors'][0]['fid']
    expected_fails('duplicate_factor_identity',lambda:engines.stream_indexed(u,rule),ValueError);mutants.append('duplicate_factor_identity')
    u=copy.deepcopy(s);u['events'][2]['tick']=0;u['events'][2]['ordinal']=0
    expected_fails('nonmonotone_event_rejected',lambda:engines.stream_indexed(u,rule),ValueError);mutants.append('nonmonotone_event_rejected')
    u=copy.deepcopy(s);u['events'][2]['status']='ELEVATED'
    expected_fails('invalid_status_rejected',lambda:engines.stream_indexed(u,rule),ValueError);mutants.append('invalid_status_rejected')
    # Effectiveness: force a scorer to lie and prove parity auditor rejects its changed output.
    saved=engines.SourceIndexedExact.score
    try:
        def corrupt(self,rule):return 1-saved(self,rule)
        engines.SourceIndexedExact.score=corrupt
        expected_fails('parity_auditor_detects_false_engine',lambda:engines.assert_parity(s,rule),AssertionError)
    finally:
        engines.SourceIndexedExact.score=saved
    mutants.append('parity_auditor_detects_false_engine')
    # Correlated bug: both engines use the same source predicate, so an independent
    # trajectory expectation (not mere A/B parity) must detect it.
    saved=engines._valid
    try:
        engines._valid=lambda state,f: (state.get(f['source']) is not None and state[f['source']][1]==f['source_version'])
        expected_fails('ignoring_stale_status',lambda:case('broken_status',s,rule,[0,0,1,0,1]),AssertionError)
    finally:engines._valid=saved
    mutants.append('ignoring_stale_status')
    try:
        engines._valid=lambda state,f: (state.get(f['source']) is not None and state[f['source']][0]=='VALID')
        e=copy.deepcopy(s);e['events'][-1]['epoch']=1
        expected_fails('ignoring_epoch_invalidation',lambda:case('broken_epoch',e,rule,[0,0,1,0,0]),AssertionError)
    finally:engines._valid=saved
    mutants.append('ignoring_epoch_invalidation')
    # A changed source name in a final source event must be caught against independent gold.
    e=copy.deepcopy(s);e['events'][-1]['source']='ghost'
    expected_fails('source_event_tamper_against_gold',lambda:engines.assert_parity(e,rule,expected_final=1),AssertionError)
    mutants.append('source_event_tamper_against_gold')
    # Frozen event-state index detects parity divergence at an intermediate prefix.
    saved=engines.SourceIndexedExact.apply
    try:
        def no_update(self,event):pass
        engines.SourceIndexedExact.apply=no_update
        expected_fails('dropped_incremental_updates',lambda:engines.assert_parity(s,rule),AssertionError)
    finally:engines.SourceIndexedExact.apply=saved
    mutants.append('dropped_incremental_updates')
    if len(cases)!=10 or len(mutants)!=8:raise AssertionError('test count '+str((len(cases),len(mutants))))
    print('PASS_H1B2A_SEMANTIC_FIXTURES='+str(len(cases))+'/'+str(len(cases)),flush=True)
    print('PASS_H1B2A_FAULT_MUTANTS='+str(len(mutants))+'/'+str(len(mutants)),flush=True)
    return {'semantic_fixtures':cases,'fault_mutants':mutants}
