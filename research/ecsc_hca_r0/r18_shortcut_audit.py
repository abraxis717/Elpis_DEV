"""R18 retrospective falsification: root-only shortcut and held-root counterfactuals.

No new training, no modification to R18 authority or outcomes, no QUAL access.
Tests frozen model behavior independently of self-scoring fixture rules.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import sys
from . import r18_bracket_binding as r

SCHEMA = 'elpis.ecsc.r18.shortcut-audit.v1'
OUT = 'R18_RETROSPECTIVE_SHORTCUT_AUDIT.json'
PINS = {
    r.TRAIN_FILE:'df5b3c9530c79bae312dc96c9d04ab13f8a9737a30f2502c4d2ea304f96ace77',
    r.PUBLIC_FILE:'8fc9c852e73a8621006ff02358ad75f26dbc4c3c0136cb14608b54475a9608f8',
    r.GOLD_FILE:'3589d1c17638983ad45c61553fca2f9fee95b4bfbc05b8134ce3d5d439a54dd1',
    r.MODEL_FILE:'6eb81f038f0ef9e89367a73767db6cae5798ea7710e3e93f74d7123308ee8d63',
    r.PRED_FILE:'7d9a9fcae51a81ecf765e4104106b37e69fec78ba17736f1873a88a39a3823fa',
    r.REPORT_FILE:'38b1b68c429f875688509c4ffaba6c2d427f2366eb927f07b5dc2717758dea4a',
}

def top_rule(text):
    """Explicit supplied-parser control: ignores all leaves and inner subtrees."""
    t = r.parse(text)
    negations = 0
    while t[0] == 'not':
        negations += 1
        t = t[1]
    r.require(t[0] in ('and', 'or'), 'ROOT_RULE_SHAPE')
    return int(t[0] == 'or') ^ (negations & 1)

def truth_probe(model):
    # The truth table is NOT inferred from training labels; formal oracle is r.val.
    out = {}
    for v in (0,1):
        t = r.leaf('Mira','amber key',v)
        p = r.predict(model, {'text':r.render(t)+'.'})
        out[f'leaf.{v}'] = {'expected':v,'prediction':int(p>=.5),'probability':round(p,9)}
    for op in ('and','or'):
        for a in (0,1):
            for b in (0,1):
                tree = (op, r.leaf('Mira','amber key',a),r.leaf('Jonah','violet folder',b))
                p = r.predict(model,{'text':r.render(tree)+'.'})
                out[f'{op}.{a}{b}'] = {'expected':r.val(tree),'prediction':int(p>=.5),'probability':round(p,9)}
    return out

def held_root_pairs(model):
    """Fixed outer operator, same tokens/order, opposite labels from inner brackets.

    These are retrospective, new, synthetic counterexamples, NOT a new blind qualification.
    """
    records=[]
    for core in ('or_and','and_or'):
        for middle in (0,1):
            a=r.leaf('Mira','amber key',core=='or_and')
            b=r.leaf('Jonah','violet folder',middle)
            c=r.leaf('Nadia','red ledger',core=='and_or')
            if core=='or_and':
                left=('and',('or',a,b),c)
                right=('or',a,('and',b,c))
            else:
                left=('or',('and',a,b),c)
                right=('and',a,('or',b,c))
            for outer in ('and','or'):
                d=r.leaf('Tessa','gold lantern',outer=='and')
                for wrapper in (0,1):
                    pair=[]
                    for inner in (left,right):
                        tree=(outer,d,inner)
                        if wrapper:tree=('not',tree)
                        text=r.render(tree)+'.'
                        prob=r.predict(model,{'text':text})
                        pair.append((text, r.val(tree),int(prob>=.5)))
                    r.require(r.bare_tokens(pair[0][0])==r.bare_tokens(pair[1][0]),'PAIR_TOKENS')
                    r.require(pair[0][1]!=pair[1][1],'PAIR_MUST_FLIP')
                    # The two trees share the outermost operator and number of NOT wrappers.
                    records.append({'core':core,'middle':middle,'root':outer,
                                    'wrappers':wrapper,'expected':[p[1] for p in pair],
                                    'predicted':[p[2] for p in pair],
                                    'both_correct':all(p[1]==p[2] for p in pair),
                                    'one_correct':sum(p[1]==p[2] for p in pair)==1,
                                    'same_predicted':pair[0][2]==pair[1][2],
                                    'input_sha256':[r.sha(p[0].encode()) for p in pair]})
    return records

def inspect(root):
    root=Path(root)
    r.provenance(root)
    inputs={}
    for name,pin in PINS.items():
        inputs[name]=r.digest_for(root/name)
        r.require(inputs[name]==pin,'PIN_DRIFT:'+name)
    train, _=r.read_json(root/r.TRAIN_FILE,r.SCHEMA+'.train',PINS[r.TRAIN_FILE])
    public, _=r.read_json(root/r.PUBLIC_FILE,r.SCHEMA+'.public',PINS[r.PUBLIC_FILE])
    model, _=r.read_json(root/r.MODEL_FILE,r.SCHEMA+'.model',PINS[r.MODEL_FILE])
    preds, _=r.read_json(root/r.PRED_FILE,r.SCHEMA+'.predictions',PINS[r.PRED_FILE])
    report,_=r.read_json(root/r.REPORT_FILE,r.SCHEMA+'.report',PINS[r.REPORT_FILE])
    r.require(model['train_sha256']==PINS[r.TRAIN_FILE] and model['public_sha256']==PINS[r.PUBLIC_FILE], 'MODEL_INPUT_BINDING')
    r.require(preds['model_sha256']==PINS[r.MODEL_FILE] and preds['train_sha256']==PINS[r.TRAIN_FILE] and preds['public_sha256']==PINS[r.PUBLIC_FILE],'PRED_INPUT_BINDING')
    r.require(report['model_sha256']==PINS[r.MODEL_FILE] and report['prediction_sha256']==PINS[r.PRED_FILE] and report['gold_sha256']==PINS[r.GOLD_FILE],'REPORT_BINDING')
    replay=r.fit(train['records']);replay.update({'train_sha256':PINS[r.TRAIN_FILE], 'public_sha256':PINS[r.PUBLIC_FILE]})
    r.require(r.wire(replay)==r.wire(model),'MODEL_REPLAY')
    r.require(len(public['records'])==len(preds['rows'])==448,'ROW_COUNT')
    matches=defaultdict(lambda:{'n':0,'match':0,'refusal':0,'binary_mismatches':0})
    for row,pred in zip(public['records'],preds['rows']):
        r.require(row['id']==pred['id'],'PRED_ORDER')
        key=row['group'];m=matches[key];m['n']+=1
        try:
            value=r.predict(model,row)
            r.require(pred=={'id':row['id'],'status':'ANSWER','answer':int(value>=.5),'p':round(value,9)},'PRED_REPLAY')
            guess=top_rule(row['text'])
            m['match']+=int(guess==pred['answer'])
            m['binary_mismatches']+=int(guess!=pred['answer'])
        except r.Hold:
            r.require(pred['status']=='REFUSED' and pred['answer'] is None and pred['p'] is None, 'REFUSAL_REPLAY')
            m['refusal']+=1;m['match']+=1
    # Gold was not needed for the root-prediction equivalence; read only for invariants.
    gold,_=r.read_json(root/r.GOLD_FILE,r.SCHEMA+'.gold',PINS[r.GOLD_FILE])
    r.require(len(gold['records'])==448 and all(a['id']==b['id'] for a,b in zip(public['records'],gold['records'])),'GOLD_ALIGN')
    table=truth_probe(model)
    cf=held_root_pairs(model)
    gates={k:round(r.sigmoid(v),9) for k,v in model['theta'].items()}
    result={
      'schema':SCHEMA,
      'disposition':'RETROSPECTIVE_SHORTCUT_REPRODUCED_NOT_SCIENTIFIC_QUAL',
      'source_pins':inputs,
      'root_only_rule':'Ignore all leaves and subtrees; AND=false, OR=true, invert per enclosing NOT; use frozen parser for syntax/refusal.',
      'root_only_comparison':dict(sorted(matches.items())),
      'all_recorded_predictions_matched':all(s['match']==s['n'] for s in matches.values()),
      'learned_gate_probabilities':gates,
      'independent_primitive_truth_table':table,
      'primitive_truth_correct':sum(v['expected']==v['prediction'] for v in table.values()),
      'primitive_truth_total':len(table),
      'fixed_root_inner_bracket_interventions':{
          'pairs':len(cf),
          'both_correct':sum(e['both_correct'] for e in cf),
          'one_correct':sum(e['one_correct'] for e in cf),
          'neither_correct':sum(not(e['one_correct'] or e['both_correct']) for e in cf),
          'same_prediction_pairs':sum(e['same_predicted'] for e in cf),
          'cases':cf},
      'notes':['No training, label modification or prediction rewrite.',
               'R18 success is matched by root-only shortcut; cannot support learned recursive evaluation.',
               'Post-hoc counterfactuals are diagnostic only; next experiment must be preregistered.']
    }
    r.require(result['all_recorded_predictions_matched'],'ROOT_CONTROL_MISMATCH')
    r.require(result['fixed_root_inner_bracket_interventions']['pairs']==16,'PAIR_COUNT')
    return result

def main():
    r.require(len(sys.argv)==2,'USAGE')
    root=r.root_for(sys.argv[1]);data=inspect(root)
    payload=r.wire(data)
    r.write_once(root/OUT,payload)
    print('R18_SHORTCUT_AUDIT_REPORT=',root/OUT,'SHA256=',r.sha(payload),flush=True)
    for k,v in data['root_only_comparison'].items():
        print('ROOT_RULE_GROUP=',k,'TOTAL=',v['n'],'IDENTICAL=',v['match'],'REFUSED=',v['refusal'],flush=True)
    h=data['fixed_root_inner_bracket_interventions']
    print('R18_FIXED_ROOT=', 'PAIRS=',h['pairs'],'BOTH=',h['both_correct'],
          'ONE=',h['one_correct'],'NEITHER=',h['neither_correct'],
          'SAME_PREDICTIONS=',h['same_prediction_pairs'],flush=True)
    print('R18_PRIMITIVE_TRUTH=',data['primitive_truth_correct'],'/',data['primitive_truth_total'],flush=True)
    print('R18_SHORTCUT_AUDIT=PASS_RETROSPECTIVE_ONLY_NOT_SCIENTIFIC_QUAL',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('R18_SHORTCUT_AUDIT=HOLD',type(exc).__name__,str(exc),file=sys.stderr,flush=True)
        sys.exit(3)
