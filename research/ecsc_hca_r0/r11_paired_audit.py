"""Read-only, post-hoc paired DEV diagnostic for frozen R10 and R11 source-turn predictions.

This is NOT an additional model, a scientific qualification, or a parameter-selection
protocol. Never reads QUAL questions through the predictor and never modifies R9-R11.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

from .r9_external_intake import DEV_IDS, digest, safe_write, wire
from .r10_blind_retrieval import load_exact

SCHEMA = 'elpis.ecsc.r11.paired-dev-audit.v1'
PINS = {
    'DEV_INPUT.json': '89b8ed64d9a3cf83b7efb87611cef017dc65a2775a634851ecd6e6f711f50672',
    'R10_DEV_PREDICTIONS.json': '019aa16568d50c4f3eb668d59bd0caeee84ad0796cea1fbcf2b9cf7ea14579b8',
    'R10_DEV_EVIDENCE_REPORT.json': '7994a487a186b2444e7e0d32b5b64e78d50c4d2444ebea6559d40b93d377ed83',
    'R11_DEV_PREDICTIONS.json': '3dee8a4d7042fa0b15f76d48df940508908d35b284c1e8bf8234f7f065b5fe5d',
    'R11_DEV_EVIDENCE_REPORT.json': '1e57390b920f2cab26f1076bb9f893652549a572702f44b06f6216d9a41eff03',
}
METHODS = ('LEXICAL_TOP8', 'LEXICAL4_ADJACENT4', 'ECSC_B0_CANDIDATE_RADIUS2_TOP8')
REPORT_FIELDS = ('queries', 'gold_references', 'matched_references', 'unresolved_references',
                 'any_hit', 'all_hit', 'empty_evidence')


class AuditError(Exception):
    pass


def require(flag, code):
    if not flag:
        raise AuditError(code)


def bucket():
    return {k:0 for k in REPORT_FIELDS}


def pair_bucket():
    return {'queries':0, 'same_selected_set':0, 'same_order':0,
            'common_selected_turns':0,
            'gold_refs_gained':0,'gold_refs_lost':0,
            'matched_more':0,'matched_less':0,'matched_equal':0,
            'any_wins':0,'any_losses':0,'any_ties':0,
            'all_wins':0,'all_losses':0,'all_ties':0}


def paired_evaluation(dev, gold, r10, r11):
    """Pure audit. Gold only enters here, after R10/R11 predictions were sealed."""
    require(dev['status']=='DEV_ONLY' and tuple(s['sample_id'] for s in dev['samples'])==DEV_IDS,
            'DEV_SCOPE')
    expected=[(s['sample_id'],q['id'],q['category']) for s in dev['samples'] for q in s['questions']]
    require(len(expected)==304 and len({qid for _,qid,_ in expected})==304,'DEV_CASE_COUNT')
    legal={s['sample_id']:{t['dialog_id'] for block in s['sessions'] for t in block['turns']}
           for s in dev['samples']}
    dev_gold={g['id']:g for g in gold['annotations'] if g['sample_id'] in DEV_IDS}
    require(len(dev_gold)==304 and set(dev_gold)=={qid for _,qid,_ in expected},'DEV_GOLD_IDS')
    require(len(r10['questions'])==len(expected) and len(r11['questions'])==len(expected),'ROWS_COUNT')
    require([p['id'] for p in r10['questions']]==[qid for _,qid,_ in expected],'R10_ORDER')
    require([p['id'] for p in r11['questions']]==[qid for _,qid,_ in expected],'R11_ORDER')
    require(r10['top_k']==8 and r11['top_k']==8 and
            r10['methods']==list(METHODS[:2]) and r11['method']==METHODS[2], 'METHODS_CONTRACT')
    by_method={m:{s:bucket() for s in ('answer_present','missing_answer')} for m in METHODS}
    by_category={str(cat):{m:bucket() for m in METHODS} for cat in range(1,6)}
    pairs={(a+'__VS__'+b):{split:pair_bucket() for split in ('answer_present','missing_answer')}
           for a,b in ((METHODS[2],METHODS[0]),(METHODS[2],METHODS[1]))}
    rows=[]
    for (sid,qid,category),p10,p11 in zip(expected,r10['questions'],r11['questions']):
        g=dev_gold[qid]
        require(g['sample_id']==sid and g['category']==category and type(g['answer_present']) is bool,
                'GOLD_METADATA')
        refs=g['evidence_original']
        unresolved=[ref for ref in refs if ref not in legal[sid]]
        require(unresolved==g['evidence_unresolved'],'EVIDENCE_ANOMALY_DRIFT')
        selected={**p10['methods'], METHODS[2]:p11['turns']}
        require(set(selected)==set(METHODS),'SELECTED_METHODS')
        split='answer_present' if g['answer_present'] else 'missing_answer'
        per={}
        for method in METHODS:
            turns=selected[method]
            require(type(turns) is list and len(turns)==8 and len(set(turns))==8 and
                    all(type(turn) is str and turn in legal[sid] for turn in turns), 'INVALID_SELECTION')
            n=sum(x in turns for x in refs)
            per[method]={'refs':n,'any':bool(n),'all':bool(refs) and n==len(refs)}
            for b in (by_method[method][split], by_category[str(category)][method]):
                b['queries']+=1
                b['gold_references']+=len(refs)
                b['matched_references']+=n
                b['unresolved_references']+=len(unresolved)
                b['any_hit']+=int(n>0)
                b['all_hit']+=int(bool(refs) and n==len(refs))
                b['empty_evidence']+=int(not refs)
        for comparator in METHODS[:2]:
            name=METHODS[2]+'__VS__'+comparator
            b=pairs[name][split]
            a=per[METHODS[2]];c=per[comparator]
            a_turns=selected[METHODS[2]];c_turns=selected[comparator]
            b['queries']+=1
            b['same_selected_set']+=int(set(a_turns)==set(c_turns))
            b['same_order']+=int(a_turns==c_turns)
            b['common_selected_turns']+=len(set(a_turns)&set(c_turns))
            b['gold_refs_gained']+=sum(x in a_turns and x not in c_turns for x in refs)
            b['gold_refs_lost']+=sum(x in c_turns and x not in a_turns for x in refs)
            b['matched_more']+=int(a['refs']>c['refs'])
            b['matched_less']+=int(a['refs']<c['refs'])
            b['matched_equal']+=int(a['refs']==c['refs'])
            for kind in ('any','all'):
                b[kind+'_wins']+=int(a[kind] and not c[kind])
                b[kind+'_losses']+=int(c[kind] and not a[kind])
                b[kind+'_ties']+=int(a[kind]==c[kind])
        rows.append({'id':qid,'category':category,'split':split,'gold_reference_count':len(refs),
                     'unresolved_references':len(unresolved),
                     'method_hits':per})
    require(sum(x['gold_reference_count'] for x in rows if x['split']=='answer_present')==310,
            'GOLD_REFERENCE_TOTAL')
    require(sum(x['split']=='answer_present' for x in rows)==235,'ANSWERED_CASE_COUNT')
    return {'methods':by_method,'by_category':by_category,
            'paired_all_questions':pairs,'question_accounting':rows}


def inspect(work):
    root=Path(work)/'research/ecsc_hca_r0/external/locomo_r0'
    for filename,pin in PINS.items():
        path=root/filename
        require(path.is_file() and not path.is_symlink() and path.stat().st_size<16*1024*1024,
                'SOURCE_MISSING_OR_LINK:'+filename)
        require(digest(path.read_bytes())==pin,'SOURCE_HASH_DRIFT:'+filename)
    # Source and gold must be in the existing attested R9 DEV split; no QUAL row is scored.
    dev,_=load_exact(root/'DEV_INPUT.json','elpis.ecsc.external-locomo-intake.v1.public-dev',
                     ('schema','status','samples'))
    r10,_=load_exact(root/'R10_DEV_PREDICTIONS.json',
                     'elpis.ecsc.r10.dev-blind-source-turn-retrieval.v1.predictions',
                     ('schema','status','source_input_sha256','methods','top_k','questions','prediction_contract'))
    r11,_=load_exact(root/'R11_DEV_PREDICTIONS.json',
                     'elpis.ecsc.r11.label-blind-candidate-hacf-b0.v1.predictions',
                     ('schema','status','dev_input_sha256','r10_predictions_sha256','source_profile',
                      'relation_authority','method','top_k','questions','work_descriptive_total'))
    report10,_=load_exact(root/'R10_DEV_EVIDENCE_REPORT.json',
                          'elpis.ecsc.r10.dev-blind-source-turn-retrieval.v1.descriptive',
                          ('schema','status','top_k','methods','denominators','abstention','relations',
                           'no_qual_scored','dev_input_sha256','predictions_sha256','gold_sha256','source_sha256'))
    report11,_=load_exact(root/'R11_DEV_EVIDENCE_REPORT.json',
                          'elpis.ecsc.r11.label-blind-candidate-hacf-b0.v1.descriptive',
                          ('schema','status','source_sha256','prediction_sha256',
                           'baseline_r10_report_sha256','methods','work_descriptive_total','claims'))
    att,_=load_exact(root/'R10_INPUT_ATTESTATION.json',
                     'elpis.ecsc.r10.dev-blind-source-turn-retrieval.v1.attestation',
                     ('schema','source_sha256','source_git_blob_sha1','dev_input_sha256',
                      'scorer_gold_sha256','r9_audit_sha256','sample_ids','qual_count','status'))
    gold,gold_hash=load_exact(root/'SCORER_GOLD.json',
                              'elpis.ecsc.external-locomo-intake.v1.gold',
                              ('schema','status','annotations'))
    require(gold_hash==att['scorer_gold_sha256'] and
            att['dev_input_sha256']==PINS['DEV_INPUT.json'] and
            att['sample_ids']==list(DEV_IDS) and att['qual_count']==8 and
            att['status']=='R9_BYTES_RECONSTRUCTED','ATTESTATION_INTEGRITY')
    require(report10['gold_sha256']==gold_hash and
            report10['predictions_sha256']==PINS['R10_DEV_PREDICTIONS.json'] and
            report11['prediction_sha256']==PINS['R11_DEV_PREDICTIONS.json'] and
            r11['r10_predictions_sha256']==PINS['R10_DEV_PREDICTIONS.json'] and
            r10['source_input_sha256']==PINS['DEV_INPUT.json'] and
            r11['dev_input_sha256']==PINS['DEV_INPUT.json'],'PREDICTION_AUTHORITY')
    require(r10['status']=='DEV_ONLY_UNSCORED' and r11['status']=='DEV_ONLY_NO_GOLD_SEEN' and
            gold['status']=='SCORER_ONLY' and report10['no_qual_scored'] is True,'STATUS_DRIFT')
    stats=paired_evaluation(dev,gold,r10,r11)
    for m in METHODS:
        for split in ('answer_present','missing_answer'):
            actual=stats['methods'][m][split]
            if m==METHODS[2]:
                recorded=report11['methods'][m][split]
                for k in REPORT_FIELDS:
                    require(recorded[k]==actual[k],'R11_SCORE_DRIFT:'+m+':'+split+':'+k)
            else:
                recorded=report10['methods'][m][split]
                synonyms={'queries':'queries','gold_references':'evidence_references_raw',
                          'matched_references':'matched_references','unresolved_references':'unresolved_evidence_references',
                          'any_hit':'any_hit_queries','all_hit':'all_hit_queries','empty_evidence':'empty_evidence'}
                for k,native in synonyms.items():
                    require(recorded[native]==actual[k],'R10_SCORE_DRIFT:'+m+':'+split+':'+k)
    return {'schema':SCHEMA,'status':'POSTHOC_DEV_PAIRED_DIAGNOSTIC_NOT_SCIENTIFIC_QUAL',
            'inputs_sha256':dict(PINS,SCORER_GOLD=gold_hash),
            'interpretation':'EVIDENCE_POINTER_RETRIEVAL_ONLY_NOT_ANSWER_QUALITY',
            'cost_notice':'R10_AND_R11_COST_UNITS_NOT_COMPARABLE; NO_END_TO_END_SPEED_CLAIM',
            'qualification':'EIGHT_QUAL_CONVERSATIONS_NOT_SCORED', **stats}


def main(work):
    report=inspect(work)
    path=Path(work)/'research/ecsc_hca_r0/external/locomo_r0/R11_DEV_PAIRED_AUDIT.json'
    safe_write(path,wire(report))
    print('R11_PAIRED_AUDIT_REPORT=',path,'SHA256=',digest(wire(report)),flush=True)
    for name,pair in report['paired_all_questions'].items():
        summary=pair['answer_present']
        print('PAIR_ANSWER_PRESENT=',name, 'QUERIES=',summary['queries'],'IDENTICAL_SET=',summary['same_selected_set'],
              'ALL_WINS=',summary['all_wins'],'ALL_LOSSES=',summary['all_losses'],
              'ANY_WINS=',summary['any_wins'],'ANY_LOSSES=',summary['any_losses'],
              'GOLD_GAINED=',summary['gold_refs_gained'],'GOLD_LOST=',summary['gold_refs_lost'],flush=True)
    print('R11_PAIRED_AUDIT=PASS_POSTHOC_NOT_SCIENTIFIC_QUAL',flush=True)


if __name__=='__main__':
    try:
        require(len(sys.argv)==2, 'USAGE: python -m research.ecsc_hca_r0.r11_paired_audit WORKTREE')
        main(sys.argv[1])
    except Exception as exc:
        print('R11_PAIRED_AUDIT=HOLD',type(exc).__name__,str(exc),file=sys.stderr,flush=True)
        raise SystemExit(3)
