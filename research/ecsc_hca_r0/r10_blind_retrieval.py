"""R10 DEV_ONLY source-turn retrieval controls, independent of LoCoMo answer labels.

predict() consumes ONLY R9 DEV_INPUT.json. score() separately consumes scorer gold.
Neither algorithm admits semantic relations, answers questions, nor tests ECS_C B0.
The next phase may use these frozen retrieval diagnostics to construct/compare
ECS_C provenance-bearing readouts; no scientific qualification follows here.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from math import log
from pathlib import Path
import json
import re
import sys

from .r9_external_intake import DEV_IDS, GIT_BLOB_SHA1, SCHEMA as R9_SCHEMA, convert, digest, gitblob, safe_write, wire

SCHEMA = 'elpis.ecsc.r10.dev-blind-source-turn-retrieval.v1'
PREDICTION_NAMES = ('LEXICAL_TOP8', 'LEXICAL4_ADJACENT4')
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
STOP = frozenset('a an and are as at be been being but by can could did do does for from had has have how i in into is it its me my of on or our she should so than that the their them there these they this those to was were what when where which who why will with would you your'.split())
K = 8
SEED_K = 4
SOURCE_SHA256 = '79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4'

class R10Error(Exception):
    pass

def need(flag, code):
    if not flag:
        raise R10Error(code)

def tokens(text):
    need(type(text) is str, 'TEXT_TYPE')
    return [s for s in TOKEN.findall(text.casefold()) if s not in STOP]

def load_exact(path, schema, allowed_keys):
    path=Path(path)
    need(path.is_file() and not path.is_symlink(), 'INPUT_MISSING_OR_LINKED:'+path.name)
    raw=path.read_bytes()
    need(len(raw)<=16*1024*1024, 'INPUT_TOO_LARGE')
    try:
        data=json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise R10Error('INPUT_BAD_JSON') from exc
    need(type(data) is dict and data.get('schema')==schema and set(data)==set(allowed_keys), 'INPUT_SCHEMA')
    return data, digest(raw)

def verify_public(data):
    """Only R9 DEV release shape permitted. Fail if any answer/evidence fields leak."""
    need(data['status']=='DEV_ONLY' and type(data['samples']) is list, 'PUBLIC_STATUS')
    need(len(data['samples'])==len(DEV_IDS), 'PUBLIC_SAMPLE_COUNT')
    need(tuple(s['sample_id'] for s in data['samples'])==DEV_IDS, 'PUBLIC_IDS_ORDER')
    qids=set()
    totals=0
    for sample in data['samples']:
        need(set(sample)=={'sample_id','speakers','sessions','questions'}, 'PUBLIC_SAMPLE_KEYS')
        need(type(sample['speakers']) is list and len(sample['speakers'])==2,'PUBLIC_SPEAKERS')
        sessions=sample['sessions']; need(type(sessions) is list and sessions, 'PUBLIC_SESSIONS')
        dids=set()
        for s in sessions:
            need(type(s) is dict and set(s)=={'session','date_time','turns'}, 'PUBLIC_SESSION_KEYS')
            need(re.fullmatch(r'session_[1-9][0-9]*',s['session']) is not None and type(s['date_time']) is str, 'PUBLIC_SESSION')
            need(type(s['turns']) is list and s['turns'], 'PUBLIC_EMPTY_SESSION')
            for turn in s['turns']:
                need(type(turn) is dict and set(turn)=={'dialog_id','speaker','text'}, 'PUBLIC_TURN_KEYS')
                did=turn['dialog_id']
                need(type(did) is str and did not in dids and type(turn['text']) is str and type(turn['speaker']) is str,'PUBLIC_TURN')
                dids.add(did)
        need(type(sample['questions']) is list, 'PUBLIC_QUESTIONS')
        for i,question in enumerate(sample['questions']):
            need(type(question) is dict and set(question)=={'id','question','category'}, 'PUBLIC_QUESTION_KEYS')
            qid=sample['sample_id']+'/q'+str(i).zfill(4)
            need(question['id']==qid and qid not in qids and type(question['question']) is str and question['question'], 'PUBLIC_QID')
            need(type(question['category']) is int and question['category'] in (1,2,3,4,5), 'PUBLIC_CATEGORY')
            qids.add(qid);totals+=1
    need(totals==304, 'DEV_QUESTION_COUNT')
    return qids

def indexed_turns(sample):
    turns=[]; adjacency={}
    for session in sample['sessions']:
        prev=None
        for t in session['turns']:
            idx=len(turns)
            turns.append((t['dialog_id'],t['text']))
            adjacency[idx]=[]
            if prev is not None:
                adjacency[idx].append(prev)
                adjacency[prev].append(idx)
            prev=idx
    need(len(turns)<=2000,'DEV_TURN_LIMIT')
    return turns,adjacency

def lexical_ranking(turns, question):
    """Fixed BM25 on source TURN text only, no gold, category, timestamps or hints."""
    documents=[Counter(tokens(t)) for _,t in turns]
    lengths=[sum(v.values()) for v in documents]
    avg=sum(lengths)/len(lengths) if documents else 1.0
    df=Counter()
    for d in documents:
        df.update(d.keys())
    qterms=set(tokens(question))
    n=len(documents)
    scores=[]
    for i,doc in enumerate(documents):
        score=0.0
        denomlen=1.2*(0.25+0.75*lengths[i]/avg)
        for term in qterms:
            f=doc.get(term,0)
            if f:
                idf=log(1+(n-df[term]+0.5)/(df[term]+0.5))
                score+=idf*f*2.2/(f+denomlen)
        scores.append(score)
    # No cutoff on zero relevance: always produce a deterministic eight-turn
    # prediction, exposing low-confidence failures rather than abstaining post-hoc.
    return sorted(range(n), key=lambda i:(-scores[i],i))

def produce_sample(sample):
    turns,adjacency=indexed_turns(sample)
    need(len(turns)>=K, 'DEV_TOO_FEW_TURNS')
    rows=[]
    for q in sample['questions']:
        ranked=lexical_ranking(turns,q['question'])
        base=ranked[:K]
        seeds=ranked[:SEED_K]
        chosen=list(seeds); have=set(chosen)
        # Temporal adjacency is a CANDIDATE navigation edge, not evidence of
        # semantic entailment. Each seed tries predecessor then successor.
        for seed in seeds:
            for nbr in sorted(adjacency[seed]):
                if len(chosen)>=K: break
                if nbr not in have:
                    chosen.append(nbr);have.add(nbr)
            if len(chosen)>=K: break
        for i in ranked:
            if len(chosen)>=K:break
            if i not in have:
                chosen.append(i);have.add(i)
        need(len(chosen)==K and len(set(chosen))==K, 'RANKING_ERROR')
        rows.append({'id':q['id'], 'methods':{
          'LEXICAL_TOP8': [turns[i][0] for i in base],
          'LEXICAL4_ADJACENT4':[turns[i][0] for i in chosen]
        }})
    return rows

def predictions_from_public(data, source_input_hash):
    verify_public(data)
    entries=[]
    for sample in data['samples']:
        entries.extend(produce_sample(sample))
    need(len(entries)==304, 'PREDICTION_COUNT')
    return {'schema':SCHEMA+'.predictions', 'status':'DEV_ONLY_UNSCORED',
            'source_input_sha256':source_input_hash, 'methods':list(PREDICTION_NAMES),
            'top_k':K, 'questions':entries,
            'prediction_contract':'NO_LABELS_NO_SEMANTIC_ADMISSION_NO_ANSWER_GENERATION'}

def evaluate(pred, gold, pub):
    """Scorer-only. Never called from predictor branch. No changes to gold labels."""
    verify_public(pub)
    need(pred.get('schema')==SCHEMA+'.predictions' and pred.get('top_k')==K and pred.get('methods')==list(PREDICTION_NAMES), 'PREDICTION_SCHEMA')
    need(type(gold) is dict and gold.get('schema')==R9_SCHEMA+'.gold' and gold.get('status')=='SCORER_ONLY','GOLD_SCHEMA')
    known={s['sample_id']:{t['dialog_id'] for session in s['sessions'] for t in session['turns']} for s in pub['samples']}
    question_ids=[q['id'] for s in pub['samples'] for q in s['questions']]
    need(type(pred['questions']) is list and [r['id'] for r in pred['questions']]==question_ids, 'PREDICTION_IDS')
    # Explicitly restrict to the two DEV samples. Do not report QUAL statistics.
    annotations={}
    for g in gold['annotations']:
        if g['sample_id'] in DEV_IDS:
            need(g['id'] not in annotations,'DUPLICATE_GOLD_ID')
            annotations[g['id']]=g
    need(set(annotations)==set(question_ids), 'DEV_GOLD_IDS')
    summaries={name:{split:{'queries':0,'answered_gold':0,'missing_gold_answer':0,'with_evidence':0,
                       'empty_evidence':0,'evidence_references_raw':0,'valid_evidence_references':0,
                       'unresolved_evidence_references':0,'matched_references':0,'any_hit_queries':0,
                       'all_hit_queries':0} for split in ('answer_present','missing_answer')} for name in PREDICTION_NAMES}
    for p in pred['questions']:
        g=annotations[p['id']]
        sid=g['sample_id'];refs=g['evidence_original']
        need(type(refs) is list and all(type(r) is str for r in refs), 'GOLD_EVIDENCE_SHAPE')
        unresolved=[ref for ref in refs if ref not in known[sid]]
        need(unresolved==g['evidence_unresolved'],'GOLD_ANOMALY_DISAGREEMENT')
        split='answer_present' if g['answer_present'] else 'missing_answer'
        need(set(p['methods'])==set(PREDICTION_NAMES),'PREDICTION_METHODS')
        for method in PREDICTION_NAMES:
            selected=p['methods'][method]
            need(type(selected) is list and len(selected)==K and len(set(selected))==K and all(s in known[sid] for s in selected),'PREDICTION_INVALID_TURN')
            stat=summaries[method][split]
            stat['queries']+=1
            stat['answered_gold']+=int(split=='answer_present')
            stat['missing_gold_answer']+=int(split=='missing_answer')
            stat['evidence_references_raw']+=len(refs)
            stat['valid_evidence_references']+=len(refs)-len(unresolved)
            stat['unresolved_evidence_references']+=len(unresolved)
            if not refs:
                stat['empty_evidence']+=1
                continue
            stat['with_evidence']+=1
            matched=sum(ref in selected for ref in refs)
            stat['matched_references']+=matched
            stat['any_hit_queries']+=int(matched>0)
            stat['all_hit_queries']+=int(matched==len(refs))
    return {'schema':SCHEMA+'.descriptive',
            'status':'DEV_ONLY_EXTERNAL_EVIDENCE_RETRIEVAL_NOT_ECSC_UTILITY_OR_SCIENTIFIC_QUAL',
            'top_k':K, 'methods':summaries,
            'denominators':'RAW_REFERENCE_RECALL_COUNTS_UNRESOLVED_AS_MISSES; EMPTY_EVIDENCE_NOT_IN_ANY_ALL',
            'abstention':'MISSING_ANSWER_QUERIES_SEPARATE; NO_ANSWER_GENERATION_OR_ABSTENTION_ACCURACY_TESTED',
            'relations':'ADJACENCY_IS_CANDIDATE_NAVIGATION_ONLY_NOT_SEMANTIC_ADMISSION',
            'no_qual_scored':True}

def paths(work):
    d=Path(work)/'research/ecsc_hca_r0/external/locomo_r0'
    return d, d/'DEV_INPUT.json',d/'SCORER_GOLD.json',d/'INTAKE_AUDIT.json'

def _attestation(d):
    receipt,_=load_exact(d/'R10_INPUT_ATTESTATION.json',SCHEMA+'.attestation',
        ('schema','source_sha256','source_git_blob_sha1','dev_input_sha256','scorer_gold_sha256','r9_audit_sha256','sample_ids','qual_count','status'))
    need(receipt['source_sha256']==SOURCE_SHA256 and receipt['source_git_blob_sha1']==GIT_BLOB_SHA1, 'ATTESTATION_SOURCE_MISMATCH')
    need(receipt['sample_ids']==list(DEV_IDS) and receipt['qual_count']==8 and receipt['status']=='R9_BYTES_RECONSTRUCTED', 'ATTESTATION_SPLIT')
    return receipt

def attest(work):
    """Separate trusted intake/attestation PROCESS; predictor never reads this raw gold."""
    d,devpath,goldpath,auditpath=paths(work)
    src=d/'locomo10.json'
    need(src.is_file() and not src.is_symlink() and src.stat().st_size<=8*1024*1024, 'R9_SOURCE_MISSING')
    raw=src.read_bytes()
    need(digest(raw)==SOURCE_SHA256 and gitblob(raw)==GIT_BLOB_SHA1, 'R9_SOURCE_BYTES_CHANGED')
    canonical_public,canonical_gold,canonical_audit=convert(raw)
    expected=((devpath,canonical_public), (goldpath,canonical_gold), (auditpath,canonical_audit))
    for path,obj in expected:
        need(path.is_file() and not path.is_symlink() and path.read_bytes()==wire(obj), 'R9_INPUT_TAMPER:'+path.name)
    need(canonical_audit['dev_sample_ids']==list(DEV_IDS) and len(canonical_audit['qual_sample_ids'])==8, 'R9_SPLIT')
    receipt={'schema':SCHEMA+'.attestation','status':'R9_BYTES_RECONSTRUCTED',
            'source_sha256':SOURCE_SHA256,'source_git_blob_sha1':GIT_BLOB_SHA1,
            'dev_input_sha256':digest(wire(canonical_public)),
            'scorer_gold_sha256':digest(wire(canonical_gold)),
            'r9_audit_sha256':digest(wire(canonical_audit)),
            'sample_ids':list(DEV_IDS), 'qual_count':8}
    safe_write(d/'R10_INPUT_ATTESTATION.json',wire(receipt))
    print('R10_ATTESTATION=PASS_PINNED_SOURCE_AND_RELEASES',
          'DEV_SHA256=',receipt['dev_input_sha256'], flush=True)

def predict(work):
    """Label-blind PROCESS: opens DEV_INPUT and attestation only."""
    d,devpath,_,_=paths(work)
    dev,dev_hash=load_exact(devpath,R9_SCHEMA+'.public-dev',('schema','status','samples'))
    receipt=_attestation(d)
    need(dev_hash==receipt['dev_input_sha256'],'R9_DEV_INPUT_CHANGED')
    result=predictions_from_public(dev,dev_hash)
    out=d/'R10_DEV_PREDICTIONS.json'
    safe_write(out,wire(result))
    print('R10_PREDICTIONS=',out, 'SHA256=',digest(wire(result)), 'DEV_CASES=',len(result['questions']),flush=True)

def score(work):
    """Separate SCORER PROCESS; only this branch reads R9 scorer gold."""
    d,devpath,goldpath,auditpath=paths(work)
    receipt=_attestation(d)
    dev,dev_hash=load_exact(devpath,R9_SCHEMA+'.public-dev',('schema','status','samples'))
    need(dev_hash==receipt['dev_input_sha256'],'R9_DEV_INPUT_CHANGED')
    pred,pred_hash=load_exact(d/'R10_DEV_PREDICTIONS.json',SCHEMA+'.predictions',('schema','status','source_input_sha256','methods','top_k','questions','prediction_contract'))
    need(pred['source_input_sha256']==dev_hash,'PREDICTION_DEV_INPUT_DRIFT')
    gold,gold_hash=load_exact(goldpath,R9_SCHEMA+'.gold',('schema','status','annotations'))
    need(gold_hash==receipt['scorer_gold_sha256'],'R9_GOLD_CHANGED')
    audit,audit_hash=load_exact(auditpath,R9_SCHEMA,('schema','status','source_commit','source_git_blob_sha1','source_sha256','sample_count','question_count','missing_answer_count','unresolved_evidence_reference_count','anomalies','dev_sample_ids','qual_sample_ids','qual_status','semantics','prohibition'))
    need(audit_hash==receipt['r9_audit_sha256'],'R9_AUDIT_CHANGED')
    result=evaluate(pred,gold,dev)
    result.update({'dev_input_sha256':dev_hash,'predictions_sha256':pred_hash,'gold_sha256':gold_hash,'source_sha256':SOURCE_SHA256})
    out=d/'R10_DEV_EVIDENCE_REPORT.json'
    safe_write(out,wire(result))
    print('R10_REPORT=',out, 'SHA256=',digest(wire(result)),flush=True)
    for method in PREDICTION_NAMES:
        x=result['methods'][method]['answer_present']
        print(method,'ANSWERED_QUERIES=',x['queries'],'GOLD_REFS=',x['evidence_references_raw'],
            'RETRIEVED_GOLD_REFS=',x['matched_references'],'ANY_HIT=',x['any_hit_queries'],
            'ALL_HIT=',x['all_hit_queries'],'UNRESOLVED=',x['unresolved_evidence_references'],flush=True)
    print('R10_DESCRIPTIVE=PASS_DEV_ONLY_NOT_SCIENTIFIC_QUAL',flush=True)

if __name__=='__main__':
    try:
        need(len(sys.argv)==3 and sys.argv[1] in ('attest','predict','score'), 'USAGE: attest|predict|score WORKTREE')
        if sys.argv[1]=='attest':attest(sys.argv[2])
        elif sys.argv[1]=='predict':predict(sys.argv[2])
        else:score(sys.argv[2])
    except Exception as exc:
        print('R10=HOLD',type(exc).__name__,str(exc),file=sys.stderr,flush=True)
        raise SystemExit(3)
