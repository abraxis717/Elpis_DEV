"""R12: DEV-only, label-blind answer-level evidence comparison.

No source/QUAL/gold in prepare or infer. No claimed semantic adjudication.
Only the separate scorer opens SCORER_GOLD. All outputs are immutable except the
append-only, hash-chained inference journal used for exact resumption.
"""
from __future__ import annotations
import argparse
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
import unicodedata
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler, ProxyHandler

from .r9_external_intake import DEV_IDS, SCHEMA as R9_SCHEMA, digest, safe_write, wire
from .r10_blind_retrieval import (SCHEMA as R10_SCHEMA, R10Error, _attestation,
     load_exact, paths, verify_public)

SCHEMA = 'elpis.ecsc.r12.dev-external-answer-comparison.v1'
METHODS = ('LEXICAL_TOP8', 'LEXICAL4_ADJACENT4', 'ECSC_B0_CANDIDATE_RADIUS2_TOP8', 'NO_MEMORY')
PINS = {
 'R10_DEV_PREDICTIONS.json':'019aa16568d50c4f3eb668d59bd0caeee84ad0796cea1fbcf2b9cf7ea14579b8',
 'R11_DEV_PREDICTIONS.json':'3dee8a4d7042fa0b15f76d48df940508908d35b284c1e8bf8234f7f065b5fe5d',
}
# Never opened or parsed by prepare/infer: these reports contain gold-derived summaries.
SCORER_ONLY_PINS = {
 'R10_DEV_EVIDENCE_REPORT.json':'7994a487a186b2444e7e0d32b5b64e78d50c4d2444ebea6559d40b93d377ed83',
 'R11_DEV_EVIDENCE_REPORT.json':'1e57390b920f2cab26f1076bb9f893652549a572702f44b06f6216d9a41eff03',
 'R11_DEV_PAIRED_AUDIT.json':'f196b6b65ab92c2b30765b3653c019862ef53d5049f14b0f307e9d00d19e47f6',
}
SYSTEM = ('You answer a factual question about a prior conversation. Use ONLY the supplied conversation turns. '
          'Never treat spatial adjacency or a retrieved passage as proof of a relationship. '
          'If the supplied turns do not establish the answer, abstain. Do not use external or memorized facts. '
          'Respond with exactly one JSON object, with keys "answer" (a short string, or null if insufficient) '
          'and "citations" (an array of turn IDs quoted from the evidence). '
          'For a non-null answer give at least one supporting citation; for null, citations must be []. '
          'No explanation, markdown, or other fields.')
DECODE = {'temperature':0,'max_tokens':128,'stream':False}
ZERO = '0'*64


def need(ok, code):
    if not ok: raise R10Error('R12_'+code)


def read_pinned(d, name):
    p=d/name
    need(p.is_file() and not p.is_symlink() and p.stat().st_size<16*1024*1024,'PIN_FILE:'+name)
    b=p.read_bytes()
    need(digest(b)==PINS[name],'PIN_DRIFT:'+name)
    return json.loads(b)


def prior_inputs(work):
    d,devpath,_,_=paths(work)
    dev,dev_sha=load_exact(devpath,R9_SCHEMA+'.public-dev',('schema','status','samples'))
    verify_public(dev)
    att=_attestation(d)
    need(att['dev_input_sha256']==dev_sha,'DEV_HASH_DRIFT')
    records={k:read_pinned(d,k) for k in PINS}
    r10=records['R10_DEV_PREDICTIONS.json']
    r11=records['R11_DEV_PREDICTIONS.json']
    need(r10['schema']==R10_SCHEMA+'.predictions' and r10['status']=='DEV_ONLY_UNSCORED' and
         r10['top_k']==8 and r10['source_input_sha256']==dev_sha and
         r10['methods']==list(METHODS[:2]),'R10_SCHEMA_DRIFT')
    need(r11['status']=='DEV_ONLY_NO_GOLD_SEEN' and r11['dev_input_sha256']==dev_sha and
         r11['method']==METHODS[2] and r11['top_k']==8,'R11_SCHEMA_DRIFT')
    ids=[q['id'] for s in dev['samples'] for q in s['questions']]
    need(len(ids)==304 and [p['id'] for p in r10['questions']]==ids and
         [p['id'] for p in r11['questions']]==ids,'QUESTION_IDS_DRIFT')
    return d,dev,dev_sha,r10,r11


def make_packets(dev, dev_sha, r10, r11):
    """Never accepts labels or a gold path. Stable ordering: question then fixed method."""
    packets=[]
    r10rows=iter(r10['questions']);r11rows=iter(r11['questions'])
    for sample in dev['samples']:
        turns={t['dialog_id']:(s['date_time'],t['speaker'],t['text'])
               for s in sample['sessions'] for t in s['turns']}
        pos={t['dialog_id']:i for i,t in enumerate(t for s in sample['sessions'] for t in s['turns'])}
        for q in sample['questions']:
            row10=next(r10rows);row11=next(r11rows)
            need(row10['id']==q['id'] and row11['id']==q['id'],'ROW_ALIGNMENT')
            shift=sha256(q['id'].encode('utf8')).digest()[0] % len(METHODS)
            order=METHODS[shift:]+METHODS[:shift]
            for method in order:
                selected=(row10['methods'][method] if method in METHODS[:2] else
                          row11['turns'] if method==METHODS[2] else [])
                need(type(selected) is list and len(selected)==(0 if method=='NO_MEMORY' else 8) and
                     len(set(selected))==len(selected) and all(type(s) is str and s in turns for s in selected),
                     'TURN_SET_INVALID')
                # Same transcript order, independently of ranked retrieval method.
                passage=[{'turn_id':did,'date_time':turns[did][0],
                          'speaker':turns[did][1],'text':turns[did][2]}
                         for did in sorted(selected,key=lambda x:pos[x])]
                packets.append({'id':q['id'],'question':q['question'],
                                'method':method,'turns':passage})
    need(len(packets)==304*len(METHODS),'PACKET_COUNT')
    return {'schema':SCHEMA+'.packets','status':'DEV_ONLY_NO_GOLD',
      'dev_input_sha256':dev_sha,'input_pins':dict(PINS),
      'prompt_sha256':digest(SYSTEM.encode('utf-8')),
      'decode':dict(DECODE),'methods':list(METHODS),'packets':packets,
      'limits':{'answer_tokens':128,'passages_per_method':8,
                'gold_or_qual_in_predictor':False}}


def messages(packet):
    lines=['QUESTION:',packet['question'],'','EVIDENCE (source turns, not semantic proofs):']
    if not packet['turns']:
        lines.append('(no memory evidence provided)')
    for t in packet['turns']:
        lines.append('['+t['turn_id']+'] '+t['date_time']+' | '+t['speaker']+': '+t['text'])
    user='\n'.join(lines)
    need(len(user)<=65536,'PROMPT_TOO_LARGE')
    return [{'role':'system','content':SYSTEM},{'role':'user','content':user}]


def prepare(work):
    d,dev,dev_sha,r10,r11=prior_inputs(work)
    packets=make_packets(dev,dev_sha,r10,r11)
    out=d/'R12_DEV_ANSWER_PACKETS.json'
    safe_write(out,wire(packets))
    pre={'schema':SCHEMA+'.preregistered-dev','status':'FROZEN_DEV_DESCRIPTIVE_NO_QUAL',
      'source_sha256':dev_sha,'input_pins':dict(PINS),
      'packets_sha256':digest(wire(packets)),
      'n_queries':304,'n_methods':len(METHODS),'methods':list(METHODS),
      'prompt_sha256':digest(SYSTEM.encode('utf8')),'decode':dict(DECODE),
      'primary_observations':['strict_exact_answer_match_on_ANSWER_PRESENT',
         'abstention_on_MISSING_ANSWER','format_refusal_and_citation_contract_failures'],
      'no_science_claim':'DEV_DESCRIPTIVE; semantic paraphrases require independent adjudication; correction/revocation unlabelled',
      'no_qual':'8 conversations remain reserved; no QUAL content used',
      'prohibitions':['NO_TUNING_AFTER_SCORING','NO_AUTOMATIC_SEMANTIC_RELATION_ADMISSION',
                      'NO_RETROACTIVE_GOLD_REPAIR','NO_HIDDEN_ANSWER_LABELS_IN_PREDICTOR']}
    safe_write(d/'R12_DEV_PREREGISTRATION.json',wire(pre))
    print('R12_PREPARED=',out,'SHA256=',digest(wire(packets)),'QUESTION_COUNT=304',
          'PREDICTION_CALLS=',len(packets['packets']),flush=True)
    return packets


def load_prepared(work):
    d,dev,dev_sha,r10,r11=prior_inputs(work)
    expected=make_packets(dev,dev_sha,r10,r11)
    packets,psha=load_exact(d/'R12_DEV_ANSWER_PACKETS.json',SCHEMA+'.packets',
      ('schema','status','dev_input_sha256','input_pins','prompt_sha256','decode','methods','packets','limits'))
    need(packets==expected,'PACKETS_CHANGED')
    pre,_=load_exact(d/'R12_DEV_PREREGISTRATION.json',SCHEMA+'.preregistered-dev',
      ('schema','status','source_sha256','input_pins','packets_sha256','n_queries','n_methods','methods',
       'prompt_sha256','decode','primary_observations','no_science_claim','no_qual','prohibitions'))
    need(pre['packets_sha256']==psha and pre['prompt_sha256']==digest(SYSTEM.encode('utf8')),
         'PREREGISTRATION_CHANGED')
    return d,packets,pre


def parse_answer(content, selected):
    """Invalid JSON is an error, NOT an abstention. No repair, no re-asking."""
    need(type(content) is str and len(content)<=16000,'RESPONSE_INVALID_LENGTH')
    try: obj=json.loads(content)
    except ValueError: return {'valid':False,'reason':'NON_JSON','answer':None,'citations':[]}
    if type(obj) is not dict or set(obj)!={'answer','citations'}:
        return {'valid':False,'reason':'SCHEMA','answer':None,'citations':[]}
    answer=obj['answer'];c=obj['citations']
    if not (answer is None or type(answer) is str and 0<len(answer.strip())<=512) or type(c) is not list or any(type(v) is not str for v in c):
        return {'valid':False,'reason':'FIELD_TYPE','answer':None,'citations':[]}
    if len(c)!=len(set(c)) or any(v not in selected for v in c) or (answer is None)!=(not c):
        return {'valid':False,'reason':'CITATION','answer':None,'citations':[]}
    return {'valid':True,'reason':None,'answer':answer.strip() if answer is not None else None,'citations':c}


def journal_entries(d, packets):
    path=d/'R12_DEV_INFERENCE_JOURNAL.jsonl'
    if not path.exists():return []
    need(path.is_file() and not path.is_symlink() and path.stat().st_size<32*1024*1024,'JOURNAL_INVALID')
    raw=path.read_bytes()
    need(not raw or raw.endswith(b'\n'),'JOURNAL_TORN_LAST_LINE')
    prev=ZERO; entries=[]
    for line in raw.splitlines():
        need(len(line)<=18000,'JOURNAL_LINE_OVERSIZE')
        row=json.loads(line)
        need(type(row) is dict and set(row)=={'index','id','method','completion_text','elapsed_ns','previous_sha256','record_sha256'},
             'JOURNAL_SCHEMA')
        body={k:row[k] for k in row if k!='record_sha256'}
        need(row['index']==len(entries) and row['index']<len(packets) and
             row['id']==packets[len(entries)]['id'] and row['method']==packets[len(entries)]['method'] and
             row['previous_sha256']==prev and row['record_sha256']==digest(wire(body)) and
             type(row['elapsed_ns']) is int and row['elapsed_ns']>=0 and
             type(row['completion_text']) is str and len(row['completion_text'])<=16000,
             'JOURNAL_CHAIN_OR_TASK_MISMATCH')
        prev=row['record_sha256'];entries.append(row)
    return entries


def check_local(url):
    parts=urlsplit(url)
    need(parts.scheme=='http' and parts.hostname=='127.0.0.1' and
         parts.path=='/v1/chat/completions' and not parts.username and not parts.password and
         not parts.query and not parts.fragment and parts.port is not None and 1<=parts.port<=65535,
         'LOOPBACK_ONLY_HTTP_ENDPOINT')


def _infer_locked(work, endpoint, model, limit):
    check_local(endpoint)
    need(type(model) is str and 1<=len(model)<=256 and '\n' not in model,'MODEL_ID_INVALID')
    need(type(limit) is int and 1<=limit<=1216,'BATCH_LIMIT_INVALID')
    d,prepared,_=load_prepared(work)
    packets=prepared['packets']; entries=journal_entries(d,packets)
    header={'schema':SCHEMA+'.inference-identity','status':'LOCAL_LOOPBACK_UNVERIFIED_MODEL_WEIGHTS',
            'endpoint':endpoint,'model':model,'decode':dict(DECODE),
            'packets_sha256':digest(wire(prepared))}
    safe_write(d/'R12_DEV_INFERENCE_HEADER.json',wire(header))
    # Reject HTTP redirects and environment proxies; never leave loopback.
    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs):
            return None
    opener=build_opener(ProxyHandler({}),NoRedirect())
    for i in range(len(entries),min(len(packets),len(entries)+limit)):
        packet=packets[i]
        payload={'model':model,'messages':messages(packet),**DECODE}
        data=json.dumps(payload,ensure_ascii=True).encode('ascii')
        start=time.monotonic_ns()
        req=Request(endpoint,data=data,headers={'Content-Type':'application/json'},method='POST')
        with opener.open(req,timeout=120) as resp:
            need(resp.status==200,'MODEL_STATUS_NON_200')
            raw=resp.read(256*1024+1)
        need(len(raw)<=256*1024,'MODEL_RESPONSE_OVERSIZE')
        response=json.loads(raw)
        need(type(response) is dict and type(response.get('choices')) is list and
             len(response['choices'])==1,'MODEL_RESPONSE_SCHEMA')
        content=response['choices'][0]['message']['content']
        need(type(content) is str and len(content)<=16000,'MODEL_CONTENT_INVALID')
        old=entries[-1]['record_sha256'] if entries else ZERO
        body={'index':i,'id':packet['id'],'method':packet['method'],
              'completion_text':content,'elapsed_ns':time.monotonic_ns()-start,
              'previous_sha256':old}
        row=dict(body,record_sha256=digest(wire(body)))
        encoded=(json.dumps(row,sort_keys=True,ensure_ascii=True,separators=(',',':'))+'\n').encode('ascii')
        path=d/'R12_DEV_INFERENCE_JOURNAL.jsonl'
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_APPEND|os.O_NOFOLLOW,0o600)
        try:
            need(os.write(fd,encoded)==len(encoded),'JOURNAL_SHORT_WRITE')
            os.fsync(fd)
        finally:os.close(fd)
        entries.append(row)
        parsed=parse_answer(content,{t['turn_id'] for t in packet['turns']})
        if (i+1)%8==0 or i+1==len(packets):
            print('R12_MODEL_CALLS=',i+1,'OF=',len(packets),'LAST_OUTPUT_VALID=',parsed['valid'],flush=True)
    print('R12_INFERENCE_PROGRESS=',len(entries),'/',len(packets),
          'STATUS=',('COMPLETE' if len(entries)==len(packets) else 'RESUMABLE_DEV_ONLY'),flush=True)


def infer(work, endpoint, model, limit):
    """Exclusive single-writer inference, with no silent conflict or repair."""
    check_local(endpoint)
    root=Path(work)/'research/ecsc_hca_r0/external/locomo_r0'
    need(root.is_dir() and not root.is_symlink(),'RESEARCH_ROOT_MISSING')
    lock=root/'R12_DEV_INFERENCE.lock'
    fd=os.open(lock,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        _infer_locked(work,endpoint,model,limit)
    finally:
        os.close(fd)


def normalized(value):
    text=unicodedata.normalize('NFKC',str(value)).casefold()
    return ' '.join(re.findall(r'\w+',text,flags=re.UNICODE))


def evaluate(prepared, entries, gold):
    packets=prepared['packets']; need(len(entries)==len(packets),'INCOMPLETE_INFERENCE')
    gmap={g['id']:g for g in gold['annotations'] if g['sample_id'] in DEV_IDS}
    need(len(gmap)==304 and {p['id'] for p in packets}==set(gmap),'DEV_GOLD_MISMATCH')
    stats={m:{'n':0,'answer_present':0,'missing_answer':0,'exact_answer':0,
              'nonexact_answer':0,'answer_abstain':0,'missing_abstain':0,
              'missing_assertion':0,'format_invalid':0,'valid_citations':0,
              'citations_in_gold':0,'latency_ns':[]} for m in METHODS}
    paired={m:{} for m in METHODS}
    for p,row in zip(packets,entries):
        g=gmap[p['id']];s=stats[p['method']]
        s['n']+=1
        s['answer_present' if g['answer_present'] else 'missing_answer']+=1
        parsed=parse_answer(row['completion_text'],{t['turn_id'] for t in p['turns']})
        s['latency_ns'].append(row['elapsed_ns'])
        if not parsed['valid']:
            s['format_invalid']+=1;paired[p['method']][p['id']]='invalid';continue
        s['valid_citations']+=len(parsed['citations'])
        s['citations_in_gold']+=sum(v in g['evidence_original'] for v in parsed['citations'])
        if g['answer_present']:
            if parsed['answer'] is None:
                s['answer_abstain']+=1;out='abstain'
            elif normalized(parsed['answer'])==normalized(g['answer']):
                s['exact_answer']+=1;out='exact'
            else:
                s['nonexact_answer']+=1;out='nonexact'
        else:
            if parsed['answer'] is None:
                s['missing_abstain']+=1;out='abstain'
            else:
                s['missing_assertion']+=1;out='assertion'
        paired[p['method']][p['id']]=out
    for s in stats.values():
        s['median_model_latency_ns']=int(statistics.median(s.pop('latency_ns')))
        # Answer-present and missing-answer denominators include invalid outputs.
        s['answer_present_total']=235
        s['missing_answer_total']=69
        need(s['n']==304 and s['answer_present']==235 and s['missing_answer']==69,'METHOD_COUNT_OR_GOLD_DENOMINATORS')
    # This strictly-normalized match is a LOWER BOUND, not a semantic judgment.
    pair={}
    for m in METHODS:
        if m=='LEXICAL4_ADJACENT4':continue
        a=paired[m]; b=paired['LEXICAL4_ADJACENT4']
        pair[m]={'better_strict_answer':sum(a[k]=='exact' and b[k]!='exact' for k in a),
                 'worse_strict_answer':sum(a[k]!='exact' and b[k]=='exact' for k in a),
                 'better_missing_abstain':sum(a[k]=='abstain' and b[k]!='abstain' for k in a if not gmap[k]['answer_present']),
                 'worse_missing_abstain':sum(a[k]!='abstain' and b[k]=='abstain' for k in a if not gmap[k]['answer_present'])}
    return {'schema':SCHEMA+'.dev-score','status':'DESCRIPTIVE_DEV_ONLY_NO_SCIENTIFIC_QUAL',
         'n_questions':304,'methods':stats,'paired_vs_lexical_adjacent':pair,
         'strict_score_warning':'Normalized exact-string agreement is not semantic correctness; do not count paraphrases as proven errors.',
         'missing_answer_warning':'Absent LoCoMo answer is benchmark-defined abstention; independent unanswerability is not verified.',
         'contradiction_status':'NOT_EVALUATED; no independent correction/revocation labels',
         'no_qual_scored':True,
         'qual_gold_present_in_scorer_source':True}


def score(work):
    d,prepared,pre=load_prepared(work)
    rows=journal_entries(d,prepared['packets'])
    need(len(rows)==1216,'PREDICTIONS_INCOMPLETE')
    header,_=load_exact(d/'R12_DEV_INFERENCE_HEADER.json',SCHEMA+'.inference-identity',
             ('schema','status','endpoint','model','decode','packets_sha256'))
    need(header['packets_sha256']==pre['packets_sha256'] and header['decode']==DECODE,
         'HEADER_DRIFT')
    gpath=d/'SCORER_GOLD.json'
    # Gold-derived predecessor reports are validated ONLY after inference is complete.
    for name, expected in SCORER_ONLY_PINS.items():
        p=d/name
        need(p.is_file() and not p.is_symlink() and digest(p.read_bytes())==expected,
             'SCORER_PREDECESSOR_DRIFT:'+name)
    gold,gsha=load_exact(gpath,R9_SCHEMA+'.gold',('schema','status','annotations'))
    att=_attestation(d)
    need(gsha==att['scorer_gold_sha256'] and gold['status']=='SCORER_ONLY','GOLD_PIN_MISMATCH')
    report=evaluate(prepared,rows,gold)
    report.update({'packets_sha256':pre['packets_sha256'],
                   'inference_header_sha256':digest(wire(header)),
                   'journal_sha256':digest((d/'R12_DEV_INFERENCE_JOURNAL.jsonl').read_bytes()),
                   'gold_sha256':gsha})
    out=d/'R12_DEV_ANSWER_SCORE.json'
    safe_write(out,wire(report))
    print('R12_DEV_SCORE=',out,'SHA256=',digest(wire(report)))
    for method,s in report['methods'].items():
        print('R12',method,'EXACT_ANSWER=',s['exact_answer'],'/235',
              'MISSING_ABSTAIN=',s['missing_abstain'],'/69',
              'INVALID=',s['format_invalid'],'NONEXACT=',s['nonexact_answer'],flush=True)
    print('R12_ANSWER_DEV=PASS_DESCRIPTIVE_NOT_SCIENTIFIC_QUAL',flush=True)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    for name in ('prepare','score'):
        x=sub.add_parser(name);x.add_argument('workspace')
    x=sub.add_parser('infer');x.add_argument('workspace');x.add_argument('--endpoint',required=True)
    x.add_argument('--model',required=True);x.add_argument('--steps',type=int,default=16)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.workspace)
    elif a.command=='infer':infer(a.workspace,a.endpoint,a.model,a.steps)
    else:score(a.workspace)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('R12_HOLD=',type(exc).__name__,str(exc),file=sys.stderr,flush=True)
        raise SystemExit(3)
