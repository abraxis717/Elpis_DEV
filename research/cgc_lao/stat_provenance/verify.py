#!/usr/bin/env python3
"""Read-only verification of typed statistical receipts and R8 methodology; no scientific execution."""
import argparse, hashlib, json, math, pathlib, zipfile
ROOT=pathlib.Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def require(x,msg):
    if not x: raise SystemExit('BLOCKER='+msg)

def exact_p(v):
    if not v:return None
    assert len(v)<=16
    observed=abs(math.fsum(v));tot=-math.fsum(v)
    count=int(abs(tot)>=observed-1e-12);prev=0
    for j in range(1,1<<len(v)):
        mask=j^(j>>1);flip=(mask^prev).bit_length()-1
        tot+=2*v[flip] if (mask>>flip)&1 else -2*v[flip]
        count+=abs(tot)>=observed-1e-12
        prev=mask
    return count/(1<<len(v))

def holm(v):
    m=len(v);order=sorted(range(m),key=lambda i:(v[i],i));out=[None]*m;hi=0.
    for j,i in enumerate(order):
        hi=max(hi,min(1.,(m-j)*v[i]));out[i]=hi
    return out

def lcg_unique(n,draws=20000):
    state=12345;seen=set()
    for _ in range(draws):
        mask=0
        for i in range(n):
            state=(1103515245*state+12345)&0xffffffff
            if (state>>9)&1:mask|=1<<i
        seen.add(mask)
    return len(seen)

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--reanalysis-zip');a=pa.parse_args()
    reg=json.loads((ROOT/'R25A_R8_ENUMERATION_REGISTRATION.json').read_bytes())
    data=json.loads((ROOT/'R25A_R7_STATISTICAL_RECEIPTS.json').read_bytes())
    require(reg['phase']=='R25A_R8' and reg['status']=='METHOD_FROZEN_BEFORE_R8_EXECUTION','REGISTRATION_DRIFT')
    require(reg['method_source_sha256']==digest(__file__),'METHOD_SOURCE_DRIFT')
    require(len(data['receipts'])==48,'RECEIPT_COUNT')
    require(exact_p([1.]*16)==2/65536,'EXACT_TAIL_CANARY')
    require(exact_p([0.]*16)==1.0,'NULL_CANARY')
    require(abs(holm([.0001,.01,.06]+[1.]*21)[0]-.0024)<1e-14,'HOLM_CANARY')
    require(lcg_unique(16)==64,'LEGACY_SAMPLER_CANARY')
    lcg_cache={0:0}
    for compiler in ('gcc','clang'):
        block=[x for x in data['receipts'] if x['compiler']==compiler]
        require(len(block)==24,'FAMILYWISE_SIZE')
        ps=[]
        for r in block:
            n=r['n_seed_pairs'];v=r['paired_deltas'];s=r['sampler_corrected'];old=r['sampler_historical']
            require(n==len(v) and n<=16,'SEED_PAIR_COUNT')
            require(r['test_scope']==('conditional_on_adapt_publications_EXPLORATORY' if r['endpoint'].startswith('E3') else 'full_16_seeds_archival_task'),'SCOPE_DRIFT')
            require(s['enumerated_assignments']==(1<<n if n else 0),'SUPPORT_MISMATCH')
            require(s['enumeration_is_complete']==(True if n else None),'COVERAGE_MISMATCH')
            require(old['intended_uniform_assignment_space']==(1<<n if n else 0),'LEGACY_SUPPORT_MISMATCH')
            if n not in lcg_cache:lcg_cache[n]=lcg_unique(n)
            require(old['distinct_sign_vectors_observed']==lcg_cache[n],'LCG_MASK_COUNT')
            require(r['current_qualification_authority']=='ARCHIVAL_SCORE_ONLY_NO_NATIVE_OR_TASK_PROMOTION','RECEIPT_AUTHORITY_DRIFT')
            require(s['algorithm']=='EXHAUSTIVE_SIGNFLIP_GRAY_2SIDED_TOL_1E12','SAMPLER_TYPE_DRIFT')
            require(r['unit_of_inference']=='paired_development_seed','INFERENCE_UNIT_DRIFT')
            p=exact_p(v)
            require(p==s['p_exact'],'EXACT_P_MISMATCH')
            ps.append(p if p is not None else 1.)
        for r,p in zip(block,holm(ps)):
            require(abs(p-r['sampler_corrected']['holm24_p'])<1e-14,'HOLM24_DRIFT')
        for family in {x['family'] for x in block}:
            e1=next(x for x in block if x['family']==family and x['endpoint']=='E1_acc_tie')
            e2=next(x for x in block if x['family']==family and x['endpoint']=='E2_regret_gain')
            require(all(abs(16*x-y)<1e-13 for x,y in zip(e1['paired_deltas'],e2['paired_deltas'])),'E1_E2_REDUNDANCY_DRIFT')
    if a.reanalysis_zip:
        p=pathlib.Path(a.reanalysis_zip)
        require(p.is_file(),'REANALYSIS_ARCHIVE_ABSENT')
        require(digest(p)==data['source_reanalysis_sha256'],'REANALYSIS_ARCHIVE_SHA_DRIFT')
        with zipfile.ZipFile(p) as z:
            require(z.testzip() is None,'REANALYSIS_ZIP_CRC')
            m=json.loads(z.read('MANIFEST.json'))
            for entry in m['files']:
                raw=z.read(entry['path'])
                require(len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],'REANALYSIS_INTERNAL_HASH_DRIFT')
            before=json.loads(z.read('results.json'))
            for c in ('gcc','clang'):
                source={(z['family'],z['endpoint']):z for z in before['results'][c]['hypotheses']}
                for r in (x for x in data['receipts'] if x['compiler']==c):
                    h=source[(r['family'],r['endpoint'])]
                    require(h['values']==r['paired_deltas'] and h['holm_p_adj_24']==r['sampler_corrected']['holm24_p'],'SOURCE_REANALYSIS_DRIFT')
    print('=== SEND BACK ===')
    print('PHASE=CGC_LAO_R24_STATISTICAL_PROVENANCE_REGISTRATION')
    print('DISPOSITION=METHOD_AND_RETROSPECTIVE_RECEIPTS_VERIFIED')
    print('RECORDED_HYPOTHESES=48 RECOMPUTED_EXACT_P=48 HOLM_FAMILIES=2x24')
    print('LEGACY_LCG_DISTINCT_MASKS_16=64 OF 65536')
    print('E1_E2_SEED_LEVEL_REDUNDANCY=16/16_FAMILY_COMPILER_PAIRS')
    print('REANALYSIS_SOURCE_VERIFIED='+('YES' if a.reanalysis_zip else 'NOT_PROVIDED_TO_VALIDATOR'))
    print('R24_HISTORICAL_DISPOSITION=UNCHANGED CURRENT_RELIANCE=HOLD')
    print('R8_PROSPECTIVE_RESULT=NOT_RUN')
    print('BLOCKER=NONE EXIT_STATUS=0')
if __name__=='__main__':main()
