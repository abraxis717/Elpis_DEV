"""Microbenchmark only: compare R10 host JSON/whole-image vs R11 C typed state ops."""
from __future__ import annotations
import json, shutil, statistics, time
from dataclasses import replace
from test_r8_ssc_real_contract import context
from native.ersa_authority_r10 import compile_authority
from native.ersa_authority_r10 import NativeAuthority
from integration.ssc_native_r10 import NativeSscAuthorityR10
from integration.ssc_native_r11 import NativeSscAuthorityR11,compile_r11

def clock_calls(bank,n=24):
    for _ in range(3):bank.read_exact(101)
    obs=[]
    for _ in range(n):
        t=time.perf_counter_ns();bank.read_exact(101);obs.append((time.perf_counter_ns()-t)/1000)
    return round(statistics.median(obs),2)

def main():
    l10=compile_authority();l11=compile_r11();results=[]
    try:
        for n in (256,1024):
            spec,reg,tr,_,_ = context()
            spec=replace(spec,max_active_entities=n,max_active_relations=n,max_ops_per_tick=60000)
            one=NativeSscAuthorityR10(spec,reg,tr,l10)
            two=NativeSscAuthorityR11(spec,reg,tr,l11)
            try:
                for i in range(1,n+1):
                    assert one.upsert_exact(i,(.125,0,0,0),'fixture:write:r0').status=='OK'
                    assert two.upsert_exact(i,(.125,0,0,0),'fixture:write:r0').status=='OK'
                r10=clock_calls(one);r11=clock_calls(two)
                results.append(dict(n=n,r10_us=r10,r11_us=r11,scope='EXACT_READ_ONLY_FULL_OWNED_STATE'))
            finally:one.close();two.close()
        spec,reg,tr,_,_=context()
        spec=replace(spec,max_active_entities=8192,max_active_relations=8192,max_ops_per_tick=20000)
        with NativeSscAuthorityR11(spec,reg,tr,l11) as b:
            for i in range(1,8193):b.upsert_exact(i,(.125,0,0,0),'fixture:write:r0')
            results.append(dict(n=8192,r10_us=None,r11_us=clock_calls(b),scope='R11_ONLY_FULL_CAP_EXACT_READ'))
        print('R11_READ_MICROBENCH='+json.dumps(results,separators=(',',':')))
    finally:
        shutil.rmtree(l10.parent,ignore_errors=True)
        shutil.rmtree(l11.parent,ignore_errors=True)
if __name__=='__main__':main()
