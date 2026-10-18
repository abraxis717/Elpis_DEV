#define _POSIX_C_SOURCE 200809L
#include "elpis/ecsg_fms.h"
#include "elpis/fms_pal_posix.h"
#include <assert.h>
#include <stdio.h>
#include <time.h>

static uint64_t ns(void) {
    struct timespec t; (void)clock_gettime(CLOCK_MONOTONIC,&t);
    return (uint64_t)t.tv_sec*1000000000u+(uint64_t)t.tv_nsec;
}
int main(void) {
    double w[216],x[384],y[64],out[64];
    for(int i=0;i<216;++i) w[i]=0.005*(i%17-8);
    for(int i=0;i<384;++i) x[i]=0.025*(i%13-6);
    for(int i=0;i<64;++i) y[i]=0.02*(i%5-2);
    const unsigned runs=100;
    elpis_ecsg_executor *e=NULL;
    assert(!elpis_ecsg_executor_create(6,36,64,w,&e));
    uint64_t query=0,learn=0;
    for(unsigned i=0;i<runs;++i) {
        uint64_t t=ns(); assert(!elpis_ecsg_executor_forward(e,x,64,out)); query+=ns()-t;
        t=ns();assert(!elpis_ecsg_executor_learn(e,x,y,64,0.002,10,NULL));learn+=ns()-t;
    }
    assert(!elpis_ecsg_executor_destroy(&e));
    printf("PERFORMANCE_ONLY NO_SCIENTIFIC_CLAIM d=6 N=36 rows=64 K=10 runs=%u mean_ns\n",runs);
    printf("resident query=%llu learn_including_exchange=%llu\n",
           (unsigned long long)(query/runs),(unsigned long long)(learn/runs));
    for(int cold=0;cold<2;++cold) {
        fms_config cfg={0};cfg.tier_budget[1]=cfg.domain_ceiling[0]=40+216*8;
        cfg.tier_budget[2]=cfg.domain_ceiling[2]=10*(40+216*8);
        cfg.high_wm=0.9f;cfg.low_wm=0.7f;cfg.max_objects=4;
        fms_ctx *ctx=elpis_fms_create_posix(&cfg,cold?"perf-cold":"perf-warm");assert(ctx);
        elpis_ecsg_fms *r=NULL;assert(!elpis_ecsg_fms_create(ctx,4,&r));
        uint8_t key[32]={1};uint64_t id;
        assert(!elpis_ecsg_fms_register(r,key,6,36,64,w,&id));
        query=learn=0;
        for(unsigned i=0;i<runs;++i) {
            if(cold) assert(!elpis_ecsg_fms_pump(r));
            uint64_t t=ns();assert(!elpis_ecsg_fms_forward(r,id,x,64,out));query+=ns()-t;
            if(cold) assert(!elpis_ecsg_fms_pump(r));
            t=ns();assert(!elpis_ecsg_fms_learn(r,id,x,y,64,0.002,10,NULL));learn+=ns()-t;
        }
        elpis_ecsg_fms_info info;elpis_ecsg_fms_metrics m;
        assert(!elpis_ecsg_fms_inspect(r,id,&info));
        assert(!elpis_ecsg_fms_stats(r,&m));
        printf("%s query_total=%llu learn_total=%llu acquire_per_call=%llu query_compute=%llu learn_compute=%llu commit_copy=%llu pump_per_call=%llu cold_reads=%llu cold_writes=%llu\n",
            cold?"cold":"warm",(unsigned long long)(query/runs),(unsigned long long)(learn/runs),
            (unsigned long long)(info.materialization_ns/(2*runs)),(unsigned long long)(info.query_ns/runs),
            (unsigned long long)(info.learn_ns/runs),(unsigned long long)(info.commit_ns/runs),
            (unsigned long long)(m.demotion_ns/(2*runs)),
            (unsigned long long)m.residency.cold_reads,(unsigned long long)m.residency.cold_writes);
        assert(!elpis_ecsg_fms_close(r,&id));assert(!elpis_ecsg_fms_destroy(&r));
    }
    return 0;
}
