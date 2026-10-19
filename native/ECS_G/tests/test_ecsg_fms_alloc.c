#include "elpis/ecsg_fms.h"
#include "elpis/fms_pal_posix.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Link-time failure injection only. Every allocation of the integration and
 * generic FMS is covered; production has no fault-control surface. */
static unsigned fail_at, calls;
void *__real_malloc(size_t);
void *__real_calloc(size_t,size_t);
void *__real_aligned_alloc(size_t,size_t);
void *__wrap_malloc(size_t n) { return ++calls==fail_at ? NULL : __real_malloc(n); }
void *__wrap_calloc(size_t n,size_t z) { return ++calls==fail_at ? NULL : __real_calloc(n,z); }
void *__wrap_aligned_alloc(size_t a,size_t n) { return ++calls==fail_at ? NULL : __real_aligned_alloc(a,n); }
static void fault(unsigned n) { calls=0; fail_at=n; }
int main(void) {
    const size_t bytes=40+216*8;
    fms_config cfg={0};
    cfg.tier_budget[1]=cfg.domain_ceiling[0]=bytes*2;
    cfg.max_objects=4;cfg.high_wm=0.9f;cfg.low_wm=0.7f;
    unsigned refused=0;
    for(unsigned n=1;n<=8;++n) {
        fms_ctx *ctx=fms_create(&cfg,fms_pal_posix_create_ram_only()); assert(ctx);
        elpis_ecsg_fms *r=NULL;
        fault(n);
        int rc=elpis_ecsg_fms_create(ctx,4,&r);
        fault(0);
        if(rc) { assert(!r);fms_destroy(ctx);++refused;continue; }
        uint8_t key[32]={1};double w[216]={0};
        uint64_t id=0;
        fault(n);
        rc=elpis_ecsg_fms_register(r,key,6,36,8,w,&id);
        fault(0);
        if(rc) { assert(!id);++refused; }
        else {
            uint8_t before[40+216*8],after[sizeof(before)];
            assert(!elpis_ecsg_fms_snapshot_write(r,id,before,sizeof(before)));
            double x[6]={0.2,0.1,0.3,0.4,0.1,0.2},y[1]={1.0};
            fault(1); /* lease allocation refuses: no W committed */
            assert(elpis_ecsg_fms_learn(r,id,x,y,1,0.002,3,NULL)==-102);
            fault(0);
            assert(!elpis_ecsg_fms_snapshot_write(r,id,after,sizeof(after)));
            assert(!memcmp(before,after,sizeof(before)));
            assert(!elpis_ecsg_fms_close(r,&id));
        }
        elpis_ecsg_fms_metrics m;
        assert(!elpis_ecsg_fms_stats(r,&m));
        assert(!m.states && !m.leases && !m.residency.pinned_bytes);
        assert(!elpis_ecsg_fms_destroy(&r));
    }
    assert(refused>=4);
    printf("allocator failure points refused=%u; no partial registration/commit or leaked leases\n",refused);
    return 0;
}
