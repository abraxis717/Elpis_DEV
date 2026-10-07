#define _POSIX_C_SOURCE 200809L
#include "elpis/ecsg_fms.h"
#include "elpis/fms_pal_posix.h"
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define OK(x) assert((x) == 0)
#define BYTES (40u + 6u * 36u * 8u)
static double w[6 * 72], x[64 * 6], y[64];
static unsigned rng = 717;
static double random_value(void) {
    rng = rng * 1664525u + 1013904223u;
    return ((double)(rng >> 8) / 16777216.0 - 0.5) * 0.2;
}
static void data(void) {
    for (size_t i = 0; i < sizeof(w)/sizeof(*w); ++i) w[i] = random_value();
    for (size_t i = 0; i < sizeof(x)/sizeof(*x); ++i) x[i] = random_value();
    for (size_t i = 0; i < sizeof(y)/sizeof(*y); ++i) y[i] = random_value();
}
static fms_config config(size_t warm, unsigned objects) {
    fms_config c = {0};
    c.tier_budget[FMS_WARM] = c.domain_ceiling[FMS_DOM_RAM] = warm;
    c.tier_budget[FMS_COLD] = c.domain_ceiling[FMS_DOM_STORAGE] = BYTES * 100;
    c.high_wm = 0.9f; c.low_wm = 0.7f; c.max_objects = objects;
    c.hot_absent_policy = c.cold_absent_policy = FMS_REJECT;
    return c;
}
static elpis_ecsg_fms *runtime(const char *root, size_t warm, unsigned objects) {
    fms_config c = config(warm, objects);
    fms_ctx *ctx = elpis_fms_create_posix(&c, root);
    assert(ctx);
    elpis_ecsg_fms *r = NULL;
    OK(elpis_ecsg_fms_create(ctx, objects, &r));
    return r;
}
static uint64_t state(elpis_ecsg_fms *r, unsigned key, size_t width) {
    uint8_t k[32] = {0}; k[0] = (uint8_t)key;
    uint64_t id = 0;
    OK(elpis_ecsg_fms_register(r, k, 6, width, 64, w, &id));
    assert(id);
    return id;
}
static void equal(elpis_ecsg_fms *r, uint64_t id, elpis_ecsg_executor *e) {
    uint8_t a[40 + 6 * 72 * 8], b[sizeof(a)];
    size_t n = elpis_ecsg_executor_snapshot_size(e);
    OK(elpis_ecsg_fms_snapshot_write(r, id, a, n));
    OK(elpis_ecsg_executor_snapshot_write(e, b, n));
    assert(!memcmp(a, b, n));
    double p[64], q[64], sm[83], tm[83];
    OK(elpis_ecsg_fms_forward(r, id, x, 64, p));
    OK(elpis_ecsg_executor_forward(e, x, 64, q));
    assert(!memcmp(p, q, sizeof(p)));
    OK(elpis_ecsg_fms_project_s3(r, id, sm, sm+6, sm+27));
    OK(elpis_ecsg_executor_project_s3(e, tm, tm+6, tm+27));
    assert(!memcmp(sm, tm, sizeof(sm)));
}
static void parity(void) {
    elpis_ecsg_fms *r = runtime("parity", BYTES * 3, 8);
    const size_t widths[] = {36,48,72}, rows[] = {1,7,8,17,64};
    const uint64_t steps[] = {1,3,17};
    unsigned cases = 0;
    for (size_t n = 0; n < 3; ++n) for (size_t row = 0; row < 5; ++row)
    for (size_t k = 0; k < 3; ++k) {
        data();
        uint64_t id = state(r, 1, widths[n]);
        elpis_ecsg_executor *e = NULL;
        OK(elpis_ecsg_executor_create(6, widths[n], 64, w, &e));
        equal(r, id, e);
        elpis_ecsg_exec_transition a, b;
        OK(elpis_ecsg_fms_learn(r,id,x,y,rows[row],0.002,steps[k],&a));
        OK(elpis_ecsg_executor_learn(e,x,y,rows[row],0.002,steps[k],&b));
        assert(!memcmp(&a,&b,sizeof(a)));
        equal(r,id,e);
        const elpis_ecsg_drive drives[] = {{3,2},{5,3}};
        OK(elpis_ecsg_fms_learn_schedule(r,id,x,y,drives,2,0.002,&a));
        OK(elpis_ecsg_executor_learn_schedule(e,x,y,drives,2,0.002,&b));
        assert(!memcmp(&a,&b,sizeof(a)));
        uint64_t ta,tb;
        OK(elpis_ecsg_fms_txn_begin(r,id,&ta));
        OK(elpis_ecsg_executor_txn_begin(e,&tb));
        OK(elpis_ecsg_fms_txn_learn_schedule(r,id,ta,x,y,drives,2,0.002,&a));
        OK(elpis_ecsg_executor_txn_learn_schedule(e,tb,x,y,drives,2,0.002,&b));
        assert(!memcmp(&a,&b,sizeof(a)));
        double p[83],q[83];
        OK(elpis_ecsg_fms_txn_project_s3(r,id,ta,p,p+6,p+27));
        OK(elpis_ecsg_executor_txn_project_s3(e,tb,q,q+6,q+27));
        assert(!memcmp(p,q,sizeof(p)));
        OK(elpis_ecsg_fms_txn_forward(r,id,ta,x,64,p));
        OK(elpis_ecsg_executor_txn_forward(e,tb,x,64,q));
        assert(!memcmp(p,q,64*sizeof(double)));
        OK(elpis_ecsg_fms_txn_commit(r,id,ta,&a));
        OK(elpis_ecsg_executor_txn_commit(e,tb,&b));
        assert(!memcmp(&a,&b,sizeof(a)));
        equal(r,id,e);
        OK(elpis_ecsg_fms_pump(r));
        equal(r,id,e);
        uint8_t blob[40+6*72*8];
        size_t size = elpis_ecsg_executor_snapshot_size(e);
        OK(elpis_ecsg_fms_snapshot_write(r,id,blob,size));
        elpis_ecsg_executor *restored = NULL;
        OK(elpis_ecsg_executor_restore(blob,size,64,&restored));
        equal(r,id,restored);
        OK(elpis_ecsg_executor_destroy(&restored));
        OK(elpis_ecsg_fms_close(r,&id));
        uint8_t key[32] = {1};
        OK(elpis_ecsg_fms_restore(r,key,blob,size,64,&id));
        equal(r,id,e);
        OK(elpis_ecsg_executor_destroy(&e));
        OK(elpis_ecsg_fms_close(r,&id));
        ++cases;
    }
    OK(elpis_ecsg_fms_destroy(&r));
    printf("bitwise differential: %u cases; forward/K/schedule/candidate/commit/S3/snapshot/epoch PASS\n",cases);
}
static void scaling(void) {
    elpis_ecsg_fms *r = runtime("scaling", BYTES*2, 12);
    uint64_t ids[10];
    double baseline[10][64], out[64];
    for (unsigned i=0; i<10; ++i) {
        data(); ids[i]=state(r,i+1,36);
        OK(elpis_ecsg_fms_learn(r,ids[i],x,y,64,0.002,i+1,NULL));
    }
    for (unsigned i=0; i<10; ++i) OK(elpis_ecsg_fms_forward(r,ids[i],x,64,baseline[i]));
    elpis_ecsg_fms_metrics m;
    OK(elpis_ecsg_fms_stats(r,&m));
    assert(m.states==10 && m.logical_bytes==10*BYTES);
    assert(m.resident_authoritative_bytes<=2*BYTES && m.logical_bytes>m.resident_authoritative_bytes);
    assert(m.residency.cold_reads && m.residency.cold_writes && !m.leases && !m.residency.pinned_bytes);
    OK(elpis_ecsg_fms_forward(r,ids[0],x,64,out));
    assert(!memcmp(out,baseline[0],sizeof(out)));
    OK(elpis_ecsg_fms_learn(r,ids[1],x,y,64,0.002,7,NULL));
    OK(elpis_ecsg_fms_pump(r));
    for (unsigned i=0; i<10; ++i) {
        OK(elpis_ecsg_fms_forward(r,ids[i],x,64,out));
        if (i==1) assert(memcmp(out,baseline[i],sizeof(out)));
        else assert(!memcmp(out,baseline[i],sizeof(out)));
        elpis_ecsg_fms_info info;
        OK(elpis_ecsg_fms_inspect(r,ids[i],&info));
        assert(info.epoch==i+1+(i==1?7:0));
    }
    printf("scaling: logical=%llu resident=%llu active_workspace=%llu WARM=%llu HOT=%llu COLD=%llu storage=%llu pins=%llu leases=%llu high_water=%llu\n",
        (unsigned long long)m.logical_bytes,(unsigned long long)m.resident_authoritative_bytes,
        (unsigned long long)m.active_workspace_bytes,
        (unsigned long long)m.residency.tier_bytes[1],(unsigned long long)m.residency.tier_bytes[0],
        (unsigned long long)m.residency.tier_bytes[2],(unsigned long long)m.residency.domain_bytes[2],
        (unsigned long long)m.residency.pinned_bytes,(unsigned long long)m.leases,
        (unsigned long long)m.resident_high_water);
    for(unsigned i=0;i<10;++i) OK(elpis_ecsg_fms_close(r,&ids[i]));
    OK(elpis_ecsg_fms_destroy(&r));
}
static void atomicity(void) {
    data();
    elpis_ecsg_fms *r=runtime("atomicity",BYTES*2,4);
    uint64_t id=state(r,1,36), other=state(r,2,36), token;
    uint8_t before[BYTES], after[sizeof(before)];
    OK(elpis_ecsg_fms_snapshot_write(r,id,before,sizeof(before)));
    double bad[64]; memcpy(bad,y,sizeof(bad)); bad[3]=NAN;
    assert(elpis_ecsg_fms_learn(r,id,x,bad,64,0.002,3,NULL)==-2);
    /* Finite input, late arithmetic refusal. */
    assert(elpis_ecsg_fms_learn(r,id,x,y,64,1e200,10,NULL)==-2);
    OK(elpis_ecsg_fms_snapshot_write(r,id,after,sizeof(after)));
    assert(!memcmp(before,after,sizeof(before)));
    OK(elpis_ecsg_fms_txn_begin(r,id,&token));
    assert(elpis_ecsg_fms_close(r,&id)==-4);
    assert(elpis_ecsg_fms_destroy(&r)==-4);
    uint64_t invalid;
    assert(elpis_ecsg_fms_txn_begin(r,id,&invalid)==-4);
    OK(elpis_ecsg_fms_txn_learn(r,id,token,x,y,64,0.002,3,NULL));
    OK(elpis_ecsg_fms_pump(r));
    elpis_ecsg_fms_info info;
    OK(elpis_ecsg_fms_inspect(r,id,&info));
    assert(info.tier==FMS_WARM && info.lease_count==1);
    assert(elpis_ecsg_fms_txn_commit(r,id,token+1,NULL)==-1);
    OK(elpis_ecsg_fms_snapshot_write(r,id,after,sizeof(after)));
    assert(!memcmp(before,after,sizeof(before)));
    OK(elpis_ecsg_fms_txn_abort(r,id,token));
    OK(elpis_ecsg_fms_txn_begin(r,id,&token));
    OK(elpis_ecsg_fms_txn_learn(r,id,token,x,y,64,0.002,2,NULL));
    OK(elpis_ecsg_fms_learn(r,id,x,y,64,0.002,1,NULL));
    OK(elpis_ecsg_fms_snapshot_write(r,id,before,sizeof(before)));
    assert(elpis_ecsg_fms_txn_commit(r,id,token,NULL)==-3);
    OK(elpis_ecsg_fms_snapshot_write(r,id,after,sizeof(after)));
    assert(!memcmp(before,after,sizeof(before)));
    OK(elpis_ecsg_fms_txn_begin(r,id,&token));
    assert(elpis_ecsg_fms_txn_learn(r,id,token,x,bad,64,0.002,2,NULL)==-2);
    OK(elpis_ecsg_fms_inspect(r,id,&info)); assert(!info.lease_count);
    uint64_t stale=id;
    OK(elpis_ecsg_fms_close(r,&id)); OK(elpis_ecsg_fms_close(r,&id));
    assert(elpis_ecsg_fms_inspect(r,stale,&info)==-1);
    id=state(r,1,36); assert(id!=stale);
    assert(elpis_ecsg_fms_forward(r,stale,x,64,y)==-1);
    OK(elpis_ecsg_fms_close(r,&id)); OK(elpis_ecsg_fms_close(r,&other));
    OK(elpis_ecsg_fms_destroy(&r)); OK(elpis_ecsg_fms_destroy(&r));
    puts("atomicity, stale transactions, lease lifetime and invalid handles PASS");
}

static int fail_ram(void *self, uint64_t bytes, void **out) {
    (void)self;(void)bytes;(void)out; return FMS_PAL_ENOMEM;
}
static int fail_read(void *self, const fms_cold_token *t, void *out, uint64_t bytes) {
    (void)self;(void)t;(void)out;(void)bytes; return FMS_PAL_EIO;
}
static int fail_put(void *self,const void *src,uint64_t bytes,fms_cold_token **out) {
    (void)self;(void)src;(void)bytes;(void)out;return FMS_PAL_EIO;
}
static void faults(void) {
    for(int mode=0;mode<5;++mode) {
        char root[32]; snprintf(root,sizeof(root),"fault-%d",mode);
        fms_pal *pal=fms_pal_posix_create(root); assert(pal);
        fms_config cfg=config(BYTES,2);
        fms_ctx *ctx=fms_create(&cfg,pal); assert(ctx);
        elpis_ecsg_fms *r=NULL; OK(elpis_ecsg_fms_create(ctx,4,&r));
        data();
        uint64_t id=state(r,1,36), other=0;
        uint8_t key[32]={2};
        int (*ram)(void *,uint64_t,void **)=pal->ram_alloc;
        if(mode==0) {
            pal->ram_alloc=fail_ram;
            /* registration allocation failure after a successful eviction */
            assert(elpis_ecsg_fms_register(r,key,6,36,64,w,&other)==-102 && !other);
            assert(elpis_ecsg_fms_forward(r,id,x,64,y)==-102);
            pal->ram_alloc=ram;
            OK(elpis_ecsg_fms_forward(r,id,x,64,y));
        } else if(mode==1) {
            uint64_t token; OK(elpis_ecsg_fms_txn_begin(r,id,&token));
            assert(elpis_ecsg_fms_register(r,key,6,36,64,w,&other)==-107 && !other);
            OK(elpis_ecsg_fms_txn_abort(r,id,token));
        } else if(mode==2) {
            OK(elpis_ecsg_fms_pump(r)); pal->cold_get=fail_read;
            assert(elpis_ecsg_fms_forward(r,id,x,64,y)==-106);
            assert(elpis_ecsg_fms_forward(r,id,x,64,y)==-108);
        } else if(mode==3) {
            OK(elpis_ecsg_fms_pump(r));
            DIR *dir=opendir(root); assert(dir); struct dirent *ent; int changed=0;
            while((ent=readdir(dir))) if(strstr(ent->d_name,".blob")) {
                int fd=openat(dirfd(dir),ent->d_name,O_WRONLY); assert(fd>=0);
                uint8_t bad=0xff; assert(pwrite(fd,&bad,1,16)==1); close(fd); ++changed;
            }
            closedir(dir); assert(changed==1);
            assert(elpis_ecsg_fms_forward(r,id,x,64,y)==-109);
            assert(elpis_ecsg_fms_forward(r,id,x,64,y)==-108);
        } else {
            pal->cold_put=fail_put;
            assert(elpis_ecsg_fms_register(r,key,6,36,64,w,&other)==-107 && !other);
            OK(elpis_ecsg_fms_forward(r,id,x,64,y)); /* intact WARM source */
        }
        elpis_ecsg_fms_metrics m; OK(elpis_ecsg_fms_stats(r,&m));
        assert(!m.leases && !m.residency.pinned_bytes && m.states==1);
        OK(elpis_ecsg_fms_close(r,&id)); OK(elpis_ecsg_fms_destroy(&r));
    }
    elpis_ecsg_fms *r=runtime("slots",BYTES*3,1);
    uint64_t id=state(r,1,36), another=0; uint8_t key[32]={2};
    assert(elpis_ecsg_fms_register(r,key,6,36,64,w,&another)==-5);
    OK(elpis_ecsg_fms_close(r,&id)); OK(elpis_ecsg_fms_destroy(&r));
    puts("registration, capacity, promotion allocation, cold read/digest/write failures PASS");
}

typedef struct { elpis_ecsg_fms *r; uint64_t id; atomic_int entered,done; } worker;
static void *learn_worker(void *arg) {
    worker *a=arg;
    atomic_store(&a->entered,1);
    int rc;
    do { rc=elpis_ecsg_fms_learn(a->r,a->id,x,y,64,0.002,2000,NULL); } while(rc==-4);
    assert(!rc); atomic_store(&a->done,1); return NULL;
}
static void concurrency(void) {
    data();
    elpis_ecsg_fms *r=runtime("concurrency",BYTES*2,4);
    worker a={.r=r,.id=state(r,1,36)};
    atomic_init(&a.entered,0); atomic_init(&a.done,0);
    uint64_t other=state(r,2,36);
    pthread_t th; assert(!pthread_create(&th,NULL,learn_worker,&a));
    unsigned busy=0; double out[64];
    while(!atomic_load(&a.entered)) {}
    while(!atomic_load(&a.done)) {
        int rc=elpis_ecsg_fms_forward(r,a.id,x,64,out);
        assert(!rc || rc==-4); if(rc==-4) ++busy;
        OK(elpis_ecsg_fms_forward(r,other,x,64,out));
        OK(elpis_ecsg_fms_pump(r));
        struct timespec t={0,100000}; nanosleep(&t,NULL);
    }
    assert(!pthread_join(th,NULL)); assert(busy);
    OK(elpis_ecsg_fms_close(r,&a.id)); OK(elpis_ecsg_fms_close(r,&other));
    OK(elpis_ecsg_fms_destroy(&r));
    printf("same-state BUSY=%u; independent state and concurrent pump PASS\n",busy);
}
int main(void) {
    assert(elpis_ecsg_fms_abi_version()==1);
    parity(); scaling(); atomicity(); faults(); concurrency();
    return 0;
}
