#include "indexed_lattice.h"
#include "bounded_lattice.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

static uint64_t checks, heap_operations;
static int track_heap;
void *__real_calloc(size_t,size_t);
void *__real_malloc(size_t);
void __real_free(void*);
void *__wrap_calloc(size_t a,size_t b){if(track_heap)heap_operations++;return __real_calloc(a,b);}
void *__wrap_malloc(size_t a){if(track_heap)heap_operations++;return __real_malloc(a);}
void __wrap_free(void*p){if(track_heap)heap_operations++;__real_free(p);}
#define REQUIRE(c) do { checks++; if(!(c)){fprintf(stderr,"FAIL_D31C line=%d condition=%s\n",__LINE__,#c);return 2;} }while(0)
static void samples(unsigned t,float k[64],float v[64],float q[64],float alpha[2],float beta[2]){
    for(unsigned j=0;j<64;j++) {
        k[j]=(float)(sin((double)(j+t)*.117)/8.0);
        v[j]=(float)(cos((double)(j+3*t)*.137)/2.0);
        q[j]=(float)(sin((double)(j+5*t)*.157)/8.0);
    }
    alpha[0]=.998f;alpha[1]=.970f;beta[0]=.73f;beta[1]=.51f;
}
static void address(unsigned t,double xy[2]){
    xy[0]=(double)((int)((t*11u)%33u)-16)+.17;
    xy[1]=(double)((int)((t*7u)%35u)-17)-.27;
}
static int compare_outputs(const d31b_arena *b,const d31c_arena *c,const double xy[2],const float q[64]){
    float bo[64],co[64];
    int br=d31b_read(b,xy,q,bo),cr=d31c_read(c,xy,q,co);
    REQUIRE(br==cr);
    if(br==D31B_OK)for(size_t i=0;i<64;i++)REQUIRE(bo[i]==co[i]);
    return 0;
}
int main(void){
    REQUIRE(d31c_create(8,D31C_REFUSE,96)==NULL);
    REQUIRE(d31c_create(9,D31C_REFUSE,1)==NULL);
    REQUIRE(d31c_create(9,D31C_REFUSE,257)==NULL);
    REQUIRE(d31c_create(9,(d31c_policy)5,96)==NULL);
    REQUIRE(d31c_cell_bytes()==d31b_cell_bytes());
    for(size_t ci=0;ci<4;ci++){
        const size_t caps[]={9,18,72,288};
        const size_t cap=caps[ci];
        for(unsigned pol=0;pol<2;pol++){
            d31b_arena *b=d31b_create(cap,(d31b_policy)pol);
            d31c_arena *c=d31c_create(cap,(d31c_policy)pol,96);
            REQUIRE(b&&c);
            const size_t bytes=d31c_allocated_bytes(c),idxbytes=d31c_index_bytes(c);
            REQUIRE(idxbytes==(size_t)195*195*4);
            REQUIRE(bytes==d31b_allocated_bytes(b)+idxbytes+sizeof(void*)*3u || bytes>idxbytes+cap*8192u);
            REQUIRE(d31c_occupied(c)==0);
            const unsigned rounds=ci==3?88:130;
            track_heap=1;
            for(unsigned t=0;t<rounds;t++){
                double xy[2];float k[64],v[64],q[64],a[2],be[2];
                address(t,xy);samples(t,k,v,q,a,be);
                int br=d31b_write(b,xy,k,v,a,be),cr=d31c_write(c,xy,k,v,a,be);
                REQUIRE(br==cr);
                REQUIRE(d31b_commits(b)==d31c_commits(c));
                REQUIRE(d31b_evictions(b)==d31c_evictions(c));
                REQUIRE(d31b_occupied(b)==d31c_occupied(c));
                REQUIRE(d31c_occupied(c)<=cap);
                REQUIRE(d31c_allocated_bytes(c)==bytes);
                REQUIRE(d31c_index_lookups(c)==9*d31c_commits(c));
                REQUIRE(compare_outputs(b,c,xy,q)==0);
                double other[2];address((t+1u)*3u,other);
                REQUIRE(compare_outputs(b,c,other,q)==0);
            }
            track_heap=0;
            REQUIRE(heap_operations==0);
            REQUIRE(d31c_cell_updates(c)==9*d31c_commits(c));
            d31b_destroy(b);d31c_destroy(c);
        }
    }
    d31c_arena *ref=d31c_create(9,D31C_REFUSE,96);
    d31c_arena *reuse=d31c_create(9,D31C_LRU_REUSE,96);
    REQUIRE(ref&&reuse);
    double first[2]={0.1,0.2},other[2]={60.1,60.2},oor[2]={97.0,0.1};
    float k[64],v[64],q[64],a[2],beta[2],before[64],out[64],sentinel[64];
    samples(0,k,v,q,a,beta);
    for(size_t j=0;j<64;j++)sentinel[j]=(float)(j+1000u);
    REQUIRE(d31c_write(ref,first,k,v,a,beta)==D31C_OK);
    REQUIRE(d31c_read(ref,first,q,before)==D31C_OK);
    REQUIRE(d31c_write(ref,other,k,v,a,beta)==D31C_ERR_CAPACITY);
    REQUIRE(d31c_read(ref,first,q,out)==D31C_OK);
    REQUIRE(memcmp(before,out,sizeof(out))==0);
    REQUIRE(d31c_evictions(ref)==0&&d31c_commits(ref)==1);
    REQUIRE(d31c_write(reuse,first,k,v,a,beta)==D31C_OK);
    REQUIRE(d31c_write(reuse,other,k,v,a,beta)==D31C_OK);
    REQUIRE(d31c_evictions(reuse)==9&&d31c_occupied(reuse)==9);
    REQUIRE(d31c_read(reuse,first,q,out)==D31C_OK);
    for(size_t j=0;j<64;j++)REQUIRE(out[j]==0.0f);
    memcpy(out,sentinel,sizeof(out));
    REQUIRE(d31c_read(ref,oor,q,out)==D31C_ERR_RANGE);
    REQUIRE(memcmp(out,sentinel,sizeof(out))==0);
    REQUIRE(d31c_write(ref,oor,k,v,a,beta)==D31C_ERR_RANGE);
    float nanv[64];memcpy(nanv,v,sizeof(nanv));nanv[6]=NAN;
    REQUIRE(d31c_write(ref,first,k,nanv,a,beta)==D31C_ERR_NUMERICAL);
    REQUIRE(d31c_read(ref,first,q,out)==D31C_OK);
    REQUIRE(memcmp(out,before,sizeof(out))==0);
    float huge[64];for(size_t j=0;j<64;j++)huge[j]=FLT_MAX;
    REQUIRE(d31c_write(ref,first,huge,huge,a,beta)==D31C_ERR_NUMERICAL);
    REQUIRE(d31c_read(ref,first,q,out)==D31C_OK);
    REQUIRE(memcmp(out,before,sizeof(out))==0);
    REQUIRE(d31c_commits(ref)==1&&d31c_evictions(ref)==0&&d31c_occupied(ref)==9);
    d31c_destroy(ref);d31c_destroy(reuse);
    printf("PASS_D31C_INDEX_MECHANICS checks=%llu heap_after_create=%llu lookup=9 capacity_limit=8192\n",
           (unsigned long long)checks,(unsigned long long)heap_operations);
    return 0;
}
