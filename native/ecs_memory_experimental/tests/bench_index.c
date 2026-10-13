#define _POSIX_C_SOURCE 200809L
#include "indexed_lattice.h"
#include "bounded_lattice.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void){struct timespec t;if(clock_gettime(CLOCK_MONOTONIC,&t)!=0)exit(2);return (double)t.tv_sec+1e-9*(double)t.tv_nsec;}
static void coords(size_t i,double p[2]){p[0]=-89.3+4.0*(double)(i%45u);p[1]=-89.7+4.0*(double)(i/45u);}
static void vec(float k[64],float v[64],float q[64]){
    for(unsigned i=0;i<64u;i++){
        k[i]=(float)(sin((double)(i+1u)*.15)/8.0);
        v[i]=(float)(cos((double)(i+1u)*.12)/2.0);
        q[i]=(float)(sin((double)(i+1u)*.15)/8.0);
    }
}
int main(void){
    size_t caps[]={9,72,288,1152};
    float k[64],v[64],q[64],a[2]={.999f,.970f},beta[2]={.73f,.51f},out[64];
    vec(k,v,q);
    double checksum=0;
    puts("capacity,provisioned_d31b_bytes,provisioned_d31c_bytes,index_bytes,reads,d31b_ns_per_read,d31c_ns_per_read,indexed_admission_scan_steps");
    for(size_t ci=0;ci<4;ci++){
        size_t cap=caps[ci],groups=cap/9u;
        d31b_arena *b=d31b_create(cap,D31B_LRU_REUSE);
        d31c_arena *c=d31c_create(cap,D31C_LRU_REUSE,96);
        if(!b||!c)return 2;
        for(size_t i=0;i<groups;i++){
            double xy[2];coords(i,xy);
            if(d31b_write(b,xy,k,v,a,beta)!=D31B_OK||d31c_write(c,xy,k,v,a,beta)!=D31C_OK)return 2;
        }
        const size_t n=256;
        double start=now();
        for(size_t i=0;i<n;i++){double xy[2];coords(i%groups,xy);if(d31b_read(b,xy,q,out)!=D31B_OK)return 2;checksum+=out[0];}
        double bend=now();
        for(size_t i=0;i<n;i++){double xy[2];coords(i%groups,xy);if(d31c_read(c,xy,q,out)!=D31C_OK)return 2;checksum+=out[0];}
        double cend=now();
        printf("%zu,%zu,%zu,%zu,%zu,%.1f,%.1f,%llu\n",cap,d31b_allocated_bytes(b),d31c_allocated_bytes(c),d31c_index_bytes(c),n,
               1e9*(bend-start)/(double)n,1e9*(cend-bend)/(double)n,
               (unsigned long long)d31c_admission_scan_steps(c));
        d31b_destroy(b);d31c_destroy(c);
    }
    if(!isfinite(checksum))return 2;
    fprintf(stderr,"BENCHMARK_CHECKSUM=%.8f\n",checksum);
    puts("PASS_D31C_INDEXED_CENSUS_DESCRIPTIVE_ONLY");return 0;
}
