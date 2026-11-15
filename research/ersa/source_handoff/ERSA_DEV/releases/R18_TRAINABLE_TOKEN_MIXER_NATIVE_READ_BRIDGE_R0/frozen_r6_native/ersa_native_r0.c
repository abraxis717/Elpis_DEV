/* ERSA R6 — fixed-capacity, stateless native CPU rank and signed reduction.
 * Original research implementation; no GDN / proprietary kernels.
 * Input records are fully validated/admitted by the upstream ERSA R5 bank.
 * This module grants NO admission rights and performs NO memory writes.
 * API errors are negative; output buffers are never committed on error.
 */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <limits.h>

#define ERSA_MAX 8192
#define ERSA_FEATURES 8
#define ERSA_VEC 4

static const double weights[8]={1.0,.5,.35,.4,-.6,.45,.3,.55};

int ersa_score_batch_v1(int64_t n, const double *features, double *scores) {
    if(n<0 || n>ERSA_MAX || !features || !scores) return -1;
    for(int64_t i=0;i<n;i++) {
        long double sum=0.0L;
        for(int j=0;j<8;j++) {
            double f=features[i*8+j];
            if(!isfinite(f)) return -2;
            sum += (long double)(weights[j]*f);
        }
        double out=(double)sum;
        if(!isfinite(out)) return -2;
    }
    for(int64_t i=0;i<n;i++){
        long double sum=0.0L;
        for(int j=0;j<8;j++) sum += (long double)(weights[j]*features[i*8+j]);
        scores[i]=(double)sum;
    }
    return 0;
}

typedef struct {double score; int64_t rid; int32_t idx;} Candidate;
static int cmp_rank(const void* lhs,const void* rhs){
    const Candidate *a=(const Candidate*)lhs,*b=(const Candidate*)rhs;
    if(a->score>b->score) return -1;
    if(a->score<b->score) return 1;
    if(a->rid<b->rid) return -1;
    if(a->rid>b->rid) return 1;
    return (a->idx>b->idx)-(a->idx<b->idx);
}
static int cmp_id(const void* lhs,const void* rhs){
    int64_t a=*(const int64_t*)lhs,b=*(const int64_t*)rhs;
    return (a>b)-(a<b);
}
int ersa_rank_canonical_v1(int64_t n,int64_t k,const double *canonical_scores,
                            const int64_t *rids,const uint8_t *eligible,
                            int32_t *out_indices,int64_t *out_count){
    if(n<0 || n>ERSA_MAX || k<1 || k>ERSA_MAX ||
       !canonical_scores || !rids || !eligible || !out_indices || !out_count) return -1;
    /* Two-pass validate / compute; never touch output until all inputs pass. */
    int64_t m=0;
    for(int64_t i=0;i<n;i++){
        if(eligible[i]>1) return -2;
        if(eligible[i]){
            if(!isfinite(canonical_scores[i])) return -2;
            m++;
        }
    }
    Candidate *c=(Candidate*)malloc((size_t)(m?m:1)*sizeof(Candidate));
    int64_t *ids=(int64_t*)malloc((size_t)(m?m:1)*sizeof(int64_t));
    if(!c || !ids){ free(c);free(ids);return -3; }
    int64_t p=0;
    for(int64_t i=0;i<n;i++) if(eligible[i]){
        c[p]=(Candidate){canonical_scores[i],rids[i],(int32_t)i};
        ids[p]=rids[i];p++;
    }
    qsort(ids,(size_t)m,sizeof(int64_t),cmp_id);
    for(int64_t i=1;i<m;i++) if(ids[i]==ids[i-1]){
        free(c);free(ids);return -4;
    }
    qsort(c,(size_t)m,sizeof(Candidate),cmp_rank);
    int64_t take=(k<m?k:m);
    for(int64_t i=0;i<take;i++) out_indices[i]=c[i].idx;
    *out_count=take;
    free(c);free(ids);
    return 0;
}

int ersa_signed_reduce_v1(int64_t n,const double *vectors,const double *weights_in,
                           double *out_salient,double *out_collective){
    if(n<0 || n>ERSA_MAX || !vectors || !weights_in ||
       !out_salient || !out_collective) return -1;
    long double z=0.0L,ws[ERSA_VEC]={0},us[ERSA_VEC]={0};
    for(int64_t i=0;i<n;i++){
        double w=weights_in[i];
        if(!isfinite(w)) return -2;
        z += fabsl((long double)w);
        for(int j=0;j<4;j++){
            double x=vectors[i*4+j];
            if(!isfinite(x))return -2;
            ws[j]+=(long double)w*x;
            us[j]+=(long double)x;
        }
    }
    /* Empty set: documented zero vector. Degenerate signed mass: mean. */
    for(int j=0;j<4;j++){
        long double mean=n?us[j]/(long double)n:0;
        long double val=z<1.0e-12L?mean:ws[j]/z;
        if(!isfinite((double)val)||!isfinite((double)mean))return -2;
    }
    for(int j=0;j<4;j++){
        long double mean=n?us[j]/(long double)n:0;
        out_salient[j]=(double)(z<1.0e-12L?mean:ws[j]/z);
        out_collective[j]=(double)mean;
    }
    return 0;
}
