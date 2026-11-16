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

/* R7: vectorized deterministic features, *not* an admission authority.
 * Caller supplies only records already validated by the host bank. Reject
 * malformed/nonfinite inputs before committing ANY output rows.
 * Feature order: qcompat, utility, freshness, credibility, contradiction,
 * goalrel, protected, geomcompat. R5 authority & revisions stay host-owned.
 */
static double cos_double(const double *a,const double *b,int64_t d){
    double aa=0.0,bb=0.0,ab=0.0;
    for(int64_t j=0;j<d;j++) {aa+=a[j]*a[j];bb+=b[j]*b[j];ab+=a[j]*b[j];}
    double na=sqrt(aa),nb=sqrt(bb);
    return (na<1.0e-14 || nb<1.0e-14)?0.0:ab/(na*nb);
}
int ersa_features_v2(int64_t n,int64_t dim,const double *keys,
    const double *goals,const double *vectors,const double *meta,
    const int64_t *issued,int64_t tick,double halflife,
    const double *qkey,const double *qgoal,const double *qdir,
    int32_t has_goal,double *out){
    if(n<0 || n>ERSA_MAX || dim<1 || dim>32 || !keys || !goals || !vectors ||
       !meta || !issued || !qkey || !qdir || !out ||
       (has_goal!=0 && has_goal!=1) || (has_goal && !qgoal) ||
       !isfinite(halflife) || halflife<=0.0 || tick<0) return -1;
    for(int64_t j=0;j<dim;j++){
        if(!isfinite(qkey[j]) || (has_goal && !isfinite(qgoal[j])))return -2;
    }
    for(int j=0;j<4;j++)if(!isfinite(qdir[j]))return -2;
    for(int64_t i=0;i<n;i++){
        if(issued[i]<0 || issued[i]>tick)return -3;
        for(int64_t j=0;j<dim;j++){
            if(!isfinite(keys[i*dim+j]) || !isfinite(goals[i*dim+j]))return -2;
        }
        for(int j=0;j<4;j++){
            if(!isfinite(vectors[i*4+j]) || !isfinite(meta[i*4+j]))return -2;
        }
        if((meta[i*4+2]!=0.0 && meta[i*4+2]!=-1.0) ||
           (meta[i*4+3]!=0.0 && meta[i*4+3]!=1.0))return -3;
    }
    size_t bytes=(size_t)(n?n:1)*8*sizeof(double);
    double *temporary=(double*)malloc(bytes);
    if(!temporary)return -4;
    for(int64_t i=0;i<n;i++){
        const double *key=keys+i*dim,*goal=goals+i*dim,*v=vectors+i*4;
        const double *m=meta+i*4;
        double age=(double)(tick-issued[i]);
        double x[8]={cos_double(key,qkey,dim),m[0],exp(-age/halflife),
                     m[1],m[2],has_goal?cos_double(goal,qgoal,dim):0.0,
                     m[3],cos_double(v,qdir,4)};
        for(int j=0;j<8;j++)if(!isfinite(x[j])){
            free(temporary);return -2;
        }
        memcpy(temporary+i*8,x,sizeof(x));
    }
    memcpy(out,temporary,(size_t)n*8*sizeof(double));
    free(temporary);
    return 0;
}
