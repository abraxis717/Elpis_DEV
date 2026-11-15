/* R36: CPU reference mirror of the GSA dual-matrix slot recurrence.
 * No persistent I/O, no global mutable state, no GPU kernels.
 * Independently implements the R35 donor's recurrence, NOT the R11 SSC bank.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <pthread.h>

typedef struct {
    uint32_t b,d,n;
    uint64_t generation;
    pthread_mutex_t lock; /* one writer per generation */
    float *hk; /* [B,D,N] */
    float *hv; /* [B,N,D] */
} ersa_slot36;

static int sane(const float *p, size_t n) {
    if (!p) return 0;
    for(size_t i=0;i<n;i++) if(!isfinite(p[i])) return 0;
    return 1;
}

void* ersa36_new(uint32_t b, uint32_t d, uint32_t n, uint64_t generation) {
    if(b<1 || b>128 || d<4 || d>32 || n<1 || n>64 || (uint64_t)b*d*n*2 > 2000000 || generation>INT_MAX)
        return NULL;
    ersa_slot36 *s=calloc(1,sizeof(*s));
    if(!s)return NULL;
    s->b=b;s->d=d;s->n=n;s->generation=generation;
    if(pthread_mutex_init(&s->lock,NULL)!=0){free(s);return NULL;}
    size_t elements=(size_t)b*d*n;
    s->hk=calloc(elements,sizeof(float));s->hv=calloc(elements,sizeof(float));
    if(!s->hk || !s->hv){free(s->hk);free(s->hv);pthread_mutex_destroy(&s->lock);free(s);return NULL;}
    return s;
}

void ersa36_free(void *p) {
    if(!p)return;
    ersa_slot36*s=p;
    size_t bytes=(size_t)s->b*s->d*s->n*sizeof(float);
    if(s->hk){memset(s->hk,0,bytes);free(s->hk);}
    if(s->hv){memset(s->hv,0,bytes);free(s->hv);}
    pthread_mutex_destroy(&s->lock);memset(s,0,sizeof(*s));free(s);
}

uint64_t ersa36_generation(const void *p){
    if(!p)return UINT64_MAX;
    ersa_slot36*s=(ersa_slot36*)p;
    pthread_mutex_lock(&s->lock);uint64_t gen=s->generation;pthread_mutex_unlock(&s->lock);
    return gen;
}

/* IMPORT: only allowed before execution, checked for complete finite image. */
int ersa36_import(void *p,const float*hk,const float*hv,uint64_t expected_generation) {
    if(!p)return -1;
    ersa_slot36*s=p;
    size_t count=(size_t)s->b*s->d*s->n;
    if(!sane(hk,count)||!sane(hv,count))return -2;
    pthread_mutex_lock(&s->lock);
    if(s->generation != expected_generation){pthread_mutex_unlock(&s->lock);return -2;}
    memcpy(s->hk,hk,count*sizeof(float));memcpy(s->hv,hv,count*sizeof(float));
    pthread_mutex_unlock(&s->lock);return 0;
}

int ersa36_export(const void *p,float*hk,float*hv,uint64_t expected_generation) {
    if(!p || !hk || !hv)return -1;
    const ersa_slot36*s=p;
    pthread_mutex_lock(&((ersa_slot36*)s)->lock);
    if(s->generation!=expected_generation){pthread_mutex_unlock(&((ersa_slot36*)s)->lock);return -2;}
    size_t count=(size_t)s->b*s->d*s->n;
    memcpy(hk,s->hk,count*sizeof(float));memcpy(hv,s->hv,count*sizeof(float));
    pthread_mutex_unlock(&((ersa_slot36*)s)->lock);return 0;
}

/* This is the reference recurrence for one token per batch:
   hk = hk*exp(g) + k*s
   p = softmax(scale * dot(q,hk), slot_axis)
   hv = hv*exp(g) + s*v
   y = sum_slot p*hv
   Both matrices commit together, only for finite inputs and finite results.
   p[slot] is included in y even for untouched slots, matching FLA naive.
   Caller must supply explicitly authorized s,g; no concealed slot routing.
 */
int ersa36_step(void *p,const float*q,const float*k,const float*v,
                const float*s,const float*g,float*y,uint64_t expected_generation) {
    if(!p || !q || !k || !v || !s || !g || !y)return -1;
    ersa_slot36*state=p;
    uint32_t b=state->b,d=state->d,n=state->n;
    if(!sane(q,(size_t)b*d)||!sane(k,(size_t)b*d)||!sane(v,(size_t)b*d)||
       !sane(s,(size_t)b*n)||!sane(g,(size_t)b*n))return -3;
    pthread_mutex_lock(&state->lock);
    if(state->generation!=expected_generation || expected_generation>=INT_MAX){pthread_mutex_unlock(&state->lock);return -2;}
    size_t total=(size_t)b*d*n;
    float *nextk=malloc(total*sizeof(float)),*nextv=malloc(total*sizeof(float));
    float *outs=calloc((size_t)b*d,sizeof(float));
    float *weights=malloc((size_t)b*n*sizeof(float));
    if(!nextk||!nextv||!outs||!weights){free(nextk);free(nextv);free(outs);free(weights);pthread_mutex_unlock(&state->lock);return -4;}
    int rc=0;
    double scale=1.0/sqrt((double)d);
    for(uint32_t bi=0;bi<b && !rc;bi++){
        double maxscore=-INFINITY;
        for(uint32_t j=0;j<n;j++){
            size_t ij=(size_t)bi*n+j;
            double decay=exp((double)g[ij]);
            if(!isfinite(decay)){rc=-5;break;}
            double score=0;
            for(uint32_t x=0;x<d;x++){
                size_t index=((size_t)bi*d+x)*n+j;
                double z=(double)state->hk[index]*decay+(double)k[(size_t)bi*d+x]*s[ij];
                nextk[index]=(float)z;
                if(!isfinite(nextk[index])){rc=-5;break;}
                score+=(double)q[(size_t)bi*d+x]*nextk[index];
            }
            if(rc)break;
            score*=scale;
            if(!isfinite(score)){rc=-5;break;}
            weights[ij]=(float)score;
            if(score>maxscore)maxscore=score;
        }
        if(rc)break;
        double sum=0;
        for(uint32_t j=0;j<n;j++){
            size_t ij=(size_t)bi*n+j;
            double e=exp((double)weights[ij]-maxscore);
            weights[ij]=(float)e;sum+=e;
        }
        if(!isfinite(sum)||sum<=0){rc=-5;break;}
        for(uint32_t j=0;j<n;j++)weights[(size_t)bi*n+j]=(float)(weights[(size_t)bi*n+j]/sum);
        for(uint32_t j=0;j<n;j++){
            size_t ij=(size_t)bi*n+j;
            double decay=exp((double)g[ij]);
            for(uint32_t x=0;x<d;x++){
                size_t index=((size_t)bi*n+j)*d+x;
                double z=(double)state->hv[index]*decay+(double)s[ij]*v[(size_t)bi*d+x];
                nextv[index]=(float)z;
                if(!isfinite(nextv[index])){rc=-5;break;}
                outs[(size_t)bi*d+x]+=(float)(weights[ij]*nextv[index]);
            }
            if(rc)break;
        }
        for(uint32_t x=0;x<d;x++)if(!isfinite(outs[(size_t)bi*d+x]))rc=-5;
    }
    if(!rc){memcpy(state->hk,nextk,total*sizeof(float));memcpy(state->hv,nextv,total*sizeof(float));
            memcpy(y,outs,(size_t)b*d*sizeof(float));state->generation++;}
    free(nextk);free(nextv);free(outs);free(weights);
    pthread_mutex_unlock(&state->lock);return rc;
}
