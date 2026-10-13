/* D31-B research-only bounded donor math. No file I/O, no inference authority. */
#include "bounded_lattice.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

typedef struct {
    int64_t x, y;
    uint64_t stamp;
    uint32_t used;
    float w[D31B_TENSOR_FLOATS];
} d31b_cell;
struct d31b_arena {
    size_t capacity, occupied;
    d31b_policy policy;
    uint64_t clock, evictions, commits, cell_updates;
    d31b_cell *cells;
};

static size_t element(size_t h,size_t v,size_t k) {
    return (h*D31B_DV+v)*D31B_DK+k;
}
static d31b_status neighborhood(const double xy[2], int64_t zx[9],
                                 int64_t zy[9],double phi[9]) {
    if (!xy || !isfinite(xy[0]) || !isfinite(xy[1]) ||
        fabs(xy[0])>1024.0 || fabs(xy[1])>1024.0) return D31B_ERR_ARGUMENT;
    double w[2][3]; int64_t coord[2][3];
    for (size_t axis=0;axis<2;axis++) {
        double lo=ceil(xy[axis]-1.5), sum=0.0;
        for(size_t j=0;j<3;j++) {
            coord[axis][j]=(int64_t)lo+(int64_t)j;
            double t=xy[axis]-(double)coord[axis][j];
            double b=fabs(t)<1.5 ? pow(cos(3.1415926535897932384626433832795*t/3.0),2.0) : 0.0;
            w[axis][j]=b; sum+=b*b;
        }
        if (!(sum>0) || !isfinite(sum)) return D31B_ERR_NUMERICAL;
        for(size_t j=0;j<3;j++)w[axis][j]/=sqrt(sum);
    }
    for(size_t i=0;i<3;i++)for(size_t j=0;j<3;j++){
        size_t p=3*i+j;zx[p]=coord[0][i];zy[p]=coord[1][j];phi[p]=w[0][i]*w[1][j];
    }
    return D31B_OK;
}
d31b_arena *d31b_create(size_t capacity,d31b_policy policy){
    if(capacity<9||capacity>D31B_CAP_MAX||(policy!=D31B_REFUSE&&policy!=D31B_LRU_REUSE))return NULL;
    d31b_arena *a=calloc(1,sizeof(*a));if(!a)return NULL;
    a->cells=calloc(capacity,sizeof(d31b_cell));
    if(!a->cells){free(a);return NULL;}
    a->capacity=capacity;a->policy=policy;return a;
}
void d31b_destroy(d31b_arena*a){if(a){free(a->cells);free(a);}}
size_t d31b_capacity(const d31b_arena*a){return a?a->capacity:0;}
size_t d31b_occupied(const d31b_arena*a){return a?a->occupied:0;}
size_t d31b_allocated_bytes(const d31b_arena*a){return a?sizeof(*a)+a->capacity*sizeof(d31b_cell):0;}
size_t d31b_cell_bytes(void){return sizeof(d31b_cell);}
uint64_t d31b_evictions(const d31b_arena*a){return a?a->evictions:0;}
uint64_t d31b_commits(const d31b_arena*a){return a?a->commits:0;}
uint64_t d31b_cell_updates(const d31b_arena*a){return a?a->cell_updates:0;}
static int finite_vector(const float *v,size_t n){
    if(!v)return 0;
    for(size_t i=0;i<n;i++)if(!isfinite(v[i]))return 0;
    return 1;
}
/* Plan mapping WITHOUT modifying arena, reserving distinct victims. */
static d31b_status plan(const d31b_arena*a,const int64_t x[9],const int64_t y[9],size_t idx[9],size_t *newcells,size_t *evict){
    unsigned char reserved[9]={0};(void)reserved;
    size_t nc=0, ev=0;
    for(size_t j=0;j<9;j++)idx[j]=SIZE_MAX;
    for(size_t j=0;j<9;j++){
        for(size_t i=0;i<a->capacity;i++){
            if(a->cells[i].used&&a->cells[i].x==x[j]&&a->cells[i].y==y[j]){idx[j]=i;break;}
        }
    }
    for(size_t j=0;j<9;j++){
        if(idx[j]!=SIZE_MAX)continue;
        size_t selected=SIZE_MAX;
        for(size_t i=0;i<a->capacity;i++){
            int already=0;for(size_t k=0;k<9;k++)if(idx[k]==i){already=1;break;}
            if(!already&&!a->cells[i].used){selected=i;break;}
        }
        if(selected==SIZE_MAX){
            if(a->policy==D31B_REFUSE)return D31B_ERR_CAPACITY;
            uint64_t least=UINT64_MAX;
            for(size_t i=0;i<a->capacity;i++){
                int already=0;for(size_t k=0;k<9;k++)if(idx[k]==i){already=1;break;}
                if(!already&&a->cells[i].stamp<least){selected=i;least=a->cells[i].stamp;}
            }
            if(selected==SIZE_MAX)return D31B_ERR_CAPACITY;
            ++ev;
        }else ++nc;
        idx[j]=selected;
    }
    *newcells=nc;*evict=ev;return D31B_OK;
}
d31b_status d31b_write(d31b_arena*a,const double xy[2],const float key[64],const float val[64],const float alpha[2],const float beta[2]){
    if(!a||!key||!val||!alpha||!beta)return D31B_ERR_ARGUMENT;
    if(!finite_vector(key,64)||!finite_vector(val,64)||!finite_vector(alpha,2)||!finite_vector(beta,2))return D31B_ERR_NUMERICAL;
    for(size_t h=0;h<2;h++)if(alpha[h]<0||alpha[h]>1||beta[h]<0||beta[h]>1)return D31B_ERR_ARGUMENT;
    if(a->clock==UINT64_MAX||a->commits==UINT64_MAX||a->cell_updates>UINT64_MAX-9||a->evictions>UINT64_MAX-9)return D31B_ERR_CLOCK;
    int64_t xs[9],ys[9];double ph[9];
    d31b_status rc=neighborhood(xy,xs,ys,ph);if(rc)return rc;
    size_t idx[9],newcells=0,evict=0;
    rc=plan(a,xs,ys,idx,&newcells,&evict);if(rc)return rc;
    /* Fixed, bounded on-stack staging: 9 * 8192 bytes, no heap allocations. */
    float staged[9][D31B_TENSOR_FLOATS];
    for(size_t j=0;j<9;j++){
        const d31b_cell *c=&a->cells[idx[j]];
        int same=c->used&&c->x==xs[j]&&c->y==ys[j];
        for(size_t h=0;h<2;h++){
            double bar[32]={0};
            if(same)for(size_t v=0;v<32;v++)for(size_t k=0;k<32;k++)
                bar[v]+=(double)c->w[element(h,v,k)]*(double)key[h*32+k];
            for(size_t v=0;v<32;v++){
                double residual=(double)val[h*32+v]-(double)alpha[h]*bar[v];
                for(size_t k=0;k<32;k++){
                    size_t at=element(h,v,k);
                    double old=same?(double)c->w[at]:0.0;
                    double next=(double)alpha[h]*old+(double)beta[h]*ph[j]*residual*(double)key[h*32+k];
                    if(!isfinite(next)||fabs(next)>3.402823466e38)return D31B_ERR_NUMERICAL;
                    staged[j][at]=(float)next;
                }
            }
        }
    }
    /* One commit point: every earlier refusal left the arena bit-identical. */
    for(size_t j=0;j<9;j++){
        d31b_cell *c=&a->cells[idx[j]];
        c->x=xs[j];c->y=ys[j];c->used=1;c->stamp=a->clock+1;
        memcpy(c->w,staged[j],sizeof(staged[j]));
    }
    a->occupied+=newcells;a->evictions+=evict;a->clock++;
    a->commits++;a->cell_updates+=9;
    return D31B_OK;
}
d31b_status d31b_read(const d31b_arena*a,const double xy[2],const float query[64],float output[64]){
    if(!a||!output||!query)return D31B_ERR_ARGUMENT;
    if(!finite_vector(query,64))return D31B_ERR_NUMERICAL;
    int64_t xs[9],ys[9];double ph[9];
    d31b_status rc=neighborhood(xy,xs,ys,ph);if(rc)return rc;
    float tmp[64];
    for(size_t h=0;h<2;h++)for(size_t v=0;v<32;v++){
        double total=0;
        for(size_t j=0;j<9;j++)for(size_t i=0;i<a->capacity;i++){
            const d31b_cell*c=&a->cells[i];
            if(!c->used||c->x!=xs[j]||c->y!=ys[j])continue;
            double one=0;
            for(size_t k=0;k<32;k++)one+=(double)c->w[element(h,v,k)]*(double)query[h*32+k];
            total+=ph[j]*one;break;
        }
        if(!isfinite(total)||fabs(total)>3.402823466e38)return D31B_ERR_NUMERICAL;
        tmp[h*32+v]=(float)total;
    }
    memcpy(output,tmp,sizeof(tmp));return D31B_OK;
}
