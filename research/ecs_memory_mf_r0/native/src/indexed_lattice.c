/* D31-C independent fixed direct index variant of pinned D31-B math. No file I/O. */
#include "indexed_lattice.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

typedef struct {
    int64_t x, y;
    uint64_t stamp;
    uint32_t used;
    float w[D31C_TENSOR_FLOATS];
} d31c_cell;
struct d31c_arena {
    size_t capacity, occupied;
    d31c_policy policy;
    uint64_t clock, evictions, commits, cell_updates;
    d31c_cell *cells;
    uint32_t *direct; /* slot+1, fixed grid, 0 means absent */
    size_t side;
    int32_t radius;
    uint64_t index_lookups, admission_scan_steps;
};

static size_t element(size_t h,size_t v,size_t k) {
    return (h*D31C_DV+v)*D31C_DK+k;
}
static d31c_status neighborhood(const double xy[2], int64_t zx[9],
                                 int64_t zy[9],double phi[9]) {
    if (!xy || !isfinite(xy[0]) || !isfinite(xy[1]) ||
        fabs(xy[0])>1024.0 || fabs(xy[1])>1024.0) return D31C_ERR_ARGUMENT;
    double w[2][3]; int64_t coord[2][3];
    for (size_t axis=0;axis<2;axis++) {
        double lo=ceil(xy[axis]-1.5), sum=0.0;
        for(size_t j=0;j<3;j++) {
            coord[axis][j]=(int64_t)lo+(int64_t)j;
            double t=xy[axis]-(double)coord[axis][j];
            double b=fabs(t)<1.5 ? pow(cos(3.1415926535897932384626433832795*t/3.0),2.0) : 0.0;
            w[axis][j]=b; sum+=b*b;
        }
        if (!(sum>0) || !isfinite(sum)) return D31C_ERR_NUMERICAL;
        for(size_t j=0;j<3;j++)w[axis][j]/=sqrt(sum);
    }
    for(size_t i=0;i<3;i++)for(size_t j=0;j<3;j++){
        size_t p=3*i+j;zx[p]=coord[0][i];zy[p]=coord[1][j];phi[p]=w[0][i]*w[1][j];
    }
    return D31C_OK;
}
d31c_arena *d31c_create(size_t capacity,d31c_policy policy,int radius){
    if (radius < 2 || radius > 256) return NULL;
    if(capacity<9||capacity>D31C_CAP_MAX||(policy!=D31C_REFUSE&&policy!=D31C_LRU_REUSE))return NULL;
    d31c_arena *a=calloc(1,sizeof(*a));if(!a)return NULL;
    a->cells=calloc(capacity,sizeof(d31c_cell));
    if(!a->cells){free(a);return NULL;}
    a->side = (size_t)(2*radius+3);
    a->radius = radius;
    a->direct = calloc(a->side*a->side,sizeof(uint32_t));
    if(!a->direct){free(a->cells);free(a);return NULL;}
    a->capacity=capacity;a->policy=policy;return a;
}
void d31c_destroy(d31c_arena*a){if(a){free(a->direct);free(a->cells);free(a);}}
size_t d31c_capacity(const d31c_arena*a){return a?a->capacity:0;}
size_t d31c_occupied(const d31c_arena*a){return a?a->occupied:0;}
size_t d31c_allocated_bytes(const d31c_arena*a){return a?sizeof(*a)+a->capacity*sizeof(d31c_cell)+a->side*a->side*sizeof(uint32_t):0;}
size_t d31c_cell_bytes(void){return sizeof(d31c_cell);}
uint64_t d31c_evictions(const d31c_arena*a){return a?a->evictions:0;}
uint64_t d31c_commits(const d31c_arena*a){return a?a->commits:0;}
uint64_t d31c_cell_updates(const d31c_arena*a){return a?a->cell_updates:0;}
size_t d31c_index_bytes(const d31c_arena*a){return a?a->side*a->side*sizeof(uint32_t):0;}
uint64_t d31c_index_lookups(const d31c_arena*a){return a?a->index_lookups:0;}
uint64_t d31c_admission_scan_steps(const d31c_arena*a){return a?a->admission_scan_steps:0;}
static size_t map_pos(const d31c_arena *a,int64_t x,int64_t y){
    int64_t i=x+(int64_t)a->radius+1,j=y+(int64_t)a->radius+1;
    if(i<0 || j<0 || i>=(int64_t)a->side || j>=(int64_t)a->side)return SIZE_MAX;
    return (size_t)i*a->side+(size_t)j;
}
static int admitted_address(const d31c_arena *a,const double xy[2]){
    return xy && isfinite(xy[0]) && isfinite(xy[1]) &&
           fabs(xy[0]) <= (double)a->radius && fabs(xy[1]) <= (double)a->radius;
}
static int finite_vector(const float *v,size_t n){
    if(!v)return 0;
    for(size_t i=0;i<n;i++)if(!isfinite(v[i]))return 0;
    return 1;
}
/* Plan mapping WITHOUT modifying arena, reserving distinct victims. */
static d31c_status plan(const d31c_arena*a,const int64_t x[9],const int64_t y[9],
                        size_t idx[9],size_t *newcells,size_t *evict,uint64_t *steps){
    size_t nc=0, ev=0; uint64_t scanned=0;
    for(size_t j=0;j<9;j++){
        size_t at=map_pos(a,x[j],y[j]);
        if(at==SIZE_MAX)return D31C_ERR_RANGE;
        idx[j]=a->direct[at]?a->direct[at]-1u:SIZE_MAX;
    }
    for(size_t j=0;j<9;j++){
        if(idx[j]!=SIZE_MAX)continue;
        size_t selected=SIZE_MAX;
        for(size_t i=0;i<a->capacity;i++){
            ++scanned;
            int already=0;for(size_t k=0;k<9;k++)if(idx[k]==i){already=1;break;}
            if(!already&&!a->cells[i].used){selected=i;break;}
        }
        if(selected==SIZE_MAX){
            if(a->policy==D31C_REFUSE)return D31C_ERR_CAPACITY;
            uint64_t least=UINT64_MAX;
            for(size_t i=0;i<a->capacity;i++){
                ++scanned;
                int already=0;for(size_t k=0;k<9;k++)if(idx[k]==i){already=1;break;}
                if(!already&&a->cells[i].stamp<least){selected=i;least=a->cells[i].stamp;}
            }
            if(selected==SIZE_MAX)return D31C_ERR_CAPACITY;
            ++ev;
        }else ++nc;
        idx[j]=selected;
    }
    *newcells=nc;*evict=ev;*steps=scanned;return D31C_OK;
}
d31c_status d31c_write(d31c_arena*a,const double xy[2],const float key[64],const float val[64],const float alpha[2],const float beta[2]){
    if(!a||!key||!val||!alpha||!beta)return D31C_ERR_ARGUMENT;
    if(!finite_vector(key,64)||!finite_vector(val,64)||!finite_vector(alpha,2)||!finite_vector(beta,2))return D31C_ERR_NUMERICAL;
    for(size_t h=0;h<2;h++)if(alpha[h]<0||alpha[h]>1||beta[h]<0||beta[h]>1)return D31C_ERR_ARGUMENT;
    if(a->clock==UINT64_MAX||a->commits==UINT64_MAX||a->cell_updates>UINT64_MAX-9||a->evictions>UINT64_MAX-9||a->index_lookups>UINT64_MAX-9)return D31C_ERR_CLOCK;
    if(!admitted_address(a,xy))return D31C_ERR_RANGE;
    int64_t xs[9],ys[9];double ph[9];
    d31c_status rc=neighborhood(xy,xs,ys,ph);if(rc)return rc;
    size_t idx[9],newcells=0,evict=0; uint64_t scan_steps=0;
    rc=plan(a,xs,ys,idx,&newcells,&evict,&scan_steps);if(rc)return rc;
    if(a->admission_scan_steps>UINT64_MAX-scan_steps)return D31C_ERR_CLOCK;
    /* Fixed, bounded on-stack staging: 9 * 8192 bytes, no heap allocations. */
    float staged[9][D31C_TENSOR_FLOATS];
    for(size_t j=0;j<9;j++){
        const d31c_cell *c=&a->cells[idx[j]];
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
                    if(!isfinite(next)||fabs(next)>3.402823466e38)return D31C_ERR_NUMERICAL;
                    staged[j][at]=(float)next;
                }
            }
        }
    }
    /* One commit point. Inverse address map is updated only after all math succeeded. */
    for(size_t j=0;j<9;j++){
        d31c_cell *c=&a->cells[idx[j]];
        if(c->used){size_t old=map_pos(a,c->x,c->y);if(old!=SIZE_MAX)a->direct[old]=0;}
    }
    for(size_t j=0;j<9;j++){
        d31c_cell *c=&a->cells[idx[j]];
        c->x=xs[j];c->y=ys[j];c->used=1;c->stamp=a->clock+1;
        a->direct[map_pos(a,xs[j],ys[j])]=(uint32_t)(idx[j]+1u);
        memcpy(c->w,staged[j],sizeof(staged[j]));
    }
    a->occupied+=newcells;a->evictions+=evict;a->clock++;
    a->commits++;a->cell_updates+=9;
    a->index_lookups+=9;a->admission_scan_steps+=scan_steps;
    return D31C_OK;
}
d31c_status d31c_read(const d31c_arena*a,const double xy[2],const float query[64],float output[64]){
    if(!a||!output||!query)return D31C_ERR_ARGUMENT;
    if(!finite_vector(query,64))return D31C_ERR_NUMERICAL;
    if(!admitted_address(a,xy))return D31C_ERR_RANGE;
    int64_t xs[9],ys[9];double ph[9];
    d31c_status rc=neighborhood(xy,xs,ys,ph);if(rc)return rc;
    size_t direct_slots[9];
    for(size_t j=0;j<9;j++){
        size_t pos=map_pos(a,xs[j],ys[j]);
        if(pos==SIZE_MAX)return D31C_ERR_RANGE;
        direct_slots[j]=a->direct[pos] ? (size_t)a->direct[pos]-1u : SIZE_MAX;
    }
    float tmp[64];
    for(size_t h=0;h<2;h++)for(size_t v=0;v<32;v++){
        double total=0;
        for(size_t j=0;j<9;j++){
            if(direct_slots[j]==SIZE_MAX)continue;
            const d31c_cell*c=&a->cells[direct_slots[j]];
            double one=0;
            for(size_t k=0;k<32;k++)one+=(double)c->w[element(h,v,k)]*(double)query[h*32+k];
            total+=ph[j]*one;
        }
        if(!isfinite(total)||fabs(total)>3.402823466e38)return D31C_ERR_NUMERICAL;
        tmp[h*32+v]=(float)total;
    }
    memcpy(output,tmp,sizeof(tmp));return D31C_OK;
}
