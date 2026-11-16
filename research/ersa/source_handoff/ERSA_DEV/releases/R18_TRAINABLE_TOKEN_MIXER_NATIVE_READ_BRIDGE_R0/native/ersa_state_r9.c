/* ERSA R9: bounded volatile exact-address shadow; NOT SSC state authority.
 * Deterministic fixed allocation, no persistent storage or global mutable state.
 */
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <limits.h>

typedef struct { int64_t address; double v[4]; uint8_t used; } Slot;
typedef struct {
    int64_t lo, hi;
    uint32_t cap, count;
    uint64_t generation;
    double bound;
    Slot slots[];
} ErsaState;

void* ersa_state_create(int64_t cap,int64_t lo,int64_t hi,double bound) {
    if(cap<1 || cap>8192 || lo>hi || !isfinite(bound) || bound<=0) return NULL;
    if((uint64_t)cap > (SIZE_MAX-sizeof(ErsaState))/sizeof(Slot)) return NULL;
    ErsaState* s=(ErsaState*)calloc(1,sizeof(ErsaState)+(size_t)cap*sizeof(Slot));
    if (!s) return NULL;
    s->cap=(uint32_t)cap; s->lo=lo; s->hi=hi; s->bound=bound;
    return s;
}
void ersa_state_free(void* ptr) {free(ptr);}
uint64_t ersa_state_generation(void* ptr) {return ptr?((ErsaState*)ptr)->generation:0;}
int64_t ersa_state_count(void* ptr) {return ptr?(int64_t)((ErsaState*)ptr)->count:-1;}
int64_t ersa_state_capacity(void* ptr) {return ptr?(int64_t)((ErsaState*)ptr)->cap:-1;}
static int locate(ErsaState* s,int64_t address) {
    for(uint32_t i=0;i<s->cap;i++) if(s->slots[i].used && s->slots[i].address==address) return (int)i;
    return -1;
}
/* 0 success, 1 stale, 2 capacity, 3 invalid value, 4 invalid address, 5 epoch overflow */
int ersa_state_upsert(void* ptr,int64_t address,const double* value,uint64_t expected) {
    if(!ptr||!value)return 3;
    ErsaState* s=(ErsaState*)ptr;
    if(s->generation!=expected)return 1;
    if(address<s->lo||address>s->hi)return 4;
    for(int j=0;j<4;j++) if(!isfinite(value[j])||fabs(value[j])>s->bound)return 3;
    if(s->generation==UINT64_MAX)return 5;
    int ix=locate(s,address);
    if(ix<0){
        if(s->count==s->cap)return 2;
        for(uint32_t i=0;i<s->cap;i++)if(!s->slots[i].used){ix=(int)i;break;}
        if(ix<0)return 2;
        s->slots[ix].used=1; s->slots[ix].address=address; s->count++;
    }
    memcpy(s->slots[ix].v,value,4*sizeof(double));
    s->generation++;
    return 0;
}
/* 0 success, 1 stale, 4 invalid address, 5 epoch overflow, 6 not found */
int ersa_state_delete(void* ptr,int64_t address,uint64_t expected){
    if(!ptr)return 4;
    ErsaState* s=(ErsaState*)ptr;
    if(s->generation!=expected)return 1;
    if(address<s->lo||address>s->hi)return 4;
    if(s->generation==UINT64_MAX)return 5;
    int ix=locate(s,address);
    if(ix<0)return 6;
    memset(&s->slots[ix],0,sizeof(Slot));s->count--;s->generation++;return 0;
}
/* 0 found, 6 not found, 4 invalid address. Never updates read markers. */
int ersa_state_peek(void* ptr,int64_t address,double* out){
    if(!ptr||!out)return 4;
    ErsaState* s=(ErsaState*)ptr;
    if(address<s->lo||address>s->hi)return 4;
    int ix=locate(s,address);
    if(ix<0)return 6;
    memcpy(out,s->slots[ix].v,4*sizeof(double));return 0;
}
