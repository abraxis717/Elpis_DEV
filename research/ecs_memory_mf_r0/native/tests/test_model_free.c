#include "model_free_memory.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    const float alpha[2] = {1.0f,1.0f};
    const float beta[2] = {0.7f,0.7f};
    ecs_mf_memory *a = ecs_mf_create(9,D31C_REFUSE,64,alpha,beta);
    ecs_mf_memory *b = ecs_mf_create(9,D31C_REFUSE,64,alpha,beta);
    assert(a && b);
    assert(ecs_mf_allocated_bytes(a) == ecs_mf_allocated_bytes(b));
    assert(ecs_mf_occupied_cells(a) == 0);
    double xy1[2],xy2[2];
    ecs_mf_address(5,3,64,xy1); ecs_mf_address(5,3,64,xy2);
    assert(memcmp(xy1,xy2,sizeof(xy1)) == 0);
    float k1[64],k2[64],value[64],r1[64],r2[64];
    ecs_mf_features(123,3,k1); ecs_mf_features(123,3,k2);
    assert(memcmp(k1,k2,sizeof(k1)) == 0);
    for (size_t i = 0; i < 64; i++) value[i] = (float)i/100.0f;
    assert(ecs_mf_write(a,5,123,3,value) == D31C_OK);
    assert(ecs_mf_write(b,5,123,3,value) == D31C_OK);
    assert(ecs_mf_occupied_cells(a) == 9);
    assert(ecs_mf_read(a,5,123,3,r1) == D31C_OK);
    assert(ecs_mf_read(b,5,123,3,r2) == D31C_OK);
    assert(memcmp(r1,r2,sizeof(r1)) == 0);
    assert(ecs_mf_commits(a) == 1);
    assert(ecs_mf_evictions(a) == 0);
    /* One nine-cell capacity cannot admit an unrelated distant footprint. */
    uint64_t other=7;
    double diff[2];
    while (other < 100000) {
        ecs_mf_address(other,3,64,diff);
        if (fabs(diff[0]-xy1[0]) > 3.5 && fabs(diff[1]-xy1[1]) > 3.5) break;
        other++;
    }
    assert(other < 100000);
    assert(ecs_mf_write(a,other,123,3,value) == D31C_ERR_CAPACITY);
    assert(ecs_mf_commits(a) == 1);
    assert(ecs_mf_read(a,5,123,3,r2) == D31C_OK);
    assert(memcmp(r1,r2,sizeof(r1)) == 0);
    ecs_mf_destroy(a);ecs_mf_destroy(b);
    assert(ecs_mf_create(9,D31C_REFUSE,64,NULL,beta) == NULL);
    printf("PASS_ECS_MODEL_FREE_ADAPTER_SMOKE\n");
    return 0;
}
