/* ERSA R10: bounded volatile CAS-owned state image.
 * This component owns session state bytes; it does NOT execute SSC semantics.
 * No persistent I/O. All read/replace operations have bounded capacity.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <pthread.h>

typedef struct {
    uint64_t generation;
    size_t length;
    size_t capacity;
    unsigned char *bytes;
    pthread_mutex_t mutex;
} ErsaAuthority;

ErsaAuthority *ersa_authority_new(size_t cap) {
    if (cap < 128 || cap > (size_t) 32*1024*1024) return NULL;
    ErsaAuthority *a = calloc(1, sizeof(*a));
    if (!a) return NULL;
    a->bytes = calloc(cap, 1);
    if (!a->bytes) { free(a); return NULL; }
    a->capacity=cap;
    if (pthread_mutex_init(&a->mutex,NULL)!=0) { free(a->bytes);free(a);return NULL; }
    return a;
}
void ersa_authority_free(ErsaAuthority *a) {
    if (!a) return;
    if (a->bytes) {
        volatile unsigned char *p=(volatile unsigned char *)a->bytes;
        for (size_t i=0;i<a->capacity;i++) p[i]=0;
        free(a->bytes);
    }
    pthread_mutex_destroy(&a->mutex);
    memset(a,0,sizeof(*a));
    free(a);
}
uint64_t ersa_authority_generation(ErsaAuthority *a) {
    if (!a) return 0;
    pthread_mutex_lock(&a->mutex);uint64_t v=a->generation;pthread_mutex_unlock(&a->mutex);
    return v;
}
size_t ersa_authority_length(ErsaAuthority *a) {
    if (!a) return 0;
    pthread_mutex_lock(&a->mutex);size_t v=a->length;pthread_mutex_unlock(&a->mutex);
    return v;
}
size_t ersa_authority_capacity(ErsaAuthority *a) { return a?a->capacity:0; }
/* 0 OK, 1 STALE, 2 SIZE, 3 EMPTY/NULL, 4 COUNTER_OVERFLOW */
int ersa_authority_replace(ErsaAuthority *a, uint64_t expected,
                           const unsigned char *data, size_t length) {
    if (!a || !data || !length) return 3;
    pthread_mutex_lock(&a->mutex);
    if (expected!=a->generation) {pthread_mutex_unlock(&a->mutex);return 1;}
    if (length>a->capacity) {pthread_mutex_unlock(&a->mutex);return 2;}
    if (a->generation==UINT64_MAX) {pthread_mutex_unlock(&a->mutex);return 4;}
    memmove(a->bytes,data,length);
    if (length<a->length) memset(a->bytes+length,0,a->length-length);
    a->length=length;
    a->generation++;
    pthread_mutex_unlock(&a->mutex);
    return 0;
}
/* 0 OK, 1 STALE, 2 SHORT BUFFER, 3 INVALID */
int ersa_authority_copy(ErsaAuthority *a, uint64_t expected,
                        unsigned char *out, size_t max_len) {
    if (!a || !out) return 3;
    pthread_mutex_lock(&a->mutex);
    if (expected!=a->generation) {pthread_mutex_unlock(&a->mutex);return 1;}
    if (max_len<a->length) {pthread_mutex_unlock(&a->mutex);return 2;}
    if (a->length) memcpy(out,a->bytes,a->length);
    pthread_mutex_unlock(&a->mutex);
    return 0;
}
