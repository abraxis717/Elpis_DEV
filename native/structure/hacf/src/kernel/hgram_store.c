/* H-gram store: READ-ONLY (docs/ELPIS_MISSION.md, Amendment 3; elpis/hgram_store.h).
 *
 * A retained H-gram file is a read-only static resource: this file probes it and reads association cells from it,
 * and nothing in it creates, preallocates, initializes, resizes or writes a file. The initializer
 * (elpis_hgram_init_once, the elpis_hgram_init executable) and the in-place association writer
 * (elpis_hgram_assoc_store_preapproved) are retired by the volatile-native migration; their code is in git history
 * only. No descriptor here is ever opened for writing. */
#define _GNU_SOURCE 1
#include "elpis/hgram_store.h"
#include <errno.h>
#include <fcntl.h>
#include <sys/file.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define MIB 1048576ULL
#define HEADER ELPIS_HGRAM_HEADER_BYTES
#define SLOT ELPIS_HGRAM_SLOT_BYTES
static const unsigned char magic[8] = {'E','L','P','H','G','R','A','1'};

static uint32_t get32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
}
static uint64_t get64(const unsigned char *p) {
    return (uint64_t)get32(p) | (uint64_t)get32(p+4)<<32;
}
static uint64_t fnv(const unsigned char *p, size_t n) {
    uint64_t h=UINT64_C(14695981039346656037);
    for(size_t i=0;i<n;i++) { h ^= p[i]; h *= UINT64_C(1099511628211); }
    return h;
}
static int header_check(const unsigned char h[HEADER], uint64_t file_len,
                        elpis_hgram_info *out) {
    if (memcmp(h,magic,8) || get32(h+8)!=ELPIS_HGRAM_ABI ||
        get32(h+12)!=HEADER || get32(h+32)!=SLOT || get32(h+36)!=0 ||
        get64(h+56)!=0 || get64(h+48)!=fnv(h,48)) return -1;
    if(file_len < MIB || file_len % MIB || get64(h+16)!=file_len ||
       get64(h+24)!=file_len-HEADER ||
       get64(h+40)!=(file_len-HEADER)/SLOT) return -1;
    for(size_t i=64;i<HEADER;i++) if(h[i]) return -1;
    if(out) { out->total_bytes=file_len; out->slots=(file_len-HEADER)/SLOT;
              out->capacity_mib=file_len/MIB; }
    return 0;
}
/* Return a dirfd + one unambiguous leaf. The caller owns the dirfd. */
static int parse_path(const char *p,int *dirfd,char leaf[NAME_MAX+1]) {
    if(!p || *p!='/' || !*p) return -1;
    size_t n=strlen(p);
    if(n==0 || n>=PATH_MAX || p[n-1]=='/') return -1;
    const char *slash=strrchr(p,'/');
    if(!slash || !slash[1] || !strcmp(slash+1,".") || !strcmp(slash+1,"..") ||
       strlen(slash+1)>NAME_MAX) return -1;
    memcpy(leaf,slash+1,strlen(slash+1)+1);
    char parent[PATH_MAX]; size_t plen=(size_t)(slash-p);
    if(plen==0) { parent[0]='/'; parent[1]=0; }
    else { memcpy(parent,p,plen); parent[plen]=0; }
    int fd=open(parent,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
    if(fd<0) return -1;
    struct stat st;
    if(fstat(fd,&st)!=0 || !S_ISDIR(st.st_mode) || st.st_uid!=geteuid() ||
       (st.st_mode & (S_IWGRP|S_IWOTH))) { close(fd);return -1; }
    *dirfd=fd;return 0;
}
static int probe_fd(int fd,elpis_hgram_info *out) {
    struct stat st;
    if(fstat(fd,&st)!=0) return ELPIS_HGRAM_IO;
    /* Owned by this user and private to it; a read-only static resource need not be writable (mode 0400 or 0600). */
    if(!S_ISREG(st.st_mode) || st.st_nlink!=1 || st.st_uid!=geteuid() ||
       (uint64_t)st.st_blocks < ((uint64_t)st.st_size+511)/512 ||
       (st.st_mode & (S_IRWXG|S_IRWXO)) || !(st.st_mode & S_IRUSR) ||
       st.st_size < (off_t)MIB) return ELPIS_HGRAM_INVALID;
    unsigned char h[HEADER];
    size_t done=0;
    while(done<sizeof h) {
        ssize_t n=pread(fd,h+done,sizeof h-done,(off_t)done);
        if(n<=0) return ELPIS_HGRAM_IO;
        done+=(size_t)n;
    }
    return header_check(h,(uint64_t)st.st_size,out)==0 ? ELPIS_HGRAM_EXISTS : ELPIS_HGRAM_INVALID;
}
int elpis_hgram_probe(const char *path,elpis_hgram_info *out) {
    if(!out) return ELPIS_HGRAM_INVALID;
    int dfd=-1; char leaf[NAME_MAX+1];
    if(parse_path(path,&dfd,leaf)!=0) return ELPIS_HGRAM_PATH;
    int fd=openat(dfd,leaf,O_RDONLY|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK);
    int e=errno;close(dfd);
    if(fd<0) return e==ENOENT ? ELPIS_HGRAM_PATH : ELPIS_HGRAM_INVALID;
    elpis_hgram_info tmp;
    int rc=probe_fd(fd,&tmp);
    if(close(fd)!=0) rc=ELPIS_HGRAM_IO;
    if(rc==ELPIS_HGRAM_EXISTS) *out=tmp;
    return rc;
}
/* R1 association cells, read only: paired 256-byte records per deterministic bucket (the format the retired
 * writer produced, unchanged). */
#define ASSOC_REC_BYTES ELPIS_HGRAM_SLOT_BYTES
#define ASSOC_PAIR_BYTES (2u * ASSOC_REC_BYTES)
#define ASSOC_VALID 1u
static const uint8_t assoc_magic[8]={'E','L','P','H','A','S','0','1'};

typedef struct assoc_cell {
    uint8_t key[ELPIS_HGRAM_ASSOC_KEY_BYTES];
    uint8_t value[ELPIS_HGRAM_ASSOC_VALUE_BYTES];
    uint64_t generation;
} assoc_cell;

/* Read-only: opened O_RDONLY by the same owner-controlled path rules as the probe, and the exact opened inode
 * validated under a shared advisory lock (an exclusive holder, any other tool, makes the read BUSY). */
static int assoc_open(const char *path, int *fd_out, elpis_hgram_info *info) {
    int dfd=-1;
    char leaf[NAME_MAX+1];
    if(parse_path(path,&dfd,leaf)!=0) return ELPIS_HGRAM_PATH;
    int fd=openat(dfd,leaf,O_RDONLY|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK);
    if(fd<0) { close(dfd); return ELPIS_HGRAM_INVALID; }
    if(flock(fd,LOCK_SH|LOCK_NB)!=0) {
        close(fd);close(dfd);return ELPIS_HGRAM_ASSOC_BUSY;
    }
    struct stat opened,current;
    int same=fstat(fd,&opened)==0 &&
        fstatat(dfd,leaf,&current,AT_SYMLINK_NOFOLLOW)==0 &&
        opened.st_dev==current.st_dev && opened.st_ino==current.st_ino;
    int rc=same ? probe_fd(fd,info) : ELPIS_HGRAM_INVALID;
    close(dfd);
    if(rc!=ELPIS_HGRAM_EXISTS) { close(fd);return rc; }
    *fd_out=fd;
    return ELPIS_HGRAM_ASSOC_OK;
}
static int assoc_blank(const uint8_t p[ASSOC_REC_BYTES]) {
    for(size_t i=0;i<ASSOC_REC_BYTES;i++) if(p[i])return 0;
    return 1;
}
static int assoc_decode(const uint8_t p[ASSOC_REC_BYTES],uint64_t bucket,
                        assoc_cell *c) {
    if(memcmp(p,assoc_magic,8)!=0 ||
       get32(p+8)!=ELPIS_HGRAM_ASSOC_RECORD_ABI ||
       get32(p+12)!=ASSOC_VALID || get64(p+16)!=bucket ||
       get64(p+24)==0 || get64(p+248)!=fnv(p,248))return -1;
    for(size_t i=192;i<248;i++)if(p[i])return -1;
    c->generation=get64(p+24);
    memcpy(c->key,p+32,ELPIS_HGRAM_ASSOC_KEY_BYTES);
    memcpy(c->value,p+64,ELPIS_HGRAM_ASSOC_VALUE_BYTES);
    return 0;
}
typedef struct assoc_pair {
    assoc_cell selected;
    int active; /* -1 if empty, 0 or 1 for most recent valid record */
    uint64_t offset;
} assoc_pair;

/* When one copy is torn, a complete surviving copy remains authoritative.
 * If both are damaged, fail closed: NEVER reinterpret corruption as vacancy.
 */
static int assoc_read_pair(int fd,uint64_t bucket,assoc_pair *pair) {
    uint8_t raw[ASSOC_PAIR_BYTES];
    uint64_t off=(uint64_t)HEADER + bucket*ASSOC_PAIR_BYTES;
    if(off>(uint64_t)INT64_MAX-sizeof raw) return ELPIS_HGRAM_INVALID;
    size_t done=0;
    while(done<sizeof raw) {
        ssize_t n=pread(fd,raw+done,sizeof raw-done,(off_t)(off+done));
        if(n<0 && errno==EINTR) continue;
        if(n<=0) return ELPIS_HGRAM_IO;
        done+=(size_t)n;
    }
    pair->offset=off;
    pair->active=-1;
    assoc_cell cells[2];
    int good[2];
    int blank[2];
    for(int i=0;i<2;i++) {
        const uint8_t *p=raw+i*ASSOC_REC_BYTES;
        blank[i]=assoc_blank(p);
        good[i]=!blank[i] && assoc_decode(p,bucket,&cells[i])==0;
    }
    if(good[0] && good[1]) {
        if(memcmp(cells[0].key,cells[1].key,sizeof cells[0].key)!=0 ||
           cells[0].generation==cells[1].generation ||
           (cells[0].generation>cells[1].generation &&
            cells[0].generation-cells[1].generation!=1) ||
           (cells[1].generation>cells[0].generation &&
            cells[1].generation-cells[0].generation!=1))
            return ELPIS_HGRAM_ASSOC_CORRUPT;
    }
    if(good[0] || good[1]) {
        pair->active=good[0] && (!good[1] ||
                      cells[0].generation>cells[1].generation)?0:1;
        pair->selected=cells[pair->active];
        return ELPIS_HGRAM_ASSOC_OK;
    }
    if(blank[0] && blank[1]) return ELPIS_HGRAM_ASSOC_NOT_FOUND;
    return ELPIS_HGRAM_ASSOC_CORRUPT;
}
static uint64_t assoc_bucket(const uint8_t key[ELPIS_HGRAM_ASSOC_KEY_BYTES],
                             const elpis_hgram_info *info) {
    return fnv(key,ELPIS_HGRAM_ASSOC_KEY_BYTES) % (info->slots/2);
}
static void assoc_output(const assoc_cell *c,elpis_hgram_assoc_record *out) {
    memcpy(out->value,c->value,sizeof out->value);
    out->generation=c->generation;
}
int elpis_hgram_assoc_get(const char *path,
                           const uint8_t key[ELPIS_HGRAM_ASSOC_KEY_BYTES],
                           elpis_hgram_assoc_record *out) {
    if(!key || !out) return ELPIS_HGRAM_INVALID;
    int fd=-1;elpis_hgram_info info;
    int rc=assoc_open(path,&fd,&info);
    if(rc!=ELPIS_HGRAM_ASSOC_OK) return rc;
    if(info.slots<2){close(fd);return ELPIS_HGRAM_INVALID;}
    assoc_pair pair;
    rc=assoc_read_pair(fd,assoc_bucket(key,&info),&pair);
    if(rc==ELPIS_HGRAM_ASSOC_OK) {
        if(memcmp(pair.selected.key,key,ELPIS_HGRAM_ASSOC_KEY_BYTES)!=0)
            rc=ELPIS_HGRAM_ASSOC_CONFLICT;
        else assoc_output(&pair.selected,out);
    }
    if(close(fd)!=0) return ELPIS_HGRAM_IO;
    return rc;
}
