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
#include <sys/statvfs.h>
#include <sys/types.h>
#include <unistd.h>

#define MIB 1048576ULL
#define GIB 1073741824ULL
#define HEADER ELPIS_HGRAM_HEADER_BYTES
#define SLOT ELPIS_HGRAM_SLOT_BYTES
static const unsigned char magic[8] = {'E','L','P','H','G','R','A','1'};

static uint32_t get32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
}
static uint64_t get64(const unsigned char *p) {
    return (uint64_t)get32(p) | (uint64_t)get32(p+4)<<32;
}
static void put32(unsigned char *p,uint32_t v) {
    for (unsigned i=0;i<4;i++) p[i]=(unsigned char)(v>>(8*i));
}
static void put64(unsigned char *p,uint64_t v) {
    for (unsigned i=0;i<8;i++) p[i]=(unsigned char)(v>>(8*i));
}
static uint64_t fnv(const unsigned char *p, size_t n) {
    uint64_t h=UINT64_C(14695981039346656037);
    for(size_t i=0;i<n;i++) { h ^= p[i]; h *= UINT64_C(1099511628211); }
    return h;
}
static void header_make(unsigned char h[HEADER], uint64_t bytes) {
    memset(h, 0, HEADER);
    memcpy(h, magic, 8);
    put32(h+8,ELPIS_HGRAM_ABI);
    put32(h+12,HEADER);
    put64(h+16,bytes);
    put64(h+24,bytes-HEADER);
    put32(h+32,SLOT);
    put64(h+40,(bytes-HEADER)/SLOT);
    put64(h+48,fnv(h,48));
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
    if(!S_ISREG(st.st_mode) || st.st_nlink!=1 || st.st_uid!=geteuid() ||
       (uint64_t)st.st_blocks < ((uint64_t)st.st_size+511)/512 ||
       (st.st_mode & (S_IRWXG|S_IRWXO)) || !(st.st_mode & S_IWUSR) ||
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
int elpis_hgram_init_once(const char *path,uint64_t size_mib,
                          elpis_hgram_info *out) {
    if(!out || !size_mib || size_mib>INT64_MAX/MIB) return ELPIS_HGRAM_INVALID;
    uint64_t bytes=size_mib*MIB;
    if(bytes < HEADER+SLOT) return ELPIS_HGRAM_INVALID;
    int dfd=-1;char leaf[NAME_MAX+1];
    if(parse_path(path,&dfd,leaf)!=0) return ELPIS_HGRAM_PATH;
    /* Do not even create a directory/file when an object already exists. */
    struct stat target;
    if(fstatat(dfd,leaf,&target,AT_SYMLINK_NOFOLLOW)==0) {
        close(dfd);
        elpis_hgram_info existing;
        int rc=elpis_hgram_probe(path,&existing);
        if(rc==ELPIS_HGRAM_EXISTS && existing.capacity_mib==size_mib) {
            *out=existing;return ELPIS_HGRAM_EXISTS;
        }
        return ELPIS_HGRAM_INVALID;
    }
    if(errno!=ENOENT) {close(dfd);return ELPIS_HGRAM_IO;}
    struct statvfs sv;
    if(fstatvfs(dfd,&sv)!=0) {close(dfd);return ELPIS_HGRAM_IO;}
    uint64_t unit=sv.f_frsize ? sv.f_frsize : sv.f_bsize;
    if(!unit || sv.f_bavail > UINT64_MAX/unit) {close(dfd);return ELPIS_HGRAM_SPACE;}
    uint64_t freebytes=(uint64_t)sv.f_bavail*unit;
    uint64_t reserve=freebytes/10;
    if(reserve<GIB) reserve=GIB;
    if(bytes>freebytes || reserve>freebytes-bytes) {close(dfd);return ELPIS_HGRAM_SPACE;}
    int fd=openat(dfd,leaf,O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
    if(fd<0) {int rc=errno==EEXIST ? ELPIS_HGRAM_RACE : ELPIS_HGRAM_IO;
              close(dfd);return rc;}
    int rc=ELPIS_HGRAM_IO;
    struct stat made;
    if(fstat(fd,&made)!=0) goto failed;
    /* Full preallocation: a sparse logical-size-only ftruncate is forbidden. */
    int alloc=posix_fallocate(fd,0,(off_t)bytes);
    if(alloc!=0) {errno=alloc;goto failed;}
    unsigned char h[HEADER];header_make(h,bytes);
    size_t done=0;
    while(done<sizeof h) {
        ssize_t n=pwrite(fd,h+done,sizeof h-done,(off_t)done);
        if(n<=0)goto failed;
        done+=(size_t)n;
    }
    if(fdatasync(fd)!=0 || fsync(dfd)!=0)goto failed;
    rc=probe_fd(fd,out);
    if(rc!=ELPIS_HGRAM_EXISTS) {rc=ELPIS_HGRAM_IO;goto failed;}
    if(close(fd)!=0) {close(dfd);return ELPIS_HGRAM_IO;}
    close(dfd);return ELPIS_HGRAM_CREATED;
failed:
    close(fd);
    /* Only unlink our own inode; another process may have replaced the path. */
    struct stat current;
    if(fstatat(dfd,leaf,&current,AT_SYMLINK_NOFOLLOW)==0 &&
       made.st_dev==current.st_dev && made.st_ino==current.st_ino) {
        unlinkat(dfd,leaf,0);
        fsync(dfd);
    }
    close(dfd);
    return rc;
}

/* R1: fixed-capacity, dual-copy, collision-refusing association mechanics.
 * The provisioner above and its ABI=1 header intentionally stay unchanged.
 * All durable per-association metadata lives inside paired 256-byte slots.
 * No LRU or eviction occurs in this phase: avoiding catastrophic churn under
 * erroneous requests is more important than accepting every association.
 */
#define ASSOC_REC_BYTES ELPIS_HGRAM_SLOT_BYTES
#define ASSOC_PAIR_BYTES (2u * ASSOC_REC_BYTES)
#define ASSOC_VALID 1u
static const uint8_t assoc_magic[8]={'E','L','P','H','A','S','0','1'};

typedef struct assoc_cell {
    uint8_t key[ELPIS_HGRAM_ASSOC_KEY_BYTES];
    uint8_t value[ELPIS_HGRAM_ASSOC_VALUE_BYTES];
    uint64_t generation;
} assoc_cell;

/* No resizing, reallocation, auxiliary files or bypass via a preopened fd.
 * Open by the same owner-controlled path rules as the R0 provisioner, and
 * validate the exact opened inode under an advisory process lock. The lock
 * does NOT defeat hostile same-UID code; only OS isolation can do that.
 */
static int assoc_open(const char *path, int writable,
                      int *fd_out, elpis_hgram_info *info) {
    int dfd=-1;
    char leaf[NAME_MAX+1];
    if(parse_path(path,&dfd,leaf)!=0) return ELPIS_HGRAM_PATH;
    int fd=openat(dfd,leaf,(writable?O_RDWR:O_RDONLY)|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK);
    if(fd<0) { close(dfd); return ELPIS_HGRAM_INVALID; }
    if(flock(fd,(writable?LOCK_EX:LOCK_SH)|LOCK_NB)!=0) {
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
static void assoc_encode(uint8_t p[ASSOC_REC_BYTES], uint64_t bucket,
                         const assoc_cell *c) {
    memset(p,0,ASSOC_REC_BYTES);
    memcpy(p,assoc_magic,8);
    put32(p+8,ELPIS_HGRAM_ASSOC_RECORD_ABI);
    put32(p+12,ASSOC_VALID);
    put64(p+16,bucket);
    put64(p+24,c->generation);
    memcpy(p+32,c->key,ELPIS_HGRAM_ASSOC_KEY_BYTES);
    memcpy(p+64,c->value,ELPIS_HGRAM_ASSOC_VALUE_BYTES);
    put64(p+248,fnv(p,248));
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
    int rc=assoc_open(path,0,&fd,&info);
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
int elpis_hgram_assoc_store_preapproved(const char *path,
             const uint8_t key[ELPIS_HGRAM_ASSOC_KEY_BYTES],
             const uint8_t value[ELPIS_HGRAM_ASSOC_VALUE_BYTES],
             uint64_t expected_generation, elpis_hgram_assoc_record *out) {
    if(!key || !value || !out) return ELPIS_HGRAM_INVALID;
    int fd=-1;elpis_hgram_info info;
    int rc=assoc_open(path,1,&fd,&info);
    if(rc!=ELPIS_HGRAM_ASSOC_OK) return rc;
    if(info.slots<2){close(fd);return ELPIS_HGRAM_INVALID;}
    uint64_t bucket=assoc_bucket(key,&info);
    assoc_pair pair;
    rc=assoc_read_pair(fd,bucket,&pair);
    if(rc==ELPIS_HGRAM_ASSOC_NOT_FOUND) {
        if(expected_generation!=0) rc=ELPIS_HGRAM_ASSOC_STALE;
    } else if(rc==ELPIS_HGRAM_ASSOC_OK) {
        if(memcmp(pair.selected.key,key,ELPIS_HGRAM_ASSOC_KEY_BYTES)!=0)
            rc=ELPIS_HGRAM_ASSOC_CONFLICT;
        else if(memcmp(pair.selected.value,value,ELPIS_HGRAM_ASSOC_VALUE_BYTES)==0 &&
                (expected_generation==pair.selected.generation ||
                 (expected_generation!=UINT64_MAX &&
                  expected_generation+1==pair.selected.generation))) {
            assoc_output(&pair.selected,out);close(fd);return ELPIS_HGRAM_ASSOC_OK;
        } else if(expected_generation!=pair.selected.generation)
            rc=ELPIS_HGRAM_ASSOC_STALE;
        else if(pair.selected.generation==UINT64_MAX)
            rc=ELPIS_HGRAM_ASSOC_STALE;
    }
    if(rc==ELPIS_HGRAM_ASSOC_OK || rc==ELPIS_HGRAM_ASSOC_NOT_FOUND) {
        assoc_cell candidate;
        memcpy(candidate.key,key,sizeof candidate.key);
        memcpy(candidate.value,value,sizeof candidate.value);
        candidate.generation=pair.active<0?1:pair.selected.generation+1;
        uint8_t wire[ASSOC_REC_BYTES];
        assoc_encode(wire,bucket,&candidate);
        int inactive=pair.active<0?0:1-pair.active;
        uint64_t off=pair.offset+(uint64_t)inactive*ASSOC_REC_BYTES;
        ssize_t n;
        do { n=pwrite(fd,wire,sizeof wire,(off_t)off); } while(n<0 && errno==EINTR);
        if(n!=(ssize_t)sizeof wire || fdatasync(fd)!=0) rc=ELPIS_HGRAM_IO;
        else {
            uint8_t check[ASSOC_REC_BYTES];
            ssize_t got=pread(fd,check,sizeof check,(off_t)off);
            if(got!=(ssize_t)sizeof check || memcmp(wire,check,sizeof wire)!=0)
                rc=ELPIS_HGRAM_IO;
            else { assoc_output(&candidate,out); rc=ELPIS_HGRAM_ASSOC_OK; }
        }
    }
    if(close(fd)!=0) return ELPIS_HGRAM_IO;
    return rc;
}
