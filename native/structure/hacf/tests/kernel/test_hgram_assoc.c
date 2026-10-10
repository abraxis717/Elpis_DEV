#define _GNU_SOURCE 1
#include "elpis/hgram_store.h"
#include <assert.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static void join(char *out,size_t cap,const char *dir,const char *leaf) {
    int n=snprintf(out,cap,"%s/%s",dir,leaf);
    assert(n>0 && (size_t)n<cap);
}
static unsigned file_count(const char *path) {
    DIR *d=opendir(path);assert(d);
    unsigned n=0;struct dirent *e;
    while((e=readdir(d))!=NULL) if(strcmp(e->d_name,".") && strcmp(e->d_name,"..")) n++;
    assert(closedir(d)==0);return n;
}
static void read_exact(int fd,void *p,size_t n,off_t off) {
    assert(pread(fd,p,n,off)==(ssize_t)n);
}
static void find_record_offsets(int fd,const uint8_t key[32],off_t *first,off_t *second) {
    uint8_t buf[256];*first=-1;*second=-1;
    for(off_t off=4096;off<1048576;off+=256) {
        read_exact(fd,buf,sizeof buf,off);
        if(memcmp(buf,"ELPHAS01",8)==0 && memcmp(buf+32,key,32)==0) {
            if(*first<0) *first=off;
            else { *second=off;return; }
        }
    }
}
static void test_corruption_survivability(int fd,const char *bin,
             const uint8_t key[32],uint8_t value[128],elpis_hgram_assoc_record *a) {
    off_t first,second;
    find_record_offsets(fd,key,&first,&second);
    assert(first>=0 && second>=0);
    /* Corrupt the inactive version: current record must still be readable. */
    uint8_t bad=0x88;
    assert(pwrite(fd,&bad,1,first+120)==1);
    assert(fdatasync(fd)==0);
    assert(elpis_hgram_assoc_get(bin,key,a)==ELPIS_HGRAM_ASSOC_OK);
    assert(a->generation==2);
    /* Rewriting the damaged copy must recover safely and preserve file size. */
    memset(value,0x45,128);
    assert(elpis_hgram_assoc_store_preapproved(bin,key,value,2,a)==ELPIS_HGRAM_ASSOC_OK);
    assert(a->generation==3);
    /* Now corrupt both versions; refuse to pretend the bucket is empty. */
    assert(pwrite(fd,&bad,1,first+119)==1);
    assert(pwrite(fd,&bad,1,second+119)==1);
    assert(fdatasync(fd)==0);
    assert(elpis_hgram_assoc_get(bin,key,a)==ELPIS_HGRAM_ASSOC_CORRUPT);
    assert(elpis_hgram_assoc_store_preapproved(bin,key,value,0,a)==ELPIS_HGRAM_ASSOC_CORRUPT);
}
int main(void) {
    char dir[]="/tmp/elpis-hgram-assoc-XXXXXX";
    assert(mkdtemp(dir)!=NULL);
    char bin[4096];join(bin,sizeof bin,dir,"hgram.bin");
    uint8_t key[32],other[32],value[128];
    memset(key,0x17,sizeof key);
    memset(value,0x72,sizeof value);
    memset(other,0x42,sizeof other);
    elpis_hgram_info info={0};
    elpis_hgram_assoc_record a={0},b={0};
    assert(elpis_hgram_assoc_get(bin,key,&a)==ELPIS_HGRAM_INVALID);
    assert(elpis_hgram_init_once(bin,1,&info)==ELPIS_HGRAM_CREATED);
    struct stat initial,after;
    assert(stat(bin,&initial)==0);
    assert(initial.st_size==1048576);
    assert(elpis_hgram_assoc_get(bin,key,&a)==ELPIS_HGRAM_ASSOC_NOT_FOUND);
    assert(elpis_hgram_assoc_store_preapproved(bin,key,value,1,&a)==ELPIS_HGRAM_ASSOC_STALE);
    assert(elpis_hgram_assoc_store_preapproved(bin,key,value,0,&a)==ELPIS_HGRAM_ASSOC_OK);
    assert(a.generation==1 && memcmp(a.value,value,128)==0);
    assert(elpis_hgram_assoc_get(bin,key,&b)==ELPIS_HGRAM_ASSOC_OK);
    assert(b.generation==1 && memcmp(b.value,value,128)==0);
    /* Replayed create of identical value is an exact no-op. */
    assert(elpis_hgram_assoc_store_preapproved(bin,key,value,0,&a)==ELPIS_HGRAM_ASSOC_OK);
    assert(a.generation==1);
    memset(value,0x11,sizeof value);
    assert(elpis_hgram_assoc_store_preapproved(bin,key,value,0,&a)==ELPIS_HGRAM_ASSOC_STALE);
    assert(elpis_hgram_assoc_store_preapproved(bin,key,value,1,&a)==ELPIS_HGRAM_ASSOC_OK);
    assert(a.generation==2);
    assert(elpis_hgram_assoc_store_preapproved(bin,key,value,1,&a)==ELPIS_HGRAM_ASSOC_OK);
    assert(a.generation==2);
    /* Fixed collision refusal: search for a different key mapping to the same
     * bucket by calling the public API (no private hash assumption). */
    uint8_t clash[32];int found=0;
    for(unsigned n=0;n<200000 && !found;n++) {
        memset(clash,0,sizeof clash);
        clash[0]=(uint8_t)n;clash[1]=(uint8_t)(n>>8);clash[2]=(uint8_t)(n>>16);
        if(memcmp(clash,key,32)==0) continue;
        if(elpis_hgram_assoc_get(bin,clash,&b)==ELPIS_HGRAM_ASSOC_CONFLICT) found=1;
    }
    assert(found);
    assert(elpis_hgram_assoc_store_preapproved(bin,clash,value,0,&a)==ELPIS_HGRAM_ASSOC_CONFLICT);
    assert(elpis_hgram_assoc_get(bin,key,&b)==ELPIS_HGRAM_ASSOC_OK);
    assert(b.generation==2);
    /* A separate valid key can be written, and remains independent. */
    assert(elpis_hgram_assoc_store_preapproved(bin,other,value,0,&a)==ELPIS_HGRAM_ASSOC_OK);
    assert(a.generation==1);
    assert(elpis_hgram_assoc_get(bin,key,&b)==ELPIS_HGRAM_ASSOC_OK);
    assert(b.generation==2);
    assert(elpis_hgram_probe(bin,&info)==ELPIS_HGRAM_EXISTS);
    /* Stress 1000 sequential in-place revisions. No append/files/extra extents.
     * Useful data for unrelated key must survive all revisions. */
    for(uint64_t i=2;i<=1001;i++) {
        memset(value,(int)(i&255),sizeof value);
        assert(elpis_hgram_assoc_store_preapproved(bin,other,value,i-1,&a)==ELPIS_HGRAM_ASSOC_OK);
        assert(a.generation==i);
    }
    assert(elpis_hgram_assoc_get(bin,key,&b)==ELPIS_HGRAM_ASSOC_OK);
    assert(b.generation==2);
    assert(stat(bin,&after)==0);
    assert(after.st_size==initial.st_size);
    assert(after.st_blocks==initial.st_blocks);
    assert(file_count(dir)==1);
    /* A second independent descriptor/process cannot race a locked writer. */
    int locked=open(bin,O_RDWR|O_CLOEXEC);assert(locked>=0);
    assert(flock(locked,LOCK_EX|LOCK_NB)==0);
    pid_t child=fork();assert(child>=0);
    if(child==0) {
        elpis_hgram_assoc_record tmp={0};
        int rc=elpis_hgram_assoc_get(bin,key,&tmp);
        _exit(rc==ELPIS_HGRAM_ASSOC_BUSY?0:1);
    }
    int status=0;assert(waitpid(child,&status,0)==child);
    assert(WIFEXITED(status) && WEXITSTATUS(status)==0);
    assert(flock(locked,LOCK_UN)==0 && close(locked)==0);
    int fd=open(bin,O_RDWR|O_CLOEXEC);assert(fd>=0);
    memset(value,0x11,sizeof value);
    test_corruption_survivability(fd,bin,key,value,&a);
    assert(close(fd)==0);
    assert(stat(bin,&after)==0 && after.st_size==initial.st_size);
    assert(file_count(dir)==1);
    assert(unlink(bin)==0 && rmdir(dir)==0);
    puts("ELPIS_HGRAM_ASSOC_R1_LOCAL_PASS");
    return 0;
}
