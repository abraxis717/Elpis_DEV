#define _GNU_SOURCE 1
#include "elpis/hgram_store.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void pathcat(char *out,size_t cap,const char *dir,const char *leaf) {
    int n=snprintf(out,cap,"%s/%s",dir,leaf);
    assert(n>0 && (size_t)n<cap);
}
static uint64_t filesize(const char *path) {
    struct stat st;
    assert(stat(path,&st)==0);
    assert(S_ISREG(st.st_mode));
    return (uint64_t)st.st_size;
}
int main(void) {
    char dir[]="/tmp/elpis-hgram-test-XXXXXX";
    assert(mkdtemp(dir)!=NULL);
    char bin[4096],bad[4096],sym[4096],sparse[4096],hard[4096];
    pathcat(bin,sizeof bin,dir,"hgram.bin");
    pathcat(bad,sizeof bad,dir,"bad.bin");
    pathcat(sym,sizeof sym,dir,"symlink.bin");
    pathcat(sparse,sizeof sparse,dir,"sparse.bin");
    pathcat(hard,sizeof hard,dir,"hard.bin");
    elpis_hgram_info a={0},b={0};
    assert(elpis_hgram_probe(bin,&a)==ELPIS_HGRAM_PATH);
    assert(elpis_hgram_init_once(bin,0,&a)==ELPIS_HGRAM_INVALID);
    assert(access(bin,F_OK)!=0);
    assert(elpis_hgram_init_once(bin,1,&a)==ELPIS_HGRAM_CREATED);
    assert(a.total_bytes==1048576ULL && a.slots==(1048576-4096)/256 && a.capacity_mib==1);
    assert(filesize(bin)==a.total_bytes);
    assert(elpis_hgram_init_once(bin,1,&b)==ELPIS_HGRAM_EXISTS);
    assert(elpis_hgram_init_once(bin,2,&b)==ELPIS_HGRAM_INVALID);
    assert(filesize(bin)==a.total_bytes);
    assert(elpis_hgram_probe(bin,&b)==ELPIS_HGRAM_EXISTS);
    assert(b.total_bytes==a.total_bytes);
    assert(link(bin,hard)==0);
    assert(elpis_hgram_probe(bin,&b)==ELPIS_HGRAM_INVALID);
    assert(elpis_hgram_init_once(bin,1,&b)==ELPIS_HGRAM_INVALID);
    assert(unlink(hard)==0);
    assert(elpis_hgram_probe(bin,&b)==ELPIS_HGRAM_EXISTS);
    /* Copy only the header into a sparse file, then extend it logically.
       Equal file length must not be mistaken for physical preallocation. */
    int original=open(bin,O_RDONLY);assert(original>=0);
    char header[4096];assert(read(original,header,sizeof header)==sizeof header);
    assert(close(original)==0);
    int fake=open(sparse,O_CREAT|O_EXCL|O_WRONLY,0600);assert(fake>=0);
    assert(write(fake,header,sizeof header)==sizeof header);
    assert(ftruncate(fake,1048576)==0);assert(close(fake)==0);
    assert(elpis_hgram_probe(sparse,&b)==ELPIS_HGRAM_INVALID);
    assert(unlink(sparse)==0);
    int fd=open(bin,O_RDWR);assert(fd>=0);
    assert(pwrite(fd,"Z",1,100)==1);
    assert(close(fd)==0);
    assert(elpis_hgram_probe(bin,&b)==ELPIS_HGRAM_INVALID);
    assert(elpis_hgram_init_once(bin,1,&b)==ELPIS_HGRAM_INVALID);
    assert(filesize(bin)==a.total_bytes);
    fd=open(bad,O_CREAT|O_EXCL|O_WRONLY,0600);assert(fd>=0);
    assert(write(fd,"bad",3)==3);assert(close(fd)==0);
    assert(elpis_hgram_init_once(bad,1,&b)==ELPIS_HGRAM_INVALID);
    assert(filesize(bad)==3);
    assert(symlink(bin,sym)==0);
    assert(elpis_hgram_probe(sym,&b)==ELPIS_HGRAM_INVALID);
    assert(elpis_hgram_init_once(sym,1,&b)==ELPIS_HGRAM_INVALID);
    assert(filesize(bin)==a.total_bytes);
    assert(unlink(sym)==0 && unlink(bad)==0 && unlink(bin)==0);
    assert(rmdir(dir)==0);
    puts("ELPIS_HGRAM_FIXED_CAPACITY_PASS");
    return 0;
}
