#define _GNU_SOURCE
#include "elpis_semantic/active_head_b2b.h"
#include "elpis/sha256.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef O_NOFOLLOW
#error "B2b requires O_NOFOLLOW"
#endif
#ifndef O_CLOEXEC
#error "B2b requires O_CLOEXEC"
#endif
static const uint8_t HEAD_TAG[16]={'E','L','P','I','S','-','H','E','A','D','-','V','1',0,0,0};
typedef struct {uint8_t tag[16];uint8_t digest[32];uint8_t check[32];} head_record;
static int eq(const hacf_digest *a,const hacf_digest *b){return memcmp(a->bytes,b->bytes,32)==0;}
static int zero(const hacf_digest *a){const uint8_t z[32]={0};return memcmp(a->bytes,z,32)==0;}
static int valid_digest(const hacf_digest *d){return d&&!zero(d);}
static int open_root(const char *p){
 if(!p||!*p||strnlen(p,PATH_MAX)>=PATH_MAX)return -1;
 return open(p,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
}
static int lock_fd(int dir,int exclusive){
 int fd=openat(dir,".semantic-head.lock",O_RDWR|O_CREAT|O_CLOEXEC|O_NOFOLLOW,0600);
 if(fd<0)return -1;
 struct stat st;
 if(fstat(fd,&st)||!S_ISREG(st.st_mode)||st.st_nlink!=1||flock(fd,exclusive?LOCK_EX:LOCK_SH)){
  close(fd);return -1;
 }
 return fd;
}
static int read_all(int fd,void *p,size_t n){
 uint8_t *b=p;size_t pos=0;
 while(pos<n){ssize_t k=read(fd,b+pos,n-pos);if(k<0&&errno==EINTR)continue;if(k<=0)return -1;pos+=(size_t)k;}
 return 0;
}
static int write_all(int fd,const void *p,size_t n){
 const uint8_t *b=p;size_t pos=0;
 while(pos<n){ssize_t k=write(fd,b+pos,n-pos);if(k<0&&errno==EINTR)continue;if(k<=0)return -1;pos+=(size_t)k;}
 return 0;
}
static int read_head(int dir,hacf_digest *out){
 int fd=openat(dir,"HEAD",O_RDONLY|O_CLOEXEC|O_NOFOLLOW);if(fd<0)return errno==ENOENT?SEMANTIC_B2B_CONFLICT:SEMANTIC_B2B_IO;
 struct stat st;head_record h;int good=0;uint8_t sum[32];
 if(!fstat(fd,&st)&&S_ISREG(st.st_mode)&&st.st_nlink==1&&st.st_size==(off_t)sizeof(h)&&
    !read_all(fd,&h,sizeof(h))&&!memcmp(h.tag,HEAD_TAG,16)){
  elpis_sha256(h.tag,48,sum);
  if(!memcmp(sum,h.check,32)){memcpy(out->bytes,h.digest,32);good=!zero(out);}
 }
 int cr=close(fd);return good&&!cr?SEMANTIC_B2B_OK:SEMANTIC_B2B_INVALID;
}
static int cas_manifest(int dir,const hacf_digest *pin,semantic_snapshot_manifest *out){
 if(!valid_digest(pin))return SEMANTIC_B2B_INVALID;
 char hex[65],name[80];elpis_hex32(pin->bytes,hex);
 if(snprintf(name,sizeof(name),"%s.snapshot",hex)!=73)return SEMANTIC_B2B_INVALID;
 int fd=openat(dir,name,O_RDONLY|O_CLOEXEC|O_NOFOLLOW);if(fd<0)return SEMANTIC_B2B_IO;
 struct stat st;semantic_snapshot_manifest *m=malloc(sizeof(*m));int rc=SEMANTIC_B2B_INVALID;
 if(m&&!fstat(fd,&st)&&S_ISREG(st.st_mode)&&st.st_nlink==1&&st.st_size==(off_t)sizeof(*m)&&
    !read_all(fd,m,sizeof(*m))&&semantic_snapshot_validate(m)==SEMANTIC_OK&&
    eq(pin,&m->manifest_digest)&&eq(&m->manifest_digest,&m->hacf_package_digest)){
  if(out)*out=*m;
  rc=SEMANTIC_B2B_OK;
 }
 if(close(fd))rc=SEMANTIC_B2B_IO;
 free(m);return rc;
}
/* Private, same-directory temp. Stale temp files do not affect recovery. */
static int persist_head(int dir,const hacf_digest *next){
 head_record h={0};memcpy(h.tag,HEAD_TAG,16);memcpy(h.digest,next->bytes,32);
 elpis_sha256(h.tag,48,h.check);
 int fd=-1;char name[80];static unsigned seq=0;
 for(unsigned i=0;i<256;i++){
  unsigned counter=__atomic_add_fetch(&seq,1,__ATOMIC_RELAXED);
  int n=snprintf(name,sizeof(name),".head-stage-%ld-%u",(long)getpid(),counter);
  if(n<1||(size_t)n>=sizeof(name))return SEMANTIC_B2B_IO;
  fd=openat(dir,name,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
  if(fd>=0)break;
  if(errno!=EEXIST)return SEMANTIC_B2B_IO;
 }
 if(fd<0)return SEMANTIC_B2B_IO;
 int rc=SEMANTIC_B2B_IO;
 if(!write_all(fd,&h,sizeof(h))&&!fsync(fd)&&!close(fd)){
  fd=-1;
  if(!renameat(dir,name,dir,"HEAD")){
   rc=fsync(dir)==0?SEMANTIC_B2B_OK:SEMANTIC_B2B_UNCERTAIN;
   return rc;
  }
 }
 if(fd>=0)close(fd);
 unlinkat(dir,name,0);
 return rc;
}
static int validate_next_segment(int dir,const char *root,
                                 const semantic_snapshot_manifest *base,
                                 const semantic_snapshot_manifest *next){
 if(next->segment_count!=base->segment_count+1 || next->segment_count>SEMANTIC_MAX_SEGMENTS||
    !eq(&next->type_registry_digest,&base->type_registry_digest)||
    !eq(&next->genesis_identity,&base->genesis_identity)||
    !eq(&next->prior_manifest_digest,&base->manifest_digest)||
    !memcmp(&next->hacf_graph_snapshot_digest,&base->hacf_graph_snapshot_digest,32))return SEMANTIC_B2B_INVALID;
 for(uint32_t i=0;i<base->segment_count;i++)
  if(!eq(&next->segment_digests[i],&base->segment_digests[i]))return SEMANTIC_B2B_INVALID;
 char hex[65],name[80];elpis_hex32(next->segment_digests[base->segment_count].bytes,hex);
 if(snprintf(name,sizeof(name),"%s.segment",hex)!=72)return SEMANTIC_B2B_INVALID;
 int fd=openat(dir,name,O_RDONLY|O_NOFOLLOW|O_CLOEXEC);
 if(fd<0)return SEMANTIC_B2B_IO;
 struct stat st;int good=!fstat(fd,&st)&&S_ISREG(st.st_mode)&&st.st_nlink==1;
 if(close(fd))good=0;
 if(!good)return SEMANTIC_B2B_INVALID;
 char path[PATH_MAX];int n=snprintf(path,sizeof(path),"%s/%s",root,name);
 if(n<0||(size_t)n>=sizeof(path))return SEMANTIC_B2B_INVALID;
 semantic_segment_record *s=malloc(sizeof(*s));hacf_digest d={0};
 if(!s)return SEMANTIC_B2B_IO;
 int rc=SEMANTIC_B2B_INVALID;
 if(semantic_segment_read(path,s,&d)==SEMANTIC_OK&&
    eq(&d,&next->segment_digests[base->segment_count])&&
    eq(&s->prior_snapshot_digest,&base->hacf_graph_snapshot_digest)&&
    eq(&s->type_registry_digest,&base->type_registry_digest)&&
    eq(&s->hacf_next_snapshot,&next->hacf_graph_snapshot_digest)&&
    base->unique_node_count<=UINT32_MAX-s->node_count&&
    base->unique_hyperedge_count<=UINT32_MAX-s->hyperedge_count&&
    base->assertion_count<=UINT32_MAX-s->assertion_count&&
    base->incidence_count<=UINT32_MAX-s->incidence_count&&
    next->unique_node_count==base->unique_node_count+s->node_count&&
    next->unique_hyperedge_count==base->unique_hyperedge_count+s->hyperedge_count&&
    next->assertion_count==base->assertion_count+s->assertion_count&&
    next->incidence_count==base->incidence_count+s->incidence_count)
  rc=SEMANTIC_B2B_OK;
 free(s);return rc;
}
int semantic_b2b_head_read(const char *root,hacf_digest *out){
 if(!out)return SEMANTIC_B2B_INVALID;
 int dir=open_root(root);if(dir<0)return SEMANTIC_B2B_IO;
 int lock=lock_fd(dir,0);if(lock<0){close(dir);return SEMANTIC_B2B_IO;}
 hacf_digest current={0};int rc=read_head(dir,&current);
 if(rc==SEMANTIC_B2B_OK)rc=cas_manifest(dir,&current,NULL);
 if(rc==SEMANTIC_B2B_OK)*out=current;
 close(lock);close(dir);return rc;
}
int semantic_b2b_head_bootstrap(const char *root,const hacf_digest *pin){
 if(!valid_digest(pin))return SEMANTIC_B2B_INVALID;
 int dir=open_root(root);if(dir<0)return SEMANTIC_B2B_IO;
 int lock=lock_fd(dir,1);if(lock<0){close(dir);return SEMANTIC_B2B_IO;}
 hacf_digest current={0};int rc=read_head(dir,&current);
 if(rc==SEMANTIC_B2B_CONFLICT){
  semantic_snapshot_manifest *m=malloc(sizeof(*m));
  if(!m)rc=SEMANTIC_B2B_IO;
  else{
   rc=cas_manifest(dir,pin,m);
   if(rc==SEMANTIC_B2B_OK&&!zero(&m->prior_manifest_digest))rc=SEMANTIC_B2B_INVALID;
   if(rc==SEMANTIC_B2B_OK)rc=persist_head(dir,pin);
   free(m);
  }
 }else if(rc==SEMANTIC_B2B_OK)rc=SEMANTIC_B2B_CONFLICT;
 close(lock);close(dir);return rc;
}
int semantic_b2b_head_cas(const char *root,const hacf_digest *expected,const hacf_digest *next){
 if(!valid_digest(expected)||!valid_digest(next)||eq(expected,next))return SEMANTIC_B2B_INVALID;
 int dir=open_root(root);if(dir<0)return SEMANTIC_B2B_IO;
 int lock=lock_fd(dir,1);if(lock<0){close(dir);return SEMANTIC_B2B_IO;}
 hacf_digest current={0};int rc=read_head(dir,&current);
 if(rc==SEMANTIC_B2B_OK&&!eq(&current,expected))rc=SEMANTIC_B2B_CONFLICT;
 if(rc==SEMANTIC_B2B_OK){
  semantic_snapshot_manifest *base=malloc(sizeof(*base)),*successor=malloc(sizeof(*successor));
  if(!base||!successor)rc=SEMANTIC_B2B_IO;
  else{
   rc=cas_manifest(dir,expected,base);
   if(rc==SEMANTIC_B2B_OK)rc=cas_manifest(dir,next,successor);
   if(rc==SEMANTIC_B2B_OK)rc=validate_next_segment(dir,root,base,successor);
   if(rc==SEMANTIC_B2B_OK)rc=persist_head(dir,next);
  }
  free(base);free(successor);
 }
 close(lock);close(dir);return rc;
}
