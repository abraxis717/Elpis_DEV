#define _GNU_SOURCE
#include "elpis_semantic/admission_commit_b2c.h"
#include "elpis_semantic/snapshot_publication.h"
#include "elpis_semantic/hypergraph.h"
#include "elpis/sha256.h"
#include <errno.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/file.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#ifndef O_NOFOLLOW
#error "B2c requires O_NOFOLLOW"
#endif
static int same(const hacf_digest *a,const hacf_digest *b){return memcmp(a,b,sizeof(*a))==0;}
static int read_exact(int fd,void *dst,size_t n){
 uint8_t *p=dst;size_t at=0;
 while(at<n){ssize_t q=read(fd,p+at,n-at);if(q<0&&errno==EINTR)continue;if(q<=0)return 0;at+=(size_t)q;}
 return 1;
}
static int safe_path(char *dst,size_t cap,const char *root,
                     const hacf_digest *id,const char *extension){
 char hex[65];elpis_hex32(id->bytes,hex);
 int n=snprintf(dst,cap,"%s/%s.%s",root,hex,extension);
 return n>0&&(size_t)n<cap;
}
/* The v1 projection identity does NOT authenticate all raw record bytes.
 * Verify exactly the one-node/one-assertion materialization, even when a
 * CAS name was already staged by a prior incomplete transaction. */
static int verify_staged_segment(int dir,const hacf_digest *id,
                                 const semantic_segment_record *seg,
                                 const semantic_hypergraph_builder *builder,int accept_orphan){
 char hex[65],name[80];elpis_hex32(id->bytes,hex);
 if(snprintf(name,sizeof(name),"%s.segment",hex)!=72)return 0;
 int fd=openat(dir,name,O_RDONLY|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK);
 if(fd<0)return 0;
 struct stat st;const elpis_semantic_node_v1 *node=semantic_builder_get_node(builder,0);
 const elpis_semantic_assertion_v1 *assertion=semantic_builder_get_assertion(builder,0);
 semantic_segment_record header;elpis_semantic_node_v1 n;elpis_semantic_assertion_v1 a;
 const size_t length=sizeof(header)+sizeof(n)+sizeof(a);
 int ok=node&&assertion&&!fstat(fd,&st)&&S_ISREG(st.st_mode)&&(st.st_nlink==1||(accept_orphan&&st.st_nlink==2))&&
       st.st_size==(off_t)length&&read_exact(fd,&header,sizeof(header))&&
       read_exact(fd,&n,sizeof(n))&&read_exact(fd,&a,sizeof(a))&&
       memcmp(&header,seg,sizeof(header))==0&&memcmp(&n,node,sizeof(n))==0&&
       memcmp(&a,assertion,sizeof(a))==0;
 if(close(fd))ok=0;
 return ok;
}
static int verify_staged_manifest(int dir,const hacf_digest *id,
                                  const semantic_snapshot_manifest *expected,int accept_orphan){
 char hex[65],name[80];elpis_hex32(id->bytes,hex);
 if(snprintf(name,sizeof(name),"%s.snapshot",hex)!=73)return 0;
 int fd=openat(dir,name,O_RDONLY|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK);
 if(fd<0)return 0;
 struct stat st;semantic_snapshot_manifest *m=malloc(sizeof(*m));
 int ok=m&&!fstat(fd,&st)&&S_ISREG(st.st_mode)&&(st.st_nlink==1||(accept_orphan&&st.st_nlink==2))&&
       st.st_size==(off_t)sizeof(*m)&&read_exact(fd,m,sizeof(*m))&&
       memcmp(m,expected,sizeof(*m))==0;
 if(close(fd))ok=0;
 free(m);return ok;
}
/* B2g: A single native admission transaction owns staging under this lock.
 * The CAS writers do not know the trusted witness; only this caller may
 * normalize a post-link / pre-unlink crash after verifying exact CAS bytes.
 * The directory MUST be private to the trusted process UID; this mechanism
 * does not attribute intent to an arbitrary same-UID adversary. */
static int stage_lock(int dir,int exclusive){
 int fd=openat(dir,".semantic-stage.lock",O_RDWR|(exclusive?O_CREAT:0)|O_CLOEXEC|O_NOFOLLOW,0600);
 if(fd<0)return -1;
 struct stat st;
 if(fstat(fd,&st)||!S_ISREG(st.st_mode)||st.st_nlink!=1||
    st.st_uid!=geteuid()||(st.st_mode&0077)!=0||flock(fd,exclusive?LOCK_EX:LOCK_SH)!=0){
  close(fd);return -1;
 }
 return fd;
}
/* A recoverable artifact has exactly TWO links, CAS name and one regular
 * same-inode mkstemp alias with the writer's exact six-character suffix.
 * No arbitrary glob deletion, no symlink traversal, no nlink>2 normalization.
 * Full expected bytes were independently checked by verify_staged_* first.
 * Return INVALID if provenance is ambiguous; UNCERTAIN after unlink/fsync
 * failure, requiring caller reconciliation rather than a blind retry. */
static int reconcile_cas_orphan(int dir,const hacf_digest *id,
                                 const char *extension,const char *prefix){
 char hex[65],name[96];elpis_hex32(id->bytes,hex);
 int k=snprintf(name,sizeof(name),"%s.%s",hex,extension);
 if(k<0||(size_t)k>=sizeof(name))return SEMANTIC_B2B_INVALID;
 int cas=openat(dir,name,O_RDONLY|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK);
 if(cas<0)return SEMANTIC_B2B_INVALID;
 struct stat cs;
 int ok=!fstat(cas,&cs)&&S_ISREG(cs.st_mode)&&cs.st_nlink==2&&cs.st_uid==geteuid();
 if(close(cas))ok=0;
 if(!ok)return SEMANTIC_B2B_INVALID;
 int dup_fd=openat(dir,".",O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
 if(dup_fd<0)return SEMANTIC_B2B_IO;
 DIR *dp=fdopendir(dup_fd);
 if(!dp){close(dup_fd);return SEMANTIC_B2B_IO;}
 char selected[NAME_MAX+1]={0};unsigned hits=0;
 const size_t plen=strlen(prefix);
 struct dirent *entry;
 errno=0;
 while((entry=readdir(dp))){
  const char *nm=entry->d_name;
  if(strncmp(nm,prefix,plen))continue;
  /* mkstemp writers use exactly six alphanumeric characters after prefix. */
  if(strlen(nm)!=plen+6){ok=0;break;}
  for(size_t i=plen;i<plen+6;i++){
   char c=nm[i];
   if(!((c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>='0'&&c<='9'))){ok=0;break;}
  }
  if(!ok)break;
  struct stat ts;
  if(fstatat(dir,nm,&ts,AT_SYMLINK_NOFOLLOW)!=0||!S_ISREG(ts.st_mode)){
   ok=0;break;
  }
  if(ts.st_dev==cs.st_dev&&ts.st_ino==cs.st_ino){
   if(ts.st_nlink!=2||ts.st_uid!=cs.st_uid||++hits!=1){ok=0;break;}
   memcpy(selected,nm,plen+7);
  }
 }
 if(errno)ok=0;
 if(closedir(dp))ok=0;
 if(!ok||hits!=1)return SEMANTIC_B2B_INVALID;
 /* Revalidate the alias and canonical inode just before unlink. */
 struct stat ts,again;
 if(fstatat(dir,selected,&ts,AT_SYMLINK_NOFOLLOW)||
    fstatat(dir,name,&again,AT_SYMLINK_NOFOLLOW)||
    !S_ISREG(ts.st_mode)||!S_ISREG(again.st_mode)||
    ts.st_dev!=cs.st_dev||ts.st_ino!=cs.st_ino||ts.st_nlink!=2||
    again.st_dev!=cs.st_dev||again.st_ino!=cs.st_ino||again.st_nlink!=2)
  return SEMANTIC_B2B_INVALID;
 if(unlinkat(dir,selected,0)!=0)return SEMANTIC_B2B_UNCERTAIN;
 if(fsync(dir)!=0)return SEMANTIC_B2B_UNCERTAIN;
 return SEMANTIC_B2B_OK;
}
static int cas_nlink(int dir,const hacf_digest *id,const char *ext){
 char hex[65],name[96];elpis_hex32(id->bytes,hex);
 int n=snprintf(name,sizeof(name),"%s.%s",hex,ext);
 if(n<0||(size_t)n>=sizeof(name))return -1;
 struct stat st;
 if(fstatat(dir,name,&st,AT_SYMLINK_NOFOLLOW)||!S_ISREG(st.st_mode))return -1;
 return st.st_nlink==1?1:st.st_nlink==2?2:-1;
}
/* A manifest's digest is not proof that all historical segment files exist
 * in the durable store. Compare every fully audited source segment against the
 * exact CAS bytes, refusing symlinks, hardlinks, streams and missing files.
 * This is a narrow B2d closure for the 2..64 segment genesis witness. */
static int verify_base_cas_closure(int dir,const semantic_b2a_witness *w){
 if(!w||!w->base||!w->base_segment_paths||w->base_segment_count<2||
    w->base_segment_count>64||w->base_segment_count!=w->base->segment_count)
    return 0;
 for(uint32_t i=0;i<w->base_segment_count;i++){
  const char *source=w->base_segment_paths[i];
  if(!source||!*source||strnlen(source,PATH_MAX)>=PATH_MAX)return 0;
  char hex[65],name[80];
  elpis_hex32(w->base->segment_digests[i].bytes,hex);
  if(snprintf(name,sizeof(name),"%s.segment",hex)!=72)return 0;
  int src=open(source,O_RDONLY|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK);
  if(src<0)return 0;
  int cas=openat(dir,name,O_RDONLY|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK);
  if(cas<0){close(src);return 0;}
  struct stat ss,cs;
  int ok=!fstat(src,&ss)&&!fstat(cas,&cs)&&
         S_ISREG(ss.st_mode)&&S_ISREG(cs.st_mode)&&cs.st_nlink==1&&
         ss.st_size>=0&&ss.st_size==cs.st_size&&
         ss.st_size>=(off_t)sizeof(semantic_segment_record);
  uint8_t a[8192],b[8192];
  while(ok){
   ssize_t na=read(src,a,sizeof(a));
   if(na<0&&errno==EINTR)continue;
   if(na<0){ok=0;break;}
   if(!na)break;
   size_t at=0;
   while(at<(size_t)na){
    ssize_t nb=read(cas,b+at,(size_t)na-at);
    if(nb<0&&errno==EINTR)continue;
    if(nb<=0){ok=0;break;}
    at+=(size_t)nb;
   }
   if(!ok||memcmp(a,b,(size_t)na)!=0){ok=0;break;}
  }
  if(ok){uint8_t trailing;ssize_t n;
   do{n=read(cas,&trailing,1);}while(n<0&&errno==EINTR);
   if(n!=0)ok=0;
  }
  int close_src=close(src),close_cas=close(cas);
  if(close_src!=0||close_cas!=0)ok=0;
  if(!ok)return 0;
 }
 return 1;
}
int semantic_b2c_publish_one(const char *root,const semantic_b2a_witness *w){
 if(!root||!w||!w->trusted_base_manifest_digest||!w->base||!w->overlay||
    !w->overlay->local_builder||!w->base_segment_paths||
    !*root||strnlen(root,PATH_MAX)>=PATH_MAX)return SEMANTIC_B2B_INVALID;
 int dir=open(root,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
 if(dir<0)return SEMANTIC_B2B_IO;
 struct stat rst;int rc=SEMANTIC_B2B_INVALID;int staged_lock=-1;
 if(fstat(dir,&rst)||!S_ISDIR(rst.st_mode)||rst.st_uid!=geteuid()||
    (rst.st_mode&0077)!=0)goto done;
 /* Prevent two trusted B2c writers from racing an orphan reconciliation. */
 staged_lock=stage_lock(dir,1);
 if(staged_lock<0){rc=SEMANTIC_B2B_IO;goto done;}
 /* This first check is advisory; the actual compare-and-swap rechecks under
  * the B2b cross-process lock after all artifacts have been staged. */
 hacf_digest head={0};rc=semantic_b2b_head_read(root,&head);
 if(rc!=SEMANTIC_B2B_OK)goto done;
 if(!same(&head,w->trusted_base_manifest_digest)){rc=SEMANTIC_B2B_CONFLICT;goto done;}
 /* A pinned base must match the actual CAS bytes, not only its name. */
 if(!verify_staged_manifest(dir,w->trusted_base_manifest_digest,w->base,0)){
  rc=SEMANTIC_B2B_INVALID;goto done;
 }
 semantic_segment_record *segment=malloc(sizeof(*segment));
 semantic_snapshot_manifest *next=malloc(sizeof(*next));
 if(!segment||!next){free(segment);free(next);rc=SEMANTIC_B2B_IO;goto done;}
 rc=SEMANTIC_B2B_INVALID;
 if(semantic_b2a_prepare_successor(w,segment,next)!=SEMANTIC_OK)goto cleanup;
 /* Publication is forbidden if HEAD's historical base references are
  * absent from this store, even when external witness files verify. */
 if(!verify_base_cas_closure(dir,w))goto cleanup;
 if(segment->node_count!=1||segment->assertion_count!=1||segment->hyperedge_count||
    segment->incidence_count||segment->hacf_op_count!=1||
    next->segment_count!=w->base->segment_count+1||
    !same(&next->prior_manifest_digest,w->trusted_base_manifest_digest)||
    !same(&next->segment_digests[w->base->segment_count],&segment->segment_identity))
    goto cleanup;
 char segment_path[PATH_MAX];
 if(!safe_path(segment_path,sizeof(segment_path),root,&segment->segment_identity,"segment"))goto cleanup;
 const semantic_hypergraph_builder *builder=w->overlay->local_builder;
 int wr=semantic_segment_write(segment,builder,segment_path,NULL);
 if(wr!=SEMANTIC_OK&&wr!=SEMANTIC_E_DUPLICATE){rc=SEMANTIC_B2B_IO;goto cleanup;}
 if(!verify_staged_segment(dir,&segment->segment_identity,segment,builder,1))goto cleanup;
 if(cas_nlink(dir,&segment->segment_identity,"segment")==2){
  rc=reconcile_cas_orphan(dir,&segment->segment_identity,"segment",".tmp_segment_");
  if(rc!=SEMANTIC_B2B_OK)goto cleanup;
 }
 rc=SEMANTIC_B2B_INVALID;
 if(!verify_staged_segment(dir,&segment->segment_identity,segment,builder,0))goto cleanup;
 /* Duplicate is recoverable only if the entire existing manifest matches. */
 wr=semantic_snapshot_publish_cas(next,root,NULL);
 if(wr!=SEMANTIC_OK&&wr!=SEMANTIC_E_DUPLICATE){rc=SEMANTIC_B2B_IO;goto cleanup;}
 if(!verify_staged_manifest(dir,&next->manifest_digest,next,1))goto cleanup;
 if(cas_nlink(dir,&next->manifest_digest,"snapshot")==2){
  rc=reconcile_cas_orphan(dir,&next->manifest_digest,"snapshot",".tmp_snap_");
  if(rc!=SEMANTIC_B2B_OK)goto cleanup;
 }
 rc=SEMANTIC_B2B_INVALID;
 if(!verify_staged_manifest(dir,&next->manifest_digest,next,0))goto cleanup;
 /* Only B2b may mutate HEAD. It rechecks the old HEAD and validates the
  * appended segment while holding its cross-process lock. */
 rc=semantic_b2b_head_cas(root,w->trusted_base_manifest_digest,&next->manifest_digest);
cleanup:
 free(segment);free(next);
done:
 if(staged_lock>=0&&close(staged_lock)&&rc==SEMANTIC_B2B_OK)rc=SEMANTIC_B2B_UNCERTAIN;
 if(close(dir)&&rc==SEMANTIC_B2B_OK)rc=SEMANTIC_B2B_UNCERTAIN;
 return rc;
}

/* B2h: The bare HEAD read verifies only its manifest. A caller retaining an
 * independently trusted B2a witness can revalidate the exact admitted bytes,
 * all historical base segment CAS files, and the currently active successor.
 * This read does not grant admission or mutate HEAD / CAS. The stage lock is
 * opened in shared mode, without creation, and only exists after B2c write.
 * No witness-independent cryptographic guarantee is implied by v1 IDs. */
int semantic_b2h_verify_published_one(const char *root,
                                      const semantic_b2a_witness *w){
 if(!root||!*root||strnlen(root,PATH_MAX)>=PATH_MAX||!w||!w->base||
    !w->trusted_base_manifest_digest||!w->overlay||
    !w->overlay->local_builder)return SEMANTIC_B2B_INVALID;
 int dir=open(root,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
 if(dir<0)return SEMANTIC_B2B_IO;
 int lock=-1,rc=SEMANTIC_B2B_INVALID;
 struct stat rst;
 if(fstat(dir,&rst)||!S_ISDIR(rst.st_mode)||rst.st_uid!=geteuid()||
    (rst.st_mode&0077)!=0)goto done;
 lock=stage_lock(dir,0);
 if(lock<0)goto done;
 hacf_digest head={0};
 rc=semantic_b2b_head_read(root,&head);
 if(rc!=SEMANTIC_B2B_OK)goto done;
 rc=SEMANTIC_B2B_INVALID;
 if(!verify_staged_manifest(dir,w->trusted_base_manifest_digest,w->base,0))goto done;
 semantic_segment_record *seg=malloc(sizeof(*seg));
 semantic_snapshot_manifest *next=malloc(sizeof(*next));
 if(!seg||!next){free(seg);free(next);rc=SEMANTIC_B2B_IO;goto done;}
 if(semantic_b2a_prepare_successor(w,seg,next)!=SEMANTIC_OK)goto verify_done;
 if(!same(&head,&next->manifest_digest)||
    !verify_base_cas_closure(dir,w)||
    seg->node_count!=1||seg->assertion_count!=1||
    seg->hyperedge_count!=0||seg->incidence_count!=0||
    seg->hacf_op_count!=1||
    !verify_staged_segment(dir,&seg->segment_identity,seg,
                           w->overlay->local_builder,0)||
    !verify_staged_manifest(dir,&next->manifest_digest,next,0))goto verify_done;
 /* A B2b-only caller may race independently of the B2c staging lock.
  * Refuse success if active HEAD changed during this verification. */
 hacf_digest latest={0};
 rc=semantic_b2b_head_read(root,&latest);
 if(rc==SEMANTIC_B2B_OK)rc=same(&latest,&head)?SEMANTIC_B2B_OK:SEMANTIC_B2B_CONFLICT;
verify_done:
 free(seg);free(next);
done:
 if(lock>=0&&close(lock)&&rc==SEMANTIC_B2B_OK)rc=SEMANTIC_B2B_UNCERTAIN;
 if(close(dir)&&rc==SEMANTIC_B2B_OK)rc=SEMANTIC_B2B_UNCERTAIN;
 return rc;
}
