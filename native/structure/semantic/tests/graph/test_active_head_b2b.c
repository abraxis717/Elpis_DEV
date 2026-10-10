/* B2b: native, executable head CAS/restart/concurrent-writer tests. */
#define _GNU_SOURCE
#include "elpis_semantic/active_head_b2b.h"
#include "elpis_semantic/snapshot_publication.h"
#include "elpis/sha256.h"
#include "elpis_semantic/hypergraph.h"
#include "elpis_semantic/type_registry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"B2B_FAIL line=%d condition=%s\n",__LINE__,#x);exit(1);}}while(0)
static semantic_snapshot_manifest *allocate(void){semantic_snapshot_manifest *m=calloc(1,sizeof(*m));CHECK(m);return m;}
static void fname(char *dst,size_t cap,const char *dir,const hacf_digest *d,const char *suffix){
 char hex[65];elpis_hex32(d->bytes,hex);int n=snprintf(dst,cap,"%s/%s%s",dir,hex,suffix);CHECK(n>0&&(size_t)n<cap);
}
static void store_manifest(const char *dir,const semantic_snapshot_manifest *m){
 char hex[65];CHECK(semantic_snapshot_publish_cas(m,dir,hex)==SEMANTIC_OK);
}
static void node_builder(semantic_hypergraph_builder *b,unsigned tag){
 elpis_semantic_node_v1 n={0};n.abi_version=SEMANTIC_ABI_VERSION;
 n.node_type=SEMANTIC_NODE_NAMESPACE|1u;n.payload_digest.bytes[0]=(uint8_t)tag;
 CHECK(elpis_semantic_node_identity(&n,&n.node_identity)==SEMANTIC_OK);
 CHECK(semantic_builder_add_node(b,&n)==SEMANTIC_OK);
 /* Node-only records do not project to HACF operations. A real admitted
  * claim carries a provenance assertion, which binds the node identity
  * into the segment's native v1 graph-projection digest. */
 elpis_semantic_assertion_v1 a={0};
 a.abi_version=SEMANTIC_ABI_VERSION;
 a.asserted_object_kind=SEMANTIC_OBJECT_KIND_NODE;
 a.asserted_object_digest=n.node_identity;
 a.provenance_digest.bytes[0]=(uint8_t)(0xa0u+tag);
 a.authority=1;
 CHECK(elpis_semantic_assertion_identity(&a,&a.assertion_identity)==SEMANTIC_OK);
 CHECK(semantic_builder_add_assertion(b,&a)==SEMANTIC_OK);
}
static void stage_segment(const char *dir,semantic_type_registry *reg,
                          semantic_snapshot_manifest *m,unsigned tag){
 semantic_hypergraph_builder *b=semantic_builder_create(reg);CHECK(b);
 node_builder(b,tag);
 const hacf_digest *prev=m->segment_count?&m->hacf_graph_snapshot_digest:&m->genesis_identity;
 semantic_segment_record seg={0};
 CHECK(semantic_segment_build(b,reg,prev,&seg)==SEMANTIC_OK);
 CHECK(seg.node_count==1 && seg.assertion_count==1 && seg.hacf_op_count==1);
 char path[512];fname(path,sizeof(path),dir,&seg.segment_identity,".segment");
 CHECK(semantic_segment_write(&seg,b,path,NULL)==SEMANTIC_OK);
 CHECK(semantic_snapshot_add_segment(m,&seg)==SEMANTIC_OK);
 semantic_builder_destroy(b);
}
int main(void){
 char dir[]="./semantic-head-b2b-XXXXXX";CHECK(mkdtemp(dir));
 semantic_type_registry *reg=semantic_type_registry_create();CHECK(reg);
 semantic_node_type_entry t={.node_type=SEMANTIC_NODE_NAMESPACE|1u,
   .semantic_flag_mask=SEMANTIC_NODE_FLAG_MASK,.min_authority=0,.max_authority=3};
 CHECK(semantic_type_registry_add_node_type(reg,&t)==SEMANTIC_OK);
 hacf_digest rd={0},gen={0};CHECK(semantic_type_registry_seal(reg,&rd)==SEMANTIC_OK);
 CHECK(semantic_genesis_identity(&rd,&gen)==SEMANTIC_OK);
 semantic_snapshot_manifest *base=allocate(),*next=allocate(),*alt=allocate();
 base->abi_version=SEMANTIC_SNAPSHOT_ABI_VERSION;base->genesis_identity=gen;
 stage_segment(dir,reg,base,5u);
 CHECK(semantic_snapshot_finalize(base)==SEMANTIC_OK);
 CHECK(semantic_snapshot_validate(base)==SEMANTIC_OK);
 store_manifest(dir,base);
 hacf_digest result={0},sentinel={0};memset(sentinel.bytes,0xa5,32);result=sentinel;
 CHECK(semantic_b2b_head_read(dir,&result)==SEMANTIC_B2B_CONFLICT);
 CHECK(!memcmp(&result,&sentinel,sizeof(result)));
 CHECK(semantic_b2b_head_bootstrap(dir,&base->manifest_digest)==SEMANTIC_B2B_OK);
 CHECK(semantic_b2b_head_bootstrap(dir,&base->manifest_digest)==SEMANTIC_B2B_CONFLICT);
 CHECK(semantic_b2b_head_read(dir,&result)==SEMANTIC_B2B_OK);
 CHECK(!memcmp(&result,&base->manifest_digest,32));
 *next=*base;next->prior_manifest_digest=base->manifest_digest;
 stage_segment(dir,reg,next,6u);
 CHECK(semantic_snapshot_finalize(next)==SEMANTIC_OK);store_manifest(dir,next);
 *alt=*base;alt->prior_manifest_digest=base->manifest_digest;
 stage_segment(dir,reg,alt,7u);
 CHECK(semantic_snapshot_finalize(alt)==SEMANTIC_OK);store_manifest(dir,alt);
 /* Competing successors must have genuinely distinct content identities,
  * not merely different uncommitted node payloads outside the v1 projection. */
 CHECK(memcmp(&next->segment_digests[base->segment_count],
              &alt->segment_digests[base->segment_count],sizeof(hacf_digest))!=0);
 CHECK(memcmp(&next->manifest_digest,&alt->manifest_digest,sizeof(hacf_digest))!=0);
 CHECK(semantic_b2b_head_cas(dir,&sentinel,&next->manifest_digest)==SEMANTIC_B2B_CONFLICT);
 CHECK(semantic_b2b_head_cas(dir,&base->manifest_digest,&sentinel)==SEMANTIC_B2B_IO);
 CHECK(semantic_b2b_head_read(dir,&result)==SEMANTIC_B2B_OK);
 CHECK(!memcmp(&result,&base->manifest_digest,32));
 /* Competing processes start from the same HEAD; exactly one may commit. */
 int go[2];CHECK(pipe(go)==0);pid_t child[2]={0};
 for(int i=0;i<2;i++){
  child[i]=fork();CHECK(child[i]>=0);
  if(child[i]==0){
   close(go[1]);char b;CHECK(read(go[0],&b,1)==1);
   const hacf_digest *target=i?&alt->manifest_digest:&next->manifest_digest;
   int rc=semantic_b2b_head_cas(dir,&base->manifest_digest,target);
   _exit(rc==SEMANTIC_B2B_OK?0:rc==SEMANTIC_B2B_CONFLICT?2:5);
  }
 }
 close(go[0]);CHECK(write(go[1],"ab",2)==2);close(go[1]);
 int success=0,conflict=0;
 for(int i=0;i<2;i++){int st;CHECK(waitpid(child[i],&st,0)==child[i]);
  CHECK(WIFEXITED(st));int rc=WEXITSTATUS(st);
  success+=(rc==0);conflict+=(rc==2);
 }
 CHECK(success==1&&conflict==1);
 CHECK(semantic_b2b_head_read(dir,&result)==SEMANTIC_B2B_OK);
 CHECK(!memcmp(&result,&next->manifest_digest,32)||!memcmp(&result,&alt->manifest_digest,32));
 CHECK(semantic_b2b_head_cas(dir,&base->manifest_digest,&next->manifest_digest)==SEMANTIC_B2B_CONFLICT);
 /* Stranded temp file cannot affect restart verification. */
 char stranded[512];CHECK(snprintf(stranded,sizeof(stranded),"%s/.head-stage-orphan",dir)>0);
 int fd=open(stranded,O_WRONLY|O_CREAT|O_EXCL,0600);CHECK(fd>=0);
 CHECK(write(fd,"partial",7)==7);CHECK(close(fd)==0);
 hacf_digest reread=sentinel;CHECK(semantic_b2b_head_read(dir,&reread)==SEMANTIC_B2B_OK);
 CHECK(!memcmp(&reread,&result,32));
 /* HEAD corruption is fail-closed, never treated as missing or resettable. */
 char head[512];CHECK(snprintf(head,sizeof(head),"%s/HEAD",dir)>0);
 fd=open(head,O_RDWR);CHECK(fd>=0);uint8_t corrupt=0xff;CHECK(pwrite(fd,&corrupt,1,0)==1);CHECK(fsync(fd)==0);CHECK(close(fd)==0);
 reread=sentinel;CHECK(semantic_b2b_head_read(dir,&reread)==SEMANTIC_B2B_INVALID);
 CHECK(!memcmp(&reread,&sentinel,32));
 CHECK(semantic_b2b_head_bootstrap(dir,&base->manifest_digest)==SEMANTIC_B2B_INVALID);
 CHECK(semantic_b2b_head_cas(dir,&base->manifest_digest,&next->manifest_digest)==SEMANTIC_B2B_INVALID);
 /* HEAD symlink replacement is not an absent file or a recoverable bootstrap. */
 CHECK(unlink(head)==0);
 CHECK(symlink(".semantic-head.lock",head)==0);
 CHECK(semantic_b2b_head_read(dir,&reread)!=SEMANTIC_B2B_OK);
 CHECK(semantic_b2b_head_bootstrap(dir,&base->manifest_digest)!=SEMANTIC_B2B_OK);
 CHECK(semantic_b2b_head_cas(dir,&base->manifest_digest,&next->manifest_digest)!=SEMANTIC_B2B_OK);
 /* Cleanup test objects, not production directory. */
 CHECK(unlink(head)==0);CHECK(unlink(stranded)==0);
 char path[512];fname(path,sizeof(path),dir,&base->manifest_digest,".snapshot");CHECK(unlink(path)==0);
 fname(path,sizeof(path),dir,&next->manifest_digest,".snapshot");CHECK(unlink(path)==0);
 fname(path,sizeof(path),dir,&alt->manifest_digest,".snapshot");CHECK(unlink(path)==0);
 for(unsigned i=0;i<3;i++){
  const semantic_snapshot_manifest *m=i==0?base:i==1?next:alt;
  uint32_t idx=i==0?0:1;
  fname(path,sizeof(path),dir,&m->segment_digests[idx],".segment");CHECK(unlink(path)==0);
 }
 CHECK(snprintf(path,sizeof(path),"%s/.semantic-head.lock",dir)>0);CHECK(unlink(path)==0);
 CHECK(rmdir(dir)==0);
 free(base);free(next);free(alt);semantic_type_registry_destroy(reg);
 puts("PASS_B2B_NATIVE_HEAD_CAS_CONCURRENCY_RECOVERY");return 0;
}
