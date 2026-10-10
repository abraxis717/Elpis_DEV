/* B1m: actual two-segment populated genesis chain, verified on disk. */
#define _POSIX_C_SOURCE 200809L
#include "elpis_semantic/admission_populated_b1l.h"
#include "elpis_semantic/type_registry.h"
#include "elpis_semantic/hypergraph.h"
#include "elpis/sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#define CHECK(x) do {if(!(x)){fprintf(stderr,"B1M_FAIL line=%d expr=%s\n",__LINE__,#x);return 1;}}while(0)
static int eq(const hacf_digest *a,const hacf_digest *b){return memcmp(a,b,sizeof(*a))==0;}
static void fill(hacf_digest *d,unsigned char c){memset(d->bytes,c,sizeof(d->bytes));}
static void node(elpis_semantic_node_v1 *n,const char *value){
 memset(n,0,sizeof(*n));n->abi_version=SEMANTIC_ABI_VERSION;
 n->node_type=SEMANTIC_NODE_NAMESPACE|1;
 elpis_sha256(value,strlen(value),n->payload_digest.bytes);
 (void)elpis_semantic_node_identity(n,&n->node_identity);
}
static int run(void){
 char dir[]="/tmp/elpis-b1m-XXXXXX";CHECK(mkdtemp(dir));
 char path0[512],path1[512],path_dup[512],sym[512];
 CHECK(snprintf(path0,sizeof(path0),"%s/first.segment",dir)>0);
 CHECK(snprintf(path1,sizeof(path1),"%s/second.segment",dir)>0);
 CHECK(snprintf(path_dup,sizeof(path_dup),"%s/duplicate.segment",dir)>0);
 CHECK(snprintf(sym,sizeof(sym),"%s/symlink.segment",dir)>0);
 semantic_type_registry *reg=semantic_type_registry_create();CHECK(reg);
 semantic_node_type_entry t={0};t.node_type=SEMANTIC_NODE_NAMESPACE|1;
 t.semantic_flag_mask=SEMANTIC_NODE_FLAG_MASK;t.max_authority=3;
 CHECK(semantic_type_registry_add_node_type(reg,&t)==SEMANTIC_OK);
 CHECK(semantic_type_registry_seal(reg,NULL)==SEMANTIC_OK);
 semantic_hypergraph_builder *a=semantic_builder_create(reg), *b=semantic_builder_create(reg), *dup=semantic_builder_create(reg);
 CHECK(a && b && dup);
 elpis_semantic_node_v1 na,nb,missing;node(&na,"first");node(&nb,"second");node(&missing,"absent");
 CHECK(semantic_builder_add_node(a,&na)==SEMANTIC_BUILDER_OK);
 CHECK(semantic_builder_add_node(b,&nb)==SEMANTIC_BUILDER_OK);
 CHECK(semantic_builder_add_node(dup,&na)==SEMANTIC_BUILDER_OK);
 semantic_snapshot_manifest *m=semantic_snapshot_create(),*bad=semantic_snapshot_create();CHECK(m&&bad);
 CHECK(semantic_type_registry_digest(reg,&m->type_registry_digest)==SEMANTIC_OK);
 CHECK(semantic_genesis_identity(&m->type_registry_digest,&m->genesis_identity)==SEMANTIC_OK);
 semantic_segment_record s0,s1,sdup;
 CHECK(semantic_segment_build(a,reg,&m->genesis_identity,&s0)==SEMANTIC_OK);
 CHECK(semantic_segment_write(&s0,a,path0,NULL)==SEMANTIC_OK);
 CHECK(semantic_snapshot_add_segment(m,&s0)==SEMANTIC_OK);
 CHECK(semantic_segment_build(b,reg,&s0.hacf_next_snapshot,&s1)==SEMANTIC_OK);
 CHECK(semantic_segment_write(&s1,b,path1,NULL)==SEMANTIC_OK);
 CHECK(semantic_snapshot_add_segment(m,&s1)==SEMANTIC_OK);
 CHECK(semantic_snapshot_finalize(m)==SEMANTIC_OK);
 CHECK(m->segment_count==2 && m->unique_node_count==2);
 const char *paths[2]={path0,path1};
 hacf_digest pin=m->manifest_digest,first,sentinel,report;
 fill(&sentinel,0x7e);
 uint32_t present=77;
 CHECK(semantic_b1m_populated_chain_node_audit(m,&pin,paths,2,&na.node_identity,&present,&first)==SEMANTIC_OK);
 CHECK(present==1 && !eq(&first,&sentinel));
 present=77;report=sentinel;
 CHECK(semantic_b1m_populated_chain_node_audit(m,&pin,paths,2,&nb.node_identity,&present,&report)==SEMANTIC_OK);
 CHECK(present==1);
 present=77;report=sentinel;
 CHECK(semantic_b1m_populated_chain_node_audit(m,&pin,paths,2,&missing.node_identity,&present,&report)==SEMANTIC_OK);
 CHECK(present==0 && !eq(&report,&first));
#define REFUSE(man,pinptr,list,target) do {present=77;report=sentinel;CHECK(semantic_b1m_populated_chain_node_audit((man),(pinptr),(list),2,(target),&present,&report)!=SEMANTIC_OK);CHECK(present==77 && eq(&report,&sentinel));}while(0)
 fill(&pin,0x34);REFUSE(m,&pin,paths,&na.node_identity);pin=m->manifest_digest;
 const char *swapped[2]={path1,path0};REFUSE(m,&pin,swapped,&na.node_identity);
 REFUSE(m,&pin,paths,&(hacf_digest){{0}});
 /* Rehash a forged manifest, proving the rejection isn't just a stale pin. */
 *bad=*m;bad->unique_node_count=3;CHECK(semantic_snapshot_finalize(bad)==SEMANTIC_OK);
 hacf_digest forged=bad->manifest_digest;REFUSE(bad,&forged,paths,&na.node_identity);
 *bad=*m;bad->prior_manifest_digest=pin;CHECK(semantic_snapshot_finalize(bad)==SEMANTIC_OK);
 forged=bad->manifest_digest;REFUSE(bad,&forged,paths,&na.node_identity);
 /* Independently valid files cannot be replayed out of graph-chain order. */
 CHECK(semantic_segment_build(dup,reg,&s0.hacf_next_snapshot,&sdup)==SEMANTIC_OK);
 CHECK(semantic_segment_write(&sdup,dup,path_dup,NULL)==SEMANTIC_OK);
 *bad=*m;bad->segment_digests[1]=sdup.segment_identity;
 bad->hacf_graph_snapshot_digest=sdup.hacf_next_snapshot;
 CHECK(semantic_snapshot_finalize(bad)==SEMANTIC_OK);forged=bad->manifest_digest;
 const char *duplicate[2]={path0,path_dup};
 REFUSE(bad,&forged,duplicate,&na.node_identity);
 int fd=open(path1,O_RDWR);CHECK(fd>=0);unsigned char saved=0;
 CHECK(pread(fd,&saved,1,0)==1);
 unsigned char flip=(unsigned char)(saved^0x80);
 CHECK(pwrite(fd,&flip,1,0)==1);
 REFUSE(m,&pin,paths,&na.node_identity);
 CHECK(pwrite(fd,&saved,1,0)==1);CHECK(close(fd)==0);
 CHECK(symlink(path1,sym)==0);const char *symlinks[2]={path0,sym};
 REFUSE(m,&pin,symlinks,&na.node_identity);CHECK(unlink(sym)==0);
 /* Outputs must not alias trusted pinned input. */
 hacf_digest before=pin;
 present=77;
 CHECK(semantic_b1m_populated_chain_node_audit(m,&pin,paths,2,&na.node_identity,&present,&pin)!=SEMANTIC_OK);
 CHECK(eq(&pin,&before));
 present=77;report=sentinel;
 CHECK(semantic_b1m_populated_chain_node_audit(m,&pin,paths,2,&na.node_identity,&present,&report)==SEMANTIC_OK);
 CHECK(present==1 && eq(&report,&first));
 CHECK(unlink(path0)==0 && unlink(path1)==0 && unlink(path_dup)==0 && rmdir(dir)==0);
 semantic_snapshot_destroy(m);semantic_snapshot_destroy(bad);
 semantic_builder_destroy(a);semantic_builder_destroy(b);semantic_builder_destroy(dup);
 semantic_type_registry_destroy(reg);
 puts("B1M_NATIVE_POPULATED_MULTISEGMENT_PASS");return 0;
}
int main(void){return run();}
