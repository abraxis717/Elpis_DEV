/* B1l: real on-disk populated genesis segment, no injected records. */
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
#define CHECK(x) do {if(!(x)){fprintf(stderr,"B1L_FAIL line=%d expr=%s\n",__LINE__,#x);return 1;}}while(0)
static void fill(hacf_digest *d,unsigned char c){memset(d->bytes,c,HACF_DIGEST_BYTES);}
static int eq(const hacf_digest *a,const hacf_digest *b){return memcmp(a->bytes,b->bytes,32)==0;}
static void make_node(elpis_semantic_node_v1 *n,const char *s){
    memset(n,0,sizeof(*n));n->abi_version=SEMANTIC_ABI_VERSION;
    n->node_type=SEMANTIC_NODE_NAMESPACE|1;
    elpis_sha256(s,strlen(s),n->payload_digest.bytes);
    (void)elpis_semantic_node_identity(n,&n->node_identity);
}
static int run(void){
    char dir[]="/tmp/elpis-b1l-XXXXXX";CHECK(mkdtemp(dir)!=NULL);
    char path[512];CHECK(snprintf(path,sizeof(path),"%s/populated.segment",dir)>0);
    semantic_type_registry *reg=semantic_type_registry_create();CHECK(reg);
    semantic_node_type_entry t={0};t.node_type=SEMANTIC_NODE_NAMESPACE|1;
    t.semantic_flag_mask=SEMANTIC_NODE_FLAG_MASK;t.max_authority=3;
    CHECK(semantic_type_registry_add_node_type(reg,&t)==SEMANTIC_OK);
    CHECK(semantic_type_registry_seal(reg,NULL)==SEMANTIC_OK);
    semantic_hypergraph_builder *builder=semantic_builder_create(reg);CHECK(builder);
    elpis_semantic_node_v1 a,b,other;make_node(&a,"alpha");make_node(&b,"beta");make_node(&other,"not-in-base");
    CHECK(semantic_builder_add_node(builder,&a)==SEMANTIC_BUILDER_OK);
    CHECK(semantic_builder_add_node(builder,&b)==SEMANTIC_BUILDER_OK);
    semantic_snapshot_manifest *m=semantic_snapshot_create();CHECK(m);
    CHECK(semantic_type_registry_digest(reg,&m->type_registry_digest)==SEMANTIC_OK);
    CHECK(semantic_genesis_identity(&m->type_registry_digest,&m->genesis_identity)==SEMANTIC_OK);
    semantic_segment_record seg;
    CHECK(semantic_segment_build(builder,reg,&m->genesis_identity,&seg)==SEMANTIC_OK);
    CHECK(seg.node_count==2);
    CHECK(semantic_segment_write(&seg,builder,path,NULL)==SEMANTIC_OK);
    CHECK(semantic_snapshot_add_segment(m,&seg)==SEMANTIC_OK);
    CHECK(semantic_snapshot_finalize(m)==SEMANTIC_OK);
    hacf_digest pin=m->manifest_digest,first,second,sentinel;
    uint32_t present=44;fill(&sentinel,0x91);first=sentinel;
    CHECK(semantic_b1l_populated_genesis_node_audit(m,&pin,path,&a.node_identity,&present,&first)==SEMANTIC_OK);
    CHECK(present==1 && !eq(&first,&sentinel));
    present=44;second=sentinel;
    CHECK(semantic_b1l_populated_genesis_node_audit(m,&pin,path,&other.node_identity,&present,&second)==SEMANTIC_OK);
    CHECK(present==0 && !eq(&second,&first));
    present=44;second=sentinel;
    CHECK(semantic_b1l_populated_genesis_node_audit(m,&pin,path,&b.node_identity,&present,&second)==SEMANTIC_OK);
    CHECK(present==1);
#define REFUSE(target) do{present=44;second=sentinel;CHECK(semantic_b1l_populated_genesis_node_audit(m,&pin,path,(target),&present,&second)!=SEMANTIC_OK);CHECK(present==44 && eq(&second,&sentinel));}while(0)
    fill(&pin,0x52);REFUSE(&a.node_identity);pin=m->manifest_digest;
    m->unique_node_count=1;CHECK(semantic_snapshot_finalize(m)==SEMANTIC_OK);pin=m->manifest_digest;REFUSE(&a.node_identity);
    m->unique_node_count=2;CHECK(semantic_snapshot_finalize(m)==SEMANTIC_OK);pin=m->manifest_digest;
    m->prior_manifest_digest=pin;CHECK(semantic_snapshot_finalize(m)==SEMANTIC_OK);pin=m->manifest_digest;REFUSE(&a.node_identity);
    memset(&m->prior_manifest_digest,0,sizeof(m->prior_manifest_digest));CHECK(semantic_snapshot_finalize(m)==SEMANTIC_OK);pin=m->manifest_digest;
    int fd=open(path,O_RDWR);CHECK(fd>=0);unsigned char saved=0;CHECK(pread(fd,&saved,1,0)==1);
    unsigned char changed=(unsigned char)(saved^0x80);CHECK(pwrite(fd,&changed,1,0)==1);
    REFUSE(&a.node_identity);
    CHECK(pwrite(fd,&saved,1,0)==1);CHECK(close(fd)==0);
    present=44;second=sentinel;
    CHECK(semantic_b1l_populated_genesis_node_audit(m,&pin,path,&a.node_identity,&present,&second)==SEMANTIC_OK);
    CHECK(present==1 && eq(&second,&first));
    char sym[512];CHECK(snprintf(sym,sizeof(sym),"%s/sym",dir)>0);
    CHECK(symlink(path,sym)==0);
    present=44;second=sentinel;
    CHECK(semantic_b1l_populated_genesis_node_audit(m,&pin,sym,&a.node_identity,&present,&second)!=SEMANTIC_OK);
    CHECK(present==44 && eq(&second,&sentinel));CHECK(unlink(sym)==0);
    /* Output alias must be refused without modifying pinned identity. */
    hacf_digest before=pin;
    CHECK(semantic_b1l_populated_genesis_node_audit(m,&pin,path,&a.node_identity,&present,&pin)!=SEMANTIC_OK);
    CHECK(eq(&pin,&before));
    CHECK(unlink(path)==0 && rmdir(dir)==0);
    semantic_snapshot_destroy(m);semantic_builder_destroy(builder);semantic_type_registry_destroy(reg);
    puts("B1L_NATIVE_POPULATED_SINGLE_SEGMENT_PASS");return 0;
}
int main(void){return run();}
