/* B1j adversarial native read-only empty-genesis-chain proof tests. */
#define _POSIX_C_SOURCE 200809L
#include "elpis_semantic/admission_base_b1j.h"
#include "elpis_semantic/type_registry.h"
#include "elpis_semantic/hypergraph.h"
#include "elpis_semantic/identity.h"
#include "elpis/sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#define CHECK(x) do { if(!(x)){fprintf(stderr,"B1J_FAIL line=%d check=%s\n",__LINE__,#x);return 1;} }while(0)
static void fill(hacf_digest *d,unsigned char c){memset(d->bytes,c,32);}
static int equal(const hacf_digest *a,const hacf_digest *b){return memcmp(a,b,sizeof(*a))==0;}
static int audit(const semantic_snapshot_manifest *m,const hacf_digest *pin,const char *p,hacf_digest *out){
    const char *paths[]={p};return semantic_b1j_empty_base_chain_audit(m,pin,paths,1,out);
}
static int run(void) {
    char dir[]="/tmp/elpis-b1j-XXXXXX";
    CHECK(mkdtemp(dir)!=NULL);
    char path[512]; CHECK(snprintf(path,sizeof(path),"%s/empty.segment",dir)>0);
    semantic_type_registry *registry=semantic_type_registry_create(); CHECK(registry);
    semantic_node_type_entry nt={0};nt.node_type=SEMANTIC_NODE_NAMESPACE|1;
    nt.semantic_flag_mask=SEMANTIC_NODE_FLAG_MASK;nt.max_authority=3;
    CHECK(semantic_type_registry_add_node_type(registry,&nt)==SEMANTIC_OK);
    CHECK(semantic_type_registry_seal(registry,NULL)==SEMANTIC_OK);
    semantic_hypergraph_builder *builder=semantic_builder_create(registry); CHECK(builder);
    semantic_snapshot_manifest *manifest=semantic_snapshot_create();CHECK(manifest);
    CHECK(semantic_type_registry_digest(registry,&manifest->type_registry_digest)==SEMANTIC_OK);
    CHECK(semantic_genesis_identity(&manifest->type_registry_digest,&manifest->genesis_identity)==SEMANTIC_OK);
    semantic_segment_record seg;
    CHECK(semantic_segment_build(builder,registry,&manifest->genesis_identity,&seg)==SEMANTIC_OK);
    CHECK(seg.node_count==0 && seg.assertion_count==0 && seg.hyperedge_count==0 && seg.incidence_count==0 && seg.hacf_op_count==0);
    CHECK(semantic_segment_write(&seg,builder,path,NULL)==SEMANTIC_OK);
    CHECK(semantic_snapshot_add_segment(manifest,&seg)==SEMANTIC_OK);
    CHECK(semantic_snapshot_finalize(manifest)==SEMANTIC_OK);
    CHECK(semantic_snapshot_validate(manifest)==SEMANTIC_OK);
    hacf_digest pin=manifest->manifest_digest,report,again,sentinel;
    fill(&sentinel,0xa5);
    report=sentinel;CHECK(audit(manifest,&pin,path,&report)==SEMANTIC_OK);
    CHECK(!equal(&report,&sentinel));
    again=sentinel;CHECK(audit(manifest,&pin,path,&again)==SEMANTIC_OK && equal(&report,&again));
#define REFUSE() do{again=sentinel;CHECK(audit(manifest,&pin,path,&again)!=SEMANTIC_OK);CHECK(equal(&again,&sentinel));}while(0)
    fill(&pin,0x7f);REFUSE();pin=manifest->manifest_digest;
    manifest->prior_manifest_digest=pin;
    CHECK(semantic_snapshot_finalize(manifest)==SEMANTIC_OK);
    pin=manifest->manifest_digest;REFUSE();
    memset(&manifest->prior_manifest_digest,0,sizeof(manifest->prior_manifest_digest));
    CHECK(semantic_snapshot_finalize(manifest)==SEMANTIC_OK);pin=manifest->manifest_digest;
    manifest->unique_node_count=1;
    CHECK(semantic_snapshot_finalize(manifest)==SEMANTIC_OK);pin=manifest->manifest_digest;REFUSE();
    manifest->unique_node_count=0;CHECK(semantic_snapshot_finalize(manifest)==SEMANTIC_OK);pin=manifest->manifest_digest;
    fill(&manifest->genesis_identity,0x43);
    CHECK(semantic_snapshot_finalize(manifest)==SEMANTIC_OK);pin=manifest->manifest_digest;REFUSE();
    CHECK(semantic_genesis_identity(&manifest->type_registry_digest,&manifest->genesis_identity)==SEMANTIC_OK);
    CHECK(semantic_snapshot_finalize(manifest)==SEMANTIC_OK);pin=manifest->manifest_digest;
    fill(&manifest->segment_digests[0],0x34);
    CHECK(semantic_snapshot_finalize(manifest)==SEMANTIC_OK);pin=manifest->manifest_digest;REFUSE();
    manifest->segment_digests[0]=seg.segment_identity;
    CHECK(semantic_snapshot_finalize(manifest)==SEMANTIC_OK);pin=manifest->manifest_digest;
    /* Raw file modification must be rejected independently of manifest rehash. */
    int fd=open(path,O_RDWR); CHECK(fd>=0);
    unsigned char raw;CHECK(pread(fd,&raw,1,0)==1);
    unsigned char corrupted=(unsigned char)(raw^0x80);
    CHECK(pwrite(fd,&corrupted,1,0)==1);REFUSE();
    CHECK(pwrite(fd,&raw,1,0)==1);CHECK(close(fd)==0);
    again=sentinel;CHECK(audit(manifest,&pin,path,&again)==SEMANTIC_OK && equal(&report,&again));
    fd=open(path,O_WRONLY|O_APPEND);CHECK(fd>=0);
    CHECK(write(fd,"x",1)==1);CHECK(close(fd)==0);REFUSE();
    struct stat st;CHECK(stat(path,&st)==0 && st.st_size>0);
    CHECK(truncate(path,st.st_size-1)==0);
    again=sentinel;CHECK(audit(manifest,&pin,path,&again)==SEMANTIC_OK && equal(&report,&again));
    char sym[512];CHECK(snprintf(sym,sizeof(sym),"%s/alias",dir)>0);
    CHECK(symlink(path,sym)==0);
    again=sentinel;CHECK(audit(manifest,&pin,sym,&again)!=SEMANTIC_OK && equal(&again,&sentinel));
    CHECK(unlink(sym)==0);
    /* Output alias with pinned authority must be rejected without mutation. */
    hacf_digest before=pin;
    CHECK(audit(manifest,&pin,path,&pin)!=SEMANTIC_OK && equal(&pin,&before));
    CHECK(unlink(path)==0);CHECK(rmdir(dir)==0);
    semantic_snapshot_destroy(manifest);
    semantic_builder_destroy(builder);
    semantic_type_registry_destroy(registry);
    puts("B1J_NATIVE_GENESIS_EMPTY_CHAIN_PASS");return 0;
}
int main(void){return run();}
