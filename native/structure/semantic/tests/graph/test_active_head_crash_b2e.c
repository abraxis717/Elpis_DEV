/* B2e: executable Linux process-crash/errno fault qualification for real B2b
 * active-head CAS. Link-time wrappers are TEST ONLY; production code is intact.
 * Process SIGKILL does not model a power cut, cache flush, or a disk defect. */
#define _GNU_SOURCE
#include "elpis_semantic/active_head_b2b.h"
#include "elpis_semantic/snapshot_publication.h"
#include "elpis_semantic/hypergraph.h"
#include "elpis_semantic/type_registry.h"
#include "elpis/sha256.h"
#include <errno.h>
#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"B2E_FAIL line=%d check=%s\n",__LINE__,#x); exit(1); } } while(0)

enum inject_mode {
    CRASH_PRE_FILE_FSYNC=1, CRASH_POST_FILE_FSYNC,
    CRASH_PRE_RENAME, CRASH_POST_RENAME,
    CRASH_PRE_DIR_FSYNC, CRASH_POST_DIR_FSYNC,
    ERROR_FILE_FSYNC, ERROR_RENAME, ERROR_DIR_FSYNC
};
static int armed=0;
static unsigned fsync_count=0,rename_count=0;
static enum inject_mode mode=0;
int __real_fsync(int fd);
int __real_renameat(int oldfd,const char *oldname,int newfd,const char *newname);
static void die(void) { kill(getpid(),SIGKILL); _exit(99); }
int __wrap_fsync(int fd) {
    if (!armed) return __real_fsync(fd);
    ++fsync_count;
    if (fsync_count==1) {
        if (mode==CRASH_PRE_FILE_FSYNC) die();
        if (mode==ERROR_FILE_FSYNC) { errno=EIO; return -1; }
        if (mode==CRASH_POST_FILE_FSYNC) {
            int rc=__real_fsync(fd); if (rc) _exit(97); die();
        }
    }
    if (fsync_count==2) {
        if (mode==CRASH_PRE_DIR_FSYNC) die();
        if (mode==ERROR_DIR_FSYNC) { errno=EIO; return -1; }
        if (mode==CRASH_POST_DIR_FSYNC) {
            int rc=__real_fsync(fd); if (rc) _exit(96); die();
        }
    }
    return __real_fsync(fd);
}
int __wrap_renameat(int oldfd,const char *oldname,int newfd,const char *newname) {
    if (!armed) return __real_renameat(oldfd,oldname,newfd,newname);
    ++rename_count;
    if (mode==CRASH_PRE_RENAME) die();
    if (mode==ERROR_RENAME) { errno=EIO; return -1; }
    if (mode==CRASH_POST_RENAME) {
        int rc=__real_renameat(oldfd,oldname,newfd,newname);
        if (rc) _exit(95);
        die();
    }
    return __real_renameat(oldfd,oldname,newfd,newname);
}
static void cas_name(char *dst,size_t n,const char *root,const hacf_digest *d,const char *suffix) {
    char hex[65];elpis_hex32(d->bytes,hex);
    int k=snprintf(dst,n,"%s/%s.%s",root,hex,suffix); CHECK(k>0&&(size_t)k<n);
}
static void stage_one(const char *dir,semantic_type_registry *reg,semantic_snapshot_manifest *manifest,unsigned token) {
    semantic_hypergraph_builder *builder=semantic_builder_create(reg); CHECK(builder);
    elpis_semantic_node_v1 node={0};
    node.abi_version=SEMANTIC_ABI_VERSION;
    node.node_type=SEMANTIC_NODE_NAMESPACE|1u;
    node.payload_digest.bytes[0]=(uint8_t)token;
    CHECK(elpis_semantic_node_identity(&node,&node.node_identity)==SEMANTIC_OK);
    CHECK(semantic_builder_add_node(builder,&node)==SEMANTIC_OK);
    elpis_semantic_assertion_v1 a={0};a.abi_version=SEMANTIC_ABI_VERSION;
    a.asserted_object_kind=SEMANTIC_OBJECT_KIND_NODE;
    a.asserted_object_digest=node.node_identity;
    a.provenance_digest.bytes[0]=(uint8_t)(token+7u);a.authority=1;
    CHECK(elpis_semantic_assertion_identity(&a,&a.assertion_identity)==SEMANTIC_OK);
    CHECK(semantic_builder_add_assertion(builder,&a)==SEMANTIC_OK);
    hacf_digest prior=manifest->segment_count?manifest->hacf_graph_snapshot_digest:manifest->genesis_identity;
    semantic_segment_record seg={0};
    CHECK(semantic_segment_build(builder,reg,&prior,&seg)==SEMANTIC_OK);
    CHECK(seg.node_count==1&&seg.assertion_count==1&&seg.hacf_op_count==1);
    char path[512];cas_name(path,sizeof(path),dir,&seg.segment_identity,"segment");
    CHECK(semantic_segment_write(&seg,builder,path,NULL)==SEMANTIC_OK);
    CHECK(semantic_snapshot_add_segment(manifest,&seg)==SEMANTIC_OK);
    semantic_builder_destroy(builder);
}
static void store_manifest(const char *root,const semantic_snapshot_manifest *m) {
    CHECK(semantic_snapshot_publish_cas(m,root,NULL)==SEMANTIC_OK);
}
static void cleanup_fixture(const char *root) {
    DIR *dp=opendir(root); CHECK(dp);
    struct dirent *de;char path[512];
    while((de=readdir(dp))!=NULL) {
        if(!strcmp(de->d_name,".")||!strcmp(de->d_name,".."))continue;
        int n=snprintf(path,sizeof(path),"%s/%s",root,de->d_name);
        CHECK(n>0&&(size_t)n<sizeof(path));
        CHECK(unlink(path)==0);
    }
    CHECK(closedir(dp)==0);CHECK(rmdir(root)==0);
}
/* Reading in a freshly forked process prevents the parent test from relying
 * on a cached pointer/struct representing HEAD. */
static void check_restart_read(const char *root,const hacf_digest *expected) {
    pid_t pid=fork();CHECK(pid>=0);
    if(pid==0){
        hacf_digest out={0};
        int rc=semantic_b2b_head_read(root,&out);
        _exit(rc==SEMANTIC_B2B_OK&&memcmp(&out,expected,sizeof(out))==0?0:71);
    }
    int st=0;CHECK(waitpid(pid,&st,0)==pid);
    CHECK(WIFEXITED(st)&&WEXITSTATUS(st)==0);
}
static void run_case(enum inject_mode how) {
    char root[]="./head-crash-b2e-XXXXXX";CHECK(mkdtemp(root));
    semantic_type_registry *reg=semantic_type_registry_create();CHECK(reg);
    semantic_node_type_entry t={.node_type=SEMANTIC_NODE_NAMESPACE|1u,
      .semantic_flag_mask=SEMANTIC_NODE_FLAG_MASK,.min_authority=0,.max_authority=3};
    CHECK(semantic_type_registry_add_node_type(reg,&t)==SEMANTIC_OK);
    hacf_digest registry={0},genesis={0};
    CHECK(semantic_type_registry_seal(reg,&registry)==SEMANTIC_OK);
    CHECK(semantic_genesis_identity(&registry,&genesis)==SEMANTIC_OK);
    semantic_snapshot_manifest *base=calloc(1,sizeof(*base));
    semantic_snapshot_manifest *next=calloc(1,sizeof(*next));CHECK(base&&next);
    base->abi_version=SEMANTIC_SNAPSHOT_ABI_VERSION;base->genesis_identity=genesis;
    stage_one(root,reg,base,8u);
    CHECK(semantic_snapshot_finalize(base)==SEMANTIC_OK);
    store_manifest(root,base);
    CHECK(semantic_b2b_head_bootstrap(root,&base->manifest_digest)==SEMANTIC_B2B_OK);
    *next=*base;next->prior_manifest_digest=base->manifest_digest;
    stage_one(root,reg,next,9u);
    CHECK(semantic_snapshot_finalize(next)==SEMANTIC_OK);
    store_manifest(root,next);
    /* Fault is armed only inside the child, after all durable artifacts exist. */
    pid_t pid=fork();CHECK(pid>=0);
    if(pid==0) {
        mode=how;fsync_count=0;rename_count=0;armed=1;
        int rc=semantic_b2b_head_cas(root,&base->manifest_digest,&next->manifest_digest);
        armed=0;
        if(how==ERROR_FILE_FSYNC||how==ERROR_RENAME)
            _exit(rc==SEMANTIC_B2B_IO?0:72);
        if(how==ERROR_DIR_FSYNC)
            _exit(rc==SEMANTIC_B2B_UNCERTAIN?0:73);
        _exit(74); /* Every crash mode must actually hit its target. */
    }
    int st=0;CHECK(waitpid(pid,&st,0)==pid);
    int crashed=how<=CRASH_POST_DIR_FSYNC;
    if(crashed)CHECK(WIFSIGNALED(st)&&WTERMSIG(st)==SIGKILL);
    else CHECK(WIFEXITED(st)&&WEXITSTATUS(st)==0);
    /* renameat is the linearization point; pre-rename leaves the old head. */
    int new_head=(how==CRASH_POST_RENAME||how==CRASH_PRE_DIR_FSYNC||
                  how==CRASH_POST_DIR_FSYNC||how==ERROR_DIR_FSYNC);
    const hacf_digest *expected=new_head?&next->manifest_digest:&base->manifest_digest;
    check_restart_read(root,expected);
    if(new_head){
        CHECK(semantic_b2b_head_cas(root,&base->manifest_digest,&next->manifest_digest)==SEMANTIC_B2B_CONFLICT);
    }else{
        /* Orphaned temporary files do not prevent safe retry from old HEAD. */
        CHECK(semantic_b2b_head_cas(root,&base->manifest_digest,&next->manifest_digest)==SEMANTIC_B2B_OK);
        check_restart_read(root,&next->manifest_digest);
    }
    cleanup_fixture(root);free(base);free(next);semantic_type_registry_destroy(reg);
    printf("B2E_CASE_PASS mode=%u\n",(unsigned)how);fflush(stdout);
}
int main(void) {
    for(unsigned k=CRASH_PRE_FILE_FSYNC;k<=ERROR_DIR_FSYNC;k++)run_case((enum inject_mode)k);
    puts("PASS_B2E_NATIVE_PROCESS_CRASH_HEAD_RECOVERY");return 0;
}
