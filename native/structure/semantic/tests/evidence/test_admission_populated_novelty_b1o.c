#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
/* B1i byte-identity materialization witness built on B1h end-to-end claim/actual retrieval package composition fixture.
 * The complete chain is materialized; downstream audits are NOT mocked.
 */
#include "elpis_semantic/admission_populated_b1o.h"
#include "elpis_semantic/admission_successor_b2a.h"
#include "elpis_semantic/admission_commit_b2c.h"
#include "elpis_semantic/snapshot_publication.h"
#include "elpis_semantic/type_registry.h"
#include "elpis_semantic/hypergraph.h"
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <dirent.h>
#include <signal.h>
#include <sys/wait.h>
#include <stddef.h>
#include <sys/stat.h>
#include "elpis/sha256.h"
#include "elpis/retrieval_bundle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if(!(x)){fprintf(stderr,"B1K_FAIL line=%d check=%s\n",__LINE__,#x);return 1;} } while(0)
static void fill(hacf_digest *d,uint8_t c){memset(d->bytes,c,32);}
static const uint8_t canonical_payload[]="single canonical claim payload bytes";
static void node(elpis_semantic_node_v1 *n){
 memset(n,0,sizeof(*n));n->abi_version=SEMANTIC_ABI_VERSION;
 n->node_type=SEMANTIC_NODE_NAMESPACE|1;elpis_sha256(canonical_payload,sizeof(canonical_payload)-1,n->payload_digest.bytes);
 (void)elpis_semantic_node_identity(n,&n->node_identity);
}
static void hex(const hacf_digest *d,char s[65]){elpis_hex32(d->bytes,s);}
static void hex_bytes(const uint8_t *p,size_t n,char *s){
 static const char h[]="0123456789abcdef";
 for(size_t i=0;i<n;i++){s[i*2]=h[p[i]>>4];s[i*2+1]=h[p[i]&15];}
 s[n*2]=0;
}
static int make_json(char *out,size_t cap,
                     const elpis_retrieval_item_attachment_v1 *a,
                     const uint8_t *text,size_t text_n){
 char chunk[65],doc[65],parent[65],td[65],corp[65],policy[65],graph[65],query[65],index[65];
 char texthex[512];
 if(text_n>255)return -1;
 hex_bytes(text,text_n,texthex);
 hex(&a->chunk_digest,chunk);hex(&a->document_digest,doc);
 hex(&a->graph_parent_digest,parent);hex(&a->text_digest,td);
 hacf_digest d;fill(&d,0x11);hex(&d,corp);fill(&d,0x17);hex(&d,policy);
 fill(&d,0);hex(&d,graph);fill(&d,0x13);hex(&d,query);fill(&d,0x19);hex(&d,index);
 return snprintf(out,cap,
  "{\"abi_version\":1,\"authority_filter\":null,\"corpus_manifest_digest\":\"%s\",\"fusion_policy_digest\":\"%s\",\"graph_snapshot_digest\":\"%s\",\"items\":[{\"authority\":\"reference\",\"chunk_digest\":\"%s\",\"dense_rank\":2,\"dense_score_key\":-12,\"doc_digest\":\"%s\",\"edge_authority\":0,\"edge_type\":0,\"final_rank\":0,\"fusion_score_key\":500,\"graph_hop\":0,\"graph_parent_digest\":\"%s\",\"item_kind\":1,\"lexical_rank\":1,\"namespace_hex\":\"7265706f\",\"source_mask\":3,\"text_hex\":\"%s\",\"text_bytes\":%zu,\"text_digest\":\"%s\"}],\"namespace_filter_hex\":null,\"query_digest\":\"%s\",\"query_text_hex\":\"6173\",\"schema\":\"%s\",\"vector_index_manifest_digest\":\"%s\"}",
  corp,policy,graph,chunk,doc,parent,texthex,text_n,td,query,ELPIS_RETRIEVAL_BUNDLE_SCHEMA,index);
}
static hacf_digest package(const uint8_t *raw,size_t len){
 hacf_digest p={0},deps[2],schema,query,pol;
 fill(&query,0x13);fill(&pol,0x17);fill(&deps[0],0x11);fill(&deps[1],0x19);
 elpis_sha256(ELPIS_RETRIEVAL_BUNDLE_SCHEMA,strlen(ELPIS_RETRIEVAL_BUNDLE_SCHEMA),schema.bytes);
 hacf_package_spec spec={0};
 spec.abi_version=1;spec.object_type=HACF_OBJ_RETRIEVAL_BUNDLE;
 spec.schema_version=1;spec.authority=HACF_AUTH_REFERENCE;
 spec.schema_digest=schema;spec.policy_digest=pol;
 spec.parents=&query;spec.parent_count=1;spec.dependencies=deps;spec.dependency_count=2;
 spec.payload=raw;spec.payload_bytes=len;
 if(hacf_digest_package(&spec,&p))memset(&p,0,sizeof(p));
 return p;
}
static void catalog_build(uint8_t catalog[SEMANTIC_B1D_CATALOG_BYTES],hacf_digest *pin,
                          const semantic_snapshot_manifest *base,
                          const elpis_evidence_admission_policy_v1 *policy,
                          const uint8_t *raw,size_t raw_n){
 hacf_digest policy_digest,package_digest=package(raw,raw_n);
 (void)elpis_admission_policy_identity(policy,&policy_digest);
 memset(catalog,0,SEMANTIC_B1D_CATALOG_BYTES);
 memcpy(catalog,"ELPIS-B1D-AUTH01",16);catalog[19]=1;
 memcpy(catalog+20,policy_digest.bytes,32);
 memcpy(catalog+52,package_digest.bytes,32);
 elpis_sha256(raw,raw_n,catalog+84);
 memcpy(catalog+116,base->manifest_digest.bytes,32);
 elpis_sha256(catalog,SEMANTIC_B1D_CATALOG_BYTES,pin->bytes);
}
/* Test fixture CAS enrollment copies exact bytes without hardlinking: CAS
 * requires single-link immutable entries. Source files stay available to the
 * full B1p admission audit for independent comparison. */
static int copy_base_fixture(const char *src,const char *dst){
 FILE *a=fopen(src,"rb"),*b=NULL;
 if(!a)return 0;
 b=fopen(dst,"wb");
 if(!b){fclose(a);return 0;}
 char data[8192];size_t n;int ok=1;
 while((n=fread(data,1,sizeof(data),a))!=0){
  if(fwrite(data,1,n,b)!=n){ok=0;break;}
 }
 if(ferror(a)||fflush(b)||fsync(fileno(b)))ok=0;
 int close_a=fclose(a),close_b=fclose(b);
 if(close_a||close_b)ok=0;
 return ok;
}
static int flip_payload_byte(const char *path){
 FILE *f=fopen(path,"r+b");if(!f)return 0;
 const long off=(long)sizeof(semantic_segment_record)+
                (long)offsetof(elpis_semantic_node_v1,payload_digest);
 int ok=fseek(f,off,SEEK_SET)==0;
 int c=ok?fgetc(f):EOF;
 if(c==EOF)ok=0;
 if(ok&&fseek(f,off,SEEK_SET))ok=0;
 if(ok&&fputc(c^1,f)==EOF)ok=0;
 if(ok&&(fflush(f)||fsync(fileno(f))))ok=0;
 if(fclose(f))ok=0;
 return ok;
}

/* B2f TEST ONLY: intercept the actual immutable CAS link and HEAD rename.
 * No production hooks or alternate admission implementation are installed. */
enum b2f_mode {
 B2F_KILL_SEG_PRE=1, B2F_KILL_SEG_POST,
 B2F_KILL_MAN_PRE, B2F_KILL_MAN_POST,
 B2F_KILL_HEAD_PRE, B2F_KILL_HEAD_POST,
 B2F_ERROR_SEG_LINK, B2F_ERROR_MAN_LINK, B2F_ERROR_HEAD_RENAME
};
static int b2f_armed=0;
static enum b2f_mode b2f_mode;
static unsigned b2f_segment_links=0,b2f_manifest_links=0,b2f_head_renames=0;
int __real_link(const char *src,const char *dst);
int __real_renameat(int oldfd,const char *oldname,int newfd,const char *newname);
static void b2f_kill(void){(void)kill(getpid(),SIGKILL);_exit(99);}
static int b2f_ends_with(const char *s,const char *end){
 size_t a=strlen(s),b=strlen(end);return a>=b&&!strcmp(s+a-b,end);
}
int __wrap_link(const char *src,const char *dst){
 if(!b2f_armed)return __real_link(src,dst);
 int seg=b2f_ends_with(dst,".segment"),man=b2f_ends_with(dst,".snapshot");
 if(seg)b2f_segment_links++;
 if(man)b2f_manifest_links++;
 if(seg && b2f_segment_links==1){
  if(b2f_mode==B2F_KILL_SEG_PRE)b2f_kill();
  if(b2f_mode==B2F_ERROR_SEG_LINK){errno=EIO;return -1;}
 }
 if(man && b2f_manifest_links==1){
  if(b2f_mode==B2F_KILL_MAN_PRE)b2f_kill();
  if(b2f_mode==B2F_ERROR_MAN_LINK){errno=EIO;return -1;}
 }
 int rc=__real_link(src,dst);
 if(!rc && seg && b2f_segment_links==1 && b2f_mode==B2F_KILL_SEG_POST)b2f_kill();
 if(!rc && man && b2f_manifest_links==1 && b2f_mode==B2F_KILL_MAN_POST)b2f_kill();
 return rc;
}
int __wrap_renameat(int oldfd,const char *oldname,int newfd,const char *newname){
 if(!b2f_armed)return __real_renameat(oldfd,oldname,newfd,newname);
 if(strcmp(newname,"HEAD"))return __real_renameat(oldfd,oldname,newfd,newname);
 b2f_head_renames++;
 if(b2f_head_renames==1){
  if(b2f_mode==B2F_KILL_HEAD_PRE)b2f_kill();
  if(b2f_mode==B2F_ERROR_HEAD_RENAME){errno=EIO;return -1;}
 }
 int rc=__real_renameat(oldfd,oldname,newfd,newname);
 if(!rc && b2f_head_renames==1 && b2f_mode==B2F_KILL_HEAD_POST)b2f_kill();
 return rc;
}
static int b2f_check_restart(const char *dir,const hacf_digest *expect){
 pid_t pid=fork();if(pid<0)return 0;
 if(!pid){hacf_digest got={0};int rc=semantic_b2b_head_read(dir,&got);
  _exit(rc==SEMANTIC_B2B_OK&&!memcmp(&got,expect,sizeof(got))?0:78);}
 int st=0;return waitpid(pid,&st,0)==pid&&WIFEXITED(st)&&WEXITSTATUS(st)==0;
}
/* B2g: no test-only orphan cleaner. The actual privileged B2c transaction
 * must normalize an exact, single verified post-link hardlink and fsync the
 * store directory, or leave HEAD pinned and fail closed. */
static int flip_first_byte(const char *path){
 FILE *f=fopen(path,"r+b");if(!f)return 0;
 int c=fgetc(f);int ok=c!=EOF;
 if(ok&&fseek(f,0,SEEK_SET))ok=0;
 if(ok&&fputc(c^1,f)==EOF)ok=0;
 if(ok&&(fflush(f)||fsync(fileno(f))))ok=0;
 if(fclose(f))ok=0;
 return ok;
}
static int b2g_has_temp(const char *root,const char *prefix){
 DIR *dp=opendir(root);if(!dp)return -1;
 struct dirent *ent;int found=0;
 while((ent=readdir(dp)))if(!strncmp(ent->d_name,prefix,strlen(prefix)))found++;
 if(closedir(dp))return -1;
 return found;
}
static int b2g_find_single_temp(const char *root,const char *prefix,
                                 char *path,size_t cap){
 DIR *dp=opendir(root);if(!dp)return 0;
 struct dirent *e;int count=0,ok=1;
 while((e=readdir(dp))){
  if(strncmp(e->d_name,prefix,strlen(prefix)))continue;
  if(++count!=1){ok=0;break;}
  int n=snprintf(path,cap,"%s/%s",root,e->d_name);
  if(n<=0||(size_t)n>=cap){ok=0;break;}
 }
 if(closedir(dp))ok=0;
 return ok&&count==1;
}
static int b2f_cleanup(const char *dir){
 DIR *dp=opendir(dir);if(!dp)return 0;
 struct dirent *ent;char path[1024];int ok=1;
 while((ent=readdir(dp))){if(!strcmp(ent->d_name,".")||!strcmp(ent->d_name,".."))continue;
  int n=snprintf(path,sizeof(path),"%s/%s",dir,ent->d_name);
  if(n<=0||(size_t)n>=sizeof(path)||unlink(path))ok=0;
 }
 if(closedir(dp))ok=0;
 if(rmdir(dir))ok=0;
 return ok;
}
/* Independent complete B2d publication attempts under process death and EIO.
 * Each child executes semantic_b2c_publish_one(), not B2b in isolation.
 * The original immutable source segments and trusted catalog live outside the
 * per-case CAS root and remain independently verified for every attempt. */
static int b2f_run_faults(const char *parent,
                           const char *source0,const char *source1,
                           const semantic_snapshot_manifest *base,
                           const semantic_snapshot_manifest *successor,
                           const semantic_segment_record *proposal,
                           const semantic_b2a_witness *w){
 for(unsigned m=B2F_KILL_SEG_PRE;m<=B2F_ERROR_HEAD_RENAME;m++){
  char dir[1024],dst[1024],hex0[65],hex1[65],hexseg[65];
  if(snprintf(dir,sizeof(dir),"%s/b2f-%02u",parent,m)<=0 || mkdir(dir,0700))return 0;
  hex(&base->segment_digests[0],hex0);hex(&base->segment_digests[1],hex1);
  if(snprintf(dst,sizeof(dst),"%s/%s.segment",dir,hex0)<=0||!copy_base_fixture(source0,dst))return 0;
  if(snprintf(dst,sizeof(dst),"%s/%s.segment",dir,hex1)<=0||!copy_base_fixture(source1,dst))return 0;
  if(semantic_snapshot_publish_cas(base,dir,NULL)!=SEMANTIC_OK||
     semantic_b2b_head_bootstrap(dir,&base->manifest_digest)!=SEMANTIC_B2B_OK)return 0;
  pid_t child=fork();if(child<0)return 0;
  if(!child){
   b2f_mode=(enum b2f_mode)m;b2f_segment_links=b2f_manifest_links=b2f_head_renames=0;
   b2f_armed=1;
   int rc=semantic_b2c_publish_one(dir,w);
   b2f_armed=0;
   if(m>=B2F_ERROR_SEG_LINK) _exit(rc==SEMANTIC_B2B_IO?0:88);
   _exit(89); /* All six crash cases must actually reach their boundary. */
  }
  int status=0;
  if(waitpid(child,&status,0)!=child)return 0;
  if(m<B2F_ERROR_SEG_LINK){if(!WIFSIGNALED(status)||WTERMSIG(status)!=SIGKILL)return 0;}
  else if(!WIFEXITED(status)||WEXITSTATUS(status)!=0)return 0;
  int advanced=(m==B2F_KILL_HEAD_POST);
  if(!b2f_check_restart(dir,advanced?&successor->manifest_digest:&base->manifest_digest))return 0;
  hex(&proposal->segment_identity,hexseg);
  if(snprintf(dst,sizeof(dst),"%s/%s.segment",dir,hexseg)<=0)return 0;
  if(m==B2F_KILL_SEG_POST||m==B2F_KILL_MAN_POST){
   /* A post-link kill leaves exactly two hardlinks. More than two must be
    * refused without deleting any artifact or advancing HEAD. */
   char orphan_cas[1024],extra[1024];
   const char *prefix=m==B2F_KILL_SEG_POST?".tmp_segment_":".tmp_snap_";
   if(m==B2F_KILL_SEG_POST){
    if(snprintf(orphan_cas,sizeof(orphan_cas),"%s/%s.segment",dir,hexseg)<=0)return 0;
   }else{
    char snapshot_hex[65];hex(&successor->manifest_digest,snapshot_hex);
    if(snprintf(orphan_cas,sizeof(orphan_cas),"%s/%s.snapshot",dir,snapshot_hex)<=0)return 0;
   }
   if(snprintf(extra,sizeof(extra),"%s/unrecognized-hardlink",dir)<=0 ||
      link(orphan_cas,extra)!=0)return 0;
   if(semantic_b2c_publish_one(dir,w)!=SEMANTIC_B2B_INVALID||
      !b2f_check_restart(dir,&base->manifest_digest)||unlink(extra))return 0;
   /* A different hardlink name with nlink==2 is NOT a writer orphan. */
   char orphan_temp[1024],alien[1024];
   if(!b2g_find_single_temp(dir,prefix,orphan_temp,sizeof(orphan_temp))||
      snprintf(alien,sizeof(alien),"%s/unrecognized-hardlink-alias",dir)<=0||
      rename(orphan_temp,alien)!=0)return 0;
   if(semantic_b2c_publish_one(dir,w)!=SEMANTIC_B2B_INVALID||
      !b2f_check_restart(dir,&base->manifest_digest)||
      rename(alien,orphan_temp)!=0)return 0;
   /* Content corruption under a valid CAS identity must be refused before
    * removing the orphan's temporary name. */
   if(m==B2F_KILL_SEG_POST){
    if(!flip_payload_byte(orphan_cas))return 0;
   }else if(!flip_first_byte(orphan_cas))return 0;
   if(semantic_b2c_publish_one(dir,w)!=SEMANTIC_B2B_INVALID||
      !b2f_check_restart(dir,&base->manifest_digest)||
      b2g_has_temp(dir,prefix)!=1)return 0;
   if(m==B2F_KILL_SEG_POST){
    if(!flip_payload_byte(orphan_cas))return 0;
   }else if(!flip_first_byte(orphan_cas))return 0;
   /* Unlike B2f, this is automatic production-path recovery. */
  }
  int rc=semantic_b2c_publish_one(dir,w);
  if(rc!=(advanced?SEMANTIC_B2B_CONFLICT:SEMANTIC_B2B_OK)){
   fprintf(stderr,"B2G_FAULT_MODE=%u publish_rc=%d expected=%d\n",m,rc,
           advanced?SEMANTIC_B2B_CONFLICT:SEMANTIC_B2B_OK);return 0;
  }
  if(m==B2F_KILL_SEG_POST && b2g_has_temp(dir,".tmp_segment_")!=0)return 0;
  if(m==B2F_KILL_MAN_POST && b2g_has_temp(dir,".tmp_snap_")!=0)return 0;
  if(!b2f_check_restart(dir,&successor->manifest_digest))return 0;
  if(semantic_b2c_publish_one(dir,w)!=SEMANTIC_B2B_CONFLICT)return 0;
  if(!b2f_cleanup(dir))return 0;
  printf("B2G_INTEGRATED_FAULT_PASS mode=%u head=%s\n",m,advanced?"NEW":"OLD_THEN_RECOVERED");
  fflush(stdout);
 }
 return 1;
}

static int run_case(int candidate_in_base){
 static const uint8_t text[]="source byte anchor raw text";
 const size_t text_n=sizeof(text)-1;
 semantic_snapshot_manifest *base=semantic_snapshot_create();
 elpis_evidence_typing_bundle_v1 *bundle=calloc(1,sizeof(*bundle));
 elpis_evidence_admission_v1 *layer=calloc(1,sizeof(*layer));
 elpis_evidence_admission_decision_v1 *d=calloc(1,sizeof(*d));
 elpis_evidence_admission_receipt_v1 *r=calloc(1,sizeof(*r));
 elpis_evidence_admission_policy_v1 *policy=calloc(1,sizeof(*policy));
 CHECK(base && bundle && layer && d && r && policy);
 elpis_admission_policy_init_default(policy);
 semantic_type_registry *reg=semantic_type_registry_create();CHECK(reg);
 semantic_node_type_entry nt={0};nt.node_type=SEMANTIC_NODE_NAMESPACE|1;
 nt.semantic_flag_mask=SEMANTIC_NODE_FLAG_MASK;nt.min_authority=0;nt.max_authority=3;
 CHECK(semantic_type_registry_add_node_type(reg,&nt)==SEMANTIC_OK);
 CHECK(semantic_type_registry_seal(reg,NULL)==SEMANTIC_OK);
 /* Two actual populated native segments. The second optionally contains the
  * proposed node, with a consistently finalized base manifest and catalog. */
 char temporary_dir[]="/tmp/elpis-b1o-XXXXXX";
 CHECK(mkdtemp(temporary_dir)!=NULL);
 char path0[512],path1[512];
 CHECK(snprintf(path0,sizeof(path0),"%s/a.segment",temporary_dir)>0);
 CHECK(snprintf(path1,sizeof(path1),"%s/b.segment",temporary_dir)>0);
 semantic_hypergraph_builder *builder0=semantic_builder_create(reg);
 semantic_hypergraph_builder *builder1=semantic_builder_create(reg);
 CHECK(builder0&&builder1);
 elpis_semantic_node_v1 existing0={0},existing1={0},candidate={0};
 existing0.abi_version=SEMANTIC_ABI_VERSION;
 existing0.node_type=SEMANTIC_NODE_NAMESPACE|1;
 elpis_sha256("existing-zero",13,existing0.payload_digest.bytes);
 CHECK(elpis_semantic_node_identity(&existing0,&existing0.node_identity)==SEMANTIC_OK);
 existing1.abi_version=SEMANTIC_ABI_VERSION;
 existing1.node_type=SEMANTIC_NODE_NAMESPACE|1;
 elpis_sha256("existing-one",12,existing1.payload_digest.bytes);
 CHECK(elpis_semantic_node_identity(&existing1,&existing1.node_identity)==SEMANTIC_OK);
 node(&candidate);
 CHECK(semantic_builder_add_node(builder0,&existing0)==SEMANTIC_OK);
 CHECK(semantic_builder_add_node(builder1,&existing1)==SEMANTIC_OK);
 if(candidate_in_base==1) CHECK(semantic_builder_add_node(builder1,&candidate)==SEMANTIC_OK);
 if(candidate_in_base==2){
    elpis_semantic_node_v1 alt=candidate;
    alt.semantic_flags=SEMANTIC_NODE_FLAG_EXTERNAL;
    CHECK(elpis_semantic_node_identity(&alt,&alt.node_identity)==SEMANTIC_OK);
    CHECK(memcmp(&alt.node_identity,&candidate.node_identity,sizeof(hacf_digest))!=0);
    CHECK(semantic_builder_add_node(builder1,&alt)==SEMANTIC_OK);
 }
 CHECK(semantic_type_registry_digest(reg,&base->type_registry_digest)==SEMANTIC_OK);
 CHECK(semantic_genesis_identity(&base->type_registry_digest,&base->genesis_identity)==SEMANTIC_OK);
 semantic_segment_record first_seg,second_seg;
 CHECK(semantic_segment_build(builder0,reg,&base->genesis_identity,&first_seg)==SEMANTIC_OK);
 CHECK(semantic_segment_write(&first_seg,builder0,path0,NULL)==SEMANTIC_OK);
 CHECK(semantic_snapshot_add_segment(base,&first_seg)==SEMANTIC_OK);
 CHECK(semantic_segment_build(builder1,reg,&first_seg.hacf_next_snapshot,&second_seg)==SEMANTIC_OK);
 CHECK(semantic_segment_write(&second_seg,builder1,path1,NULL)==SEMANTIC_OK);
 CHECK(semantic_snapshot_add_segment(base,&second_seg)==SEMANTIC_OK);
 CHECK(semantic_snapshot_finalize(base)==SEMANTIC_OK);
 CHECK(semantic_snapshot_validate(base)==SEMANTIC_OK);
 semantic_builder_destroy(builder0);semantic_builder_destroy(builder1);
 const char *segment_paths[]={path0,path1};
 hacf_digest base_pin=base->manifest_digest,base_report;
 uint32_t base_present=99;
 CHECK(semantic_b1n_global_unique_node_audit(base,&base_pin,segment_paths,2,
       &candidate.node_identity,&base_present,&base_report)==SEMANTIC_OK);
 CHECK(base_present==(uint32_t)(candidate_in_base==1));
 if(candidate_in_base==2) CHECK(base_present==0); /* predecessor reports novel identity */
 hacf_digest query;fill(&query,11);
 semantic_query_overlay *ov=semantic_overlay_create(base,reg,&query);CHECK(ov);
 elpis_semantic_node_v1 nd;node(&nd);
 CHECK(semantic_overlay_add_node(ov,&nd)==SEMANTIC_OK);
 elpis_retrieval_item_attachment_v1 att={0};
 att.abi_version=RETRIEVAL_ITEM_ATTACHMENT_ABI_VERSION;
 fill(&att.evidence_node_digest,23);
 fill(&att.retrieval_requirement_digest,25);
 fill(&att.chunk_digest,22);fill(&att.document_digest,24);
 elpis_sha256("repo",4,att.namespace_digest.bytes);
 elpis_sha256(text,text_n,att.text_digest.bytes);
 att.item_authority=1;att.item_kind=1;
 att.source_mask=3;att.lexical_rank=1;att.dense_rank=2;
 att.dense_score_key=-12;att.fusion_score_key=500;
 att.graph_edge_provenance_status=GRAPH_PROVENANCE_NOT_APPLICABLE;
 char raw[8192];int n=make_json(raw,sizeof(raw),&att,text,text_n);
 CHECK(n>0&&(size_t)n<sizeof(raw));
 att.retrieval_bundle_package_digest=package((const uint8_t*)raw,(size_t)n);
 elpis_sha256(raw,(size_t)n,att.retrieval_bundle_digest.bytes);
 CHECK(elpis_attachment_digest(&att,&att.attachment_digest)==SEMANTIC_OK);
 hacf_digest retrieval,expanded,typer;fill(&retrieval,4);fill(&expanded,5);fill(&typer,6);
 elpis_evidence_span_v1 sp={0};sp.abi_version=EVIDENCE_SPAN_ABI_VERSION;
 sp.retrieval_expansion_digest=retrieval;
 sp.retrieval_bundle_digest=att.retrieval_bundle_digest;
 sp.retrieval_bundle_package_digest=att.retrieval_bundle_package_digest;
 sp.retrieval_item_attachment_digest=att.attachment_digest;
 sp.evidence_node_digest=att.evidence_node_digest;
 sp.chunk_digest=att.chunk_digest;sp.item_text_digest=att.text_digest;
 sp.byte_start=1;sp.byte_end_exclusive=(uint32_t)text_n-1;
 sp.span_flags=EVIDENCE_SPAN_FLAG_PRIMARY;
 elpis_sha256(text+1,text_n-2,sp.span_bytes_digest.bytes);
 CHECK(elpis_evidence_span_identity(&sp,&sp.span_identity)==SEMANTIC_OK);
 elpis_semantic_assertion_v1 assertion={0};
 assertion.abi_version=SEMANTIC_ABI_VERSION;
 assertion.asserted_object_kind=SEMANTIC_OBJECT_KIND_NODE;
 assertion.asserted_object_digest=nd.node_identity;
 assertion.provenance_digest=sp.span_identity;assertion.authority=1;
 CHECK(elpis_semantic_assertion_identity(&assertion,&assertion.assertion_identity)==SEMANTIC_OK);
 CHECK(semantic_overlay_add_assertion(ov,&assertion)==SEMANTIC_OK);
 CHECK(semantic_overlay_finalize(ov)==SEMANTIC_OK);
 elpis_evidence_claim_candidate_v1 claim={0};
 claim.abi_version=EVIDENCE_CLAIM_CANDIDATE_ABI_VERSION;
 claim.typer_profile_digest=typer;claim.claim_type=1;
 elpis_sha256(canonical_payload,sizeof(canonical_payload)-1,claim.claim_payload_digest.bytes);
 claim.claim_payload_object_digest=claim.claim_payload_digest;
 claim.source_span_count=1;claim.source_span_digests[0]=sp.span_identity;
 claim.claim_polarity=CLAIM_POLARITY_AFFIRMATIVE;
 claim.claim_modality=CLAIM_MODALITY_ASSERTED;claim.confidence_key=9000;
 CHECK(elpis_claim_candidate_identity(&claim,&claim.candidate_identity)==SEMANTIC_OK);
 hacf_digest policy_pin;CHECK(elpis_admission_policy_identity(policy,&policy_pin)==SEMANTIC_OK);
 elpis_typing_bundle_init(bundle);
 bundle->base_snapshot_digest=base->manifest_digest;
 bundle->query_overlay_digest=ov->overlay_identity;
 bundle->retrieval_expansion_digest=retrieval;
 bundle->retrieval_expanded_view_digest=expanded;
 bundle->typer_profile_digest=typer;
 bundle->evidence_span_count=1;bundle->evidence_span_digests[0]=sp.span_identity;
 bundle->claim_candidate_count=1;bundle->claim_candidate_digests[0]=claim.candidate_identity;
 fill(&bundle->typing_bundle_policy_digest,7);
 CHECK(elpis_typing_bundle_identity(bundle,&bundle->typing_bundle_digest)==SEMANTIC_OK);
 CHECK(elpis_typing_bundle_validate(bundle)==SEMANTIC_OK);
 elpis_admission_decision_init(d);
 d->candidate_kind=CANDIDATE_KIND_CLAIM;d->candidate_digest=claim.candidate_identity;
 d->typing_bundle_digest=bundle->typing_bundle_digest;
 d->admission_policy_digest=policy_pin;
 d->validation_stage_reached=VALIDATION_STAGE_COMPLETE;
 d->decision_disposition=DISPOSITION_ADMITTED_NEW_OBJECT;
 d->semantic_object_kind=SEMANTIC_OBJECT_KIND_CLAIM;
 d->semantic_object_digest=nd.node_identity;d->effective_authority=1;
 d->source_span_count=1;d->source_span_digests[0]=sp.span_identity;
 d->source_attachment_count=1;d->source_attachment_digests[0]=att.attachment_digest;
 CHECK(elpis_admission_decision_identity(d,&d->decision_identity)==SEMANTIC_OK);
 elpis_admission_receipt_init(r);
 r->base_snapshot_digest=base->manifest_digest;r->query_overlay_digest=ov->overlay_identity;
 r->retrieval_expansion_digest=retrieval;r->retrieval_expanded_view_digest=expanded;
 r->typing_bundle_digest=bundle->typing_bundle_digest;r->typer_profile_digest=typer;
 r->candidate_digest=claim.candidate_identity;r->admission_policy_digest=policy_pin;
 r->admission_decision_digest=d->decision_identity;r->semantic_object_digest=nd.node_identity;
 r->source_span_count=1;r->source_span_digests[0]=sp.span_identity;
 r->retrieval_bundle_count=1;
 r->retrieval_bundle_package_digests[0]=att.retrieval_bundle_package_digest;
 r->retrieval_item_attachment_count=1;
 r->retrieval_item_attachment_digests[0]=att.attachment_digest;
 CHECK(elpis_admission_receipt_identity(r,&r->receipt_digest)==SEMANTIC_OK);
 elpis_evidence_admission_init(layer);
 layer->base_snapshot_digest=base->manifest_digest;
 layer->query_overlay_digest=ov->overlay_identity;
 layer->retrieval_expansion_digest=retrieval;
 layer->retrieval_expanded_view_digest=expanded;
 layer->typing_bundle_digest=bundle->typing_bundle_digest;
 layer->admission_policy_digest=policy_pin;
 layer->admission_decision_count=1;
 layer->admission_decision_digests[0]=d->decision_identity;
 layer->admission_receipt_count=1;
 layer->admission_receipt_digests[0]=r->receipt_digest;
 layer->admitted_claim_count=1;
 CHECK(elpis_evidence_admission_identity(layer,&layer->admission_layer_digest)==SEMANTIC_OK);
 semantic_b1b_claim_source source={&claim,&sp,&att,text,text_n};
 uint8_t catalog[SEMANTIC_B1D_CATALOG_BYTES];hacf_digest pin;
 catalog_build(catalog,&pin,base,policy,(const uint8_t*)raw,(size_t)n);
 hacf_digest report,prior,sentinel,expected;
 fill(&sentinel,0xa5);
 CHECK(semantic_b1i_identity_projection_audit(catalog,sizeof(catalog),&pin,
       base,ov,bundle,policy,layer,d,r,&source,(const uint8_t*)raw,(size_t)n,
       canonical_payload,sizeof(canonical_payload)-1,&prior)==SEMANTIC_OK);
#define AUDIT() semantic_b1o_populated_claim_novelty_audit(\
    catalog,sizeof(catalog),&pin,base,&base_pin,segment_paths,2,\
    ov,bundle,policy,layer,d,r,&source,(const uint8_t*)raw,(size_t)n,\
    canonical_payload,sizeof(canonical_payload)-1,&report)
 report=sentinel;
 if(candidate_in_base==2){
     semantic_segment_record checked;
     hacf_digest checked_digest;
     uint32_t clashes=99;
     CHECK(semantic_segment_read_typed_payload_occurrences(path1,
         candidate.node_type,&candidate.payload_digest,
         &checked,&checked_digest,&clashes)==SEMANTIC_OK);
     CHECK(clashes==1&&memcmp(&checked_digest,&second_seg.segment_identity,32)==0);
 }
 /* B2a: in-memory successor, backed by a fresh full B1p audit. */
 semantic_b2a_witness w={0};
 w.catalog=catalog;w.catalog_bytes=sizeof(catalog);
 w.trusted_catalog_sha256=&pin;w.base=base;
 w.trusted_base_manifest_digest=&base_pin;
 w.base_segment_paths=segment_paths;w.base_segment_count=2;
 w.registry=reg;w.overlay=ov;w.typing_bundle=bundle;
 w.policy=policy;w.layer=layer;w.decision=d;w.receipt=r;
 w.claim_source=&source;w.raw_bundle_json=(const uint8_t*)raw;
 w.raw_bundle_bytes=(size_t)n;w.canonical_payload=canonical_payload;
 w.canonical_payload_bytes=sizeof(canonical_payload)-1;
 semantic_segment_record proposal,before_proposal;
 semantic_snapshot_manifest *successor=malloc(sizeof(*successor));
 CHECK(successor);
 memset(&proposal,0xa5,sizeof(proposal));before_proposal=proposal;
 memset(successor,0xa5,sizeof(*successor));
#define PREP() semantic_b2a_prepare_successor(&w,&proposal,successor)
 if(candidate_in_base){
     CHECK(PREP()!=SEMANTIC_OK);
     CHECK(memcmp(&proposal,&before_proposal,sizeof(proposal))==0);
     CHECK(successor->abi_version==0xa5a5a5a5u);
 }else{
     CHECK(PREP()==SEMANTIC_OK);
     CHECK(proposal.node_count==1 && proposal.assertion_count==1);
     CHECK(successor->segment_count==base->segment_count+1);
     CHECK(memcmp(&proposal.prior_snapshot_digest,&base->hacf_graph_snapshot_digest,32)==0);
     CHECK(memcmp(&successor->prior_manifest_digest,&base->manifest_digest,32)==0);
     CHECK(semantic_snapshot_validate(successor)==SEMANTIC_OK);
     semantic_segment_record saved=proposal;
     semantic_snapshot_manifest *saved_next=malloc(sizeof(*saved_next));
     CHECK(saved_next);*saved_next=*successor;
     CHECK(PREP()==SEMANTIC_OK && memcmp(&saved,&proposal,sizeof(saved))==0);
     CHECK(memcmp(saved_next,successor,sizeof(*successor))==0);
     base_pin.bytes[0]^=1;
     CHECK(PREP()!=SEMANTIC_OK && memcmp(saved_next,successor,sizeof(*successor))==0);
     base_pin.bytes[0]^=1;
     CHECK(semantic_b2a_prepare_successor(&w,&proposal,base)!=SEMANTIC_OK);
     CHECK(memcmp(&base->manifest_digest,&base_pin,32)==0);
     /* Test-only staging: native CAS writes, validated readback, no active head. */
     char segment_path[512],manifest_path[512],hex_out[65];
     CHECK(snprintf(segment_path,sizeof(segment_path),"%s/proposed.segment",temporary_dir)>0);
     CHECK(semantic_segment_write(&proposal,ov->local_builder,segment_path,NULL)==SEMANTIC_OK);
     semantic_segment_record read_seg={0};hacf_digest read_digest={0};
     CHECK(semantic_segment_read(segment_path,&read_seg,&read_digest)==SEMANTIC_OK);
     CHECK(memcmp(&read_seg,&proposal,sizeof(proposal))==0);
     CHECK(memcmp(&read_digest,&proposal.segment_identity,32)==0);
     CHECK(semantic_snapshot_publish_cas(successor,temporary_dir,hex_out)==SEMANTIC_OK);
     CHECK(snprintf(manifest_path,sizeof(manifest_path),"%s/%s.snapshot",temporary_dir,hex_out)>0);
     semantic_snapshot_manifest *read_manifest=malloc(sizeof(*read_manifest));
     CHECK(read_manifest);
     CHECK(semantic_snapshot_read(manifest_path,read_manifest)==SEMANTIC_OK);
     CHECK(memcmp(read_manifest,successor,sizeof(*successor))==0);
     CHECK(semantic_snapshot_publish_cas(successor,temporary_dir,NULL)==SEMANTIC_E_DUPLICATE);
     /* B2c: make an actual immutable successor the active HEAD, beginning
      * with a separately bootstrapped, pinned, preexisting base. */
     char base_cas_name[512],head_path[512],lock_path[512],stage_lock_path[512];
     char base_hex[65];
     CHECK(semantic_snapshot_publish_cas(base,temporary_dir,base_hex)==SEMANTIC_OK);
     CHECK(semantic_b2b_head_bootstrap(temporary_dir,&base->manifest_digest)==SEMANTIC_B2B_OK);
     hacf_digest active={0};
     CHECK(semantic_b2b_head_read(temporary_dir,&active)==SEMANTIC_B2B_OK);
     CHECK(memcmp(&active,&base->manifest_digest,sizeof(active))==0);
     /* The old B2c could publish while the base CAS inventory was absent. */
     CHECK(semantic_b2c_publish_one(temporary_dir,&w)==SEMANTIC_B2B_INVALID);
     CHECK(semantic_b2b_head_read(temporary_dir,&active)==SEMANTIC_B2B_OK);
     CHECK(memcmp(&active,&base->manifest_digest,sizeof(active))==0);
     char cas0[512],cas1[512],h0[65],h1[65];
     hex(&first_seg.segment_identity,h0);hex(&second_seg.segment_identity,h1);
     CHECK(snprintf(cas0,sizeof(cas0),"%s/%s.segment",temporary_dir,h0)>0);
     CHECK(snprintf(cas1,sizeof(cas1),"%s/%s.segment",temporary_dir,h1)>0);
     CHECK(copy_base_fixture(path0,cas0)&&copy_base_fixture(path1,cas1));
     /* A self-consistent v1 projection digest does not bind node payloads. */
     CHECK(flip_payload_byte(cas0));
     CHECK(semantic_b2c_publish_one(temporary_dir,&w)==SEMANTIC_B2B_INVALID);
     CHECK(flip_payload_byte(cas0));
     CHECK(unlink(cas1)==0);
     CHECK(symlink(path1,cas1)==0);
     CHECK(semantic_b2c_publish_one(temporary_dir,&w)==SEMANTIC_B2B_INVALID);
     CHECK(unlink(cas1)==0);
     CHECK(link(path1,cas1)==0);
     CHECK(semantic_b2c_publish_one(temporary_dir,&w)==SEMANTIC_B2B_INVALID);
     CHECK(unlink(cas1)==0);
     CHECK(copy_base_fixture(path1,cas1));
     /* Recovery authority is unavailable in a group-writable store. */
     CHECK(chmod(temporary_dir,0770)==0);
     CHECK(semantic_b2c_publish_one(temporary_dir,&w)==SEMANTIC_B2B_INVALID);
     CHECK(chmod(temporary_dir,0700)==0);
     CHECK(b2f_run_faults(temporary_dir,path0,path1,base,successor,&proposal,&w));
     hacf_digest broken_catalog_pin=pin;broken_catalog_pin.bytes[0]^=0x40;
     w.trusted_catalog_sha256=&broken_catalog_pin;
     CHECK(semantic_b2c_publish_one(temporary_dir,&w)==SEMANTIC_B2B_INVALID);
     CHECK(semantic_b2b_head_read(temporary_dir,&active)==SEMANTIC_B2B_OK);
     CHECK(memcmp(&active,&base->manifest_digest,sizeof(active))==0);
     w.trusted_catalog_sha256=&pin;
     /* A staged CAS segment with the expected filename but altered payload
      * must be refused even though the v1 header projection is unchanged.
      * Restore it in place, then verify exact orphan reuse succeeds. */
     char canonical_seg_path[512],seg_hex[65];
     hex(&proposal.segment_identity,seg_hex);
     CHECK(snprintf(canonical_seg_path,sizeof(canonical_seg_path),"%s/%s.segment",temporary_dir,seg_hex)>0);
     CHECK(semantic_segment_write(&proposal,ov->local_builder,canonical_seg_path,NULL)==SEMANTIC_OK);
     FILE *poison=fopen(canonical_seg_path,"r+b");CHECK(poison);
     const long payload_offset=(long)sizeof(semantic_segment_record)+
                               (long)offsetof(elpis_semantic_node_v1,payload_digest);
     CHECK(fseek(poison,payload_offset,SEEK_SET)==0);
     int original_byte=fgetc(poison);CHECK(original_byte!=EOF);
     CHECK(fseek(poison,payload_offset,SEEK_SET)==0);
     CHECK(fputc(original_byte^1,poison)!=EOF && fflush(poison)==0 && fclose(poison)==0);
     CHECK(semantic_b2c_publish_one(temporary_dir,&w)==SEMANTIC_B2B_INVALID);
     CHECK(semantic_b2b_head_read(temporary_dir,&active)==SEMANTIC_B2B_OK);
     CHECK(memcmp(&active,&base->manifest_digest,sizeof(active))==0);
     poison=fopen(canonical_seg_path,"r+b");CHECK(poison);
     CHECK(fseek(poison,payload_offset,SEEK_SET)==0);
     CHECK(fputc(original_byte,poison)!=EOF && fflush(poison)==0 && fclose(poison)==0);
     /* The successor's CAS manifest and exact segment now both exist,
      * simulating durable orphans left by an interrupted attempt. */
     CHECK(semantic_b2c_publish_one(temporary_dir,&w)==SEMANTIC_B2B_OK);
     CHECK(semantic_b2b_head_read(temporary_dir,&active)==SEMANTIC_B2B_OK);
     CHECK(memcmp(&active,&successor->manifest_digest,sizeof(active))==0);
     /* B2h: a bare HEAD read is insufficient as a retrieval-integrity gate.
      * Reopen the committed successor against the trusted admission witness. */
     CHECK(semantic_b2h_verify_published_one(temporary_dir,&w)==SEMANTIC_B2B_OK);
     pid_t verify_pid=fork();CHECK(verify_pid>=0);
     if(verify_pid==0){
      int v=semantic_b2h_verify_published_one(temporary_dir,&w);
      _exit(v==SEMANTIC_B2B_OK?0:96);
     }
     int verify_status=0;
     CHECK(waitpid(verify_pid,&verify_status,0)==verify_pid &&
           WIFEXITED(verify_status) && WEXITSTATUS(verify_status)==0);
     /* Corrupt a historical CAS payload after successful HEAD advancement.
      * HEAD still verifies the manifest; strict witness read MUST refuse it. */
     CHECK(flip_payload_byte(cas0));
     CHECK(semantic_b2b_head_read(temporary_dir,&active)==SEMANTIC_B2B_OK);
     CHECK(semantic_b2h_verify_published_one(temporary_dir,&w)==SEMANTIC_B2B_INVALID);
     CHECK(flip_payload_byte(cas0));
     CHECK(semantic_b2h_verify_published_one(temporary_dir,&w)==SEMANTIC_B2B_OK);
     /* Corrupt admitted bytes without changing the v1 header projection. */
     CHECK(flip_payload_byte(canonical_seg_path));
     CHECK(semantic_b2h_verify_published_one(temporary_dir,&w)==SEMANTIC_B2B_INVALID);
     CHECK(flip_payload_byte(canonical_seg_path));
     CHECK(semantic_b2h_verify_published_one(temporary_dir,&w)==SEMANTIC_B2B_OK);
     /* Disappearing historical CAS cannot be mistaken for persisted memory. */
     CHECK(unlink(cas1)==0);
     CHECK(semantic_b2h_verify_published_one(temporary_dir,&w)==SEMANTIC_B2B_INVALID);
     CHECK(copy_base_fixture(path1,cas1));
     CHECK(semantic_b2h_verify_published_one(temporary_dir,&w)==SEMANTIC_B2B_OK);
     /* Authority pins must remain independent of the HEAD file. */
     w.trusted_base_manifest_digest=&broken_catalog_pin;
     CHECK(semantic_b2h_verify_published_one(temporary_dir,&w)==SEMANTIC_B2B_INVALID);
     w.trusted_base_manifest_digest=&base_pin;
     CHECK(semantic_b2h_verify_published_one(temporary_dir,&w)==SEMANTIC_B2B_OK);
     CHECK(semantic_b2c_publish_one(temporary_dir,&w)==SEMANTIC_B2B_CONFLICT);
     CHECK(semantic_b2b_head_read(temporary_dir,&active)==SEMANTIC_B2B_OK);
     CHECK(memcmp(&active,&successor->manifest_digest,sizeof(active))==0);
     CHECK(snprintf(base_cas_name,sizeof(base_cas_name),"%s/%s.snapshot",temporary_dir,base_hex)>0);
     CHECK(snprintf(head_path,sizeof(head_path),"%s/HEAD",temporary_dir)>0);
     CHECK(snprintf(lock_path,sizeof(lock_path),"%s/.semantic-head.lock",temporary_dir)>0);
     CHECK(unlink(canonical_seg_path)==0 && unlink(base_cas_name)==0);
     CHECK(snprintf(stage_lock_path,sizeof(stage_lock_path),"%s/.semantic-stage.lock",temporary_dir)>0);
     CHECK(unlink(head_path)==0 && unlink(lock_path)==0 && unlink(stage_lock_path)==0);
     CHECK(unlink(cas0)==0 && unlink(cas1)==0);
     CHECK(unlink(segment_path)==0 && unlink(manifest_path)==0);
     free(read_manifest);free(saved_next);
 }
#undef PREP
 free(successor);
 if(candidate_in_base){
     CHECK(AUDIT()!=SEMANTIC_OK);
     CHECK(memcmp(&report,&sentinel,sizeof(report))==0);
 }else{
     CHECK(AUDIT()==SEMANTIC_OK);
     expected=report;
     report=sentinel;
     CHECK(AUDIT()==SEMANTIC_OK && memcmp(&report,&expected,sizeof(report))==0);
     base_pin.bytes[0]^=1;
     report=sentinel;CHECK(AUDIT()!=SEMANTIC_OK && memcmp(&report,&sentinel,32)==0);
     base_pin.bytes[0]^=1;
     const char *swapped[2]={path1,path0};
     report=sentinel;
     CHECK(semantic_b1o_populated_claim_novelty_audit(catalog,sizeof(catalog),&pin,
        base,&base_pin,swapped,2,ov,bundle,policy,layer,d,r,&source,
        (const uint8_t*)raw,(size_t)n,canonical_payload,sizeof(canonical_payload)-1,
        &report)!=SEMANTIC_OK && memcmp(&report,&sentinel,32)==0);
     /* Complete-package remains valid when base segment is corrupted. */
     FILE *fp=fopen(path1,"r+b");CHECK(fp);
     int original=fgetc(fp);CHECK(original!=EOF);
     CHECK(fseek(fp,0,SEEK_SET)==0);
     CHECK(fputc((original^0x80)&255,fp)!=EOF && fflush(fp)==0 && fclose(fp)==0);
     report=sentinel;CHECK(AUDIT()!=SEMANTIC_OK && memcmp(&report,&sentinel,32)==0);
     fp=fopen(path1,"r+b");CHECK(fp);
     CHECK(fputc(original,fp)!=EOF && fflush(fp)==0 && fclose(fp)==0);
     report=sentinel;CHECK(AUDIT()==SEMANTIC_OK && memcmp(&report,&expected,32)==0);
     /* Output aliases never mutate a trusted manifest pin. */
     hacf_digest original_pin=base_pin;
     CHECK(semantic_b1o_populated_claim_novelty_audit(catalog,sizeof(catalog),&pin,
        base,&base_pin,segment_paths,2,ov,bundle,policy,layer,d,r,&source,
        (const uint8_t*)raw,(size_t)n,canonical_payload,sizeof(canonical_payload)-1,
        &base_pin)!=SEMANTIC_OK && memcmp(&original_pin,&base_pin,32)==0);
 }
#undef AUDIT
 semantic_overlay_destroy(ov);semantic_type_registry_destroy(reg);
 semantic_snapshot_destroy(base);free(bundle);free(layer);free(d);free(r);free(policy);
 CHECK(unlink(path0)==0 && unlink(path1)==0 && rmdir(temporary_dir)==0);
 return 0;
}
int main(void){
 if(run_case(0)!=0)return 1;
 if(run_case(1)!=0)return 1;
 if(run_case(2)!=0)return 1;
 puts("B1O_NATIVE_POPULATED_NOVELTY_PASS");
 return 0;
}
