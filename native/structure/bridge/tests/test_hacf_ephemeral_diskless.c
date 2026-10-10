#define _POSIX_C_SOURCE 200809L
#include "retrieval_bridge.h"
#include "elpis/corpus.h"
#include "elpis/chunking.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr,"FAIL_HACF_DISKLESS line=%d condition=%s\n",__LINE__,#expr); return 1; \
} } while (0)

static int absent(const char *path) {
    struct stat st;
    return lstat(path,&st)!=0;
}

int main(int argc, char **argv) {
    CHECK(argc==2);
    const char *root=argv[1];
    CHECK(root[0]=='/');
    CHECK(absent(root));
    elpis_corpus *c=NULL;
    CHECK(elpis_corpus_open_ephemeral(&c)==0 && c!=NULL);
    const char content[]="Bounded source document with exact identity.";
    elpis_ingest_meta meta={"local.docs","reference",ELPIS_MT_TEXT,"read-in-place-source"};
    elpis_ingest_result ir={0};
    CHECK(elpis_corpus_ingest_bytes(c,content,strlen(content),&meta,&ir)==0);
    uint64_t docs=0,chunks=0;
    CHECK(elpis_corpus_counts(c,&docs,&chunks)==0 && docs==1 && chunks>0);
    void *data=NULL;
    size_t n=0;
    CHECK(elpis_corpus_document_bytes(c,ir.doc_digest,&data,&n)==0);
    CHECK(n==strlen(content) && memcmp(data,content,n)==0);
    elpis_free(data);
    uint64_t ok=0,bad=0;
    CHECK(elpis_corpus_verify(c,&ok,&bad,NULL,0)==0 && ok==1 && bad==0);
    CHECK(absent(root));
    elpis_corpus_close(c);
    CHECK(absent(root));
    const char *labels[]={"one","two"};
    const char *texts[]={"Alpha ordinary context","Beta independent context"};
    const char *namespaces[]={"local.docs","local.docs"};
    const char *authorities[]={"reference","reference"};
    for(int i=0;i<100;i++) {
        char error[256]={0};
        elpis_retrieval_env_t *env=elpis_retrieval_env_create(
            root,labels,texts,namespaces,authorities,2,0,NULL,0,error);
        if(!env) fprintf(stderr,"bridge_create_error=%s\n",error);
        CHECK(env!=NULL);
        elpis_retrieval_env_destroy(env);
        CHECK(absent(root));
    }
    /* Oversized document is refused, before any persistent allocation. */
    char *large=(char*)malloc((1u<<20)+2);
    CHECK(large!=NULL);
    memset(large,'x',(1u<<20)+1); large[(1u<<20)+1]=0;
    const char *bad_texts[]={large};
    char err[256]={0};
    elpis_retrieval_env_t *failed=elpis_retrieval_env_create(
        root,labels,bad_texts,namespaces,authorities,1,0,NULL,0,err);
    CHECK(failed==NULL);
    free(large);
    CHECK(absent(root));
    printf("PASS_HACF_RETRIEVAL_DISKLESS_100_EPOCHS\n");
    return 0;
}
