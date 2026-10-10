#define _POSIX_C_SOURCE 200809L
/* Track B0: a snapshot is an immutable content-addressed artifact.
 * This test never claims semantic promotion or admission authorization. */
#include "elpis_semantic/snapshot_publication.h"
#include "elpis/sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/wait.h>

#define CHECK(test) do { if (!(test)) { fprintf(stderr, "FAIL line=%d: %s\n",__LINE__,#test); failed=1; } } while(0)

static int count_files(const char *dir) {
    DIR *d = opendir(dir);
    if (!d) return -1;
    int count = 0;
    struct dirent *e;
    while ((e = readdir(d))) if (strcmp(e->d_name,".") && strcmp(e->d_name,"..")) ++count;
    closedir(d);
    return count;
}

static int test_content_addressed_no_replace(void) {
    int failed = 0;
    char dir[] = "./b0-snapshot-XXXXXX";
    if (!mkdtemp(dir)) return 1;
    semantic_snapshot_manifest *m = semantic_snapshot_create();
    semantic_snapshot_manifest *readback = semantic_snapshot_create();
    if (!m || !readback) { free(m); free(readback); rmdir(dir); return 1; }
    m->segment_count = 1;
    CHECK(semantic_snapshot_finalize(m) == SEMANTIC_OK);
    char hex[65], old_hex[65], path[512];
    memset(hex, 'Z', sizeof(hex));
    CHECK(semantic_snapshot_publish_cas(m,dir,hex) == SEMANTIC_OK);
    elpis_hex32(m->manifest_digest.bytes,old_hex);
    CHECK(strcmp(hex,old_hex) == 0);
    snprintf(path,sizeof(path),"%s/%s.snapshot",dir,old_hex);
    CHECK(semantic_snapshot_read(path,readback) == SEMANTIC_OK);
    CHECK(memcmp(m,readback,sizeof(*m)) == 0);
    CHECK(count_files(dir) == 1);

    /* Duplicate does not overwrite the artifact or output buffer. */
    memset(hex, 'Q', sizeof(hex));
    CHECK(semantic_snapshot_publish_cas(m,dir,hex) == SEMANTIC_E_DUPLICATE);
    CHECK(hex[0] == 'Q' && hex[64] == 'Q');
    CHECK(semantic_snapshot_read(path,readback) == SEMANTIC_OK);
    CHECK(memcmp(m,readback,sizeof(*m)) == 0);
    CHECK(count_files(dir) == 1);

    /* A distinct valid successor receives a distinct digest/path. */
    m->assertion_count++;
    CHECK(semantic_snapshot_finalize(m) == SEMANTIC_OK);
    CHECK(semantic_snapshot_publish_cas(m,dir,hex) == SEMANTIC_OK);
    CHECK(strcmp(hex,old_hex) != 0);
    CHECK(semantic_snapshot_read(path,readback) == SEMANTIC_OK);
    CHECK(readback->assertion_count == 0);
    CHECK(count_files(dir) == 2);
    char next_path[512];
    snprintf(next_path,sizeof(next_path),"%s/%s.snapshot",dir,hex);
    CHECK(semantic_snapshot_read(next_path,readback) == SEMANTIC_OK);
    CHECK(memcmp(m,readback,sizeof(*m)) == 0);

    /* Invalid manifests create no additional storage state. */
    m->reserved[0] = 1;
    CHECK(semantic_snapshot_publish_cas(m,dir,NULL) == SEMANTIC_E_RESERVATION);
    CHECK(count_files(dir) == 2);
    CHECK(semantic_snapshot_write(m,path,NULL) == SEMANTIC_E_RESERVATION);
    CHECK(count_files(dir) == 2);
    m->reserved[0] = 0;
    /* Noncanonical binary aliases cannot masquerade as the same CAS digest. */
    m->hacf_package_digest.bytes[0] ^= 1;
    CHECK(semantic_snapshot_publish_cas(m,dir,NULL) == SEMANTIC_E_DIGEST);
    m->hacf_package_digest.bytes[0] ^= 1;
    m->segment_digests[1].bytes[0] = 1;
    CHECK(semantic_snapshot_publish_cas(m,dir,NULL) == SEMANTIC_E_RESERVATION);
    CHECK(count_files(dir) == 2);

    unlink(next_path);
    unlink(path);
    CHECK(rmdir(dir) == 0);
    semantic_snapshot_destroy(m);
    semantic_snapshot_destroy(readback);
    return failed;
}

static int test_symlink_and_race(void) {
    int failed=0;
    char dir[]="./b0-race-XXXXXX";
    if (!mkdtemp(dir)) return 1;
    semantic_snapshot_manifest *m=semantic_snapshot_create();
    semantic_snapshot_manifest *readback=semantic_snapshot_create();
    if (!m || !readback) { free(m); free(readback); rmdir(dir); return 1; }
    m->segment_count=1;
    CHECK(semantic_snapshot_finalize(m)==SEMANTIC_OK);
    char hex[65],path[512],symlink_path[512];
    elpis_hex32(m->manifest_digest.bytes,hex);
    snprintf(path,sizeof(path),"%s/%s.snapshot",dir,hex);
    snprintf(symlink_path,sizeof(symlink_path),"%s/attack.snapshot",dir);
    CHECK(symlink("missing-target",symlink_path)==0);
    char oldout[65]; memset(oldout,'X',sizeof(oldout));
    CHECK(semantic_snapshot_write(m,symlink_path,oldout)==SEMANTIC_E_DUPLICATE);
    char target[64]={0};
    CHECK(readlink(symlink_path,target,sizeof(target))==14);
    CHECK(memcmp(target,"missing-target",14)==0);
    CHECK(oldout[0]=='X' && oldout[64]=='X');
    unlink(symlink_path);

    /* Eight parallel publishers, same valid digest path, exactly one winner. */
    int gate[2];
    if (pipe(gate)!=0) { free(m);free(readback);rmdir(dir);return 1; }
    pid_t children[8];
    unsigned n=0;
    for (;n<8;++n) {
        pid_t pid=fork();
        if (pid<0) { failed=1;break; }
        if(pid==0) {
            close(gate[1]);
            char c;
            /* EOF (the parent closing the write end) releases every child. */
            if (read(gate[0],&c,1) < 0) _exit(4);
            close(gate[0]);
            char out[65]; memset(out,'X',sizeof(out));
            int rc=semantic_snapshot_publish_cas(m,dir,out);
            if(rc==SEMANTIC_OK) _exit(strcmp(out,hex)==0?0:3);
            if(rc==SEMANTIC_E_DUPLICATE) _exit(out[0]=='X'?2:3);
            _exit(3);
        }
        children[n]=pid;
    }
    close(gate[0]);close(gate[1]);
    unsigned winners=0;
    for(unsigned i=0;i<n;++i) {
        int status=0;
        if(waitpid(children[i],&status,0)!=children[i] || !WIFEXITED(status)) { failed=1;continue; }
        int code=WEXITSTATUS(status);
        if(code==0)++winners;
        else if(code!=2)failed=1;
    }
    CHECK(winners==1);
    CHECK(semantic_snapshot_read(path,readback)==SEMANTIC_OK);
    CHECK(memcmp(m,readback,sizeof(*m))==0);
    CHECK(count_files(dir)==1);
    unlink(path);
    CHECK(rmdir(dir)==0);
    semantic_snapshot_destroy(m);semantic_snapshot_destroy(readback);
    return failed;
}

int main(void) {
    int a=test_content_addressed_no_replace();
    int b=test_symlink_and_race();
    printf("TRACK_B0_SNAPSHOT_STORAGE=%s\n",a||b?"FAIL":"PASS");
    return (a||b)?1:0;
}
