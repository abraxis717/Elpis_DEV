/* Explicit one-time operator provisioner, never invoked during pip install.
 * The data file stays outside both Git repositories and all source bundles.
 */
#include "elpis/hgram_store.h"
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/statvfs.h>

int main(int argc,char **argv) {
    if(argc!=2 || argv[1][0]!='/') {
        fprintf(stderr,"usage: elpis_hgram_init /absolute/runtime/path/hgram.bin\n");
        return 2;
    }
    elpis_hgram_info info={0};
    int rc=elpis_hgram_probe(argv[1],&info);
    if(rc==ELPIS_HGRAM_EXISTS) {
        printf("HGRAM_ALREADY_EXISTS=YES\nHGRAM_BYTES=%" PRIu64 "\nHGRAM_CAPACITY_MiB=%" PRIu64 "\n",info.total_bytes,info.capacity_mib);
        return 0;
    }
    if(rc!=ELPIS_HGRAM_PATH) {
        fprintf(stderr,"HGRAM_INIT_REFUSED=EXISTING_FILE_INVALID_OR_UNSAFE (status=%d)\n",rc);
        return 1;
    }
    /* Read initial size only from the trusted operator's controlling stdin. */
    char line[128];
    printf("How large should the HACF binary be? (MiB): "); fflush(stdout);
    if(!fgets(line,sizeof line,stdin)) { fputs("HGRAM_INIT_REFUSED=NO_SIZE\n",stderr);return 1; }
    size_t n=strlen(line);
    if(n==0 || line[n-1]!='\n' || line[0]<'0' || line[0]>'9') {
        fputs("HGRAM_INIT_REFUSED=INVALID_SIZE\n",stderr);return 1;
    }
    line[n-1]=0;
    if(!line[0]) { fputs("HGRAM_INIT_REFUSED=INVALID_SIZE\n",stderr);return 1; }
    for(size_t i=0;line[i];i++) if(line[i]<'0'||line[i]>'9') {
        fputs("HGRAM_INIT_REFUSED=INVALID_SIZE\n",stderr);return 1;
    }
    errno=0;char *end=NULL;
    uintmax_t mib=strtoumax(line,&end,10);
    if(errno || !end || *end || mib==0 || mib>INT64_MAX/1048576ULL) {
        fputs("HGRAM_INIT_REFUSED=INVALID_SIZE\n",stderr);return 1;
    }
    printf("HGRAM_REQUEST_MiB=%" PRIuMAX "\nHGRAM_PREALLOCATION_BYTES=%" PRIuMAX "\n",
           mib,(uintmax_t)(mib*1048576ULL));
    fflush(stdout);
    rc=elpis_hgram_init_once(argv[1],(uint64_t)mib,&info);
    if(rc!=ELPIS_HGRAM_CREATED && rc!=ELPIS_HGRAM_EXISTS) {
        fprintf(stderr,"HGRAM_INIT_REFUSED=CAPACITY_PATH_OR_IO (status=%d, errno=%d)\n",rc,errno);
        return 1;
    }
    printf("HGRAM_INIT_STATUS=%s\nHGRAM_BYTES=%" PRIu64 "\nHGRAM_SLOTS=%" PRIu64 "\n",
           rc==ELPIS_HGRAM_CREATED?"CREATED":"EXISTS",info.total_bytes,info.slots);
    return 0;
}
