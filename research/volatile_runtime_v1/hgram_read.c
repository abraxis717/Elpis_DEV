/*
 * research/volatile_runtime_v1: read a retained H-gram (a read-only static resource) and report what was read.
 *
 *     hgram_read FILE KEYBYTE...
 *
 * Probes FILE, then reads the association cell of each 32-byte key filled with KEYBYTE (hex). Prints one line:
 * "probe=<status> mib=<n>" followed by "key=<b> status=<s> generation=<g> value0=<v>" per key. Opens nothing for
 * writing (elpis/hgram_store.h); the harness traces it.
 */
#include "elpis/hgram_store.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: hgram_read FILE KEYBYTE...\n");
        return 2;
    }
    elpis_hgram_info info;
    memset(&info, 0, sizeof info);
    const int probe = elpis_hgram_probe(argv[1], &info);
    printf("probe=%d mib=%llu\n", probe, (unsigned long long)info.capacity_mib);
    for (int i = 2; i < argc; ++i) {
        uint8_t key[ELPIS_HGRAM_ASSOC_KEY_BYTES];
        memset(key, (int)strtoul(argv[i], NULL, 16), sizeof key);
        elpis_hgram_assoc_record record;
        memset(&record, 0, sizeof record);
        const int rc = elpis_hgram_assoc_get(argv[1], key, &record);
        printf("key=%s status=%d generation=%llu value0=%u\n", argv[i], rc, (unsigned long long)record.generation,
               (unsigned)record.value[0]);
    }
    return probe == ELPIS_HGRAM_EXISTS ? 0 : 1;
}
