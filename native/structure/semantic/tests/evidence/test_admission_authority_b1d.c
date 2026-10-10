/* B1d strict native catalog/origin-byte gate: descriptive, never a permit. */
#include "elpis_semantic/admission_authority_b1d.h"
#include "elpis/sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(e) do { if (!(e)) {fprintf(stderr,"FAIL B1d line %d: %s\n",__LINE__,#e); return 1;} } while (0)
static void fill(hacf_digest *p, uint8_t b) { memset(p->bytes, b, 32); }
static int run(void) {
    semantic_snapshot_manifest *base=semantic_snapshot_create();
    elpis_evidence_admission_policy_v1 *policy=calloc(1,sizeof(*policy));
    CHECK(base && policy);
    base->segment_count=1;
    CHECK(semantic_snapshot_finalize(base)==SEMANTIC_OK);
    elpis_admission_policy_init_default(policy);
    hacf_digest pol, catalog_pin, pinned_pol, pinned_package, report, sentinel;
    CHECK(elpis_admission_policy_identity(policy,&pol)==SEMANTIC_OK);
    uint8_t artifact[]="retrieval-byte-fixture:B1d";
    const size_t artifact_bytes=sizeof(artifact)-1;
    uint8_t catalog[SEMANTIC_B1D_CATALOG_BYTES]={0};
    memcpy(catalog,"ELPIS-B1D-AUTH01",16);
    catalog[19]=1;
    memcpy(catalog+20,pol.bytes,32);
    memset(catalog+52,0x41,32); /* separately recorded HACF package identity */
    elpis_sha256(artifact,artifact_bytes,catalog+84);
    memcpy(catalog+116,base->manifest_digest.bytes,32);
    elpis_sha256(catalog,sizeof(catalog),catalog_pin.bytes);
    fill(&sentinel,0xa5);pinned_pol=pinned_package=report=sentinel;
#define AUDIT() semantic_b1d_authority_bytes_audit(catalog,sizeof(catalog),&catalog_pin,base,policy,artifact,artifact_bytes,&pinned_pol,&pinned_package,&report)
#define RESET() do { pinned_pol=pinned_package=report=sentinel; } while(0)
#define REFUSE() do { CHECK(AUDIT()!=SEMANTIC_OK);CHECK(memcmp(&pinned_pol,&sentinel,32)==0);CHECK(memcmp(&pinned_package,&sentinel,32)==0);CHECK(memcmp(&report,&sentinel,32)==0); } while(0)
    CHECK(AUDIT()==SEMANTIC_OK);
    CHECK(memcmp(&pinned_pol,&pol,32)==0);
    CHECK(memcmp(pinned_package.bytes,catalog+52,32)==0);
    hacf_digest good=report;
    RESET(); CHECK(AUDIT()==SEMANTIC_OK);CHECK(memcmp(&report,&good,32)==0);
    RESET();artifact[4]^=1;REFUSE();artifact[4]^=1;
    RESET();catalog_pin.bytes[0]^=1;REFUSE();catalog_pin.bytes[0]^=1;
    RESET();policy->minimum_distinct_source_spans=2;REFUSE();policy->minimum_distinct_source_spans=1;
    RESET();base->manifest_digest.bytes[0]^=1;REFUSE();base->manifest_digest.bytes[0]^=1;
    RESET();catalog[19]=2;REFUSE();catalog[19]=1;
    RESET();catalog[52]=0;REFUSE();catalog[52]=0x41;
    RESET();catalog[20]^=1;REFUSE();catalog[20]^=1;
    RESET();catalog[116]^=1;REFUSE();catalog[116]^=1;
    RESET();CHECK(semantic_b1d_authority_bytes_audit(catalog,sizeof(catalog)+1,&catalog_pin,base,policy,artifact,artifact_bytes,&pinned_pol,&pinned_package,&report)!=SEMANTIC_OK);
    CHECK(memcmp(&report,&sentinel,32)==0);
    RESET();CHECK(semantic_b1d_authority_bytes_audit(catalog,sizeof(catalog),&catalog_pin,base,policy,artifact,0,&pinned_pol,&pinned_package,&report)!=SEMANTIC_OK);
    CHECK(memcmp(&report,&sentinel,32)==0);
    RESET();CHECK(semantic_b1d_authority_bytes_audit(catalog,sizeof(catalog),&catalog_pin,base,policy,artifact,artifact_bytes,&pinned_pol,&pinned_pol,&report)!=SEMANTIC_OK);
    CHECK(memcmp(&report,&sentinel,32)==0);
    RESET();CHECK(AUDIT()==SEMANTIC_OK);CHECK(memcmp(&report,&good,32)==0);
    semantic_snapshot_destroy(base);free(policy);
    return 0;
}
int main(void) {int rc=run();puts(rc?"B1D_FAIL":"PASS_B1D_NATIVE_READ_ONLY_AUTHORITY_BYTES");return rc;}
