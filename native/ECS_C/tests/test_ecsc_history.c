#include "elpis/ecsc_history.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void test_receipt_payload(void)
{
    static const char digest[] =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    static const char unicode_value[] =
        "\xc3\xa9\xf0\x9f\x98\x80";

    const elpis_ecsc_binding_view bindings[] = {
        {"alpha", 5u, "quote\"slash\\", 12u},
        {"zeta", 4u, unicode_value, sizeof unicode_value - 1u}
    };

    static const char expected[] =
        "{\"bindings\":{\"alpha\":\"quote\\\"slash\\\\\","
        "\"zeta\":\"\\u00e9\\ud83d\\ude00\"},"
        "\"digest\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\","
        "\"kind\":\"cognition.turn\","
        "\"schema\":\"elpis.runtime.receipt-record.v1\","
        "\"subsystem\":\"ecs_g\"}";

    uint8_t out[1024];
    size_t need = 0u;
    size_t written = 0u;

    assert(elpis_ecsc_history_abi_version() == 1u);

    assert(
        elpis_ecsc_receipt_payload_size(
            "ecs_g", 5u,
            "cognition.turn", 14u,
            digest, 64u,
            bindings, 2u,
            &need
        ) == ELPIS_ECSC_OK
    );

    assert(need == sizeof expected - 1u);

    assert(
        elpis_ecsc_receipt_payload_write(
            "ecs_g", 5u,
            "cognition.turn", 14u,
            digest, 64u,
            bindings, 2u,
            out, sizeof out,
            &written
        ) == ELPIS_ECSC_OK
    );

    assert(written == need);
    assert(memcmp(out, expected, need) == 0);

    written = 999u;
    assert(
        elpis_ecsc_receipt_payload_write(
            "ecs_g", 5u,
            "cognition.turn", 14u,
            digest, 64u,
            bindings, 2u,
            out, need - 1u,
            &written
        ) == ELPIS_ECSC_CAPACITY
    );
    assert(written == 0u);
}

static void test_rejects_unsorted_bindings(void)
{
    static const char digest[] =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    const elpis_ecsc_binding_view bad[] = {
        {"z", 1u, "1", 1u},
        {"a", 1u, "2", 1u}
    };

    size_t need = 123u;

    assert(
        elpis_ecsc_receipt_payload_size(
            "ecs_g", 5u,
            "cognition.turn", 14u,
            digest, 64u,
            bad, 2u,
            &need
        ) == ELPIS_ECSC_INVALID
    );
    assert(need == 0u);
}

static void test_digest_bytes(void)
{
    char out[65];

    assert(
        elpis_ecsc_digest_bytes("abc", 3u, out) ==
        ELPIS_ECSC_OK
    );

    assert(
        strcmp(
            out,
            "a7f4b10121b3f5e59f82bbca1954f7c"
            "80228376d60d6d24f4c10e6016de4eee3"
        ) == 0
    );
}

static void test_rejects_bad_utf8(void)
{
    static const char digest[] =
        "cccccccccccccccccccccccccccccccc"
        "cccccccccccccccccccccccccccccccc";

    static const char invalid_utf8[] = "\xed\xa0\x80";

    const elpis_ecsc_binding_view bad[] = {
        {"value", 5u, invalid_utf8, sizeof invalid_utf8 - 1u}
    };

    size_t need = 0u;

    assert(
        elpis_ecsc_receipt_payload_size(
            "ecs_g", 5u,
            "cognition.turn", 14u,
            digest, 64u,
            bad, 1u,
            &need
        ) == ELPIS_ECSC_UTF8
    );
}

int main(void)
{
    test_receipt_payload();
    test_rejects_unsorted_bindings();
    test_digest_bytes();
    test_rejects_bad_utf8();

    puts("PASS_ECSC_HISTORY_CODEC");
    return 0;
}
