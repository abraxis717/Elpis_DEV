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


static void test_message_envelope_and_frame(void)
{
    static const char sender[] =
        "11111111111111111111111111111111"
        "11111111111111111111111111111111";

    static const char receiver[] =
        "22222222222222222222222222222222"
        "22222222222222222222222222222222";

    static const uint8_t payload[] = "{\"x\":1}";

    static const char expected_payload_digest[] =
        "e82fca748aced016cdc5c2b9dfbf0dfd"
        "4ce3187e0f4a33d71bfaa7e4619423ef";

    static const char expected_message_id[] =
        "7f3771bdab5943d0098f704020e9c516"
        "b77a318b4fc8eb7fc81aa4818ab47d8e";

    static const char expected_envelope[] =
        "{\"logical_clock\":11,"
        "\"message_id\":\"7f3771bdab5943d0098f704020e9c516b77a318b4fc8eb7fc81aa4818ab47d8e\","
        "\"payload_digest\":\"e82fca748aced016cdc5c2b9dfbf0dfd4ce3187e0f4a33d71bfaa7e4619423ef\","
        "\"payload_hex\":\"7b2278223a317d\","
        "\"receiver_entity_id\":\"2222222222222222222222222222222222222222222222222222222222222222\","
        "\"schema\":\"ecs.message.v1\","
        "\"sender_entity_id\":\"1111111111111111111111111111111111111111111111111111111111111111\","
        "\"sequence\":7}";

    char message_id[65];
    char payload_digest[65];
    uint8_t envelope[1024];
    uint8_t frame[2048];
    size_t envelope_size = 0u;
    size_t written = 0u;
    size_t frame_size = 0u;

    assert(
        elpis_ecsc_message_id(
            sender,
            receiver,
            7u,
            payload,
            sizeof payload - 1u,
            message_id,
            payload_digest
        ) == ELPIS_ECSC_OK
    );

    assert(strcmp(message_id, expected_message_id) == 0);
    assert(strcmp(payload_digest, expected_payload_digest) == 0);

    assert(
        elpis_ecsc_envelope_size(
            sender,
            receiver,
            7u,
            payload,
            sizeof payload - 1u,
            11u,
            &envelope_size
        ) == ELPIS_ECSC_OK
    );

    assert(envelope_size == sizeof expected_envelope - 1u);

    assert(
        elpis_ecsc_envelope_write(
            sender,
            receiver,
            7u,
            payload,
            sizeof payload - 1u,
            11u,
            envelope,
            sizeof envelope,
            &written,
            message_id,
            payload_digest
        ) == ELPIS_ECSC_OK
    );

    assert(written == envelope_size);
    assert(memcmp(envelope, expected_envelope, written) == 0);

    assert(
        elpis_ecsc_event_frame_size(
            written,
            &frame_size
        ) == ELPIS_ECSC_OK
    );

    assert(frame_size == written + 8u);

    assert(
        elpis_ecsc_event_frame_write(
            envelope,
            written,
            frame,
            sizeof frame,
            &frame_size
        ) == ELPIS_ECSC_OK
    );

    assert(frame_size == written + 8u);

    assert(frame[0] == 0u);
    assert(frame[1] == 0u);
    assert(frame[2] == 0u);
    assert(frame[3] == 0u);
    assert(frame[4] == 0u);
    assert(frame[5] == 0u);
    assert(frame[6] == 0x01u);
    assert(frame[7] == 0xacu);

    assert(memcmp(frame + 8u, envelope, written) == 0);
}

static void test_message_and_frame_refusals(void)
{
    static const char sender[] =
        "11111111111111111111111111111111"
        "11111111111111111111111111111111";

    static const char receiver[] =
        "22222222222222222222222222222222"
        "22222222222222222222222222222222";

    static const uint8_t payload[] = "x";

    char message_id[65];
    char payload_digest[65];
    uint8_t out[32];
    size_t written = 99u;
    size_t size = 99u;

    assert(
        elpis_ecsc_message_id(
            sender,
            receiver,
            0u,
            payload,
            sizeof payload - 1u,
            message_id,
            payload_digest
        ) == ELPIS_ECSC_INVALID
    );

    assert(
        elpis_ecsc_message_id(
            sender,
            receiver,
            UINT64_MAX,
            payload,
            sizeof payload - 1u,
            message_id,
            payload_digest
        ) == ELPIS_ECSC_INVALID
    );

    assert(
        elpis_ecsc_event_frame_size(
            0u,
            &size
        ) == ELPIS_ECSC_INVALID
    );
    assert(size == 0u);

    assert(
        elpis_ecsc_event_frame_size(
            262145u,
            &size
        ) == ELPIS_ECSC_INVALID
    );
    assert(size == 0u);

    assert(
        elpis_ecsc_event_frame_write(
            payload,
            sizeof payload - 1u,
            out,
            8u,
            &written
        ) == ELPIS_ECSC_CAPACITY
    );
    assert(written == 0u);
}


static void test_committed_message_events(void)
{
    static const char sender[] =
        "11111111111111111111111111111111"
        "11111111111111111111111111111111";

    static const char receiver[] =
        "22222222222222222222222222222222"
        "22222222222222222222222222222222";

    static const char before_root[] =
        "33333333333333333333333333333333"
        "33333333333333333333333333333333";

    static const char after_root[] =
        "44444444444444444444444444444444"
        "44444444444444444444444444444444";

    static const char previous[] =
        "55555555555555555555555555555555"
        "55555555555555555555555555555555";

    static const uint8_t payload[] = "receipt";

    uint8_t event[4096];
    size_t need = 0u;
    size_t written = 0u;
    char event_digest[65];
    char intent_digest[65];
    char message_id[65];

    assert(
        elpis_ecsc_enqueue_event_size(
            sender,
            receiver,
            3u,
            payload,
            sizeof payload - 1u,
            9u,
            before_root,
            after_root,
            previous,
            &need
        ) == ELPIS_ECSC_OK
    );

    assert(need > 0u && need < sizeof event);

    assert(
        elpis_ecsc_enqueue_event_write(
            sender,
            receiver,
            3u,
            payload,
            sizeof payload - 1u,
            9u,
            before_root,
            after_root,
            previous,
            event,
            sizeof event,
            &written,
            event_digest,
            intent_digest,
            message_id
        ) == ELPIS_ECSC_OK
    );

    assert(written == need);
    assert(written < sizeof event);
    event[written] = 0u;

    assert(strlen(event_digest) == 64u);
    assert(strlen(intent_digest) == 64u);
    assert(strlen(message_id) == 64u);

    assert(
        strstr(
            (const char *)event,
            "\"event_kind\":\"MESSAGE_ENQUEUED\""
        ) != NULL
    );

    assert(
        elpis_ecsc_processed_event_size(
            receiver,
            message_id,
            10u,
            after_root,
            before_root,
            event_digest,
            &need
        ) == ELPIS_ECSC_OK
    );

    assert(
        elpis_ecsc_processed_event_write(
            receiver,
            message_id,
            10u,
            after_root,
            before_root,
            event_digest,
            event,
            sizeof event,
            &written,
            event_digest,
            intent_digest
        ) == ELPIS_ECSC_OK
    );

    assert(written < sizeof event);
    event[written] = 0u;

    assert(
        strstr(
            (const char *)event,
            "\"event_kind\":\"MESSAGE_PROCESSED\""
        ) != NULL
    );
}


static void test_history_identity_primitives(void)
{
    char genesis_v1[65];
    char genesis_v2[65];
    char entity[65];
    char initial_state[65];

    assert(
        elpis_ecsc_genesis_digest(
            "elpis.runtime.history.v1",
            sizeof("elpis.runtime.history.v1") - 1u,
            ELPIS_ECSC_SCHEDULER_V1,
            genesis_v1
        ) == ELPIS_ECSC_OK
    );

    assert(
        elpis_ecsc_genesis_digest(
            "elpis.runtime.history.v1",
            sizeof("elpis.runtime.history.v1") - 1u,
            ELPIS_ECSC_SCHEDULER_V2,
            genesis_v2
        ) == ELPIS_ECSC_OK
    );

    assert(strlen(genesis_v1) == 64u);
    assert(strlen(genesis_v2) == 64u);

    /* Scheduler protocol is part of genesis authority. */
    assert(strcmp(genesis_v1, genesis_v2) != 0);

    assert(
        elpis_ecsc_entity_id(
            0u,
            "history",
            sizeof("history") - 1u,
            genesis_v2,
            entity
        ) == ELPIS_ECSC_OK
    );

    assert(strlen(entity) == 64u);

    assert(
        elpis_ecsc_initial_state_digest(
            entity,
            initial_state
        ) == ELPIS_ECSC_OK
    );

    assert(strlen(initial_state) == 64u);
}

static void test_history_identity_refusals(void)
{
    char out[65];
    char invalid_digest[65];

    memset(invalid_digest, 'g', 64u);
    invalid_digest[64] = '\0';

    assert(
        elpis_ecsc_genesis_digest(
            "",
            0u,
            ELPIS_ECSC_SCHEDULER_V2,
            out
        ) == ELPIS_ECSC_INVALID
    );

    assert(
        elpis_ecsc_genesis_digest(
            "history",
            sizeof("history") - 1u,
            99u,
            out
        ) == ELPIS_ECSC_INVALID
    );

    assert(
        elpis_ecsc_entity_id(
            UINT64_MAX,
            "history",
            sizeof("history") - 1u,
            "11111111111111111111111111111111"
            "11111111111111111111111111111111",
            out
        ) == ELPIS_ECSC_INVALID
    );

    assert(
        elpis_ecsc_initial_state_digest(
            invalid_digest,
            out
        ) == ELPIS_ECSC_INVALID
    );
}

int main(void)
{
    test_receipt_payload();
    test_rejects_unsorted_bindings();
    test_digest_bytes();
    test_rejects_bad_utf8();
    test_message_envelope_and_frame();
    test_message_and_frame_refusals();
    test_committed_message_events();
    test_history_identity_primitives();
    test_history_identity_refusals();

    puts("PASS_ECSC_HISTORY_CODEC");
    return 0;
}
