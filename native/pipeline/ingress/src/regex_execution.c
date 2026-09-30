#include "regex_execution.h"
#include "streaming_regex_ingress_v2.h"
#include <string.h>

elpis_exec_status elpis_regex_execute(const elpis_exec_buffer *input, size_t cap,
                                    elpis_exec_buffer **out) {
    if (!input || !out || !cap) return ELPIS_EXEC_INVALID;
    *out = NULL;
    elpis_streaming_regex_stream_v2 *stream = NULL;
    elpis_streaming_regex_result_v1 *result = NULL;
    int code = elpis_streaming_regex_stream_create_v2(NULL, &stream);
    const unsigned char *bytes = elpis_exec_buffer_data(input);
    size_t size = elpis_exec_buffer_size(input);
    for (size_t offset = 0; !code && offset < size;) {
        size_t n = size - offset;
        if (n > 65536) n = 65536;
        code = elpis_streaming_regex_stream_feed_v2(stream, bytes + offset, n);
        offset += n;
    }
    if (!code) code = elpis_streaming_regex_stream_finalize_v2(stream, &result);
    elpis_streaming_regex_stream_destroy_v2(stream);
    elpis_exec_status status = ELPIS_EXEC_INVALID;
    if (!code) {
        const char *json = elpis_streaming_regex_result_json_v1(result);
        size_t n = strlen(json) + 1;
        if (n <= cap) {
            *out = elpis_exec_buffer_alloc(n);
            status = *out ? ELPIS_EXEC_OK : ELPIS_EXEC_INTERNAL;
            if (*out) memcpy(elpis_exec_buffer_mutable_data(*out), json, n);
        }
    } else if (code == ELPIS_STREAMING_REGEX_E_NOMEM) status = ELPIS_EXEC_INTERNAL;
    elpis_streaming_regex_result_destroy_v1(result);
    return status;
}
