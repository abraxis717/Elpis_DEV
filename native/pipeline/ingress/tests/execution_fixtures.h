#ifndef ELPIS_EXECUTION_FIXTURES_H
#define ELPIS_EXECUTION_FIXTURES_H
#include "regex_execution.h"
#include "streaming_regex_ingress_v2.h"
#include <cassert>
#include <cstring>
#include <string>
#include <vector>

/* Families from test_regex_stream.cpp / test_query_ingress_bounded.cpp. Large
 * cases scale the existing no-match and whitespace fixtures, not a new kernel. */
inline std::vector<std::string> execution_fixtures(const std::string& family) {
    if (family == "large") return {
        std::string(1024*1024, 'z') + "; exactly 1; maximum end.",
        std::string("at") + std::string(1024*1024, ' ') + "least 1",
        std::string(1024*1024, 'z') + "; touching endpoints do not merge; touching endpoints may merge; maximum end."
    };
    return {"", "irrelevant prose", "at least 1", "no less than -2.5",
            "touching endpoints may merge; maximum end.",
            "touching endpoints do not merge; touching endpoints may merge; maximum end.",
            "at\n\t\r least\v\f-1.5", "\xcf\x80 at least 1 \xe2\x98\x83 \xf0\x9f\x98\x80",
            std::string("maximum end\0minimum end", 23), "\xff invalid utf-8"};
}
inline elpis_exec_buffer *input_buffer(const std::string& s) {
    auto *b = elpis_exec_buffer_alloc(s.size()); assert(b);
    std::memcpy(elpis_exec_buffer_mutable_data(b), s.data(), s.size());
    return b;
}
inline std::pair<int, std::string> original(const std::string& s) {
    elpis_streaming_regex_stream_v2 *stream = nullptr;
    elpis_streaming_regex_result_v1 *result = nullptr;
    int rc = elpis_streaming_regex_stream_create_v2(nullptr, &stream);
    for (size_t i=0; !rc && i<s.size();) {
        size_t n = std::min(size_t(65536), s.size()-i);
        rc = elpis_streaming_regex_stream_feed_v2(stream,
                    reinterpret_cast<const uint8_t*>(s.data()+i), n); i += n;
    }
    if (!rc) rc = elpis_streaming_regex_stream_finalize_v2(stream, &result);
    elpis_streaming_regex_stream_destroy_v2(stream);
    std::string json = rc ? "" : elpis_streaming_regex_result_json_v1(result);
    elpis_streaming_regex_result_destroy_v1(result);
    return {rc, json};
}
#endif
