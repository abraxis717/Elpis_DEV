#ifndef ELPIS_CENSUS_FIXTURES_H
#define ELPIS_CENSUS_FIXTURES_H
// The exact test_regex_execution fixture list, in its submission order, with names.
#include "execution_fixtures.h"

struct CensusFixture {
    std::string name, source;
};
inline std::vector<CensusFixture> census_fixtures() {
    static const char *small_names[] = {"small.empty", "small.irrelevant_prose", "small.at_least_1",
                                        "small.no_less_than_neg", "small.touching_may_merge",
                                        "small.touching_do_not_merge_and_may", "small.control_whitespace",
                                        "small.unicode_mixed", "small.embedded_nul", "small.invalid_utf8"};
    static const char *large_names[] = {"large.1mib_nomatch_then_exactly", "large.1mib_whitespace_at_least",
                                        "large.1mib_nomatch_then_lexical_phrase"};
    std::vector<CensusFixture> out;
    auto small = execution_fixtures("small");
    auto large = execution_fixtures("large");
    for (size_t i = 0; i < small.size(); ++i) out.push_back({small_names[i], small[i]});
    for (size_t i = 0; i < large.size(); ++i) out.push_back({large_names[i], large[i]});
    out.push_back({"long_token_rejection", std::string("at least ") + std::string(128, 'x')});
    std::string many;
    for (unsigned i = 0; i < 4097; ++i) many += "exactly 1; ";
    out.push_back({"evidence_limit_4097", many});
    return out;
}
#endif
