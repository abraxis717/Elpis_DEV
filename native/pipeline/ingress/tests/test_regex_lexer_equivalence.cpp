// Differential oracle: the R1 lexer against the frozen public-main (R0) lexer.
//
// Both engines run the same grammar over the same bytes with the same chunking
// and evidence limit. Every observable must be identical: which feed/finish call
// fails and how (exception class and message), and on success every match's
// pattern, offsets, inline text, digest, omission flag and captures, plus the
// byte count and source digest. The V2 result JSON is a pure function of these.
// Where the oracle runs one chunking, R1 must reproduce that outcome for every
// other chunking too (the failing call index is compared only for the same one).
//
// R0 is the slow engine R1 replaces, so the default (ctest) run bounds its work:
// the oracle runs once per input and R1 covers the chunkings, spans stop just past
// the inline-text limit, and 60 random cases. `full` runs the exhaustive campaign
// (every chunking and split against the oracle, spans past window compaction, 400
// random cases by default); use it for lexer changes and seeded fuzzing.
//
//   test_regex_lexer_equivalence [RANDOM_CASES] [SEED] [full]
#include "incremental_lexer.h"
#include "incremental_lexer_r0.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {
struct Outcome {
    std::string error;  // empty on success
    size_t failed_call = 0;
    std::vector<elpis_regex_v2::Match> matches;
    uint64_t bytes = 0;
    std::string source;
};
template <class Lexer, class RangeError, class Match>
Outcome run(const std::vector<std::string>& expressions, uint32_t max_evidence, const std::string& text,
            const std::vector<size_t>& cuts) {
    Outcome o;
    size_t call = 0;
    try {
        Lexer lexer(expressions, max_evidence);
        size_t offset = 0;
        for (size_t cut : cuts) {
            size_t n = std::min(cut, text.size() - offset);
            ++call;
            lexer.feed(reinterpret_cast<const uint8_t*>(text.data() + offset), n);
            offset += n;
        }
        ++call;
        if (offset < text.size()) lexer.feed(reinterpret_cast<const uint8_t*>(text.data() + offset), text.size() - offset);
        ++call;
        std::vector<Match> m = lexer.finish();
        for (auto& x : m) {
            elpis_regex_v2::Match y;
            y.pattern = x.pattern; y.start = x.start; y.end = x.end; y.text = x.text; y.digest = x.digest;
            y.text_omitted = x.text_omitted; y.captures = x.captures;
            o.matches.push_back(std::move(y));
        }
        o.bytes = lexer.bytes();
        o.source = lexer.source_digest();
    } catch (const std::bad_alloc&) {
        o.error = "NOMEM"; o.failed_call = call;
    } catch (const RangeError&) {
        o.error = "RANGE"; o.failed_call = call;
    } catch (const std::exception& e) {
        o.error = std::string("ERR:") + e.what(); o.failed_call = call;
    }
    return o;
}
std::vector<std::string> expressions;
uint64_t cases = 0, matches_compared = 0, failures_compared = 0;

std::string show(const std::string& s) {
    std::string out;
    for (unsigned char c : s.substr(0, 300)) {
        if (c >= 32 && c < 127) out += char(c);
        else { char b[8]; std::snprintf(b, sizeof b, "\\x%02x", c); out += b; }
    }
    return out + (s.size() > 300 ? "...(" + std::to_string(s.size()) + " bytes)" : "");
}
Outcome r1(const std::string& text, const std::vector<size_t>& cuts, uint32_t max_evidence) {
    return run<elpis_regex_v2::Lexer, elpis_regex_v2::RangeError, elpis_regex_v2::Match>(expressions, max_evidence,
                                                                                         text, cuts);
}
// `a` is the reference. The failing call is compared only for the same chunking.
void compare(const Outcome& a, const Outcome& b, const std::string& text, const std::vector<size_t>& cuts,
             uint32_t max_evidence, bool same_chunking) {
    ++cases;
    auto fail = [&](const std::string& what) {
        std::cerr << "MISMATCH " << what << (same_chunking ? "" : " (R1 chunking invariance)")
                  << "\nmax_evidence=" << max_evidence << " cuts=" << cuts.size()
                  << "\ntext=" << show(text) << "\nreference error=" << a.error << " at " << a.failed_call
                  << " matches=" << a.matches.size() << "\nR1 error=" << b.error << " at " << b.failed_call
                  << " matches=" << b.matches.size() << "\n";
        std::exit(1);
    };
    if (a.error != b.error || (same_chunking && a.failed_call != b.failed_call)) fail("failure");
    if (!a.error.empty()) { ++failures_compared; return; }
    if (a.bytes != b.bytes || a.source != b.source) fail("source");
    if (a.matches.size() != b.matches.size()) fail("match count");
    for (size_t i = 0; i < a.matches.size(); ++i) {
        const auto& x = a.matches[i];
        const auto& y = b.matches[i];
        if (x.pattern != y.pattern || x.start != y.start || x.end != y.end) fail("match span " + std::to_string(i));
        if (x.text != y.text || x.text_omitted != y.text_omitted) fail("match text " + std::to_string(i));
        if (x.digest != y.digest) fail("match digest " + std::to_string(i));
        if (x.captures != y.captures) fail("match captures " + std::to_string(i));
        ++matches_compared;
    }
}
Outcome r0(const std::string& text, const std::vector<size_t>& cuts, uint32_t max_evidence) {
    return run<elpis_regex_v2_r0::Lexer, elpis_regex_v2_r0::RangeError, elpis_regex_v2_r0::Match>(
        expressions, max_evidence, text, cuts);
}
void check(const std::string& text, const std::vector<size_t>& cuts, uint32_t max_evidence) {
    compare(r0(text, cuts, max_evidence), r1(text, cuts, max_evidence), text, cuts, max_evidence, true);
}
// One oracle run with `cuts`; R1 must match it there and under every other chunking.
void check_chunkings(const std::string& text, const std::vector<size_t>& cuts,
                     const std::vector<std::vector<size_t>>& others, uint32_t max_evidence) {
    Outcome a = r0(text, cuts, max_evidence);
    compare(a, r1(text, cuts, max_evidence), text, cuts, max_evidence, true);
    for (const auto& c : others) compare(a, r1(text, c, max_evidence), text, c, max_evidence, false);
}
// No oracle: R1 must give the same outcome under every chunking.
void check_r1_invariance(const std::string& text, const std::vector<std::vector<size_t>>& chunkings,
                         uint32_t max_evidence) {
    Outcome a = r1(text, chunkings[0], max_evidence);
    for (size_t i = 1; i < chunkings.size(); ++i)
        compare(a, r1(text, chunkings[i], max_evidence), text, chunkings[i], max_evidence, false);
}
std::vector<size_t> cuts_every(size_t size, size_t k) { return std::vector<size_t>(size / k + 1, k); }

const std::vector<std::string> words = {
    "at", "least", "no", "less", "than", "greater", "or", "equal", "to", "more", "strictly", "above", "most",
    "below", "other", "not", "unequal", "exactly", "clamp", "bound", "restrict", "keep", "between", "from", "and",
    "through", "is", "as", "the", "lower", "upper", "limit", "minimum", "maximum", "floor", "ceiling", "touch",
    "touching", "endpoint", "endpoints", "boundaries", "contact", "do", "does", "don't", "doesn't", "merge",
    "strict", "overlap", "overlapping", "positive", "width", "positive-width", "interior", "interiors", "abut",
    "abutting", "intervals", "interval", "ranges", "range", "may", "can", "should", "must", "will", "also", "max",
    "min", "greatest", "farthest", "farther", "larger", "smallest", "nearest", "nearer", "lesser", "end", "ending",
    "coordinate", "right", "edge", "AT", "Least", "EXACTLY", "x", "value", "lo", "hi", "ab"};
const std::vector<std::string> exotic = {
    "\xcf\x80", "\xe2\x84\xaa", "\xc5\xbf", "\xd9\xa1", "\xc3\xa9", "\xe2\x98\x83", "\xf0\x9f\x98\x80",
    "\xc2\xa0", "\xe2\x80\x83", std::string("\0", 1), "\xef\xbc\x91"};
const std::string spaces = " \t\n\v\f\r";

std::string generate(std::mt19937_64& r) {
    auto pick = [&](size_t n) { return size_t(r() % n); };
    std::string s;
    size_t tokens = 1 + pick(r() % 8 == 0 ? 120 : 30);
    for (size_t t = 0; t < tokens; ++t) {
        size_t kind = pick(100);
        if (kind < 55) s += words[pick(words.size())];
        else if (kind < 70) {  // number
            if (pick(3) == 0) s += "+-"[pick(2)];
            size_t d = pick(r() % 6 == 0 ? 40 : 5);
            for (size_t i = 0; i < d; ++i) s += char('0' + pick(10));
            if (pick(3) == 0) { s += '.'; size_t f = pick(r() % 6 == 0 ? 20 : 4); for (size_t i = 0; i < f; ++i) s += char('0' + pick(10)); }
        } else if (kind < 78) {  // identifier, sometimes at the 127/128 boundary
            size_t n = pick(4) == 0 ? 125 + pick(6) : 1 + pick(12);
            s += "_abcXYZ"[pick(7)];
            for (size_t i = 1; i < n; ++i) s += "abcxyzQ019_"[pick(11)];
        } else if (kind < 84) s += exotic[pick(exotic.size())];
        else if (kind < 92) s += ";,.-+'!"[pick(7)];
        else if (kind < 94 && pick(4) == 0) s += std::string("\x80\xff\xc0\xe2\x82\xed\xa0\x80").substr(pick(8), 1 + pick(2));
        else if (kind < 96) {  // a long whitespace run
            size_t n = pick(4) == 0 ? 4000 + pick(300) : 100 + pick(900);
            for (size_t i = 0; i < n; ++i) s += spaces[pick(pick(3) ? 1 : spaces.size())];
        }
        size_t sep = pick(10);
        if (sep < 7) s += ' ';
        else if (sep < 9) { size_t n = 1 + pick(4); for (size_t i = 0; i < n; ++i) s += spaces[pick(spaces.size())]; }
    }
    return s;
}
std::vector<size_t> random_cuts(std::mt19937_64& r, size_t size) {
    std::vector<size_t> cuts;
    size_t mode = r() % 4;
    if (mode == 0) return cuts;  // one feed
    for (size_t used = 0; used < size;) {
        size_t k = mode == 1 ? 1 + r() % 3 : mode == 2 ? 1 + r() % 64 : 1 + r() % 5000;
        cuts.push_back(k); used += k;
    }
    return cuts;
}
}  // namespace

int main(int argc, char** argv) {
    const bool full = argc > 3 && std::string(argv[3]) == "full";
    size_t random_cases = argc > 1 ? std::stoul(argv[1]) : full ? 400 : 60;
    uint64_t seed = argc > 2 ? std::stoull(argv[2]) : 0x5eed0001u;
    expressions = elpis_regex_v2::grammar_expressions();
    // 1. Fixture corpus (test_regex_stream + execution fixtures), each with several chunkings.
    std::vector<std::string> corpus = {
        "", "irrelevant prose", "at least 1", "no less than -2.5", "greater than or equal to +.25", "more than 1",
        "greater than 2", "strictly greater than 3", "above 4", "at most 5", "no greater than 6",
        "less than or equal to 7", "less than 8", "strictly less than 9", "below 10", "other than 1",
        "not equal to 2", "unequal to 3", "exactly equal to 4", "equal to 5", "exactly 6",
        "clamp x between lower and upper", "bound VALUE from LO through HI", "restrict a between b to c",
        "keep a from b and c", "x is the lower bound", "x as lower limit", "x is minimum", "x as floor",
        "x is the upper bound", "x as upper limit", "x is maximum", "x as ceiling", "touching endpoints do not merge",
        "touch boundaries does not merge", "endpoint contact don't merge",
        "touch endpoints these five other long words doesn't merge", "strict overlap", "strictly overlapping",
        "positive-width overlap", "positive width overlap", "interior overlap", "interiors overlap",
        "overlap or touch may merge", "overlapping or abutting ranges can also merge", "touch endpoints should merge",
        "abutting boundaries will merge", "maximum end", "max endpoint", "greatest ending coordinate",
        "farthest right edge", "farther end", "larger end", "minimum end", "min endpoint", "smallest ending coordinate",
        "nearest right edge", "nearer end", "lesser end",
        "touching endpoints do not merge; touching endpoints may merge; maximum end; minimum end",
        "at least 1; at least 1; exactly 2; x is floor; y is floor",
        "touch endpoints do not merge do not merge; touching endpoints do not merge", "at least +1.23456789012345678",
        "at least 1.2X", "at least 1.X", "at least 12345678901234567890123456789012345",
        "positive  width overlap; positive\twidth overlap", "at\n\t\r least\v\f-1.5",
        "\xcf\x80 at least 1 \xe2\x98\x83 \xf0\x9f\x98\x80", "at\xc2\xa0least\xe2\x80\x83+.5",
        "\xe2\x84\xaa is floor; \xc5\xbf is ceiling", "at least \xd9\xa1", std::string("maximum end\0minimum end", 23),
        std::string("at least ") + std::string(128, 'x'),
        std::string("keep ") + std::string(128, 'a') + " from " + std::string(128, 'b') + " to " + std::string(128, 'c'),
        "touching may merge; maximum end.", "\xff invalid utf-8", "exactly 1\x80", "\xe2\x82", "exactly 1 \xed\xa0\x80"};
    for (const auto& text : corpus) {
        if (full) {
            check(text, {}, 4096);
            for (size_t k : {1u, 2u, 7u, 13u}) check(text, cuts_every(text.size(), k), 4096);
            for (size_t split = 0; split <= text.size() && text.size() < 200; ++split) check(text, {split}, 4096);
            continue;
        }
        std::vector<std::vector<size_t>> others;
        for (size_t k : {1u, 2u, 7u, 13u}) others.push_back(cuts_every(text.size(), k));
        for (size_t split = 0; split <= text.size() && text.size() < 200; ++split) others.push_back({split});
        check_chunkings(text, {}, others, 4096);
    }
    // 2. Long spans: whitespace across the inline-text limit and the window, idle and
    //    word runs, captures that survive long whitespace, evidence limits.
    for (size_t n : {1u, 511u, 512u, 513u, 4087u, 4095u, 4096u, 4097u, 5000u, 20000u, 40000u}) {
        std::string ws;
        for (size_t i = 0; i < n; ++i) ws += spaces[i % 3 == 0 ? (i / 3) % spaces.size() : 0];
        size_t variant = 0;
        for (const std::string& t : {"at" + ws + "least 1", "x" + ws + "is" + ws + "floor", "clamp v" + ws + "between lo" + ws +
                                                                                        "and hi",
                                     "touching" + ws + "endpoints do" + ws + "not merge",
                                     std::string(n, 'z') + " exactly 1 " + std::string(n, 'z')}) {
            ++variant;
            if (!full) {
                // Oracle up to just past the inline-text limit (4096), two of the five
                // shapes (plain \s+ run, captures across runs) at the 4k sizes; past
                // window compaction R1 must be chunk-invariant.
                std::vector<std::vector<size_t>> chunkings = {cuts_every(t.size(), 4093), {},
                                                              cuts_every(t.size(), 1 + n / 3)};
                if (n <= 513 || (n <= 4097 && n != 4087 && (variant == 1 || variant == 3)))
                    check_chunkings(t, chunkings[0], {chunkings[1], chunkings[2]}, 4096);
                else if (n > 5000 && (variant == 1 || variant == 5))
                    check_r1_invariance(t, chunkings, 4096);
                continue;
            }
            if (n <= 5000) {
                check(t, {}, 4096);
                check(t, cuts_every(t.size(), 4093), 4096);
                check(t, cuts_every(t.size(), 1 + n / 3), 4096);
                continue;
            }
            // Spans past one or more window compactions: the (slow) R0 oracle runs once,
            // with feeds that straddle the window; R1 must give the same outcome for the
            // other chunkings.
            check(t, cuts_every(t.size(), 4093), 4096);
            Outcome a = r1(t, cuts_every(t.size(), 4093), 4096);
            compare(a, r1(t, {}, 4096), t, {}, 4096, false);
            compare(a, r1(t, cuts_every(t.size(), 1 + n / 3), 4096), t, cuts_every(t.size(), 1 + n / 3), 4096, false);
        }
    }
    // Non-ASCII whitespace is never batched: the per-codepoint path must compact its window.
    for (size_t n : {3000u, 12000u}) {
        std::string nbsp;
        for (size_t i = 0; i < n; ++i) nbsp += i % 2 ? "\xc2\xa0" : "\xe2\x80\x83";
        for (const std::string& t : {"at" + nbsp + "least 1", "x " + nbsp + "is floor", "clamp v" + nbsp + "between a and b"}) {
            if (full) {
                check(t, {}, 4096);
                check(t, cuts_every(t.size(), 4099), 4096);
            } else if (n == 3000) {
                check_chunkings(t, cuts_every(t.size(), 4099), {{}}, 4096);
            } else {
                check_r1_invariance(t, {cuts_every(t.size(), 4099), {}}, 4096);
            }
        }
    }
    {
        const unsigned reps = full ? 300 : 60;
        std::string many;
        for (unsigned i = 0; i < reps; ++i) many += "exactly 1; ";
        for (uint32_t cap : {1u, 2u, reps - 1, reps, reps + 1}) {
            if (full) { check(many, {}, cap); check(many, cuts_every(many.size(), 17), cap); }
            else check_chunkings(many, {}, {cuts_every(many.size(), 17)}, cap);
        }
    }
    // 3. Randomized grammar-aware inputs, chunkings and evidence limits.
    std::mt19937_64 r(seed);
    for (size_t i = 0; i < random_cases; ++i) {
        std::string text = generate(r);
        uint32_t cap = r() % 8 == 0 ? uint32_t(1 + r() % 4) : 4096;
        check(text, random_cuts(r, text.size()), cap);
    }
    std::cout << "PASS R1/R0 lexer equivalence cases=" << cases << " matches=" << matches_compared
              << " failures=" << failures_compared << " random=" << random_cases << " seed=" << seed
              << (full ? " full" : " bounded") << "\n";
}
