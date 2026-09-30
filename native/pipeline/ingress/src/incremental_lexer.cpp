#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include "incremental_lexer.h"
#include <algorithm>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

// Engine layout (R1). The grammar parser and the compiled Thompson program are
// unchanged from R0, so the ordered-closure semantics (first arrival per PC
// wins, accept cuts lower priority, greedy paths may replace a winner) are the
// same program executed by a different state representation:
//
//  * Threads are fixed-size POD: pc, capture-open mask and absolute source spans
//    per capture. A capture's bytes are exactly the source bytes of its span,
//    because every thread of a candidate consumes the same codepoints.
//  * Candidate text, capture bytes and match hashes read one shared, bounded
//    window of recent source bytes instead of per-thread string copies. Bytes
//    that can still be referenced but would leave the window are materialized
//    first (closed capture spans, a won match's inline text); hashes are brought
//    up to date in bulk.
//  * Closure scratch (stack, ready list, next list) and the per-program
//    first-arrival table (generation counters) are owned by the lexer and reused.
//  * Predicates are evaluated lazily per codepoint: one 64-bit ASCII signature
//    table, and a direct-mapped cache of PCRE2 results for other codepoints.
//  * Runs of identical-signature ASCII bytes are applied in bulk once one step
//    has proven every live candidate is at a fixed point for that signature
//    (no accept, no death, identical thread list, no boundary change). Idle
//    ASCII text is scanned without per-codepoint dispatch.
//  * Candidates that cannot survive their first codepoint are not created.
//  * Diagnostics are counters maintained at state changes, not walks.
namespace elpis_regex_v2 {
namespace {
std::string digest(elpis_sha256_ctx c) {
    uint8_t d[32]; elpis_sha256_final(&c,d);
    static const char h[]="0123456789abcdef";
    std::string s(64, '0');
    for(size_t i=0;i<32;++i) { s[2*i]=h[d[i]>>4]; s[2*i+1]=h[d[i]&15]; }
    return s;
}
using Code = std::shared_ptr<pcre2_code>;
using Data = std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)>;
struct Predicate {
    Code code{nullptr, pcre2_code_free};
    Data data{nullptr, pcre2_match_data_free};
    std::array<bool,128> ascii{};
    Predicate(const Predicate& p):code(p.code),data(pcre2_match_data_create(1,nullptr),pcre2_match_data_free),ascii(p.ascii) {
        if(!data) throw std::bad_alloc();
    }
    Predicate(Predicate&&)=default;
    Predicate& operator=(Predicate&&)=default;
    explicit Predicate(const std::string& atom) {
        std::string expr="\\A(?:"+atom+")\\z";
        int error; PCRE2_SIZE offset;
        code.reset(pcre2_compile(reinterpret_cast<PCRE2_SPTR>(expr.data()), expr.size(),
                   PCRE2_UTF|PCRE2_UCP|PCRE2_CASELESS, &error, &offset, nullptr),pcre2_code_free);
        if(!code) {
            if(error==PCRE2_ERROR_HEAP_FAILED) throw std::bad_alloc();
            throw std::runtime_error("V2_PREDICATE_COMPILE");
        }
        data.reset(pcre2_match_data_create_from_pattern(code.get(),nullptr));
        if(!data) throw std::bad_alloc();
        for(unsigned i=0;i<128;++i) {
            char c=static_cast<char>(i); ascii[i]=test(&c,1);
        }
    }
    bool test(const char* s, size_t n) {
        int rc=pcre2_match(code.get(),reinterpret_cast<PCRE2_SPTR>(s),n,0,0,data.get(),nullptr);
        if(rc==PCRE2_ERROR_NOMATCH) return false;
        if(rc==PCRE2_ERROR_NOMEMORY) throw std::bad_alloc();
        if(rc<0) throw std::runtime_error("V2_PREDICATE_MATCH");
        return true;
    }
};
struct Node {
    enum Kind { atom, boundary, sequence, alternate, repeat, capture } kind;
    int arg=0, lo=0, hi=0;
    std::vector<Node> children;
    explicit Node(Kind k):kind(k) {}
};
struct Parser {
    const std::string& s; size_t pos=0;
    std::function<int(const std::string&)> predicate;
    Node expression() {
        Node n(Node::alternate); n.children.push_back(sequence());
        while(pos<s.size() && s[pos]=='|') { ++pos; n.children.push_back(sequence()); }
        return n;
    }
    Node sequence() {
        Node n(Node::sequence);
        while(pos<s.size() && s[pos]!=')' && s[pos]!='|') n.children.push_back(piece());
        return n;
    }
    int number() {
        int v=0; size_t begin=pos;
        while(pos<s.size() && s[pos]>='0' && s[pos]<='9') v=v*10+s[pos++]-'0';
        if(begin==pos || v>128) throw std::runtime_error("V2_GRAMMAR_BOUND");
        return v;
    }
    Node piece() {
        Node n(Node::atom);
        if(s[pos]=='(') {
            ++pos;
            if(s.compare(pos,2,"?:")==0) { pos+=2; n=expression(); }
            else if(s.compare(pos,2,"?<")==0) {
                pos+=2; size_t end=s.find('>',pos);
                const std::array<std::string,5> names={"scalar","value","lower","upper","subject"};
                auto it=std::find(names.begin(),names.end(),s.substr(pos,end-pos));
                if(it==names.end()) throw std::runtime_error("V2_GRAMMAR_CAPTURE");
                n=Node(Node::capture); n.arg=static_cast<int>(it-names.begin());
                pos=end+1; n.children.push_back(expression());
            } else throw std::runtime_error("V2_GRAMMAR_GROUP");
            if(pos>=s.size() || s[pos++]!=')') throw std::runtime_error("V2_GRAMMAR_CLOSE");
        } else if(s.compare(pos,2,"\\b")==0) { pos+=2; n=Node(Node::boundary); }
        else {
            size_t begin=pos++;
            if(s[begin]=='[') {
                while(pos<s.size() && s[pos]!=']') ++pos;
                if(pos==s.size()) throw std::runtime_error("V2_GRAMMAR_CLASS");
                ++pos;
            } else if(s[begin]=='\\') ++pos;
            n.arg=predicate(s.substr(begin,pos-begin));
        }
        if(pos<s.size() && (s[pos]=='?' || s[pos]=='+' || s[pos]=='{')) {
            Node r(Node::repeat);
            char q=s[pos++];
            if(q=='?') { r.lo=0; r.hi=1; }
            else if(q=='+') {
                // No other infinite repetition is accepted by this compiler.
                if(n.kind!=Node::atom || n.arg!=predicate("\\s"))
                    throw std::runtime_error("V2_UNBOUNDED_NONSPACE");
                r.lo=1; r.hi=-1;
            } else {
                r.lo=number();
                if(pos>=s.size() || s[pos++]!=',') throw std::runtime_error("V2_GRAMMAR_RANGE");
                r.hi=number();
                if(pos>=s.size() || s[pos++]!='}' || r.hi<r.lo)
                    throw std::runtime_error("V2_GRAMMAR_RANGE");
            }
            r.children.push_back(std::move(n)); return r;
        }
        return n;
    }
};
// Upper bound on consumed non-whitespace scalars. Whitespace is the only
// infinite loop; every new match start consumes a non-whitespace scalar.
size_t nonspace_bound(const Node& n, int space) {
    if(n.kind==Node::atom) return n.arg==space ? 0u : 1u;
    if(n.kind==Node::boundary) return 0;
    size_t total=0;
    for(const auto& c:n.children) {
        size_t b=nonspace_bound(c,space);
        if(n.kind==Node::alternate) total=std::max(total,b);
        else total+=b;
    }
    if(n.kind==Node::repeat) return n.hi<0 ? 0u : total*static_cast<size_t>(n.hi);
    return total;
}
constexpr size_t max_nonspace=512;
constexpr size_t max_instructions=4096;
constexpr uint64_t capture_limit=512;
constexpr size_t max_predicates=64;
struct Instruction {
    enum Op { consume, boundary, split, open, close, accept } op;
    int next=0, other=0, arg=0;
};
struct Program {
    std::vector<Instruction> code;
    int entry=0;
    int add(Instruction i) { code.push_back(i); return static_cast<int>(code.size()-1); }
    int compile(const Node& n, int next) {
        switch(n.kind) {
        case Node::atom: return add({Instruction::consume,next,0,n.arg});
        case Node::boundary: return add({Instruction::boundary,next,0,0});
        case Node::capture: {
            int end=add({Instruction::close,next,0,n.arg});
            int begin=compile(n.children[0],end);
            return add({Instruction::open,begin,0,n.arg});
        }
        case Node::sequence:
            for(auto i=n.children.rbegin();i!=n.children.rend();++i) next=compile(*i,next);
            return next;
        case Node::alternate: {
            int tail=compile(n.children.back(),next);
            for(size_t i=n.children.size()-1;i>0;--i) {
                int first=compile(n.children[i-1],next);
                tail=add({Instruction::split,first,tail,0});
            }
            return tail;
        }
        case Node::repeat:
            if(n.hi<0) {
                int loop=add({Instruction::split,0,next,0});
                int body=compile(n.children[0],loop);
                code[loop].next=body;
                return body;
            }
            for(int i=n.lo;i<n.hi;++i) {
                int body=compile(n.children[0],next);
                next=add({Instruction::split,body,next,0});
            }
            for(int i=0;i<n.lo;++i) next=compile(n.children[0],next);
            return next;
        }
        throw std::runtime_error("V2_GRAMMAR_OPCODE");
    }
    // Skip chains. `A{lo,hi}` of a single atom compiles its optional copies to
    //   S_j = split(B_j, S_{j-1}),  B_j = consume(A) -> S_{j-1},  S_0 = C.
    // In the ordered closure, first arrival at S_j marks exactly S_1..S_j (and
    // B_1..B_j) as seen: the DFS runs down the chain until it meets the highest
    // S already seen, and reaches C only if none was. So one watermark per chain
    // reproduces the seen set. Of the ready threads B_j..B_{w+1} that visit
    // yields, only B_j matters: all share A's predicate and the arriving thread's
    // captures (so any capture-limit error is raised by B_j first), and after the
    // consume every lower successor S_{i-1} is dropped at the next closure, since
    // B_j's successor S_{j-1} is processed before it and marks everything below.
    // Links are chained only while each middle split is referenced solely from
    // inside the chain, so every arrival from outside enters at the top.
    std::vector<int32_t> chain_of;       // per PC: chain id of a chain split, else -1
    std::vector<uint32_t> chain_index;   // per PC: j of S_j
    std::vector<int> chain_bottom;       // per chain: C
    void analyse_chains() {
        size_t n=code.size();
        std::vector<std::vector<int>> refs(n);
        refs[entry].push_back(-1);
        for(size_t i=0;i<n;++i) {
            const auto& x=code[i];
            if(x.op==Instruction::accept) continue;
            refs[x.next].push_back(static_cast<int>(i));
            if(x.op==Instruction::split) refs[x.other].push_back(static_cast<int>(i));
        }
        auto link=[&](int s) {
            if(code[s].op!=Instruction::split) return false;
            int b=code[s].next;
            return code[b].op==Instruction::consume && code[b].next==code[s].other &&
                   refs[b].size()==1 && refs[b][0]==s;
        };
        // down(s): the next lower link of the same chain, if it is internal-only.
        auto down=[&](int s) {
            int t=code[s].other, b=code[s].next;
            if(t==s || !link(t) || code[code[t].next].arg!=code[b].arg) return -1;
            for(int r:refs[t]) if(r!=s && r!=b) return -1;
            return t;
        };
        chain_of.assign(n,-1); chain_index.assign(n,0);
        std::vector<bool> below(n,false);
        for(size_t i=0;i<n;++i) if(link(static_cast<int>(i))) { int d=down(static_cast<int>(i)); if(d>=0) below[d]=true; }
        for(size_t i=0;i<n;++i) {
            int s=static_cast<int>(i);
            if(!link(s) || below[s]) continue;
            std::vector<int> members;
            for(int x=s;x>=0;x=down(x)) members.push_back(x);
            int id=static_cast<int>(chain_bottom.size());
            chain_bottom.push_back(code[members.back()].other);
            for(size_t k=0;k<members.size();++k) {
                chain_of[members[k]]=id;
                chain_index[members[k]]=static_cast<uint32_t>(members.size()-k);
            }
        }
    }
    // Predicates of the consume instructions reachable from entry when the entry
    // position is a word boundary (candidates only start there), and whether an
    // accept is reachable without consuming. A new candidate survives its first
    // codepoint iff an accept is reachable or one of these predicates holds.
    uint64_t start_predicates=0;
    bool start_accepts=false;
    void analyse_start() {
        std::vector<bool> seen(code.size(),false);
        std::vector<int> stack{entry};
        while(!stack.empty()) {
            int pc=stack.back(); stack.pop_back();
            if(seen[pc]) continue;
            seen[pc]=true;
            const auto& i=code[pc];
            if(i.op==Instruction::consume) start_predicates|=uint64_t(1)<<i.arg;
            else if(i.op==Instruction::accept) start_accepts=true;
            else if(i.op==Instruction::split) { stack.push_back(i.other); stack.push_back(i.next); }
            else stack.push_back(i.next);
        }
    }
};
constexpr size_t capture_count=5;
// Fixed-layout thread. Open captures always end at the current position; a
// capture that was never opened is the empty span [0,0).
struct Thread {
    uint32_t pc;
    uint32_t open;
    uint64_t begin[capture_count];
    uint64_t end[capture_count];
};
static_assert(std::is_trivially_copyable<Thread>::value, "POD thread");
static_assert(sizeof(Thread)==8+16*capture_count, "no padding: memcmp compares state");
struct Saved { uint64_t begin, end; size_t offset; };
struct Winner {
    uint64_t end=0;
    bool omitted=false, text_saved=false;
    elpis_sha256_ctx hash{};
    Thread thread{};
    std::string text;
};
// Candidates that start at the same codepoint consume identical bytes for as long
// as they live, so they share one incremental hash of [start, hashed).
struct HashStream {
    uint64_t hashed=0;
    elpis_sha256_ctx ctx{};
    uint32_t references=0;
};
struct Candidate {
    uint64_t start=0;
    uint32_t stream=0;
    bool omitted=false, done=false, won=false;
    Winner winner;
    std::vector<Thread> threads;
    std::string arena;            // materialized closed capture spans
    std::vector<Saved> saved;
};
// Codepoint predicate results for non-ASCII input, keyed by scalar value.
struct CacheEntry { uint32_t scalar=UINT32_MAX; uint64_t known=0, value=0; };
constexpr size_t cache_entries=1024;
// The window keeps at least the last `keep` source bytes while candidates live:
// a non-omitted candidate spans at most inline_limit bytes and an open capture
// at most capture_limit bytes. Compaction runs when the window exceeds `compact_at`.
constexpr uint64_t keep=inline_limit;
constexpr size_t compact_at=16*1024;
}

struct Lexer::Impl {
    std::map<std::string,int> predicate_ids;
    std::vector<Predicate> predicates;
    std::vector<Program> programs;
    std::array<uint64_t,128> ascii_signature{};
    std::array<bool,128> ascii_word{};
    std::vector<std::vector<uint32_t>> rows;          // candidate ids per program, start order
    std::vector<Candidate> pool;
    std::vector<uint32_t> free_ids;
    std::vector<HashStream> streams;
    std::vector<uint32_t> free_streams;
    uint32_t start_stream=UINT32_MAX;                  // stream opened at the current codepoint
    std::vector<std::vector<uint32_t>> seen;          // first-arrival generation per PC
    std::vector<uint32_t> generation;
    std::vector<std::vector<uint32_t>> chain_generation, chain_watermark;
    std::unique_ptr<Thread[]> stack, ready, next;
    size_t scratch=0;
    std::vector<Match> evidence;
    uint32_t limit;
    int word=0;
    uint64_t total=0, position=0;
    elpis_sha256_ctx source{};
    std::array<char,4> decoder{};
    size_t used=0, needed=0;
    bool previous_word=false, active=false;
    // Current codepoint.
    const char* current=nullptr;
    size_t current_size=0;
    bool current_ascii=false;
    uint64_t current_signature=0;
    CacheEntry* current_entry=nullptr;
    std::unique_ptr<CacheEntry[]> cache;
    // Window of recent source bytes: absolute positions [window_base, window_base+size).
    std::vector<uint8_t> window;
    uint64_t window_base=0;
    // Diagnostics, maintained at state changes.
    uint64_t live_candidates=0, live_threads=0, saved_text_bytes=0, saved_capture_bytes=0;
    Stats measured;

    int predicate(const std::string& s) {
        auto it=predicate_ids.find(s);
        if(it!=predicate_ids.end()) return it->second;
        int id=static_cast<int>(predicates.size());
        predicates.emplace_back(s); predicate_ids.emplace(s,id); return id;
    }
    Impl(const std::vector<std::string>& expressions, uint32_t max, bool use_cache=true):limit(max) {
        elpis_sha256_init(&source);
        if(use_cache) {
            // Fixed grammar compilation is shared and immutable. Match scratch,
            // Unicode predicate results and all stream state remain per handle.
            static const Impl grammar(expressions,0,false);
            predicates.reserve(grammar.predicates.size());
            for(const auto& p:grammar.predicates) predicates.emplace_back(p);
            programs=grammar.programs; word=grammar.word;
            ascii_signature=grammar.ascii_signature; ascii_word=grammar.ascii_word;
            measured.program_instructions=grammar.measured.program_instructions;
            prepare();
            return;
        }
        word=predicate("\\w");
        for(const auto& expression:expressions) {
            Parser parser{expression,0,[this](const std::string& s){return predicate(s);}};
            Node tree=parser.expression();
            if(parser.pos!=expression.size()) throw std::runtime_error("V2_GRAMMAR_TRAILING");
            if(nonspace_bound(tree,predicate("\\s"))>max_nonspace)
                throw std::runtime_error("V2_GRAMMAR_NONSPACE_BOUND");
            Program p; int end=p.add({Instruction::accept,0,0,0}); p.entry=p.compile(tree,end);
            if(p.code.size()>max_instructions) throw std::runtime_error("V2_PROGRAM_BOUND");
            p.analyse_start(); p.analyse_chains();
            measured.program_instructions+=p.code.size(); programs.push_back(std::move(p));
        }
        if(predicates.size()>max_predicates) throw std::runtime_error("V2_PREDICATE_BOUND");
        for(unsigned c=0;c<128;++c) {
            uint64_t s=0;
            for(size_t k=0;k<predicates.size();++k) if(predicates[k].ascii[c]) s|=uint64_t(1)<<k;
            ascii_signature[c]=s; ascii_word[c]=predicates[word].ascii[c];
        }
    }
    void prepare() {
        rows.resize(programs.size());
        seen.resize(programs.size());
        chain_generation.resize(programs.size()); chain_watermark.resize(programs.size());
        generation.assign(programs.size(),0);
        size_t largest=0;
        for(size_t i=0;i<programs.size();++i) {
            seen[i].assign(programs[i].code.size(),0);
            chain_generation[i].assign(programs[i].chain_bottom.size(),0);
            chain_watermark[i].assign(programs[i].chain_bottom.size(),0);
            largest=std::max(largest,programs[i].code.size());
        }
        // A closure pushes at most two entries per first-visited PC on top of the
        // candidate's threads (at most one per consume PC).
        scratch=3*largest+8;
        stack.reset(new Thread[scratch]); ready.reset(new Thread[scratch]); next.reset(new Thread[scratch]);
        cache.reset(new CacheEntry[cache_entries]);
    }

    // ---- codepoint predicates -------------------------------------------------
    void set_current(const char* bytes,size_t n) {
        current=bytes; current_size=n;
        current_ascii= n==1;
        if(current_ascii) { current_signature=ascii_signature[static_cast<unsigned char>(*bytes)]; return; }
        const auto* u=reinterpret_cast<const unsigned char*>(bytes);
        uint32_t scalar= n==2 ? ((u[0]&0x1fu)<<6)|(u[1]&0x3fu)
                       : n==3 ? ((u[0]&0x0fu)<<12)|((u[1]&0x3fu)<<6)|(u[2]&0x3fu)
                       : ((u[0]&0x07u)<<18)|((u[1]&0x3fu)<<12)|((u[2]&0x3fu)<<6)|(u[3]&0x3fu);
        current_entry=&cache[scalar%cache_entries];
        if(current_entry->scalar!=scalar) *current_entry=CacheEntry{scalar,0,0};
    }
    bool holds(int id) {
        uint64_t bit=uint64_t(1)<<id;
        if(current_ascii) return current_signature&bit;
        if(!(current_entry->known&bit)) {
            if(predicates[id].test(current,current_size)) current_entry->value|=bit;
            current_entry->known|=bit;
        }
        return current_entry->value&bit;
    }

    // ---- window -----------------------------------------------------------------
    uint64_t window_end() const { return window_base+window.size(); }
    const char* at(uint64_t absolute) const {
        return reinterpret_cast<const char*>(window.data()+(absolute-window_base));
    }
    void flush_hash(Candidate& c,uint64_t to) {
        HashStream& h=streams[c.stream];
        if(h.hashed>=to) return;
        elpis_sha256_update(&h.ctx,at(h.hashed),to-h.hashed);
        h.hashed=to;
    }
    void save_span(Candidate& c,std::vector<Saved>& out,std::string& arena,uint64_t b,uint64_t e) {
        for(const auto& s:out) if(s.begin==b && s.end==e) return;
        size_t offset=arena.size();
        if(b>=window_base) arena.append(at(b),e-b);
        else {
            auto it=std::find_if(c.saved.begin(),c.saved.end(),[&](const Saved& s){ return s.begin==b && s.end==e; });
            if(it==c.saved.end()) throw std::logic_error("V2_CAPTURE_WINDOW");
            arena.append(c.arena,it->offset,e-b);
        }
        out.push_back({b,e,offset});
    }
    // Materialize everything still referenced before `base`, then drop it.
    void compact(uint64_t base) {
        if(base<=window_base) return;
        for(auto& ids:rows) for(uint32_t id:ids) {
            Candidate& c=pool[id];
            if(!c.done) flush_hash(c,window_end());  // a finished candidate's hash is never read again
            if(c.won && !c.winner.omitted && !c.winner.text_saved && c.start<base) {
                c.winner.text.assign(at(c.start),c.winner.end-c.start);
                c.winner.text_saved=true; saved_text_bytes+=c.winner.text.size();
            }
            std::vector<Saved> kept; std::string arena;
            auto keep_thread=[&](const Thread& t,bool frozen) {
                for(size_t k=0;k<capture_count;++k)
                    if((frozen || !(t.open&(1u<<k))) && t.begin[k]<t.end[k] && t.begin[k]<base)
                        save_span(c,kept,arena,t.begin[k],t.end[k]);
            };
            for(const auto& t:c.threads) keep_thread(t,false);
            if(c.won) keep_thread(c.winner.thread,true);
            saved_capture_bytes+=arena.size(); saved_capture_bytes-=c.arena.size();
            c.arena.swap(arena); c.saved.swap(kept);
        }
        window.erase(window.begin(),window.begin()+static_cast<std::ptrdiff_t>(base-window_base));
        window_base=base;
    }
    std::string span(const Candidate& c,uint64_t b,uint64_t e) const {
        if(b==e) return std::string();
        if(b>=window_base) return std::string(at(b),e-b);
        for(const auto& s:c.saved) if(s.begin==b && s.end==e) return c.arena.substr(s.offset,e-b);
        throw std::logic_error("V2_CAPTURE_WINDOW");
    }

    // ---- candidates -------------------------------------------------------------
    uint32_t create(uint32_t entry) {
        uint32_t id;
        if(!free_ids.empty()) { id=free_ids.back(); free_ids.pop_back(); }
        else { id=static_cast<uint32_t>(pool.size()); pool.emplace_back(); }
        Candidate& c=pool[id];
        c.start=position;
        if(start_stream==UINT32_MAX) {
            if(!free_streams.empty()) { start_stream=free_streams.back(); free_streams.pop_back(); }
            else { start_stream=static_cast<uint32_t>(streams.size()); streams.emplace_back(); }
            HashStream& h=streams[start_stream];
            h.hashed=position; elpis_sha256_init(&h.ctx); h.references=0;
        }
        c.stream=start_stream; ++streams[start_stream].references;
        c.omitted=c.done=c.won=false;
        c.winner.text.clear(); c.winner.text_saved=false;
        c.threads.clear(); c.arena.clear(); c.saved.clear();
        Thread t{}; t.pc=entry; c.threads.push_back(t);
        ++live_candidates; ++live_threads;
        return id;
    }
    void release(uint32_t id) {
        Candidate& c=pool[id];
        // The stream opened at this codepoint may still be joined by a later program.
        if(!--streams[c.stream].references && c.stream!=start_stream) free_streams.push_back(c.stream);
        live_threads-=c.threads.size(); --live_candidates;
        if(c.won && c.winner.text_saved) saved_text_bytes-=c.winner.text.size();
        saved_capture_bytes-=c.arena.size();
        c.threads.clear(); c.arena.clear(); c.saved.clear(); c.winner.text.clear();
        free_ids.push_back(id);
    }
    Match materialize(Candidate& c,size_t pattern) {
        Match m;
        m.pattern=pattern; m.start=c.start; m.end=c.winner.end;
        m.text_omitted=c.winner.omitted;
        if(!c.winner.omitted)
            m.text=c.winner.text_saved ? c.winner.text : std::string(at(c.start),c.winner.end-c.start);
        m.digest=digest(c.winner.hash);
        for(size_t k=0;k<capture_count;++k) m.captures[k]=span(c,c.winner.thread.begin[k],c.winner.thread.end[k]);
        return m;
    }
    // Ordered Thompson closure: first arrival at a PC wins. Accept cuts only
    // lower-priority paths; greedy higher-priority paths can replace the winner.
    // `stable` is cleared unless the step leaves the candidate unchanged.
    bool step(Candidate& c,size_t pi,bool boundary,bool eof,bool& stable) {
        if(c.done) return false;
        const Program& prog=programs[pi];
        const Instruction* code=prog.code.data();
        const int32_t* chain_of=prog.chain_of.data();
        uint32_t* first=seen[pi].data();
        uint32_t* chain_seen=chain_generation[pi].data();
        uint32_t* watermark=chain_watermark[pi].data();
        uint32_t g=++generation[pi];
        if(!g) {
            std::fill(seen[pi].begin(),seen[pi].end(),0u);
            std::fill(chain_generation[pi].begin(),chain_generation[pi].end(),0u);
            g=generation[pi]=1;
        }
        size_t top=0, nready=0;
        for(size_t i=c.threads.size();i>0;--i) stack[top++]=c.threads[i-1];
        bool accepted=false;
        while(top) {
            Thread t=stack[--top];
            if(int32_t ch=chain_of[t.pc]; ch>=0) {
                uint32_t j=prog.chain_index[t.pc];
                uint32_t w=chain_seen[ch]==g ? watermark[ch] : 0;
                if(j<=w) continue;
                chain_seen[ch]=g; watermark[ch]=j;
                Thread b=t; b.pc=static_cast<uint32_t>(code[t.pc].next); ready[nready++]=b;
                if(!w) { t.pc=static_cast<uint32_t>(prog.chain_bottom[ch]); stack[top++]=t; }
                continue;
            }
            if(first[t.pc]==g) continue;
            first[t.pc]=g;
            const Instruction& i=code[t.pc];
            switch(i.op) {
            case Instruction::consume: ready[nready++]=t; continue;
            case Instruction::accept:
                flush_hash(c,position);
                c.won=true; accepted=true;
                if(c.winner.text_saved) { saved_text_bytes-=c.winner.text.size(); c.winner.text.clear(); }
                c.winner.end=position; c.winner.omitted=c.omitted; c.winner.text_saved=false;
                c.winner.hash=streams[c.stream].ctx; c.winner.thread=t;
                top=0; continue;
            case Instruction::boundary:
                if(!boundary) continue;
                break;
            case Instruction::split: {
                Thread other=t; other.pc=static_cast<uint32_t>(i.other); stack[top++]=other;
                break;
            }
            case Instruction::open:
                t.open|=1u<<i.arg; t.begin[i.arg]=t.end[i.arg]=position;
                break;
            case Instruction::close:
                t.open&=~(1u<<i.arg);
                break;
            }
            t.pc=static_cast<uint32_t>(i.next); stack[top++]=t;
        }
        size_t nnext=0;
        if(!eof) for(size_t r=0;r<nready;++r) {
            Thread t=ready[r];
            const Instruction& i=code[t.pc];
            if(!holds(i.arg)) continue;
            for(size_t k=0;k<capture_count;++k) if(t.open&(1u<<k)) {
                if(t.end[k]-t.begin[k]+current_size>capture_limit) throw std::runtime_error("V2_CAPTURE_BOUND");
                t.end[k]+=current_size;
            }
            t.pc=static_cast<uint32_t>(i.next); next[nnext++]=t;
        }
        if(stable && (accepted || nnext!=c.threads.size() ||
                      std::memcmp(next.get(),c.threads.data(),nnext*sizeof(Thread))!=0)) stable=false;
        live_threads+=nnext; live_threads-=c.threads.size();
        if(!nnext) { c.threads.clear(); c.done=true; return true; }
        c.threads.assign(next.get(),next.get()+nnext);
        if(!c.omitted && position+current_size-c.start>inline_limit) c.omitted=true;
        return false;
    }
    void retire(size_t pi) {
        auto& ids=rows[pi];
        while(!ids.empty() && pool[ids.front()].done) {
            Candidate& f=pool[ids.front()];
            if(!f.won) { release(ids.front()); ids.erase(ids.begin()); continue; }
            uint64_t end=f.winner.end;
            if(evidence.size()>=limit) throw RangeError{};
            evidence.push_back(materialize(f,pi));
            release(ids.front()); ids.erase(ids.begin());
            ids.erase(std::remove_if(ids.begin(),ids.end(),[&](uint32_t id){
                if(pool[id].start>=end) return false;
                release(id); return true;
            }),ids.end());
        }
        ids.erase(std::remove_if(ids.begin(),ids.end(),[&](uint32_t id){
            if(!pool[id].done || pool[id].won) return false;
            release(id); return true;
        }),ids.end());
    }
    void close_start_stream() {
        if(start_stream!=UINT32_MAX && !streams[start_stream].references) free_streams.push_back(start_stream);
        start_stream=UINT32_MAX;
    }
    bool can_start(size_t pi) {
        const Program& p=programs[pi];
        if(p.start_accepts) return true;
        for(uint64_t m=p.start_predicates;m;m&=m-1)
            if(holds(__builtin_ctzll(m))) return true;
        return false;
    }
    void measure() {
        measured.peak_candidates=std::max(measured.peak_candidates,live_candidates);
        measured.peak_threads=std::max(measured.peak_threads,live_threads);
        measured.peak_inline_bytes=std::max(measured.peak_inline_bytes,uint64_t(window.size())+saved_text_bytes);
        measured.peak_capture_bytes=std::max(measured.peak_capture_bytes,saved_capture_bytes);
        measured.evidence_count=evidence.size();
    }
    // One codepoint. Returns true when every live candidate was provably left
    // unchanged by a non-boundary ASCII codepoint (the run fixed point).
    bool codepoint(const char* bytes,size_t n,bool eof=false) {
        bool next_word=false;
        if(!eof) { set_current(bytes,n); next_word=holds(word); }
        else { current=nullptr; current_size=0; current_ascii=true; current_signature=0; }
        bool boundary=previous_word!=next_word;
        bool start=boundary && next_word;
        if(!active && !start) { previous_word=next_word; position+=n; return false; }
        if(!eof) {
            if(window.empty()) window_base=position;
            window.insert(window.end(),bytes,bytes+n);
        }
        bool stable=!boundary && !eof && current_ascii;
        active=false; close_start_stream();
        for(size_t pi=0;pi<programs.size();++pi) {
            auto& ids=rows[pi];
            // Every expression begins with a word-boundary + word character.
            // No new starts can accumulate in an infinite whitespace run.
            if(start) {
                if(ids.size()>=max_nonspace+1) throw RangeError{};
                if(can_start(pi)) ids.push_back(create(static_cast<uint32_t>(programs[pi].entry)));
            }
            // Retirement only changes when a candidate of this program finishes.
            bool finished=false;
            for(uint32_t id:ids) finished|=step(pool[id],pi,boundary,eof,stable);
            if(finished) retire(pi);
            active=active || !ids.empty();
        }
        close_start_stream();
        previous_word=next_word; position+=n;
        if(!active) { window.clear(); window_base=position; }
        else if(window.size()>compact_at) compact(position-keep);
        measure();
        return stable && active;
    }
    // Apply `n` further bytes of the same ASCII signature as the fixed-point
    // codepoint just processed: no candidate changes except by consuming them.
    void run(const uint8_t* bytes,size_t n) {
        uint64_t from=position, to=position+n;
        for(auto& ids:rows) for(uint32_t id:ids) {
            Candidate& c=pool[id];
            if(c.done) continue;
            HashStream& h=streams[c.stream];
            if(h.hashed<to) { flush_hash(c,from); elpis_sha256_update(&h.ctx,bytes,n); h.hashed=to; }
            if(!c.omitted && to-c.start>inline_limit) c.omitted=true;
        }
        uint64_t base= to>keep ? to-keep : 0;
        if(base>=from) {
            compact(from);
            window.clear(); window_base=base;
            window.insert(window.end(),bytes+(base-from),bytes+n);
        } else {
            window.insert(window.end(),bytes,bytes+n);
            if(window.size()>compact_at) compact(base);
        }
        position=to;
        measure();
    }
    void feed(const uint8_t* data,size_t n) {
        elpis_sha256_update(&source,data,n); total+=n;
        size_t i=0;
        while(i<n) {
            unsigned c=data[i];
            if(used==0 && c<128) {
                if(!active) {
                    // Idle ASCII: only a word start can begin a candidate.
                    size_t j=i; bool w=previous_word;
                    while(j<n && data[j]<128) {
                        bool x=ascii_word[data[j]];
                        if(x && !w) break;
                        w=x; ++j;
                    }
                    position+=j-i; previous_word=w; i=j;
                    if(i==n || data[i]>=128) continue;
                }
                bool fixed=codepoint(reinterpret_cast<const char*>(data+i),1);
                uint64_t signature=ascii_signature[data[i]];
                ++i;
                if(fixed) {
                    size_t j=i;
                    while(j<n && data[j]<128 && ascii_signature[data[j]]==signature) ++j;
                    if(j>i) { run(data+i,j-i); i=j; }
                }
                continue;
            }
            if(used==0) {
                if(c>=0xc2 && c<=0xdf) needed=2;
                else if(c>=0xe0 && c<=0xef) needed=3;
                else if(c>=0xf0 && c<=0xf4) needed=4;
                else throw std::runtime_error("INVALID_UTF8");
            } else {
                if((c&0xc0)!=0x80) throw std::runtime_error("INVALID_UTF8");
                unsigned lead=static_cast<unsigned char>(decoder[0]);
                if(used==1 && ((lead==0xe0 && c<0xa0) || (lead==0xed && c>=0xa0) ||
                   (lead==0xf0 && c<0x90) || (lead==0xf4 && c>=0x90)))
                    throw std::runtime_error("INVALID_UTF8");
            }
            decoder[used++]=static_cast<char>(c); ++i;
            if(used==needed) { codepoint(decoder.data(),used); used=0; }
        }
    }
};
Lexer::Lexer(const std::vector<std::string>& e,uint32_t n):impl(new Impl(e,n)) {}
Lexer::~Lexer()=default;
void Lexer::feed(const uint8_t* p,size_t n) { impl->feed(p,n); }
std::vector<Match> Lexer::finish() {
    if(impl->used) throw std::runtime_error("INVALID_UTF8_EOF");
    impl->codepoint(nullptr,0,true);
    return std::move(impl->evidence);
}
Stats Lexer::stats() const { return impl->measured; }
uint64_t Lexer::bytes() const { return impl->total; }
std::string Lexer::source_digest() const { return digest(impl->source); }
}
