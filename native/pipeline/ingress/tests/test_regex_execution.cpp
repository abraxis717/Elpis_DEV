#include "execution_fixtures.h"
#include <iostream>

int main() {
    auto inputs = execution_fixtures("small");
    auto large = execution_fixtures("large");
    inputs.insert(inputs.end(), large.begin(), large.end());
    inputs.push_back(std::string("at least ")+std::string(128,'x'));
    std::string many;
    for (unsigned i=0; i<4097; ++i) many += "exactly 1; ";
    inputs.push_back(many); // Default evidence-limit rejection remains identical.
    std::vector<std::pair<int,std::string>> oracle;
    for (auto& s : inputs) oracle.push_back(original(s));
    std::cout<<"PASS direct oracle fixtures"<<std::endl;
    for (unsigned workers=1; workers<=4; ++workers) {
        elpis_exec_config c{workers, 4, 2*1024*1024, 1024*1024, 0, nullptr};
        elpis_exec_runtime *r=nullptr;
        assert(elpis_exec_create(&c,&r)==ELPIS_EXEC_OK);
        size_t submitted=0, retired=0;
        while (retired<inputs.size()) {
            while (submitted<inputs.size() && submitted-retired<4) {
                auto *b=input_buffer(inputs[submitted]);
                elpis_exec_task t{ELPIS_EXEC_REGEX, 1, static_cast<unsigned>(submitted),
                                  ELPIS_EXEC_PURE, submitted, elpis_regex_execute};
                uint64_t seq;
                assert(elpis_exec_submit(r,&t,&b,&seq)==ELPIS_EXEC_OK && !b && seq==submitted);
                ++submitted;
            }
            elpis_exec_result result{};
            assert(elpis_exec_take(r,60000,&result)==ELPIS_EXEC_OK);
            assert(result.sequence==retired && result.tag==retired);
            assert((result.status==ELPIS_EXEC_OK)==(oracle[retired].first==0));
            if (result.status==ELPIS_EXEC_OK) {
                assert(elpis_exec_buffer_size(result.output)==oracle[retired].second.size()+1);
                assert(std::strcmp(static_cast<const char*>(elpis_exec_buffer_data(result.output)),
                                   oracle[retired].second.c_str())==0);
            } else assert(!result.output);
            elpis_exec_buffer_release(result.output);
            ++retired;
        }
        elpis_exec_destroy(r);
        std::cout<<"PASS parity workers="<<workers<<std::endl;
    }
    auto *b=input_buffer("exactly 1"); elpis_exec_buffer *out=nullptr;
    assert(elpis_regex_execute(b,1,&out)==ELPIS_EXEC_INVALID && !out);
    elpis_exec_buffer_release(b);
    std::cout<<"PASS Regex ordered byte parity, 1..4 workers, "<<inputs.size()<<" fixture families\n";
}
