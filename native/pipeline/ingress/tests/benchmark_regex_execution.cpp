#include "execution_fixtures.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sys/resource.h>

using Clock=std::chrono::steady_clock;
static double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
int main(int argc,char **argv) {
    unsigned workers=argc>1?static_cast<unsigned>(std::stoul(argv[1])):0;
    std::string family=argc>2?argv[2]:"small";
    unsigned count=argc>3?static_cast<unsigned>(std::stoul(argv[3])):32;
    if (workers>4 || !count || (family!="small" && family!="large")) return 2;
    const auto inputs=execution_fixtures(family);
    std::vector<double> latency, compute, queue;
    std::vector<Clock::time_point> accepted(count);
    size_t bytes=0, output_bytes=0, failures=0;
    double first=0;
    elpis_exec_metrics metrics{};
    auto start=Clock::now(); const auto cpu=std::clock();
    if (!workers) {
        for (unsigned i=0;i<count;++i) {
            auto t=Clock::now(); const auto& source=inputs[i%inputs.size()];
            auto result=original(source);
            latency.push_back(seconds(t)); compute.push_back(latency.back()); queue.push_back(0);
            bytes+=source.size(); output_bytes+=result.second.empty()?0:result.second.size()+1;
            failures+=result.first!=0;
            if (!i) first=seconds(start);
        }
    } else {
        elpis_exec_config c{workers,8,2*1024*1024,1024*1024,0,nullptr};
        elpis_exec_runtime *r=nullptr;
        assert(elpis_exec_create(&c,&r)==ELPIS_EXEC_OK);
        unsigned submitted=0,retired=0;
        while (retired<count) {
            while (submitted<count && submitted-retired<c.capacity) {
                const auto& source=inputs[submitted%inputs.size()];
                accepted[submitted]=Clock::now(); // Includes producer allocation and source fill.
                auto *b=input_buffer(source);
                elpis_exec_task t{ELPIS_EXEC_REGEX,1,submitted,ELPIS_EXEC_PURE,submitted,elpis_regex_execute};
                uint64_t seq;
                assert(elpis_exec_submit(r,&t,&b,&seq)==ELPIS_EXEC_OK && seq==submitted);
                bytes+=source.size(); ++submitted;
            }
            elpis_exec_result result{};
            assert(elpis_exec_take(r,60000,&result)==ELPIS_EXEC_OK && result.sequence==retired);
            latency.push_back(seconds(accepted[retired]));
            queue.push_back(result.queue_ns/1e9); compute.push_back(result.compute_ns/1e9);
            output_bytes+=elpis_exec_buffer_size(result.output); failures+=result.status!=ELPIS_EXEC_OK;
            elpis_exec_buffer_release(result.output);
            if (!retired) first=seconds(start);
            ++retired;
        }
        elpis_exec_get_metrics(r,&metrics);
        elpis_exec_destroy(r);
    }
    double wall=seconds(start), cpu_s=double(std::clock()-cpu)/CLOCKS_PER_SEC;
    auto quantile=[](std::vector<double> v,double q) { std::sort(v.begin(),v.end()); return v[static_cast<size_t>(q*(v.size()-1))]*1000; };
    struct rusage usage{}; getrusage(RUSAGE_SELF,&usage);
    std::cout<<"{\"workers\":"<<workers<<",\"family\":\""<<family<<"\",\"tasks\":"<<count
      <<",\"source_bytes\":"<<bytes<<",\"output_bytes\":"<<output_bytes<<",\"failures\":"<<failures
      <<",\"wall_s\":"<<wall<<",\"cpu_s\":"<<cpu_s<<",\"tasks_s\":"<<count/wall
      <<",\"mib_s\":"<<bytes/(1024.0*1024*wall)<<",\"first_result_ms\":"<<first*1000
      <<",\"latency_p50_ms\":"<<quantile(latency,.5)<<",\"latency_p95_ms\":"<<quantile(latency,.95)
      <<",\"compute_p50_ms\":"<<quantile(compute,.5)<<",\"queue_p95_ms\":"<<quantile(queue,.95)
      <<",\"queue_total_ns\":"<<metrics.queue_ns<<",\"high_water\":"<<metrics.high_water
      <<",\"rss_kib\":"<<usage.ru_maxrss<<"}\n";
}
