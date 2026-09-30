#ifndef ELPIS_REGEX_EXECUTION_H
#define ELPIS_REGEX_EXECUTION_H
#include "elpis/execution.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Opt-in native adapter for independent complete V2 streams. Task operation must
 * be ELPIS_EXEC_REGEX. Input is exact source bytes. Output is canonical result
 * JSON including its terminating NUL; existing V2 evidence/source limits apply.
 * No HACF retrieval, authority publication, file I/O or Python callbacks. Each
 * stream's bounded 64 KiB feeds remain serial on one worker. Invalid UTF8/source
 * grammar limits => INVALID, allocation failures => INTERNAL. Output cap failure
 * => INVALID and no partial output. Original synchronous APIs remain unchanged. */
elpis_exec_status elpis_regex_execute(const elpis_exec_buffer *, size_t max_output,
                                    elpis_exec_buffer **);
#ifdef __cplusplus
}
#endif
#endif
