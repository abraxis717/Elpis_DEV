"""ECS_G Retention R0 laboratory. RESEARCH_ONLY. NO_RUNTIME_AUTHORITY. NO_LANGUAGE_CLAIM. SEMANTICS=NONE.

Never imported by ``src/``. The numerical profile is single-threaded: BLAS and
OpenMP thread counts are pinned to 1 before numpy loads (when this package is
imported first), and the recorded profile states what was in effect.
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
