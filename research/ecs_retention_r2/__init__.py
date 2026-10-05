"""ECS_G Retention R2 laboratory. RESEARCH_ONLY. NO_RUNTIME_AUTHORITY. NO_LANGUAGE_CLAIM. SEMANTICS=NONE.

Never imported by ``src/``. BLAS and OpenMP thread counts are pinned to 1 before NumPy loads (when this
package is imported first); the laboratory reads the effective thread count and refuses to record evidence
unless it is 1 (specs/ecsg-retention-r2.v1.spec.json, numerical_binding).
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
