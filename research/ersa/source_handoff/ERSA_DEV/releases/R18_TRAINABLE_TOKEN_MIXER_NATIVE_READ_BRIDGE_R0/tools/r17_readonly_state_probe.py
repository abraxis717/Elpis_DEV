"""R17 R1: independent deterministic bitwise query/state isolation probe.

All state comparisons use torch.equal, not numerical tolerances. Read-only
query outputs are allowed floating-point roundoff across suffix shapes.
"""
from __future__ import annotations
import torch
from integration import gdn_layer_reference_r17 as g


def main():
    count = 0
    for nthreads in (1, 2, 4, 8):
        torch.set_num_threads(nthreads)
        for seed in range(1, 21):
            model = g.new_model(seed)
            torch.manual_seed(seed + 324)
            x = torch.randn(2, 24, 8)
            y = torch.randn(2, 24, 4)
            q = torch.randn(2, 4, 8)
            allow = (torch.rand(2, 24) > .2) if seed % 2 else None
            with torch.no_grad():
                first, s1 = model(x, y, q[:, :1], allow=allow, return_state=True)
                multi, s4 = model(x, y, q, allow=allow, return_state=True)
                _, changed = model(x, y, q + 13., allow=allow, return_state=True)
            if not torch.equal(s1, s4) or not torch.equal(s1, changed):
                raise RuntimeError(f'QUERY_READ_MUTATED_STATE:threads={nthreads}:seed={seed}')
            if not torch.allclose(first[:, 0], multi[:, 0], atol=2e-6, rtol=2e-6):
                raise RuntimeError(f'QUERY_PREFIX_DRIFT:threads={nthreads}:seed={seed}')
            count += 1
    assert count == 80
    print(f'READ_ONLY_STATE_BITWISE_PARITY={count}/80')
    print('READ_ONLY_QUERY_WRITES=0;THREADS=1,2,4,8;MAX_STATE_DELTA=0')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
