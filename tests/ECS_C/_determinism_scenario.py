"""Deterministic ECS scenario, run in fresh processes by the determinism test.

JSON output contains no local paths. Invoked as a script with the elpis
package importable from the child environment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from elpis.ECS_C.kernel import Kernel
from elpis.ECS_C import canonical


def result(k):
    state = k.state
    return {
        "snapshot": k.snapshot(),
        "full_projection_digest": canonical.digest({
            "entities": state.registry.as_sorted_list(),
            "mailboxes": state.mailboxes.as_sorted_list(),
            "watermarks": state.watermarks.as_sorted_dict(),
        }),
        "event_bytes_sha256": hashlib.sha256(Path(k.log_path).read_bytes()).hexdigest(),
    }


def execute(directory, restart_every=0, replay=False):
    k = Kernel(str(directory)).open()
    try:
        if replay:
            return result(k)
        count = 0
        def do(fn):
            nonlocal count
            value = fn()
            count += 1
            if restart_every and count % restart_every == 0:
                before = result(k)
                k.close()
                k.open()
                if not (result(k) == before):
                    raise AssertionError("restart changed the canonical projection")
            return value
        a = do(lambda: k.found_entity("alpha-α"))
        b = do(lambda: k.found_entity("beta-β"))
        c = do(lambda: k.found_entity("gamma-γ"))
        for _ in range(3):
            do(k.step)
        for i in range(8):
            do(lambda: k.entity_port(a).propose(b, f"message-{i}".encode()))
            do(lambda: k.entity_port(b).propose(c, bytes([0, 255, i])))
            do(k.step)
        do(lambda: k.dormant(c))
        do(k.checkpoint)
        do(lambda: k.entity_port(a).propose(c, b"while-dormant"))
        do(lambda: k.reactivate(c))
        while do(k.step):
            pass
        do(lambda: k.terminate(b))
        do(lambda: k.found_entity("next-founding"))
        do(k.step)
        return result(k)
    finally:
        k.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("storage", type=Path)
    parser.add_argument("--restart-every", type=int, default=0)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    print(json.dumps(execute(args.storage, args.restart_every, args.replay), sort_keys=True))


if __name__ == "__main__":
    main()
