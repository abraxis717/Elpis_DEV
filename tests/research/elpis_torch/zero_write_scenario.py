"""A complete session lifetime, run in its own interpreter by test_session.py (never collected by pytest).

open -> observe -> generate -> adapt (stream and support set) -> a refused update -> reset -> an exception
inside the session -> close. ``--negative-control PATH`` adds one deliberate file write. With ``--audit`` a Python audit hook records every event that could create,
modify or remove a filesystem object between the two markers; the list is printed as JSON on stdout. Under
strace the same markers (written to stderr) delimit the system calls that are inspected.
"""
import json
import os
import sys

import torch

from research.elpis_torch.config import ModelConfig
from research.elpis_torch.model import ElpisTorchLM
from research.elpis_torch.session import SessionConfig, SupportSet, VolatileSession
from research.elpis_torch.training import freeze

WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
MUTATING = ("os.remove", "os.rename", "os.mkdir", "os.rmdir", "os.truncate", "os.link", "os.symlink", "os.chmod",
            "os.chown", "os.utime", "shutil.", "tempfile.", "pickle.", "marshal.dumps", "sqlite3.connect", "mmap.")
events = []
armed = False


def hook(event, args):
    if not armed:
        return
    if event == "open":
        path, mode, flags = args
        if (isinstance(mode, str) and set(mode) & set("wax+")) or (isinstance(flags, int) and flags & WRITE_FLAGS):
            events.append([event, str(path), str(mode), flags])
    elif event.startswith(MUTATING):
        events.append([event, repr(args)[:200]])


def main():
    global armed
    torch.set_num_threads(1)
    model = freeze(ElpisTorchLM(ModelConfig(vocab_size=259, d_model=32, n_layers=2, n_heads=2, max_seq_len=32,
                                            channel="cgc", cgc_bottleneck=8, cgc_channels=16, seed=0)))
    if "--audit" in sys.argv:
        sys.addaudithook(hook)
    armed = True
    os.write(2, b"ELPIS_SESSION_BEGIN\n")
    applied = 0
    try:
        with VolatileSession(model, SessionConfig(lr=0.1, max_steps=4, momentum=0.5)) as session:
            session.observe([256, 65, 97, 61, 66, 98, 61, 67])
            session.generate(4)
            applied += sum(r.applied for r in session.adapt_on_stream(steps=2))
            support = SupportSet.observed(torch.randint(0, 259, (4, 16), generator=torch.Generator().manual_seed(0)),
                                          source="scenario")
            applied += sum(r.applied for r in session.adapt(support, steps=3))     # the third is refused: budget
            session.observe([99, 61])
            session.reset()
            session.observe([256, 65])
            if "--negative-control" in sys.argv:    # a deliberate write: the detectors must see it
                with open(sys.argv[sys.argv.index("--negative-control") + 1], "w") as handle:
                    handle.write("not volatile")
            raise RuntimeError("a fault inside a live session")
    except RuntimeError:
        pass
    os.write(2, b"ELPIS_SESSION_END\n")
    armed = False
    print(json.dumps({"applied_updates": applied, "write_events": events}))


if __name__ == "__main__":
    main()
