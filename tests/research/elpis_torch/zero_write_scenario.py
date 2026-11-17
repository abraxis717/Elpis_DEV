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


def scroll_main():
    """The same lifetime for a scrolling session: stream past the horizon, adapt the last block, reset, fail."""
    global armed
    from research.elpis_torch.scroll import ScrollConfig, ScrollLM
    from research.elpis_torch.scroll_session import ScrollSession
    torch.set_num_threads(1)
    widths = ModelConfig(vocab_size=259, d_model=32, n_layers=2, n_heads=2, max_seq_len=32, channel="cgc",
                         cgc_bottleneck=8, cgc_channels=16, seed=0)
    model = freeze(ScrollLM(ScrollConfig(widths, window=48, block=8, chunk=32, horizon=256, global_layers=(1,),
                                         global_topk=4, fine_blocks=2, sinks=2)))
    if "--audit" in sys.argv:
        sys.addaudithook(hook)
    armed = True
    os.write(2, b"ELPIS_SESSION_BEGIN\n")
    applied, position = 0, 0
    stream = [256] + [65 + (i * 7) % 5 for i in range(1200)]
    try:
        with ScrollSession(model, SessionConfig(lr=0.1, max_steps=3, momentum=0.5, fast_scope="last")) as session:
            session.observe(stream[:900])                       # 3.5 horizons
            session.generate(4)
            applied += sum(r.applied for r in session.adapt_on_stream(steps=4))      # the fourth is refused: budget
            session.observe(stream[900:1100])
            position = session.position
            session.reset()
            session.observe(stream[:40])
            if "--negative-control" in sys.argv:
                with open(sys.argv[sys.argv.index("--negative-control") + 1], "w") as handle:
                    handle.write("not volatile")
            raise RuntimeError("a fault inside a live session")
    except RuntimeError:
        pass
    os.write(2, b"ELPIS_SESSION_END\n")
    armed = False
    print(json.dumps({"applied_updates": applied, "position": position, "write_events": events}))


def evidence_main():
    """R5: stream an evidence model past its ring capacity with retrieval at query positions, reset, a fault."""
    global armed
    from research.elpis_torch.ersa_model import ErsaConfig
    from research.elpis_torch.evidence_model import EvidenceConfig, EvidenceLM
    torch.set_num_threads(1)
    widths = ModelConfig(vocab_size=700, d_model=32, n_layers=2, n_heads=2, max_seq_len=32, channel="cgc",
                         cgc_bottleneck=8, cgc_channels=16, seed=0)
    model = freeze(EvidenceLM(ErsaConfig(widths, ersa_heads=2, key_dim=8, value_dim=8, conv_kernel=4, chunk=64),
                              EvidenceConfig(block=16, capacity=256, per_block=2, key_dim=16, coarse_top=8,
                                             evidence=2, fine_top=4, overlap=4, encoder_dim=32, fine_dim=8,
                                             read_heads=2, read_head_dim=8)))
    if "--audit" in sys.argv:
        sys.addaudithook(hook)
    armed = True
    os.write(2, b"ELPIS_SESSION_BEGIN\n")
    tokens = torch.randint(0, 700, (2, 900), generator=torch.Generator().manual_seed(0))
    position = 0
    try:
        with torch.no_grad():
            state = model.new_state(2)
            for at in range(0, 900, 64):                       # 3.5 ring capacities
                model(tokens[:, at:at + 64], state, logit_positions=torch.tensor([0, 31, 63])[:tokens[:, at:at + 64].shape[1] // 32 + 1])
            position = state.position
            state.reset()
            model(tokens[:, :64], state)
            if "--negative-control" in sys.argv:
                with open(sys.argv[sys.argv.index("--negative-control") + 1], "w") as handle:
                    handle.write("not volatile")
            raise RuntimeError("a fault inside a live stream")
    except RuntimeError:
        pass
    os.write(2, b"ELPIS_SESSION_END\n")
    armed = False
    print(json.dumps({"applied_updates": 0, "position": position, "write_events": events}))


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
    scroll_main() if "--scroll" in sys.argv else evidence_main() if "--evidence" in sys.argv else main()
