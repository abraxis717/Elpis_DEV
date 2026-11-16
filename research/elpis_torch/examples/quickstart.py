#!/usr/bin/env python3
"""Train a small CGC transformer, generate from text, adapt in a volatile session, reset. About 20 s on a CPU.

    python -m research.elpis_torch.examples.quickstart

Writes nothing. The symbols of the task mean nothing (SEMANTICS=NONE): this shows mechanics, not language.
"""
import torch

from ..config import ModelConfig
from ..data import build_splits, generate, make_world, rewire
from ..model import ElpisTorchLM, parameter_digest
from ..session import SessionConfig, SupportSet, VolatileSession
from ..tokenizer import ByteTokenizer
from ..training import TrainConfig, freeze, rollout, train


def main() -> None:
    torch.set_num_threads(2)
    tokenizer = ByteTokenizer()
    world = make_world(states=6, ops=3, seed=1)
    splits = build_splits(world, train=1024, val=128, comp=128, length=64, seed=1)

    # 1. A decoder-only transformer whose channel transformation is the CGC operator (configuration B).
    config = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1,
                         max_seq_len=64, channel="cgc", cgc_bottleneck=16, cgc_channels=32, cgc_units=4, seed=1)
    model = ElpisTorchLM(config)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 2. Offline training: teacher-forced next-token cross-entropy on sequences such as "Ka=Fbc=D...".
    report = train(model, splits.train, TrainConfig(steps=500, batch_size=32, lr=4e-3, warmup_steps=20),
                   val=splits.val)
    held_out = report.history[-1]["val"]
    print(f"training loss {report.first_loss:.3f} -> {report.final_loss:.3f}; held-out loss {held_out['loss']:.3f}, "
          f"held-out single-step accuracy {held_out['single']['accuracy']:.3f}")
    print(f"free-running rollouts exact: {rollout(model, splits.val)['exact_trajectories']:.3f}")

    # 3. The trained model becomes an immutable checkpoint (in memory here; save_checkpoint is the explicit
    #    offline operation that would write one).
    freeze(model)
    digest = parameter_digest(model)

    # 4. Autoregressive generation from tokenized text, through the KV cache.
    prompt = "A" + "a="
    ids = torch.tensor([tokenizer.encode(prompt, bos=True)])
    print(f"prompt {prompt!r} -> {tokenizer.decode(model.generate(ids, 12)[0].tolist())!r}")

    # 5. A session in a world that changed after training: operation 'c' was rewired.
    shifted = rewire(world, op=2, seed=0)
    evidence = generate(shifted, 8, 64, seed=100, p_double=0.0)
    unseen = generate(shifted, 64, 64, seed=101, p_double=0.0)
    targets = unseen.target_mask()
    with VolatileSession(model, SessionConfig(lr=0.1, max_steps=16)) as session:
        before = session.loss(unseen.tokens, targets)
        results = session.adapt(SupportSet.observed(evidence.tokens, mask=evidence.target_mask(),
                                                    source="observed trajectories of the changed world"), steps=16)
        after = session.loss(unseen.tokens, targets)
        print(f"session: {sum(r.applied for r in results)} bounded updates, |dW| = {session.delta_norm():.3f}; "
              f"loss on unseen trajectories of the changed world {before:.3f} -> {after:.3f}")
        session.reset()
        print(f"after reset: {session.loss(unseen.tokens, targets):.3f} (the checkpoint's own loss again)")
    print(f"checkpoint untouched: {parameter_digest(model) == digest}")


if __name__ == "__main__":
    main()
