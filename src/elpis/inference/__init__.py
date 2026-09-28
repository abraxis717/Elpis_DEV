"""Inference behind contracts: proposals in, verified target steps out.

Driver-neutral mechanics:

* ``contracts`` / ``target`` — typed failures, banks and row identities, the
  target records (tensors, latent inputs, target state, step receipts) and the
  :class:`~elpis.inference.target.Target` protocol a driver implements;
* ``associative`` / ``rows`` — n-gram address schemes and bounded row lookup
  over substrate file assets;
* ``context`` / ``global_context`` — immutable context lifetimes and snapshots,
  and sparse global candidate selection;
* ``experts`` — digest-bound expert tensors executed from file assets;
* ``prefetch`` — physical range predictions that can never select content;
* ``structural`` — address proposals from ingress exports and retrieval
  bundles;
* ``transaction`` — atomic, replay-verified decoding over a target;
* ``speculative`` — draft proposals verified token by token by the target
  (greedy equivalence only);
* ``steering`` / ``steered`` — a read-only observer over completed decode
  epochs and future-only latent steering applied through the target's own
  latent-input path.

Drivers live in ``elpis.inference.drivers``; the first is ``dsv4``. Nothing
here downloads or ships weights, and no model output is authority: every
proposal record carries fixed zero authority flags.
"""
