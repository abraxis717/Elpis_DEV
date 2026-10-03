"""Donor router and SwiGLU order, with bounded sequential FMS materialization."""
import numpy as np
from elpis.inference.contracts import Code, require
from .numerics import F32, linear, sigmoid, softmax


def route(x, weights, bias, c):
    logits = linear(x, weights) / F32(c.gate_temp)
    if c.score_func == "softmax":
        scores = softmax(logits)
    elif c.score_func == "sigmoid":
        scores = sigmoid(logits)
    else:
        # torch softplus uses its linear branch beyond threshold 20.
        scores = np.sqrt(np.where(logits > 20, logits, np.logaddexp(F32(0), logits)))
    selection = scores + bias
    chosen = np.argsort(-selection, kind="stable")[:c.active_experts]
    values = scores[chosen].copy()
    if c.norm_topk_prob and c.active_experts > 1:
        values /= np.sum(values, dtype=F32) + F32(1e-20)
    values *= F32(c.route_scale)
    return chosen, values


def expert(x, tensors, weight, limit):
    w1, w3, w2 = tensors
    gate, up = linear(x, w1), linear(x, w3)
    if limit > 0:
        gate = np.minimum(gate, F32(limit))
        up = np.clip(up, -F32(limit), F32(limit))
    activation = (gate * sigmoid(gate)) * up
    if weight is not None:
        activation *= F32(weight)  # before the down projection, as in donor
    return linear(activation, w2)


class MoE:
    def __init__(self, c, layer, store):
        self.config, self.store = c, store
        prefix = f"layers.{layer}.ffn."
        self.router, self.bias = store.dense[prefix + "gate.weight"], store.dense[prefix + "gate.bias"]
        self.roles = tuple(tuple(prefix + f"experts.{e}.{r}" for r in ("w1", "w3", "w2"))
                           for e in (*map(str, range(c.expert_count)), "shared"))
        self.last_route = ()

    def apply(self, x):
        c = self.config
        chosen, weights = route(x, self.router, self.bias, c)
        y = np.zeros(c.dimension, dtype=F32)
        # Donor accumulates in ascending expert ID, not top-k score order.
        for index in np.argsort(chosen):
            with self.store.expert(self.roles[int(chosen[index])]) as tensors:
                y += expert(x, tensors, weights[index], c.swiglu_limit)
        with self.store.expert(self.roles[-1]) as tensors:
            y += expert(x, tensors, None, c.swiglu_limit)
        require(np.all(np.isfinite(y)), Code.ENCODING, "MoE finite output")
        self.last_route = tuple(int(i) for i in chosen)
        return y, self.last_route
