"""Principal-only DSV4.1 tower: explicit immutable parameters, transient caches.

Legacy NeuralState transactions are intentionally not implemented by this new
driver: they have a different persistence contract. The principal protocol is
the executable boundary. No ECS, HACF, tokenizer or provenance work per token.
"""
from dataclasses import dataclass
from time import perf_counter_ns
from types import MappingProxyType
import numpy as np

from elpis.identity import content_digest
from ...associative import AddressScheme, DSV41Parameters
from ...contracts import Code, RowIdentity, integer, require
from ...rows import RowEngine
from ...target import WindowStep, numerical_profile
from ...text import V41Tokenizer
from .attention import AttentionState, SharedAttention
from .config import TowerConfig
from .engram import build_parameters
from .layer import Layer
from .native_backend import DSV41NativeBackend, NativeAttentionState
from .provider_stream import DSV41StreamProvider, ProviderAttention
from .numerics import F32, hc_pre, linear, rms, rope_frequencies
from .parameters import ConditioningProjection, ParameterManifest, TensorStore, TowerAdmission


@dataclass
class TowerState:
    owner: object
    position: int
    history: object
    attention: tuple
    layer_streams: np.ndarray
    logits: np.ndarray
    step: WindowStep | None = None
    selected: tuple = ()
    closed: bool = False
    provider: object = None  # YTS-R0 provider stream (provider-owned recurrent state)
    conditioning: object = None  # frozen projected turn conditioning [dimension], or None

    @property
    def nbytes(self):
        return self.layer_streams.nbytes + self.logits.nbytes + sum(a.nbytes for a in self.attention)


class DSV41Target:
    def __init__(self, config, manifest, tokenizer, parameters, rows, provider, *, expected_manifest,
                 resident_budget, staging_budget, state_budget, native_backend=None, provider_stream=None,
                 conditioning_projection=None):
        require(type(config) is TowerConfig and type(manifest) is ParameterManifest, detail="tower config/manifest")
        require(native_backend is None or type(native_backend) is DSV41NativeBackend,
                Code.IDENTITY, "DSV4.1 native backend")
        require(provider_stream is None or type(provider_stream) is DSV41StreamProvider,
                Code.IDENTITY, "DSV4.1 YTS-R0 provider")
        require(native_backend is None or provider_stream is None, Code.UNSUPPORTED,
                "one DSV4.1 execution backend per target")
        require(conditioning_projection is None or (
                    type(conditioning_projection) is ConditioningProjection and
                    conditioning_projection.model == config.model and
                    conditioning_projection.tensor.shape[0] == config.dimension),
                Code.IDENTITY, "DSV4.1 turn-conditioning projection")
        require(type(tokenizer) is V41Tokenizer and (tokenizer.identity, tokenizer.vocab_size) ==
                (config.tokenizer, config.vocab), Code.IDENTITY, "tower tokenizer/vocabulary")
        require(type(parameters) is DSV41Parameters, Code.UNSUPPORTED, "DSV41 backbone address scheme")
        # Re-derive normalization/map/layout before execution, not just raw vocab length.
        derived = build_parameters(tokenizer, config)
        require(parameters.digest == derived.digest == manifest.address, Code.IDENTITY, "donor Engram parameters")
        scheme = AddressScheme(parameters, expected_digest=manifest.address,
                               tokenizer=config.tokenizer, scheme=parameters.schema)
        require(type(rows) is dict and set(rows) == set(config.engram_layers), Code.MISSING, "Engram layer banks")
        require(all(type(r) is RowEngine for r in rows.values()), detail="Engram row engines")
        require(manifest.row_banks == tuple((i, rows[i].table.bank.digest) for i in config.engram_layers),
                Code.IDENTITY, "Engram bank manifest")
        for layer, row in rows.items():
            require(type(row) is RowEngine and row.table.bank.layer == layer and row.table.bank.model == config.model,
                    Code.IDENTITY, "Engram model/layer")
            scheme.validate_bank(row.table.bank)
            require(row.workers == 1, Code.UNSUPPORTED, "no hidden row thread pool")
        integer(state_budget, 1)
        # Logical preallocated tensor capacity, with a conservative scratch allowance.
        cache = config.layers * config.local_window * config.head_dim
        for i in config.kv_sources:
            ratio = config.compress_ratios[i]
            cache += (config.max_tokens // ratio) * (config.head_dim + config.index_dim)
            if ratio > 1:
                cache += 2 * ratio * config.head_dim
        persistent_work = 4 * (cache + config.layers * config.hc_mult * config.dimension + config.vocab)
        scratch = 4 * (config.heads * (config.local_window + config.index_topk) * 4 +
                       config.index_heads * config.max_tokens * 4 + config.hc_mult * config.dimension * 16 +
                       config.expert_dim * 16 + config.heads * config.head_dim * 8 +
                       config.index_heads * config.index_dim * 8 + config.hash_columns * config.engram_dim * 4 +
                       max(config.compress_ratios) * config.head_dim * 4 +
                       (config.local_window + config.index_topk) * config.head_dim * 4) + config.vocab
        require(persistent_work + scratch <= state_budget, Code.LIMIT, "sequence state/scratch budget")
        store = TensorStore(config, manifest, provider, expected_manifest=expected_manifest,
                            resident_budget=resident_budget, staging_budget=staging_budget)
        self.config, self.store, self.scheme, self.native_backend = config, store, scheme, native_backend
        self.rows = MappingProxyType(dict(rows))
        self.numerical_profile = content_digest("elpis.inference.dsv41.numerical.v1", dict(
            profile=config.numerical_profile, execution=numerical_profile(), linear="einsum-optimize-false"))
        identity_fields = dict(manifest=expected_manifest, architecture="elpis.inference.dsv41.tower-spec.v1")
        if conditioning_projection is not None:
            # Only a conditioned target changes identity; unconditioned identities are unchanged.
            identity_fields["conditioning"] = conditioning_projection.digest
        self.model_identity = content_digest("elpis.inference.dsv41.target.v1", identity_fields)
        self.conditioning_projection = conditioning_projection
        self._conditioning_weights = None if conditioning_projection is None else conditioning_projection.tensor.array()
        frequencies = {compressed: rope_frequencies(config, compressed) for compressed in (False, True)}
        self.layers = tuple(Layer(config, i, store, rows.get(i), frequencies[bool(config.compress_ratios[i])],
                                  native_backend=native_backend)
                            for i in range(config.layers))
        self.provider_stream = provider_stream
        if provider_stream is not None:
            # The provider executes the tower; its pinned identity and kernel profile are part
            # of the numerical identity, so its commits never replay on another backend.
            provider_stream.admit(self, frequencies)
            self.numerical_profile = content_digest("elpis.inference.dsv41.numerical.v1", dict(
                profile=config.numerical_profile, execution=numerical_profile(), linear="einsum-optimize-false",
                provider=provider_stream.profile))
        self._row_index = {layer: i for i, layer in enumerate(config.engram_layers)}
        self._initial_pre = np.zeros(config.hc_mult, dtype=F32)
        self._initial_pre[0] = 1
        self._initial_pre.flags.writeable = False
        self.state_bytes, self.scratch_bound_bytes = persistent_work, scratch
        self.principal_working_set = (config.attention_capacity, max(config.head_dim, config.index_dim))
        self.last_metrics = {}

    def admit_stream(self, *, resident_experts=None):
        require(resident_experts is None, Code.UNSUPPORTED,
                "tower expert residency is declared by tensor bindings, not legacy overrides")
        return TowerAdmission(self)

    @property
    def accepts_conditioning(self):
        return self.conditioning_projection is not None

    def conditioning_vector(self, conditioning):
        """Project admitted turn conditioning once, at a sequence boundary. Read-only F32 [dimension]."""
        from ...conditioning import TurnConditioning
        require(self.conditioning_projection is not None, Code.UNSUPPORTED, "target admits no turn conditioning")
        require(type(conditioning) is TurnConditioning and len(conditioning.values) ==
                self.conditioning_projection.width, Code.ENCODING, "turn conditioning width")
        vector = linear(np.asarray(conditioning.values, dtype=F32), self._conditioning_weights)
        require(bool(np.all(np.isfinite(vector))), Code.ENCODING, "projected turn conditioning")
        vector.flags.writeable = False
        return vector

    def window_initial(self, conditioning=None):
        c = self.config
        vector = None if conditioning is None else self.conditioning_vector(conditioning)
        if self.provider_stream is not None:
            stream = self.provider_stream.open_stream(conditioning=vector)
            return TowerState(self, 0, self.scheme.initial(), tuple(ProviderAttention() for _ in range(c.layers)),
                              np.full((c.layers, c.hc_mult, c.dimension), np.nan, dtype=F32),
                              np.empty(c.vocab, dtype=F32), provider=stream, conditioning=vector)
        attention = (tuple(AttentionState.create(c, i) for i in range(c.layers))
                     if self.native_backend is None else
                     tuple(self.native_backend.create_attention_state(c, i) for i in range(c.layers)))
        return TowerState(self, 0, self.scheme.initial(), attention,
                          np.empty((c.layers, c.hc_mult, c.dimension), dtype=F32),
                          np.empty(c.vocab, dtype=F32), conditioning=vector)

    def window_step(self, state, token, *, experts):
        require(type(state) is TowerState and state.owner is self and not state.closed, Code.STALE, "tower sequence")
        require(type(experts) is TowerAdmission and experts.target is self, Code.IDENTITY, "tower sequence admission")
        c, w = self.config, self.store.dense
        integer(token, 0, c.vocab - 1)
        require(state.position < c.max_tokens and state.position == state.history.position, Code.LIMIT, "tower position")
        if self.provider_stream is not None:
            return self._provider_step(state, token)
        start = perf_counter_ns()
        metrics = dict(engram_ns=0, attention_ns=0, mhc_norm_ns=0, moe_ns=0)
        self.store.last_materialize_ns = self.store.last_bytes = 0
        hashed = self.scheme.stream_hash(state.history, (token,))
        metrics["address_ns"] = perf_counter_ns() - start
        embedded = w["embed"][token] if state.conditioning is None else w["embed"][token] + state.conditioning
        stream = np.broadcast_to(embedded, (c.hc_mult, c.dimension)).copy()
        pre = self._initial_pre
        shared = SharedAttention()
        route, selections = [], []
        for i, layer in enumerate(self.layers):
            row_ids = hashed.rows[0][self._row_index[i]] if i in self._row_index else ()
            stream, pre, chosen, selected = layer.apply(stream, pre, state.attention[i], shared, state.position,
                                                       row_ids, metrics)
            state.layer_streams[i] = stream
            route.extend(i * (c.expert_count + 1) + e for e in chosen)
            route.append(i * (c.expert_count + 1) + c.expert_count)  # shared expert trace code
            selections.append(selected)
        begin = perf_counter_ns()
        if self.native_backend is None:
            hidden = rms(hc_pre(stream, pre), w["norm"], c.norm_eps)
            linear(hidden, w["head"], out=state.logits)
        else:
            self.native_backend.final_head(stream, pre, w["norm"], w["head"], state.logits, c)
        require(np.all(np.isfinite(state.logits)), Code.ENCODING, "tower logits")
        metrics["head_ns"] = perf_counter_ns() - begin
        state.history = hashed.history
        state.position += 1
        state.step = WindowStep(token, hashed.rows[0], tuple(route))
        state.selected = tuple(selections)
        metrics.update(total_ns=perf_counter_ns()-start, expert_materialize_ns=self.store.last_materialize_ns,
                       expert_file_bytes=self.store.last_bytes, staged_high_water=self.store.high_water,
                       working_bytes=state.nbytes, compressed_positions=sum(a.count for a in state.attention))
        self.last_metrics = metrics
        return state

    def _provider_step(self, state, token):
        """YTS-R0: host preparation (hash, every Engram row), then one provider token transaction."""
        c, stream = self.config, state.provider
        require(stream is not None and stream.state == "BOUNDARY", Code.STALE, "YTS-R0 provider stream")
        start = perf_counter_ns()
        self.store.last_materialize_ns = self.store.last_bytes = 0
        try:
            hashed = self.scheme.stream_hash(state.history, (token,))
            address_ns = perf_counter_ns() - start
            begin = perf_counter_ns()
            segments = []
            for layer in c.engram_layers:
                engine = self.rows[layer]
                ids = hashed.rows[0][self._row_index[layer]]
                rows = engine.lookup(tuple(RowIdentity(engine.table.bank.digest, r) for r in ids))
                segments.append((layer, np.ascontiguousarray(rows, dtype=F32)))
            engram_ns = perf_counter_ns() - begin
        except BaseException:
            stream.release()  # PREPARING: the provider stream is untouched, so a normal release
            raise
        complete = stream.token(state.position, token, segments, self.store)
        state.logits[:] = complete.logits
        if complete.layer_streams is not None:
            state.layer_streams[:] = complete.layer_streams
        route = []
        for i in range(c.layers):
            route.extend(i * (c.expert_count + 1) + e for e in complete.selected[i])
            route.append(i * (c.expert_count + 1) + c.expert_count)  # shared expert trace code
        for attention, count in zip(state.attention, complete.counts):
            attention.count = count
        state.history = hashed.history
        state.position += 1
        state.step = WindowStep(token, hashed.rows[0], tuple(route))
        state.selected = tuple(complete.positions)
        metrics = dict(address_ns=address_ns, engram_ns=engram_ns, total_ns=perf_counter_ns() - start,
                       expert_materialize_ns=self.store.last_materialize_ns, expert_file_bytes=self.store.last_bytes,
                       staged_high_water=self.store.high_water, working_bytes=state.nbytes,
                       compressed_positions=sum(a.count for a in state.attention))
        metrics.update(stream.last)
        self.last_metrics = metrics
        return state

    def release_window(self, state):
        """Sequence finalization hook; no caches survive in a retained sequence handle."""
        if state is not None:
            require(type(state) is TowerState and state.owner is self, Code.IDENTITY, "release tower state")
            if state.provider is not None:
                state.provider.release()  # never raises: provider failure quarantines instead
                state.provider = None
            for attention in state.attention:
                if type(attention) is NativeAttentionState:
                    attention.close()
            state.attention = ()
            state.layer_streams = state.logits = np.empty(0, dtype=F32)
            state.selected = ()
            state.history = None
            state.closed = True
