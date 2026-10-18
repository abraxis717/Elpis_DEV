# ECS <-> DSV4 Codec R0: interface and qualification specification

Status: **NOT RUN** · Subject: **UNQUALIFIED** · Classification: `RESEARCH_ONLY`

No ECS codec map is defined or qualified, and none is installed by this document: every canonical cognitive operation
refuses `ECS_CODEC_UNQUALIFIED` without one, and text generation is unavailable (docs/ELPIS_MISSION.md, "Missing
interfaces fail closed"). This specification states the exact interface a candidate map must implement, the mechanical
gates that tell an answer computed by ECS from an answer computed elsewhere (a hidden model, the text alone, a side
channel), and the scientific question a real map would have to answer. The DSV4 tokenizer is the communication
mechanism; it is never a substitute cognitive model.

## The interface (current code)

1. **Token boundary.** `elpis.inference.text` (the digest-bound DeepSeek recipe V4.1 tokenizer): `encode(text) ->
   tuple[int]`, an incremental `decoder()` (at most three carry bytes) and `vocab_size`. It carries no ECS state.
2. **Encode into admitted ECS stimuli** (`elpis.runtime.cognition`):
   * QUERY: `ECSQueryCodecMap.encode_query(tokens) -> QueryStimulus(x, dim)`: query rows of the state's dimension;
     admitted under total fuel (query rows <= 4096, ECS work units, output tokens) before any native call.
   * LEARN: `ECSCodecMap.encode(tokens) -> Stimulus(x, y, schedule, dim)`: an ordered experience schedule of
     `(rows, steps)` pairs (<= 64 experiences, <= 256 rows each, <= 16384 rows, <= 2^20 steps, <= 2^30 work units).
3. **Operation-specific authority.** A map runs only as an `AdmittedCodec` of a deployment-pinned `CodecAuthority`
   (`elpis.runtime.codec_authority`; the pin comes from `RuntimeConfig.codec_authority_sha256`, never from the map),
   which grants `QUERY_ENCODE`, `QUERY_DECODE`, `LEARN_ENCODE` and (legacy only) `LEARN_DECODE` separately and fixes the
   classification every result carries. A LEARN additionally needs an explicit `LearnAuthority`; a QUERY has no
   parameter that can carry one. A composite's memory additionally needs an explicit `ObservationAuthority` and is
   never written by a codec.
4. **State-dependent readouts.** QUERY: `QueryReadout(values = f_W(x) per row, dim, width, state_digest | live_generation)`
   from one native call over the authoritative (unmanaged) or live (managed) state. LEARN produces no text; the legacy
   learned turn's S3 decode is classified `LEGACY_LEARNED_TURN`. A composite's exact memory LOOKUP is not routed to a
   codec: there is no protocol method for it (a proposal, not done: `decode_query(readout, lookups)` as a new protocol
   version with its own capability).
5. **Output.** `decode_query(readout) -> tuple[int]`: in-vocabulary token ids, at most `max_output_tokens` (<= 4096),
   rendered by the tokenizer's incremental decoder. A token distribution is not part of the protocol (a proposal, not
   done: a bounded probability vector over the vocabulary returned by the map, sampled outside it by a seeded sampler,
   as a new protocol version).
6. **ECS dependence.** The mechanical gates D1-D4 below. They are necessary, not sufficient: they exclude answers that
   do not depend on ECS computation; they cannot show that an answer means anything.

## Mechanical gates (for every candidate map, on DEV and on QUAL)

* **D1 determinism.** The same text over the same ECS state yields the same tokens, in a fresh map instance and a fresh
  process.
* **D2 readout sufficiency.** Decoding the recorded readout alone, in a fresh map instance that never saw the tokens,
  reproduces the answer: no channel from the encode side, the text, the call history or any global state into the
  answer.
* **D3 counterfactual sensitivity.** Over pre-registered counterfactual ECS states (the same text; W of other seeds or
  training runs; an ablated W = 0; H and a reset), the answers differ. An answer that ignores the ECS state is the same
  for all of them.
* **D4 closure.** The map's implementation source is pinned (`implementation_sha256`) and its import closure contains no
  `elpis.inference` model-execution module, no `research` module, no network or model loader (the mission gate's
  rules); its parameters are one digest-bound artifact of declared provenance that holds no DSV model weights.

`tests/integration/test_codec_dependence_probe.py` proves D1-D3 discriminate on TEST_ONLY fixtures: a readout-bound map
passes all three; a readout-blind map (an answer from the text alone, as a hidden model would give) fails D2 and D3; a
map that smuggles encode-side tokens into its answer passes D3 and fails D2; and the repository's own interface fixture
is readout-blind by design and fails D3, so it can never be read as an ECS-dependent answer.

## Question

For one pinned candidate map M (encode and decode, D4-clean) and K1 states trained only through M's LEARN encode on a
frozen DEV corpus: do QUERY answers decoded by M on held-out prompts depend on the trained ECS state in the way M's
pre-registered hypothesis predicts, beyond the controls below, while passing D1-D4?

## Inputs (none exist yet: these are the blockers)

* A frozen, digest-bound corpus of (prompt, response) pairs with disjoint DEV and QUAL partitions and a held-out
  paraphrase set, written before DEV.
* The pinned map M (implementation and parameters), its hypothesis and its thresholds, written before DEV and frozen
  after it.
* Controls, run on the same prompts: the readout-blind decode of the same tokens (D3 must fail for it); an untrained
  W; a shuffled-state control (another run's W); a byte-matched exact lookup table of the training pairs (the lesson of
  the memory's H4: exact storage must be beaten at equal bytes); a text-only n-gram baseline at matched parameter bytes.
* One K1 shape, one learning regime, bounded: at most 64 LEARNs per world, at most 64 QUAL prompts, at most 8 worlds.

## Gates (pre-registered)

* G1 (mechanics): D1-D4 hold for M on DEV and QUAL.
* G2 (dependence): QUAL answers change under every counterfactual state on at least a fraction fixed on DEV.
* G3 (advantage): the held-out score exceeds every control (lookup table, n-gram, untrained, shuffled, readout-blind) by
  a margin fixed on DEV, on a fraction of QUAL worlds fixed on DEV.
* G4 (no laundering): no answer, readout or map output is admitted as a codec authority, an observation, a learning
  instruction or durable state by the run itself.

Falsifiers: any D-gate failure; a held-out score at or below the lookup-table or n-gram control; answers unchanged
under counterfactual states.

## Would establish / would not establish

Passing would establish, for one pinned map, corpus and regime, that M's answers are ECS computation in the sense of
D1-D4 and that the stated hypothesis held on QUAL. It would **not** establish language understanding, generalization
beyond the corpus, any claim about other maps or regimes, the qualification of a production codec, or that Elpis
generates text in general; promotion of M into the canonical runtime would be a separate, explicit authority decision.

## Evidence

`research/qualification/ecs_codec_r0/evidence/` (write-once; not present: the run has not happened).
