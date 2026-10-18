# ECS_C R14 — native learned polarity projection, controlled DEV only

## Scope and authority

This experiment introduces **one trainable Elpis-owned feature projector**. It does not use an LLM, pretrained embedding, network or external inference service. The learner is a sparse linear sigmoid readout over bounded lexical unigram/bigram features, trained with 288 controlled sentences. It emits source-addressable **candidate** polarity hypotheses, not `ADMITTED` relational evidence and not cognitive truth. Its weights live only in the research authority; it is **not** a canonical runtime change or a full Elpis language model.

R13 has already demonstrated that B0's structural readout ignores affirmative/negative source prose when supplied topology is fixed. R14 tests whether a small learning component can discriminate polarity without hard-coded positive/negative keyword logic. It does not claim that a binary polarity readout supplies a compositional semantic representation.

### Frozen data and protocol

Training: 6 distinct agent names × 6 distinct objects × 4 paired affirmative/negative statement templates × 2 polarities = **288** labeled, balanced synthetic instances. Training labels come from an explicitly authored controlled grammar and are **not** human-adjudicated natural language.

Held-out evaluation: 4 disjoint names × 4 disjoint objects, grouped into (1) familiar templates with new entities, 128; (2) new templates with familiar negation syntax, 64; (3) new negation vocabulary, 64; (4) shifted negation scope with "not false" versus "false", 32. Total **288** held-out balanced instances. The same held-out names/objects appear across evaluation groups; groups are not statistically independent. No R9 LoCoMo gold, reserved QUAL examples, or model responses are used to construct training labels or predictors.

Training examples and held-out cases are created by `prepare` as three immutable, separately hashed files. `infer` reads *only* labeled training and public held-out text, not the withheld gold; `score` reads gold only after both model and predictions have been saved. This is a source-level process separation, **not an operating-system access-control guarantee**. All source/templates and splits are declared in the code before evaluating them.

Weights trained with 32 deterministic passes, step 0.11 and small active-feature L2; no pretrained weights, random seed, imported NLP model, external provider, tuning against observed held-out outcomes, or dynamic model choice. Determinism, source-byte binding and readout are tested. The held-out evaluation accepts unfavorable outcomes and never changes thresholds or training examples post hoc.

### R13 source-backed integration control

Rebuild the two existing R13 affirmative/negated source variants from temporary copies of frozen DEV input, use the original canonical HACF source resolver and B0 graph, and classify their **identical text strings** using the learned projector. The source-identity binding stays explicit. Projection does **not** create an `ADMITTED` edge; even a correct R14 prediction is a mere candidate and must not be used as proof of the proposition it describes.

The untouched R13 report, previous source, and canonical runtime remain frozen. No creation under `/home/joe/` beyond the downloadable mailbox tarball. All persistent work is under the existing `research/ecsc_hca_r0` authority. The temporary source files are deleted.

### Interpretation, failures, and stopping

Compare each group against a balanced constant-label baseline (50%). High scores on reused syntax may indicate a shallow negation-token association. Failure on unseen markers and scope is expected and must be reported. These fixture labels are **not independent natural-language semantics**, nor are results a scientific qualification of compositional understanding, ECS_G behavior, adaptive EDEN-ECS, or general purpose language modeling.

A meaningful next scientific gate would require independent labeled natural-language contexts, a preregistered train/validation/held-out protocol, careful negation scope controls, calibrated abstention, and **candidate-to-admitted relation verification independent of the learner**. No automatic promotion into canonical ECS_C is authorized.
