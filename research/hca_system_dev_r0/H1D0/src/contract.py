"""H1D0 SOURCE_EVOLUTION_RECOVERY_GATE_R0 - frozen task contract.

Classification: PROSPECTIVE DEV CONTRACT, NOT QUAL. No scientific result, no
canonical promotion, no B3 authorization. Written and frozen BEFORE any learned
model is trained or any sealed DEV is generated.

============================================================================
1. Objects
============================================================================
A WORLD is a finite set of typed n-ary factors over opaque entity tokens, a
finite set of opaque source tokens, and an ordered event log.

  ENTITY, SOURCE, TOKEN   opaque ascii tokens, unique within a world.
  RELATION TYPE           one of R0, R1, R2, RX.  RX is a legal relation type
                          that never participates in a witness (decoy).
  FACTOR                  f = (id, rel, roles, source, version)
      roles               = (from, to, scope, kind)  -- arity 4, ordered,
                           role names are distinct and always present.
      source             = the provenance source token.
      version            = 0 or 1.  The factor is only live when the epoch of
                           its current source equals this version.
  QUERY                   q = (id, from, to, scope, kind)
      A query asks whether a live witness chain exists:
          R0(a->b, scope, kind) and R1(b->c, scope, kind) and R2(c->d, scope, kind)
          with a = q.from and d = q.to.
      The witness is a 3-hop chain of 4-ary factors sharing the SAME scope and
      the SAME kind binding.  It is not expressible as a clique or a binary
      adjacency without losing the kind binding.

  EVENT                   e = (step, type, ...)
      type SOURCE :  (step, 'SOURCE', source, status, epoch)
      type EMIT   :  (step, 'EMIT',   factor)
      type DROP   :  (step, 'DROP',   factor)
      type REBIND :  (step, 'REBIND', factor, source)
      step is a strictly increasing integer nuisance coordinate.  It is NOT
      part of the causal semantics: state is folded over the log in LOG ORDER.

  status  in {VALID, STALE, UNKNOWN};  epoch in {0, 1}.

============================================================================
2. Transition operator (declared, independently implementable)
============================================================================
State after the first n log entries:

  status[s] = (status, epoch) of the LAST SOURCE event for s in log order,
              or (UNKNOWN, 0) if s has no SOURCE event.
  present[f] = True after an EMIT for f, False after a DROP for f (last wins),
               False before any EMIT.
  src[f]     = the source of f, updated by REBIND.

  LIVE(f, n)  <=>  present[f] and status[src[f]] == (VALID, version[f])

The answer to query q at prefix n is 1 iff a witness chain of LIVE factors
exists.  There is no other rule.  Nothing about step values, insertion order,
token spelling, or generator metadata is predictive except through this fold.

============================================================================
3. Information regimes (never pooled into one metric)
============================================================================
A  EXACT REPLAY / RECONCILIATION
   The full log is visible.  Every method must reproduce the oracle.

B  RETAINED-STATE DAMAGE
   After a declared prefix k, a declared number of units of the method's OWN
   retained computational state are corrupted.  The external evidence (the log)
   is unchanged.  The method continues over the remaining log and answers.
   Damage units are declared per method class; stateless methods have no
   retained state, so damage is a declared no-op for them (a structural
   property, not a privilege).

C  INCOMPLETE OBSERVATION
   A subset of log events is withheld from every method.
   C1 (traceable): the withheld events are published as a public PENDING pool.
       Ground truth is the full log.  A decision is IDENTIFIABLE iff every
       subset of the pending pool inserted at every possible position yields
       the same answer; otherwise UNDERDETERMINED.
   C2 (untraceable): the withheld events leave no trace.  No method can
       detect them.  Only the information-theoretic identifiability cap is
       reported; deterministic accuracy on underdetermined decisions is not
       scored as an ordinary mistake.

============================================================================
4. Splits and freeze
============================================================================
TRAIN and TUNE only in H1D0.  No SEALED split is generated or read.
Generator, oracle, checker, splits, seeds, baselines, model specification,
acceptance criteria and budgets are frozen before any learned model is fit.

============================================================================
5. Acceptance criteria (declared before measurement)
============================================================================
G1  oracle == exact replay == exact incremental on every TRAIN/TUNE prefix
    decision (mechanical).
G2  every paired intervention has opposite gold on its target query and equal
    nuisance fields (mechanical).
G3  the TRAIN-only symbolic rule inducer does NOT reach 100% on TUNE.  If it
    does, the task is HOLD_ASSAY_SOLVED_BY_SIMPLE_TRAIN_ONLY_SYMBOLIC_BASELINE
    and no learned-model claim may be made on this task.
G4  mutation tests must demonstrate actual rejection of the forbidden
    behaviour.
G5  a learned local recurrence may only be considered promising if it matches
    exact correctness at strictly lower total cost on regime B or C.
"""

SCHEMA = "elpis.hca.h1d0.source_evolution.public.v1"
CONTRACT_VERSION = "H1D0_SOURCE_EVOLUTION_R0/v1"
# The complete set of fields a public case may carry.  Anything outside this
# set is leakage by construction.
PUBLIC_FIELDS = ("case_id", "pair_id", "kind", "target_query_id",
                 "factors", "queries", "events", "pending", "damage_prefix")
CLASSIFICATION = "PROSPECTIVE DEV CONTRACT, NOT QUAL. No B3 authorization."
STATUSES = ("VALID", "STALE", "UNKNOWN")
EPOCHS = (0, 1)
RELS = ("R0", "R1", "R2", "RX")
ROLES = ("from", "to", "scope", "kind")
EVENT_TYPES = ("SOURCE", "EMIT", "DROP", "REBIND")
KINDS = ("source_swap", "event_order", "epoch_rebind", "structural",
         "cancellation", "provenance")
SPLITS = ("TRAIN", "TUNE")
MAX_PENDING = 3
MAX_EVENTS = 40
MAX_FACTORS = 64
MAX_QUERIES = 10
MAX_ENTITIES = 40
MAX_SOURCES = 12

# Declared damage budget for regime B (units per case).
DAMAGE_UNITS = 4

STATUS_CLASSES = ("VALID_MATCH", "VALID_MISMATCH", "STALE", "UNKNOWN",
                  "ABSENT", "DROPPED")


def status_class(status, epoch, version, present):
    """Declared six-way provenance class of a factor at a prefix."""
    if not present:
        return "DROPPED" if status is not None else "ABSENT"
    if status is None:
        return "ABSENT"
    if status == "VALID":
        return "VALID_MATCH" if epoch == version else "VALID_MISMATCH"
    if status == "STALE":
        return "STALE"
    return "UNKNOWN"


def case_digest(case):
    import json
    import hashlib
    return hashlib.sha256(json.dumps(case, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
