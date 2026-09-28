"""Pipeline: bounded ingress, and structural proposals to one canonical publication.

* ``ingress`` — bytes -> bounded Regex lexer -> read-only HACF retrieval ->
  zero-authority context proposal -> atomic query-local proposal batch.

The canonical writer path follows. Each stage is its own authority boundary
and consumes only the previous stage's typed output:

* ``adjudication`` — structural-group proposal sets -> adjudication records and
  inert capability review requests;
* ``capability`` — review request -> authority decision -> one granted,
  unconsumed structural-influence capability;
* ``consumption`` — capability consumption -> inert structural-influence
  artifact and receipt;
* ``application`` — artifact applied to shadow capability state, recorded in
  a durable SQLite application ledger;
* ``promotion`` — read-only, advisory promotion plan and decision;
* ``canonical`` — promotion authority -> candidate -> publisher, the only
  writer of canonical Grid81 state.

Every stage fails closed; no stage can perform another's role.
"""
