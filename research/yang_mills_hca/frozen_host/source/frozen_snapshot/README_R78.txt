YM/AWR R78: FINITE-HORIZON SO(2)xSO(2) TANGENT GROWTH
===================================================

Mechanistic retrospective analysis; frozen six 88-parameter R77 checkpoints.
Parent R77 report and byte-pinned R30/R62/R63/R64 mechanics supplied IN THIS ZIP.
No prior Downloads artifact, training, runtime mutation or network access needed.

The HCA's native state is (A:R4,B:R4) across four relation nodes.
For each native frozen trajectory, at H8 and H12 we perturb one channel
at a time along the SO(2)xSO(2) 90-degree generator J acting on each R4
node. For epsilon=1e-5 and 5e-6 and both signs we replay the original
source-pinned recurrence and compute central finite-difference tangent norms.

Time windows: H8->H12, H8->H20, H12->H20.
Columns: A-only, B-only. Records: 4 R77 graphs x 6 frozen models x
4 R77 evaluation seeds x 3 existing source conditions = 288.
Paired samples from weak/strong sources are BOTH included in the tangent norm.
No teacher state or H20 target is used to construct the intervention.

Controls:
* H8/H12/H20 native output parity to ORIGINAL R77 tensors: exact locally.
* Finite-difference epsilon halving agreement <1e-10 in log gain locally.
* Sum of A-only and B-only derivatives equals the global gauge generator
  on the unperturbed final state, max error <2e-11 locally.
* No transient weights or H-grams written at inference.

Offline labels are imported from the R77 sealed H20/H12 target NSE only.
Association is Spearman by graph vs log(H20_NSE/H12_NSE), with a
predeclared 3/4 positive-graph criterion; selecting the best A/B channel
within each graph is a retrospective diagnostic, NOT a trained detector.
Because R77 graphs/seeds are reused, this is not prospective OOD evidence.

RESULT: Association sign varies by graph. Three graphs show positive
association (their strongest of A/B channels), but D2 reverses.
Even positive association would not establish causal prediction or a
self-sufficient stability controller; no new learned utility is claimed.
The geometric gauge generator can grow with state norm independent of
prediction error. This experiment measures one tangent direction and
cannot estimate full Jacobian spectral norm/Lyapunov exponents.

Scientific disposition:
RETROSPECTIVE_FROZEN_HCA_FINITE_HORIZON_TANGENT_DIAGNOSTIC;
NO AUTONOMOUS CORRECTION; NO R52 REPRODUCTION; NO PHYSICAL SPARK GDN;
NO PRODUCTION HACF AUTHORITY; CGC SHADOW ONLY.

Modes:
  verify : stdlib-only source pins, R77 checkpoint pins, 288 rows,
           3,456 finite-difference intervals, independent statistics,
           twelve adversarial mutations.
  spot   : verify plus one complete graph's 72 PyTorch tangent cases,
           and independent native SO(2)xSO(2) covariance preflight.
  run    : verify plus full frozen-inference reproduction and
           byte-for-byte report comparison, then a second verify.

Execution: no persistent operation state; extraction into /dev/shm only.
Downloads serves exclusively as a transient ZIP mailbox.
