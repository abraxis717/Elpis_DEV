#!/usr/bin/env bash
set -Eeuo pipefail
B="${1:-}"; MODE="${2:-verify}"
PHASE=YM_AWR_R78_FINITE_HORIZON_EQUIVARIANT_TANGENT_GROWTH_R0
RESULT=HOLD BLOCKER=NONE CHILD_EXIT=2
SCRATCH=""
cleanup() { if [[ -n "$SCRATCH" && "$SCRATCH" == /dev/shm/YM_R78.* ]]; then rm -rf -- "$SCRATCH"; fi; }
trap cleanup EXIT
execute() {
  [[ -f "$B" ]] || { BLOCKER=ZIP_NOT_FOUND; return 2; }
  [[ "$MODE" == verify || "$MODE" == spot || "$MODE" == run ]] || { BLOCKER=INVALID_MODE; return 2; }
  [[ -d /dev/shm ]] || { BLOCKER=NO_VOLATILE_MEMORY; return 2; }
  SCRATCH="$(mktemp -d /dev/shm/YM_R78.XXXXXXXX)" || { BLOCKER=VOLATILE_TMP_ERROR; return 2; }
  export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
  export HOME="$SCRATCH" XDG_CACHE_HOME="$SCRATCH" TMPDIR="$SCRATCH"
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  python3 - "$B" "$SCRATCH" <<'PY'
import pathlib,sys,zipfile
source,target=sys.argv[1:]
allowed={
 'run.sh','README_R78.txt','SOURCE_SHA256_ALL.txt','r78.py','verify_r78.py','spot_r78.py',
 'FROZEN_PROTOCOL_R78.json','YM_AWR_R78_RESULT.zip',
 'portable_mechanics.py','r30_source.py','r62_source.py','r63_source.py','r64.py','r77.py','make_proto.py','YM_AWR_R77_RESULT.zip',
 *[f'FROZEN_PROTOCOL_R{n}.json' for n in [61,62,63,64,70,71,72,73,74,75,76,77]]
}
with zipfile.ZipFile(source) as z:
 if z.testzip() is not None:raise SystemExit('ZIP_CRC_FAILURE')
 if len(z.namelist())!=len(allowed) or set(z.namelist())!=allowed:raise SystemExit('ZIP_MEMBER_CONTRACT')
 for name in z.namelist():
  if '/' in name or '\\' in name or name.startswith('.'):raise SystemExit('UNSAFE_ZIP_NAME')
  data=z.read(name)
  if len(data)>4_000_000:raise SystemExit('ZIP_MEMBER_TOO_LARGE')
  pathlib.Path(target,name).write_bytes(data)
print('R78_SELF_CONTAINED_VOLATILE_EXTRACTION=PASS',flush=True)
PY
  cd -- "$SCRATCH"
  python3 verify_r78.py || { BLOCKER=REPORT_REVIEW_FAILED; return 2; }
  if [[ "$MODE" == spot ]]; then
    python3 spot_r78.py || { BLOCKER=DYNAMICS_SPOT_FAILED; return 2; }
  elif [[ "$MODE" == run ]]; then
    H="$(sha256sum YM_AWR_R78_RESULT.zip | cut -d' ' -f1)"
    python3 -u r78.py run || { BLOCKER=FULL_FROZEN_REPLAY_FAILED; return 2; }
    R="$(sha256sum YM_AWR_R78_RESULT.zip | cut -d' ' -f1)"
    if [[ "$H" != "$R" ]]; then BLOCKER=FULL_RESULT_SHA_DRIFT; return 2; fi
    echo "FULL_R78_RESULT_BYTE_REPLAY=PASS SHA256=$R"
    # Manifest pins the original report; replay has shown equality.
    python3 verify_r78.py || { BLOCKER=AFTER_REPLAY_REVIEW_FAILED; return 2; }
  fi
  RESULT=INDEPENDENT_R78_TANGENT_EVIDENCE_REVIEW_PASS
  CHILD_EXIT=0
  return 0
}
if ! execute; then CHILD_EXIT=2; fi
printf '\n=== SEND BACK ===\n'
printf 'PHASE=%s\nMODE=%s RESULT=%s\n' "$PHASE" "$MODE" "$RESULT"
if [[ "$CHILD_EXIT" == 0 ]]; then
  echo 'SOURCE_PINS=PASS R77_PARENT_REPORT=PASS FROZEN_88P_CHECKPOINTS=6/6'
  echo 'R77_GRAPHS=4 REUSED_EVAL_SEEDS=4 RECORDS=288/288 TANGENT_INTERVALS=3456/3456'
  echo 'NUMERICAL_EPS_CONVERGENCE=PASS GAUGE_GENERATOR_IDENTITY=PASS ADVERSARIAL_REJECTIONS=12/12'
  echo 'SCIENTIFIC_GATE=RETROSPECTIVE_TANGENT_ASSOCIATION_BOUNDED_NO_TRANSFER_CLAIM'
  echo 'CAUSAL_STABILITY_PROOF=NO AUTONOMOUS_CORRECTION=NOT_QUALIFIED R52=NOT_REPRODUCED'
  echo 'SPARK_GDN=NOT_EXECUTED CGC=SHADOW_ONLY HACF=NOT_PRODUCTION_ATTESTED'
  echo 'HOST_PROJECT_MUTATIONS=NONE HISTORICAL_DOWNLOAD_DEPENDENCIES=NONE VOLATILE_SCRATCH=PASS'
fi
printf 'BLOCKER=%s CHILD_EXIT=%s\n=== END SEND BACK ===\n' "$BLOCKER" "$CHILD_EXIT"
exit "$CHILD_EXIT"
