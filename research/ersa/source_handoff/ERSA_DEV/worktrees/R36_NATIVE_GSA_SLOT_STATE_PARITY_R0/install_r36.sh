#!/usr/bin/env bash
set -eu
umask 077
ROOT=${ERSA_R36_ROOT:-/mnt/primesauce/RALPH/DEV/Hermes/ERSA_DEV}
SRC=$(cd -- "$(dirname -- "$0")" && pwd -P)
DEST="$ROOT/worktrees/R36_NATIVE_GSA_SLOT_STATE_PARITY_R0"
case "$SRC" in /dev/shm/*) ;; *) echo 'R36 source must execute from /dev/shm' >&2; exit 1;; esac
python3 -B "$SRC/qualify_r36.py" --gate ready
if [ -e "$DEST" ] || [ -L "$DEST" ]; then
  echo 'BLOCKED: R36 destination exists; no source overwrites permitted' >&2;exit 1
fi
mkdir -p -- "$ROOT/worktrees"
STAGE=$(mktemp -d "$ROOT/worktrees/.r36-promote.XXXXXX")
trap 'rm -rf -- "$STAGE"' EXIT
(cd "$SRC" && while IFS=' ' read -r hash path; do
  test -n "$path" || exit 1
  mkdir -p -- "$STAGE/$(dirname -- "$path")"
  cp -- "$path" "$STAGE/$path"
done < R36_FILES.sha256)
cp -- "$SRC/R36_FILES.sha256" "$STAGE/R36_FILES.sha256"
(cd "$STAGE" && sha256sum -c R36_FILES.sha256 >/dev/null)
mv -T -- "$STAGE" "$DEST"
trap - EXIT
printf 'R36_SOURCE_PROMOTED=%s\n' "$DEST"
