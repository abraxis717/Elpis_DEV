#!/usr/bin/env bash
set -eu
umask 077
ROOT=${ERSA_R35_ROOT:-/mnt/primesauce/RALPH/DEV/Hermes/ERSA_DEV}
SRC=$(cd -- "$(dirname -- "$0")" && pwd -P)
DEST="$ROOT/worktrees/R35_DONOR_BACKED_SLOT_WRITE_READ_R0"
case "$SRC" in /dev/shm/*) ;; *) echo 'R35 source must run from tmpfs' >&2; exit 1;; esac
python3 -B "$SRC/qualify_r35.py" --gate ready
if [ -e "$DEST" ] || [ -L "$DEST" ]; then
  echo 'R35 destination already exists; installer refuses overwrite' >&2
  exit 1
fi
mkdir -p -- "$ROOT/worktrees"
STAGE=$(mktemp -d "$ROOT/worktrees/.r35-promote.XXXXXX")
trap 'rm -rf -- "$STAGE"' EXIT
(cd "$SRC" && while IFS='  ' read -r sha rel; do
  test -n "$rel" || exit 1
  mkdir -p -- "$STAGE/$(dirname -- "$rel")"
  cp -- "$rel" "$STAGE/$rel"
done < R35_FILES.sha256)
cp -- "$SRC/R35_FILES.sha256" "$STAGE/R35_FILES.sha256"
(cd "$STAGE" && sha256sum -c R35_FILES.sha256 >/dev/null)
mv -T -- "$STAGE" "$DEST"
trap - EXIT
printf 'R35_SOURCE_PROMOTED=%s\n' "$DEST"
