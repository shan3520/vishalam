#!/usr/bin/env bash
# Wait until a kernel has finished, then push another (retrying while Kaggle has no free slot).
#   bash tools/wait_then_push.sh <kernel slug to wait for> <folder to push> [--sources a,b,...]
PY=.venv/Scripts/python.exe
WAIT=$1; shift
while :; do
  s=$($PY tools/kaggle_driver.py status "$WAIT" 2>&1 | tail -1); echo "$(date '+%m-%d %H:%M') $WAIT $s"
  case "$s" in RUNNING|QUEUED|NEW|""|*[!A-Z_]*) sleep 600;; *) break;; esac
done
[ "$s" = COMPLETE ] || { echo "$WAIT ended with $s - not pushing $1"; exit 1; }
exec bash tools/push_when_free.sh "$@"
