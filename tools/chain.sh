#!/usr/bin/env bash
# Ping-pong pretraining sessions on Kaggle, as a plain shell loop (long-lived Python watchers were
# being killed on Windows). Usage:  bash tools/chain.sh <size> <running_side> [max_sessions]
#   e.g. bash tools/chain.sh 125m a 12   -> waits for vishalam-pretrain-125m-a, then pushes -b, ...
set -u
SIZE=$1; CUR=$2; MAX=${3:-12}
PY=.venv/Scripts/python.exe
D="$PY tools/kaggle_driver.py"
status() { $D status "vishalam-pretrain-$SIZE-$1" 2>&1 | tail -1; }
for n in $(seq 1 "$MAX"); do
  while :; do
    s=$(status "$CUR"); echo "$(date '+%m-%d %H:%M') $SIZE-$CUR $s"
    case "$s" in RUNNING|QUEUED|NEW|*rror*|*imeout*|*onnection*|*Exception*) sleep 600;; *) break;; esac
  done
  if [ "$s" != "COMPLETE" ]; then echo "session $SIZE-$CUR ended with $s - chain stopped"; $D logs "vishalam-pretrain-$SIZE-$CUR" | tail -20; exit 1; fi
  last=$($D logs "vishalam-pretrain-$SIZE-$CUR" | tail -25); echo "$last" | grep -E "step|finished|paused|resumed" | tail -4
  if echo "$last" | grep -q "^finished"; then echo "$SIZE pretraining finished"; exit 0; fi
  NEXT=$([ "$CUR" = a ] && echo b || echo a)
  echo "pushing $SIZE-$NEXT (resume from $SIZE-$CUR)"
  $D push "vishalam_pretrain_${SIZE}_$NEXT" --sources "vishalam-data-1,vishalam-data-2,vishalam-pretrain-$SIZE-$CUR" 2>&1 | tail -1
  CUR=$NEXT; sleep 120
done
