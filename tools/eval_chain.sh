#!/usr/bin/env bash
# Keep an M1 evaluation going until every set is done: when a session ends without its results table
# (ran out of Kaggle's 12 h) or reports "incomplete", push the other eval kernel with that output
# attached, which skips the finished sets (vishalam.eval.run --skip-done).
#   bash tools/eval_chain.sh 350m vishalam-eval-m1-350m vishalam-sft-m1-350m-a >> runs/m1_350m.log 2>&1
set -u
SIZE=$1; CUR=$2; SFT=$3; M1=${M1:-vishalam-m1-examples}
PY=.venv/Scripts/python.exe
D="$PY tools/kaggle_driver.py"
stamp() { date '+%m-%d %H:%M'; }
for n in 1 2 3 4; do
  while :; do
    s=$($D status "$CUR" 2>&1 | tail -1)
    case "$s" in RUNNING|QUEUED|NEW|""|*[!A-Z_]*) sleep 600;; *) break;; esac
  done
  last=$($D logs "$CUR")
  echo "$(stamp) $CUR $s"; echo "$last" | grep -E "pass so far|^M1 bar|incomplete|out of time" | tail -4
  if echo "$last" | grep -q "^M1 bar" && ! echo "$last" | grep -q "^incomplete"; then
    echo "$(stamp) evaluation complete in $CUR"; exit 0
  fi
  case "$CUR" in *-b) NEXT=vishalam_eval_m1_$SIZE;; *) NEXT=vishalam_eval_m1_${SIZE}_b;; esac
  bash tools/push_when_free.sh "$NEXT" --sources "vishalam-data-1,$M1,$SFT,$CUR"
  CUR=${NEXT//_/-}; sleep 120
done
echo "$(stamp) evaluation still incomplete after $n sessions - stopped"
