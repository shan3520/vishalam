#!/usr/bin/env bash
# Once 350M pretraining has finished (tools/chain350.sh logs "350m pretraining finished") and the M1
# data kernel ($M1, default vishalam-m1-examples) is COMPLETE, run the M1 agent fine-tuning (ping-pong
# sft-a/b sessions until it prints "finished"), then the evaluation (tools/eval_cpu.sh).
#   /usr/bin/bash tools/after_pretrain_350m.sh >> runs/m1_350m.log 2>&1
set -u
export M1=${M1:-vishalam-gen-episodes}
PY=.venv/Scripts/python.exe
D="$PY tools/kaggle_driver.py"
stamp() { date '+%m-%d %H:%M'; }
wait_done() {  # wait_done <slug>; prints the final status
  local s
  while :; do
    s=$($D status "$1" 2>&1 | tail -1)
    case "$s" in RUNNING|QUEUED|NEW|""|*[!A-Z_]*) sleep 600;; *) echo "$s"; return;; esac
  done
}

until grep -q "^350m pretraining finished" runs/chain_350m.log; do sleep 600; done
# the kernel holding the final weights is the last pretraining session the chain saw COMPLETE
FINAL=$(grep -oE "^[0-9-]+ [0-9:]+ vishalam-pretrain-350m-[a-z-]+ COMPLETE" runs/chain_350m.log | tail -1 | awk '{print $3}')
echo "$(stamp) pretraining finished in $FINAL"
until [ "$($D status "$M1" 2>&1 | tail -1)" = COMPLETE ]; do sleep 600; done
echo "$(stamp) $M1 COMPLETE"
SRC="vishalam-data-1,vishalam-data-2,$M1,$FINAL"

SIDE=a; EXTRA=""
for n in 1 2 3 4 5 6; do
  /usr/bin/bash tools/push_when_free.sh "vishalam_sft_m1_350m_$SIDE" --sources "$SRC$EXTRA"
  CUR="vishalam-sft-m1-350m-$SIDE"; sleep 120
  s=$(wait_done "$CUR"); echo "$(stamp) $CUR $s"
  last=$($D logs "$CUR")
  echo "$last" | grep -E "^step|paused|finished" | tail -3
  if echo "$last" | grep -q "^finished"; then
    exec /usr/bin/bash tools/eval_cpu.sh 350m "$CUR"  # Kaggle CPU sessions: no GPU quota needed
  elif echo "$last" | grep -q "^paused at"; then
    EXTRA=",$CUR"; [ "$SIDE" = a ] && SIDE=b || SIDE=a
  elif [ -z "$(echo "$last" | tr -d '[:space:]')" ]; then
    echo "$(stamp) $CUR never ran - retrying"
  else
    echo "$(stamp) $CUR ended without a pause - stopped, check its log"; exit 1
  fi
done
