#!/usr/bin/env bash
# One 350M run across Kaggle's GPU (2xT4, 30 h/week) and TPU (v5e-8, 20 h/week) sessions: when a
# session ends, the next resumes from its output (checkpoints are interchangeable). The TPU is ~6x
# faster, so it goes first; the GPU fills in when the TPU quota is used up. If both are out of quota,
# wait for the weekly reset. A session that did not save a checkpoint stops the chain (the next one
# would otherwise silently start from scratch).
#   bash tools/chain350.sh <running kernel slug> [max sessions]
#   e.g. bash tools/chain350.sh vishalam-pretrain-350m-a 40
set -u
CUR=$1; MAX=${2:-40}
PY=.venv/Scripts/python.exe
D="$PY tools/kaggle_driver.py"
DATA="vishalam-data-1,vishalam-data-2"
stamp() { date '+%m-%d %H:%M'; }
status() { $D status "$1" 2>&1 | tail -1; }

push() {  # push <folder> <prev slug>; prints the push result line
  $D push "$1" --sources "$DATA,$2" 2>&1 | tail -1
}

for n in $(seq 1 "$MAX"); do
  while :; do
    s=$(status "$CUR"); echo "$(stamp) $CUR $s"
    # a real Kaggle status is one upper-case word; anything else (DNS/HTTP error text) is transient
    case "$s" in RUNNING|QUEUED|NEW|""|*[!A-Z_]*) sleep 600;; *) break;; esac
  done
  last=$($D logs "$CUR" | tail -30)
  echo "$last" | grep -E "^step|saved|paused|finished|exit code" | tail -4
  if echo "$last" | grep -q "^finished"; then echo "350m pretraining finished"; exit 0; fi
  if ! echo "$last" | grep -q "^paused at step"; then
    echo "$(stamp) $CUR ended ($s) without a clean pause - chain stopped, check its log"; exit 1
  fi
  case "$CUR" in *-tpu-a) TPU=vishalam_pretrain_350m_tpu_b;; *) TPU=vishalam_pretrain_350m_tpu_a;; esac
  case "$CUR" in *350m-a) GPU=vishalam_pretrain_350m_b;; *) GPU=vishalam_pretrain_350m_a;; esac
  while :; do
    r=$(push "$TPU" "$CUR"); echo "$(stamp) TPU push: $r"
    if echo "$r" | grep -q "successfully pushed"; then NEXT=$TPU; break; fi
    r=$(push "$GPU" "$CUR"); echo "$(stamp) GPU push: $r"
    if echo "$r" | grep -q "successfully pushed"; then NEXT=$GPU; break; fi
    echo "$(stamp) no quota on either - retrying in 1 h"; sleep 3600
  done
  CUR=${NEXT//_/-}; sleep 120
done
