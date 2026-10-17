#!/usr/bin/env bash
# One 350M run across Kaggle's GPU (2xT4, 30 h/week) and TPU (v5e-8, 20 h/week) sessions: when a
# session ends, the next resumes from its output (checkpoints are interchangeable). The TPU is ~6x
# faster, so it goes first; the GPU fills in when the TPU quota is used up. If both are out of quota,
# wait for the weekly reset. A session that never started (Kaggle's TPU queue can time out after a
# day, leaving an empty log) is retried from the last checkpoint; a session that ran but did not
# save one stops the chain (the next would otherwise silently start from scratch). A TPU session
# still queued after TPU_WAIT_H hours is abandoned for a GPU session when the GPU has quota (Kaggle's
# TPU queue has taken up to a day; the CLI cannot cancel a queued run, so it may still start later:
# if it does, the chain waits for it and continues from whichever of the two got further).
#   bash tools/chain350.sh <running kernel slug> [kernel holding the latest checkpoint] [max sessions]
#   e.g. bash tools/chain350.sh vishalam-pretrain-350m-a
#   ABANDONED=<queued TPU slug> passes on a TPU session abandoned before a restart
set -u
CUR=$1; PREV=${2:-$1}; MAX=${3:-40}; TPU_WAIT_H=${TPU_WAIT_H:-8}; ABANDONED=${ABANDONED:-}
PY=.venv/Scripts/python.exe
D="$PY tools/kaggle_driver.py"
DATA="vishalam-data-1,vishalam-data-2"
stamp() { date '+%m-%d %H:%M'; }
status() { $D status "$1" 2>&1 | tail -1; }

push() {  # push <folder> <prev slug>; prints the push result line
  $D push "$1" --sources "$DATA,$2" 2>&1 | tail -1
}

ckpt_step() { echo "$1" | grep -oE "^(paused at|saved checkpoint at) step [0-9]+" | grep -oE "[0-9]+$" | tail -1; }

gpu_for() { case "$1" in *350m-a) echo vishalam_pretrain_350m_b;; *) echo vishalam_pretrain_350m_a;; esac; }

for n in $(seq 1 "$MAX"); do
  since=$(date +%s)
  while :; do
    s=$(status "$CUR"); echo "$(stamp) $CUR $s"
    if [ "$s" = QUEUED ] && [[ "$CUR" == *-tpu-* ]] && [ $(( $(date +%s) - since )) -gt $(( TPU_WAIT_H * 3600 )) ]; then
      r=$(push "$(gpu_for "$PREV")" "$PREV"); echo "$(stamp) TPU queued > ${TPU_WAIT_H} h, GPU push: $r"
      if echo "$r" | grep -q "successfully pushed"; then
        echo "$(stamp) abandoned queued $CUR"; ABANDONED=$CUR; NEXT=$(gpu_for "$PREV"); CUR=${NEXT//_/-}; since=$(date +%s); sleep 120; continue
      fi
      since=$(( $(date +%s) - TPU_WAIT_H * 3600 + 3600 ))  # no GPU quota either: look again in 1 h
    fi
    # a real Kaggle status is one upper-case word; anything else (DNS/HTTP error text) is transient
    case "$s" in RUNNING|QUEUED|NEW|""|*[!A-Z_]*) sleep 600;; *) break;; esac
  done
  last=$($D logs "$CUR")
  echo "$last" | grep -E "^step|saved|paused|finished|exit code" | tail -4
  if echo "$last" | grep -q "^finished"; then echo "350m pretraining finished"; exit 0; fi
  if echo "$last" | grep -q "^paused at step"; then
    PREV=$CUR
  elif echo "$last" | grep -q "^saved checkpoint at step"; then
    # crashed, but after an (atomically written) checkpoint: continue from that one
    echo "$(stamp) $CUR crashed after saving a checkpoint - continuing from it"; PREV=$CUR
  elif [ -z "$(echo "$last" | tr -d '[:space:]')" ]; then
    echo "$(stamp) $CUR never ran ($s, empty log) - retrying from $PREV"
  else
    echo "$(stamp) $CUR ended ($s) without a clean pause - chain stopped, check its log"; exit 1
  fi
  if [ -n "$ABANDONED" ]; then
    # the abandoned TPU session may have started after all: wait for it, keep the further checkpoint
    while :; do
      a=$(status "$ABANDONED")
      case "$a" in RUNNING|NEW|""|*[!A-Z_]*) echo "$(stamp) abandoned $ABANDONED $a - waiting for it"; sleep 600;; *) break;; esac
    done
    if [ "$a" != QUEUED ]; then
      sa=$(ckpt_step "$($D logs "$ABANDONED")"); sc=$(ckpt_step "$last")
      echo "$(stamp) abandoned $ABANDONED ended $a at step ${sa:-none}; $CUR at step ${sc:-none}"
      if [ -n "$sa" ] && [ "$sa" -gt "${sc:-0}" ]; then PREV=$ABANDONED; fi
      ABANDONED=""
    fi
  fi
  # a kernel cannot take its own output as input, hence the a/b pairs
  case "$PREV" in *-tpu-a) TPU=vishalam_pretrain_350m_tpu_b;; *) TPU=vishalam_pretrain_350m_tpu_a;; esac
  GPU=$(gpu_for "$PREV")
  while :; do
    r=$(push "$TPU" "$PREV"); echo "$(stamp) TPU push: $r"
    if echo "$r" | grep -q "successfully pushed"; then NEXT=$TPU; break; fi
    r=$(push "$GPU" "$PREV"); echo "$(stamp) GPU push: $r"
    if echo "$r" | grep -q "successfully pushed"; then NEXT=$GPU; break; fi
    echo "$(stamp) no quota on either - retrying in 1 h"; sleep 3600
  done
  CUR=${NEXT//_/-}; [ "$CUR" = "$ABANDONED" ] && ABANDONED=""  # its queued run was replaced
  sleep 120
done
