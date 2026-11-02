#!/usr/bin/env bash
# Evaluate on Kaggle CPU sessions (no weekly quota): push the 5 shard kernels, wait for them, download
# their results.json and join them into results/m1-<size>.json.
#   bash tools/eval_cpu.sh 350m vishalam-sft-m1-350m-a >> runs/m1_350m.log 2>&1
set -u
SIZE=$1; SFT=$2; M1=${M1:-vishalam-gen-episodes}; K=5; KUSER=${KUSER:-shantanupradhan3520}
PY=.venv/Scripts/python.exe
D="$PY tools/kaggle_driver.py"
stamp() { date '+%m-%d %H:%M'; }
for k in $(seq 0 $((K - 1))); do
  bash tools/push_when_free.sh "vishalam_eval_m1_${SIZE}_cpu$k" --sources "vishalam-data-1,$M1,$SFT"
done
sleep 300
parts=""
for k in $(seq 0 $((K - 1))); do
  slug="vishalam-eval-m1-$SIZE-cpu$k"
  while :; do
    s=$($D status "$slug" 2>&1 | tail -1)
    case "$s" in RUNNING|QUEUED|NEW|""|*[!A-Z_]*) sleep 600;; *) break;; esac
  done
  echo "$(stamp) $slug $s"
  dir="runs/eval_$SIZE/cpu$k"; rm -rf "$dir"; mkdir -p "$dir"
  for i in 1 2 3; do
    .venv/Scripts/kaggle.exe kernels output "$($PY -c "import sys; sys.path.insert(0, 'notebooks'); from make_notebooks import KAGGLE_USER; print(KAGGLE_USER)")/$slug" \
      -p "$dir" --file-pattern "results.json" > /dev/null 2>&1
    [ -f "$dir/results.json" ] && break; sleep 60
  done
  [ -f "$dir/results.json" ] && parts="$parts,$dir/results.json" || echo "$(stamp) $slug: no results.json"
done
[ -n "$parts" ] || { echo "$(stamp) no shard results - stopped"; exit 1; }
mkdir -p results
$PY -m vishalam.eval.run --merge "${parts#,}" --out "results/m1-$SIZE.json"
