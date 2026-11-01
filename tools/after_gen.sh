#!/usr/bin/env bash
# After the task-generation kernels (Qwen writes candidate tasks on GPU): verify them and build the
# mixed episode set on CPU (vishalam-gen-episodes), then - if enough tasks survived - 350M agent SFT
# and the CPU evaluation.
#   bash tools/after_gen.sh vishalam-gen-tasks,vishalam-gen-tasks-2 >> runs/m1_350m.log 2>&1
set -u
GENS=$1; MIN_TASKS=${MIN_TASKS:-4000}; KUSER=${KUSER:-shantanupradhan3520}
PY=.venv/Scripts/python.exe
D="$PY tools/kaggle_driver.py"
stamp() { date '+%m-%d %H:%M'; }
wait_done() {
  local s
  while :; do
    s=$($D status "$1" 2>&1 | tail -1)
    case "$s" in RUNNING|QUEUED|NEW|""|*[!A-Z_]*) sleep 600;; *) echo "$s"; return;; esac
  done
}
for g in ${GENS//,/ }; do
  s=$(wait_done "$g"); echo "$(stamp) $g $s"
  [ "$s" = COMPLETE ] || exit 1
done
bash tools/push_when_free.sh vishalam_gen_episodes --sources "$GENS,vishalam-m1-examples"
sleep 120
s=$(wait_done vishalam-gen-episodes); echo "$(stamp) vishalam-gen-episodes $s"
[ "$s" = COMPLETE ] || exit 1
dir=runs/gen_episodes; rm -rf "$dir"; mkdir -p "$dir"
.venv/Scripts/kaggle.exe kernels output "$KUSER/vishalam-gen-episodes" -p "$dir" --file-pattern "gen_stats.json" > /dev/null 2>&1
cat "$dir/m1/gen_stats.json"
tasks=$($PY -c "import json; print(json.load(open('$dir/m1/gen_stats.json'))['tasks'])")
if [ "$tasks" -lt "$MIN_TASKS" ]; then echo "$(stamp) only $tasks tasks (< $MIN_TASKS) - stopped before SFT"; exit 1; fi
M1=vishalam-gen-episodes exec bash tools/after_pretrain_350m.sh
