#!/usr/bin/env bash
# After vishalam-gen-tasks (Qwen writes candidate tasks on GPU): verify them and build the mixed
# episode set on CPU (vishalam-gen-episodes), then 350M agent SFT and the CPU evaluation.
#   bash tools/after_gen.sh >> runs/m1_350m.log 2>&1
set -u
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
s=$(wait_done vishalam-gen-tasks); echo "$(stamp) vishalam-gen-tasks $s"
[ "$s" = COMPLETE ] || exit 1
bash tools/push_when_free.sh vishalam_gen_episodes --sources vishalam-gen-tasks,vishalam-m1-examples
sleep 120
s=$(wait_done vishalam-gen-episodes); echo "$(stamp) vishalam-gen-episodes $s"
$D logs vishalam-gen-episodes 2>&1 | grep -E '^\{' | tail -1
[ "$s" = COMPLETE ] || exit 1
M1=vishalam-gen-episodes exec bash tools/after_pretrain_350m.sh
