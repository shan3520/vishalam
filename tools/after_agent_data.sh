#!/usr/bin/env bash
# Start rewording once a fresh agent-data build (all 6 parts) has finished.
# Kaggle's status only describes a kernel's latest run, so this first waits until part 5's new run
# has been pushed (tools/push_when_free.sh logs it in runs/agent_p5.log after SINCE), then waits for
# every part to be COMPLETE again, then pushes the reword kernel (retrying while there is no GPU quota).
#   bash tools/after_agent_data.sh 10-15
set -u
SINCE=$1   # MM-DD of the rebuild: only a part-5 push logged on/after this day counts
PY=.venv/Scripts/python.exe
D="$PY tools/kaggle_driver.py"
stamp() { date '+%m-%d %H:%M'; }
until awk -v s="$SINCE" '$1 >= s && /successfully pushed/' runs/agent_p5.log | grep -q .; do sleep 600; done
echo "$(stamp) part 5 of the new build was pushed"
sleep 900
for i in 0 1 2 3 4 5; do
  while :; do
    s=$($D status "vishalam-agent-data-p$i" 2>&1 | tail -1)
    case "$s" in COMPLETE) break;; RUNNING|QUEUED|NEW|""|*[!A-Z_]*) sleep 600;; *) echo "$(stamp) part $i ended $s - stopping"; exit 1;; esac
  done
  echo "$(stamp) part $i COMPLETE"
done
exec bash tools/push_when_free.sh vishalam_reword --sources vishalam-agent-data-p0,vishalam-agent-data-p1,vishalam-agent-data-p2,vishalam-agent-data-p3,vishalam-agent-data-p4,vishalam-agent-data-p5
