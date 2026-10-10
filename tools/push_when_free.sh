#!/usr/bin/env bash
# Retry a kernel push every 20 min until Kaggle accepts it (e.g. "Maximum batch CPU session count").
#   bash tools/push_when_free.sh <folder> [--sources a,b]
PY=.venv/Scripts/python.exe
until r=$($PY tools/kaggle_driver.py push "$@" 2>&1 | grep -v '^\s*$' | tail -3); echo "$(date '+%m-%d %H:%M') $1: $r"; echo "$r" | grep -q "successfully pushed"; do
  sleep 1200
done
