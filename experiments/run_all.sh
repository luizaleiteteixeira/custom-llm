#!/bin/bash
set -e
cd /Users/luizateixeira/Desktop/MBA/custom-llm
LOG=experiments/run_all.log
: > "$LOG"
for label in steps5k steps20k steps100k bigarch20k; do
  before=$(ls -d llm_runs/2026* 2>/dev/null | grep -v '\.zip$' | sort)
  echo "=== starting $label at $(date) ===" | tee -a "$LOG"
  .venv/bin/python "_exp_${label}.py" >> "$LOG" 2>&1
  after=$(ls -d llm_runs/2026* 2>/dev/null | grep -v '\.zip$' | sort)
  newdir=$(comm -13 <(echo "$before") <(echo "$after"))
  echo "${label} -> ${newdir}" >> experiments/run_map.txt
  echo "=== finished $label -> $newdir at $(date) ===" | tee -a "$LOG"
done
echo "ALL DONE" | tee -a "$LOG"
