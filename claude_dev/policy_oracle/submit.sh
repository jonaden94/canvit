#!/bin/bash
# Greedy GT oracle on ADE20K val, split over two jupyter V100 jobs (1000 images each).
#
#   bash claude_dev/policy_oracle/submit.sh            # submit both shards
#   bash claude_dev/policy_oracle/submit.sh merge      # print mIoU/CE per t once both are done
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OUT=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp36_1_oracle

if [ "${1:-}" = merge ]; then
    exec .venv-cu126/bin/python "$OUT/oracle.py" merge "$OUT"
fi

mkdir -p "$OUT"
cp claude_dev/policy_oracle/oracle.py "$OUT/oracle.py"   # the copy the jobs run
for range in "0 1000" "1000 2000"; do
    read -r START END <<< "$range"
    START=$START END=$END sbatch --time=0-10:00:00 --job-name="oracle-$START" \
        --output="$OUT/job-%j-$START.log" --export=ALL claude_dev/policy_oracle/oracle.sbatch
done
