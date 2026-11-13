#!/bin/bash
# Held-out A/B oracle test (oracle.py --split-block), one jupyter V100 job that then stays
# allocated for interactive use until its 48 h limit. Results: $OUT/RESULTS.txt.
#
#   bash claude_dev/policy_oracle/submit_split.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OUT=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp36_1_oracle_split
mkdir -p "$OUT"
cp claude_dev/policy_oracle/oracle.py "$OUT/oracle.py"   # the copy the job runs
sbatch --job-name=oracle-split --output="$OUT/job-%j.log" --export=ALL \
    claude_dev/policy_oracle/split_then_sleep.sbatch
