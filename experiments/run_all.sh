#!/usr/bin/env bash
# Run every experiment in order. Sequential on purpose: the latency figures are
# only meaningful when nothing else is competing for the cores.
#   bash experiments/run_all.sh            # everything (about two hours on 4 CPU cores)
set -u
cd "$(dirname "$0")"
mkdir -p artifacts
for step in "e1_banking77_ladder.py" "e2_teacher_distillation.py" "e5_cascade.py" "e4_tabular.py --tabicl-max 2000" \
            "e1_banking77_ladder.py --only-modernbert" "e3_compression.py" "e6_unfair_terms.py"; do
  name="${step%% *}"; name="${name%.py}"
  [ -f "$name.py" ] || { echo "skip $name (not present)"; continue; }
  tag="$name"; [[ "$step" == *only-modernbert* ]] && tag="e1b_modernbert"
  echo "=== $step  $(date -u +%H:%M:%S)"
  python3 $step > "artifacts/$tag.log" 2>&1 && echo "    ok" || echo "    FAILED (see artifacts/$tag.log)"
done
echo "=== done $(date -u +%H:%M:%S)"
