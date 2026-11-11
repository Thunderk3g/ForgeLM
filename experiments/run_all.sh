#!/usr/bin/env bash
# Run every experiment in order. Sequential on purpose: the latency figures are
# only meaningful when nothing else is competing for the cores.
#
#   bash experiments/run_all.sh           # run every step whose result file is missing
#   FORCE=1 bash experiments/run_all.sh   # re-run everything
#
# A step is skipped when its result JSON already exists, so an interrupted run
# can be resumed by running the script again. Total on 4 CPU cores: about four
# hours, most of it the ModernBERT fine-tune (E1b) and the LLM teacher (E2, E6).
set -u
cd "$(dirname "$0")"
mkdir -p artifacts results

# step | result file it produces | log name
STEPS=(
  "e1_banking77_ladder.py|e1_banking77_ladder|e1_banking77_ladder"
  "e2_teacher_distillation.py|e2_teacher_distillation|e2_teacher_distillation"
  "e5_cascade.py|e5_cascade|e5_cascade"
  "e4_tabular.py --tabicl-max 2000|e4_tabular|e4_tabular"
  "e6_unfair_terms.py|e6_unfair_terms|e6_unfair_terms"
  "e1_banking77_ladder.py --only-modernbert|e1b_modernbert_full|e1b_modernbert"
  "e3_compression.py|e3_compression|e3_compression"
  "e7_generative_cpu.py|e7_generative_cpu|e7_generative_cpu"
  "e8_prompt_or_train.py|e8_prompt_or_train|e8_prompt_or_train"
  "e9_retrieval.py|e9_retrieval|e9_retrieval"
  "e1c_serving.py|e1c_serving|e1c_serving"
)

for entry in "${STEPS[@]}"; do
  IFS="|" read -r step result log <<< "$entry"
  script="${step%% *}"
  if [ ! -f "$script" ]; then echo "skip $step (script not present)"; continue; fi
  if [ -f "results/$result.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "skip $step (results/$result.json exists)"; continue
  fi
  echo "=== $step  $(date -u +%H:%M:%S)"
  # shellcheck disable=SC2086
  python3 $step > "artifacts/$log.log" 2>&1 && echo "    ok" || echo "    FAILED (see artifacts/$log.log)"
done
echo "=== done $(date -u +%H:%M:%S)"
