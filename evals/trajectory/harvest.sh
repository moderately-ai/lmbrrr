#!/usr/bin/env bash
# Overnight / batch trajectory harvest for accept-predictor training.
# Usage: harvest.sh OUT_DIR [N_TOKENS=128] [REPS=1]
set -euo pipefail
OUT=${1:?out dir}
N=${2:-128}
REPS=${3:-1}
BIN=${BIN:-./target/release/lmbrrr}
GGUF=${GGUF:-$HOME/models/Ternary-Bonsai-27B/Ternary-Bonsai-27B-Q2_0.gguf}
DRAFTER=${DRAFTER:-$HOME/models/Ternary-Bonsai-27B/Ternary-Bonsai-27B-dspark-Q8_0.gguf}
mkdir -p "$OUT"
export LMBRRR_ORACLE_LOG=1
export LMBRRR_ACCEPT_PROBE=1

prompts=(
  "prose|Explain quantum computing in simple terms for a curious teenager."
  "prose|Describe how a city subway system works end to end."
  "code|Write a Python function that merges two sorted lists into one sorted list."
  "code|Implement binary search in Rust with tests."
  "math|Solve step by step: if x^2 + 5x + 6 = 0, find x."
  "math|A train travels 120 km in 1.5 hours. What is its average speed?"
  "factual|What year did the Apollo 11 mission land on the Moon?"
  "factual|Who wrote Pride and Prejudice?"
  "chat|Hey, how are you doing today? Tell me something interesting."
  "reason|A bat and ball cost \$1.10. The bat costs \$1 more than the ball. How much is the ball?"
  "reason|If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies?"
)

i=0
for rep in $(seq 1 "$REPS"); do
  for entry in "${prompts[@]}"; do
    cls=${entry%%|*}
    prompt=${entry#*|}
    i=$((i+1))
    tag=$(printf "%s_r%02d_%03d" "$cls" "$rep" "$i")
    echo "[harvest] $tag"
    out="$OUT/${tag}.json"
    if "$BIN" gguf spec --gguf "$GGUF" --drafter "$DRAFTER" \
        --prompt "$prompt" --max-new-tokens "$N" \
        >"$out" 2>"$OUT/${tag}.err"; then
      :
    else
      echo "[harvest] FAIL $tag" >&2
    fi
  done
done
echo "[harvest] done -> $OUT"
