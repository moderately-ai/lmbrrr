#!/usr/bin/env bash
# One-command correctness gate battery. Usage:
#   evals/run_gate_battery.sh [--extended] [--bless]
#
# Steps (fail-fast, failing step named on exit):
#   build -> tests -> rebuild -> stub oracle -> tree-check -> golden ids -> determinism
#   --extended adds: robustness smokes + drafter-load smoke + drafter golden
#   --bless (re)writes golden files for the CURRENT fork pin. Bless only after
#   an intentionally output-changing ship, and commit the new goldens with the
#   reason in the commit message. Drafter goldens depend on the STS/cost-model
#   artifacts — re-bless them after any refit.
#
# Artifacts live in artifacts/ (NOT target/ — cargo clean must never cost us
# data again). Goldens live in evals/references/ (committed, durable).
set -euo pipefail
cd "$(dirname "$0")/.."

EXTENDED=0
BLESS=0
for arg in "$@"; do
  case "$arg" in
    --extended) EXTENDED=1 ;;
    --bless) BLESS=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

MANIFEST=artifacts/minicpm-v46-q4k-full-text/manifest.json
DRAFTER=artifacts/dspark-drafter-round4
PIN=$(grep -m1 'rev = "' Cargo.toml | sed 's/.*rev = "\([^"]*\)".*/\1/')
REFS=evals/references
BIN=./target/release/lmbrrr
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
step() { echo "== GATE: $* =="; }
fail() { echo "GATE FAILED at: $*" >&2; exit 1; }

[ -f "$MANIFEST" ] || fail "preflight — $MANIFEST missing (see artifacts recovery notes in tickets)"
[ -d "$DRAFTER" ] || fail "preflight — $DRAFTER missing"

step "licenses/sources/advisories (cargo-deny)"
command -v cargo-deny >/dev/null 2>&1 \
  || fail "cargo-deny not installed (cargo install cargo-deny --locked)"
cargo deny check licenses sources advisories || fail "cargo-deny"

step "build"
cargo build --release || fail "build"

step "tests (nextest)"
cargo nextest run || fail "tests"
# Rebuild is ~0.2s when nothing changed; keeps the historical
# nextest-clobbers-the-binary trap permanently defused.
cargo build --release || fail "rebuild after tests"

step "stub oracle (invariance, dev <= 0.75)"
"$BIN" dspark-run --quantized-manifest "$MANIFEST" --quantize-lm-head q4k \
  --prompt "Explain how tides work." --max-new-tokens 128 \
  --output "$TMP/oracle.json" || fail "stub oracle run"
python3 - "$TMP/oracle.json" <<'EOF' || exit 1
import json, sys
r = json.load(open(sys.argv[1]))
assert r["invariance_oracle_passed"] is True, "invariance oracle failed"
assert r["max_trajectory_deviation"] <= 0.75, f"dev {r['max_trajectory_deviation']} > 0.75"
print(f"   oracle ok (dev {r['max_trajectory_deviation']})")
EOF
[ $? -eq 0 ] || fail "stub oracle assertions"

step "tree-check"
"$BIN" tree-check --quantized-manifest "$MANIFEST" --quantize-lm-head q4k \
  > "$TMP/tree.log" 2>&1 || { cat "$TMP/tree.log"; fail "tree-check"; }
echo "   tree-check ok"

golden_ids() { # golden_ids <report.jsonl> <golden-path> <label>
  local report=$1 golden=$2 label=$3
  python3 - "$report" "$golden" "$BLESS" "$label" <<'EOF'
import json, sys, pathlib
report, golden, bless, label = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3] == "1", sys.argv[4]
text = open(report).read()
try:
    row = json.loads(text)          # dspark reports: one pretty-printed object
except json.JSONDecodeError:
    row = json.loads(text.splitlines()[0])  # bench reports: JSONL, first row

ids = row.get("generated_token_ids") or row.get("committed_token_ids")
if ids is None:  # dspark reports carry text, not ids
    ids = row["committed_text"]
if bless:
    golden.parent.mkdir(parents=True, exist_ok=True)
    golden.write_text(json.dumps(ids))
    print(f"   BLESSED {label} -> {golden}")
elif not golden.exists():
    print(f"   NO GOLDEN for {label} at {golden} — run with --bless (then commit it)", file=sys.stderr)
    sys.exit(1)
elif json.loads(golden.read_text()) != ids:
    print(f"   {label} DIVERGES from golden {golden}", file=sys.stderr)
    sys.exit(1)
else:
    print(f"   {label} matches golden")
EOF
}

step "golden ids (pin $PIN)"
"$BIN" bench --quantized-manifest "$MANIFEST" --quantize-lm-head q4k \
  --warmup 0 --iterations 1 --prompt "Explain how tides work." \
  --max-new-tokens 256 --output "$TMP/g1.jsonl" || fail "golden bench run"
golden_ids "$TMP/g1.jsonl" "$REFS/golden-tides-$PIN.json" "greedy-tides" || fail "golden ids"

step "determinism (run twice, identical ids)"
"$BIN" bench --quantized-manifest "$MANIFEST" --quantize-lm-head q4k \
  --warmup 0 --iterations 1 --prompt "Explain how tides work." \
  --max-new-tokens 256 --output "$TMP/g2.jsonl" || fail "determinism run"
python3 - "$TMP/g1.jsonl" "$TMP/g2.jsonl" <<'EOF' || fail "determinism"
import json, sys
a = json.loads(open(sys.argv[1]).readline())["generated_token_ids"]
b = json.loads(open(sys.argv[2]).readline())["generated_token_ids"]
assert a == b, "two identical runs produced different ids — nondeterminism"
print("   deterministic")
EOF

if [ "$EXTENDED" = "1" ]; then
  step "extended: robustness smokes"
  "$BIN" run --quantized-manifest "$MANIFEST" --quantize-lm-head q4k \
    --prompt "What is 2+2? Answer with one word." --max-new-tokens 64 >/dev/null || fail "smoke: eos-early"
  "$BIN" run --quantized-manifest "$MANIFEST" --quantize-lm-head q4k \
    --prompt "hi" --max-new-tokens 1 >/dev/null || fail "smoke: max-tokens-1"
  "$BIN" run --quantized-manifest "$MANIFEST" --quantize-lm-head q4k \
    --prompt "Translate to French: the cat 🐈 sat on the mat. 中文也可以。" \
    --max-new-tokens 64 >/dev/null || fail "smoke: unicode"
  "$BIN" run --quantized-manifest "$MANIFEST" --quantize-lm-head q4k \
    --prompt "" --max-new-tokens 8 >/dev/null || fail "smoke: empty prompt"
  "$BIN" bench --quantized-manifest "$MANIFEST" --quantize-lm-head q4k \
    --warmup 0 --iterations 1 --prompt "Write a detailed essay about the history of navigation." \
    --max-new-tokens 1000 --output "$TMP/long.jsonl" || fail "smoke: 1000-token"

  step "extended: drafter smoke + golden"
  "$BIN" dspark-run --drafter "$DRAFTER" --quantized-manifest "$MANIFEST" \
    --quantize-lm-head q4k --drafter-quantize q8-0 --gamma 6 \
    --prompt "Explain how tides work." --max-new-tokens 128 \
    --output "$TMP/drafter.json" || fail "drafter smoke"
  golden_ids "$TMP/drafter.json" "$REFS/golden-drafter-tides-$PIN.json" "drafter-tides" \
    || fail "drafter golden (re-bless after STS/cost refits — see header)"
fi

echo "== GATE BATTERY GREEN (pin $PIN)$([ "$EXTENDED" = "1" ] && echo ", extended") =="
