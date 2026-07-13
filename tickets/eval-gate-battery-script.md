---
id: eval-gate-battery-script
title: "EVAL INFRA: one-command gate battery (evals/run_gate_battery.sh) + blessed golden texts"
status: todo
priority: p1
dependencies: []
related: []
scopes: [evals]
shared_scopes: []
paths: []
tags: [eval-wave, infrastructure]
---
WHY: every eval ticket repeats the same 5-step gate chain by hand (nextest -> rebuild -> stub oracle -> tree-check -> text-compare), and the chain has two traps that have burned sessions (nextest silently rebuilds WITHOUT the metal feature and clobbers target/release/lmbrrr; golden text lives nowhere, so 'byte-identical' is checked against whatever binary happens to be lying around). A single script makes the gates cheap, ordered, and trap-proof for an executor with zero context.

BUILD: `evals/run_gate_battery.sh [--extended] [--bless]` doing, in order:
1. `cargo build --release` (fail fast on compile).
2. `cargo nextest run` — expect 59 passed; then UNCONDITIONALLY `cargo build --release` again (nextest clobbers the metal binary — print a loud comment saying why).
3. Stub oracle: `./target/release/lmbrrr dspark-run --quantized-manifest target/minicpm-v46-q4k-full-text/manifest.json --quantize-lm-head q4k --prompt "Explain how tides work." --max-new-tokens 128` -> assert (jq) invariance_oracle_passed==true and max deviation <= 0.75.
4. `./target/release/lmbrrr tree-check --quantized-manifest target/minicpm-v46-q4k-full-text/manifest.json --quantize-lm-head q4k` -> assert pass.
5. Golden-text gate: run `run --prompt "Explain how tides work." --max-new-tokens 256` (same manifest args) and diff generated text against `evals/references/golden-tides-<FORK_PIN_SHA>.txt` where FORK_PIN_SHA is parsed from Cargo.toml's rev= pin. Missing golden for the current pin -> FAIL with instructions; `--bless` writes it (bless ONLY after an intentionally text-changing ship, and commit the new golden + note the reason in the commit message).
6. Determinism: run step 5's command twice, diff — must be byte-identical (our kernels are atomics-free; any nondeterminism is a bug).
--extended adds: (a) 6-class mini-suite text identity — `python3 evals/run_spec_suite.py --arm gate=target/dspark-drafter-round4 --split validation --per-class 1 --reps 1 --gamma 6 --tag gate-battery` and diff committed texts vs blessed copies in evals/references/suite-<PIN>/; (b) robustness smokes, each asserting exit 0 + sane output: empty prompt, --max-new-tokens 1, a unicode/emoji prompt, an EOS-early prompt ("What is 2+2? Answer with one word."), --max-new-tokens 1000 (campaign-length smoke).

RULES: script never uses `cargo clean`; artifacts under target/ (manifest, drafter) are load-bearing and unrebuildable without hours of work. Exit non-zero on first failure with the failing step named.
DONE-WHEN: script committed, one green `--extended` run pasted here, goldens blessed for the deployed pin (ff134429), and the eval-wave tickets' gate sections can be executed as `evals/run_gate_battery.sh`.
