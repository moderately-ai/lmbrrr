# Research rigor protocol

The standard that turns an *observation* into a *finding* on this project. Harness-agnostic: any agent, any session, applies this before a number is allowed to influence a decision. This is the enforced instantiation of the campaign goal's meta-process (see `AGENTS.md`). The Metal-specific mechanics live in `metal_notes.md` (§10 reproducible protocol, §15 the confound playbook + case studies); this file is the general contract.

## Two rules everything serves

1. **A result is not a finding until you can say WHY it happened.** Empirically observing that X is faster/slower is the *start* of the work, not the end. The bar for "understood": you can (a) name the limiting mechanism, (b) predict a second, independent observation from it, and (c) move the number by intervening on that mechanism.
2. **A published number is evidence for us only if its regime matches ours.** A proven speedup carries a regime — (hardware, batch/M, precision, memory- vs compute-bound). Outside its regime it is a *hypothesis to test*, not evidence. Regime mismatch is the primary way "proven" misleads (worked example below).

## The gates (apply in order; a result that skips one is provisional)

1. **Regime-tag every external number.** Record hardware, batch/M, precision, and memory-vs-compute-bound for any cited result before treating it as relevant. No regime tag → not admissible as evidence, only as a candidate.
2. **Assume the measurement, not the world.** A striking result is presumed a measurement artifact until its confounds are enumerated and cleared. Standing confound list (extend per domain): DVFS/thermal clock droop, harness/dispatch-path difference, threadgroup 2D shape, capture pollution (setup/RNG in-frame), wrong code-path dispatched, wrong regime. Two sub-gates:
   - **Validate the ruler:** a known-equal pair MUST read equal first. If it doesn't, you are measuring a confound — stop and fix it.
   - **Verify you measure what you think:** confirm the actual code path, the shapes, and kernel isolation before the number counts.
3. **Triangulate before "finding."** A conclusion must converge across at least: black-box wall time, white-box instrumentation (GPU-trace counters / profiles), and the underlying source/algebra. A result held by only one view is provisional. If the empirical magnitude and a first-principles bound (roofline, instruction count) do not reconcile, one of them is wrong and you do not yet have a finding.
4. **Prove the mechanism by moving the number.** A counter/limiter names a *candidate* cause, never a confirmed one. Intervene on the candidate and re-measure: if speed moves, the diagnosis holds; if it doesn't, the candidate is disproved (a real result, not a null). Never mass-apply a fix off the diagnosis alone.
5. **One variable per experiment; controls first; interleave.** Run the known control before the treatment; round-robin variants (do not run them sequentially) so DVFS/thermal drift hits all arms equally. A strip-probe proves only that the *stripped* part is small — state what remains unattributed.
6. **Log every result with its provenance.** Regime + measured-vs-inferred tag + explicitly *what it proves and what it does NOT prove*. The ledger is the ticket comments (append-only, per-route); the campaign dashboard is the route-map epic `rollup`. A number in prose with no ledger entry did not happen.
7. **Record negatives as first-class.** A refuted route is closed (`wontdo`) with its verdict note, not deleted — so it stays queryable and never resurfaces or gets re-explored. "Do NOT retry" lists are load-bearing.
8. **Adversarially verify striking claims.** Spawn independent skeptics tasked to *refute* (not confirm); the claim survives only if the refutations fail. Give diverse verifiers distinct lenses (correctness, repro, regime) rather than N identical ones.
9. **Gate a result before it becomes a default.** Teacher-forced PPL-vs-greedy (catches shifted/near-tie divergences that eyeballing cannot), the quality reference battery, and control-normalized benching stand between "measured once" and "shipped as the new baseline."

## Self-application — this session's failures are why these are gates

- **Capture pollution (gate 2).** A first MLX gpudebug capture read "not saturated, has headroom." It was ~80% `mx.random.normal` RNG generation, not the matmul; the tell (`gpu_write_bandwidth ≈ gpu_read_bandwidth`, wrong for a weight-heavy GEMM) was in the data and got glossed. The isolated two-phase re-capture *inverted* it to f32-saturated at 91.55%. Believe only the isolated-kernel capture.
- **Wrong code-path (gate 2).** MLX's dispatch was labelled "qmm" when at m≤8 it actually runs `qmv`; the black-box wall time was reported as the answer without verifying which kernel dispatched. Corrected only after it was flagged.
- **Regime mismatch (gate 1).** Three research agents rated "dequant→bf16→dense simdgroup_matrix" *do-first* on a proven 1.7× — which is M4-Pro **prefill** (large-M). At M3 **m≤8 verify** it is 2.8× slower (measured). The paper was right; the regime was wrong.
- **Move-the-number (gate 4).** Occupancy at 39% *looked* binding; raising it to 51% moved speed zero → disproved occupancy, pointed at instruction-throughput. A limiter is a candidate until moved.

## The ruler caveat (standing)

The tok/s metrics are currently **v1 and known-biased** — a ~+6% device-chain steady-state inflation and EOS-overshoot phantom token-times (see ticket `eval-harness-validity-fixes`, the campaign's task-0). Under gate 2, **no throughput number is admissible as a finding until the harness is fixed, the metrics are v2-versioned, and a fresh quiet, rotated baseline is established.** Until then, cross-arm ratios are only fair when both arms used identical N and READBACK_EVERY — note it in every log entry.
