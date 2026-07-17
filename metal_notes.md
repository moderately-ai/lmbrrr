# Headless Metal capture and performance analysis on macOS 27

This guide describes the new macOS 27 GPU command-line workflow for coding agents, scripts, CI experiments, and human-driven performance investigations. It incorporates an end-to-end validation on an Apple M3 Pro using a candle/Metal LLM decode engine, including a one-token GPU capture analyzed entirely without the Xcode GPU debugger UI.

> This is the Metal-specific mechanics. The general method that governs how a measurement becomes a *finding* (regime-match, confounds-cleared, triangulate, move-the-number) is `docs/research/rigor-protocol.md`; the runnable capture harness is `evals/profiling/`; project orientation is `AGENTS.md`.

## What macOS 27 adds

macOS 27 ships three GPU tools as OS-level binaries in `/usr/bin`:

- `gpucapture` creates replayable `.gputrace` captures from enabled Metal processes.
- `gpudebug` navigates, inspects, fetches resources from, and profiles `.gputrace` captures using a scriptable terminal interface.
- `metalperftrace` collects live or historical performance data for `CAMetalLayer`/drawable sessions and emits text, JSON, or `.atrc` traces.

These tools complement rather than replace existing APIs and Xcode tools:

- `MTLCaptureManager` is the most precise capture mechanism for applications you can instrument.
- `xctrace` with Metal System Trace remains the best external tool for live CPU/GPU scheduling, compute-only workloads, performance-state residency, driver activity, and system correlation.
- `MTLCommandBuffer` timestamps provide exact in-process GPU execution intervals.
- `powermetrics` provides coarse system and per-process corroboration.

The validated environment was:

```text
Apple M3 Pro, Mac15,6
macOS 27.0 (26A5378n)
Xcode 26.6 (17F113)
gpucapture 2027.0.33
gpudebug 1.0
```

The new GPU binaries are supplied by macOS 27. Xcode 27 is not required. A full Xcode installation is still needed for `xctrace`, SDKs, device services, and associated developer workflows; Xcode 26.6 works.

## 1. Verify the host before collecting anything

Capture tool and environment versions with every experiment:

```sh
sw_vers
xcodebuild -version
xcode-select -p

command -v gpucapture
command -v gpudebug
command -v metalperftrace

gpucapture --version
gpudebug --version
gpucapture --list-devices
gpudebug --list-devices

MANPAGER=cat man gpucapture > gpucapture.man.txt
MANPAGER=cat man gpudebug > gpudebug.man.txt
MANPAGER=cat man metalperftrace > metalperftrace.man.txt
```

If Xcode tools are unavailable despite Xcode being installed:

```sh
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -runFirstLaunch
```

Do not assume flags from a beta manual match the installed OS. Treat installed `man` pages and `--help` output as authoritative and retain them with the artifacts.

## 2. Choose the right capture mechanism

### Prefer `MTLCaptureManager` for an application you control

For a self-instrumented CLI, ML inference engine, renderer, or benchmark, `MTLCaptureManager` provides exact semantic boundaries. It avoids races and external boundary-ID discovery and can capture exactly one token, N dispatches, one request, or any application-defined region.

The application must enable Metal capture. Use the `MetalCaptureEnabled` property-list key or launch with:

```sh
MTL_CAPTURE_ENABLED=1 ./target [arguments]
```

Programmatic outline:

```swift
let manager = MTLCaptureManager.shared()
guard manager.supportsDestination(.gpuTraceDocument) else {
    fatalError("GPU trace documents are unsupported")
}

let descriptor = MTLCaptureDescriptor()
descriptor.captureObject = device       // queue or MTLCaptureScope also works
descriptor.destination = .gpuTraceDocument
descriptor.outputURL = outputURL         // must end in .gputrace

try manager.startCapture(with: descriptor)
// Encode and commit precisely the work to capture.
manager.stopCapture()
```

Command buffers need to be created after capture starts and committed before it stops. Captures produced this way are fully compatible with `gpudebug`.

For Rust projects, expose the capture APIs through the existing Objective-C Metal binding or a small platform-specific bridge. Keep capture control outside hot loops except around the exact desired region.

### Use `gpucapture` for an external enabled process

The target must start with `MTL_CAPTURE_ENABLED=1`. For short CLI workloads, also use `MTLCAPTURE_WAIT_FOR_SIGNAL=1`; it pauses at `MTLDevice` creation until `gpucapture` registers, preventing the workload from finishing before capture begins:

```sh
mkdir -p ~/gpu-runs/run-001

MTL_CAPTURE_ENABLED=1 \
MTLCAPTURE_WAIT_FOR_SIGNAL=1 \
  ./target [arguments] \
  >~/gpu-runs/run-001/target.stdout \
  2>~/gpu-runs/run-001/target.stderr &

TARGET_PID=$!
```

Discover the process and available boundaries:

```sh
gpucapture list
gpucapture boundaries --pid "$TARGET_PID"
```

Boundary types include:

- Device: one or more complete command buffers.
- Queue: command buffers from a particular `MTLCommandQueue`.
- Layer: presented frames from a `CAMetalLayer`.
- Scope: application-defined `MTLCaptureScope` begin/end regions.

For a CLI workload with an unknown number of command buffers:

```sh
gpucapture start \
  --pid "$TARGET_PID" \
  --until-exit \
  --output ~/gpu-runs/run-001/workload.gputrace
```

For a bounded capture:

```sh
gpucapture start \
  --pid "$TARGET_PID" \
  --boundary BOUNDARY_ID \
  --count 1 \
  --output ~/gpu-runs/run-001/workload.gputrace
```

Prefer a stable application-assigned queue, layer, or scope label when available:

```sh
gpucapture start \
  --pid "$TARGET_PID" \
  --label 'Decode Token' \
  --count 1 \
  --output ~/gpu-runs/run-001/workload.gputrace
```

Boundary IDs may change between launches. Labels make automation more durable, but ambiguous duplicate labels require falling back to the reported ID.

For GUI apps, execute the bundle binary directly when reliable environment propagation matters:

```sh
MTL_CAPTURE_ENABLED=1 \
  /path/MyApp.app/Contents/MacOS/MyApp [arguments] &
```

### Capture a Python / MLX (or any non-Rust) Metal workload

`gpucapture` refuses any target that is not **debuggable** — `error: invalid PID … Processes must be debuggable … entitled with com.apple.security.get-task-allow`. A `cargo`-built binary has this for free (ad-hoc debuggable), which is why the lmbrrr kernels capture directly. Stock `/usr/bin/python3` (an Xcode/CommandLineTools shim), Homebrew python, and system frameworks are signed **without** `get-task-allow`, so gpucapture rejects them. MLX's in-process `mx.metal.start_capture()` needs no entitlement (a process capturing itself is allowed) **but it writes a streaming bundle gpudebug cannot open** (`cannot open trace: Assertion failed: archive != NULL` — loose `MTLBuffer-*` files, not the sealed `capture`/`index`/`metadata`/`store0` archive gpucapture produces). So the only route to a gpudebug-openable MLX trace is the external `gpucapture` flow against a **debuggable python**.

Make a uv-managed CPython debuggable (no sudo, one-time). uv's `python-build-standalone` interpreters are `adhoc, linker-signed`, **no hardened runtime, no library validation** (`codesign -dv` → `flags=0x20002`) — the easy case: re-signing with `get-task-allow` does not break MLX's dylib loading (there's no library validation to violate).

```sh
# get-task-allow.plist: a <dict> with com.apple.security.get-task-allow = <true/>
P=~/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/bin/python3.12   # the real Mach-O the venv symlinks to
codesign -f -s - --entitlements get-task-allow.plist "$P"      # re-sign in place; adhoc keeps working
codesign -d --entitlements - "$P"                             # verify the key is present
python -c 'import mlx.core as mx; mx.eval(mx.zeros((8,8))@mx.zeros((8,8)))'  # verify import still works
```

Re-signing the shared uv interpreter makes every venv built from it debuggable (harmless — `get-task-allow` only permits task-port access for debugging). This is set up on the M3: the `~/lmbrrr-work/lmbrrr` workspace venv (`.venv/bin/python`, MLX 0.32.0) points at the re-signed cpython-3.12.13.

**Isolate the kernel — gpudebug counters are timeline-GLOBAL, so the captured region must contain ONLY the kernel of interest.** A naive `mx.random.normal(...)` + `mx.quantize(...)` + qmm-loop capture is ~80% Box-Muller RNG (`rbitsc`/`ErfInv`/`Divide`/`Multiply`) and ~6% `affine_quantize`, with the `affine_qmm_t` matmul only ~14% — and the aggregated counters then read the RNG, not the matmul (the tell: `gpu_write_bandwidth ≈ gpu_read_bandwidth`, wrong for a weight-heavy GEMM). Use **two phases**: phase 1 (uncaptured) generates + quantizes + `mx.save_safetensors` the inputs; phase 2 (captured) does `mx.load` + a warmup + an N-iteration kernel loop only. A clean qmm-only capture reads `gpu_write_bandwidth ≪ gpu_read_bandwidth` and a single dominant compute limiter (see §15.E).

## 3. Analyze a capture using a persistent `gpudebug` session

Large captures can take seconds or minutes to load and prepare for replay. Create one persistent session and reuse it; do not repeatedly invoke one-shot mode for a multi-command investigation.

```sh
mkdir -p ~/gpu-runs/run-001/debug-output

gpudebug \
  --json \
  -t ~/gpu-runs/run-001/workload.gputrace \
  -o ~/gpu-runs/run-001/debug-output \
  -c 'list' \
  2>&1 | tee ~/gpu-runs/run-001/gpudebug-open.stream
```

The output includes a line such as `Session 412 created.` Retain the session ID and reuse it:

```sh
gpudebug --json -s 412 -c 'status'
gpudebug --json -s 412 -c 'list'
```

The root normally exposes four branches:

- `commands`: command buffers, render/compute/blit encoders, draws, dispatches, attachments, and bindings.
- `api_calls`: a flat Metal API-call list with cross-links.
- `resources`: buffers, textures, shader libraries, pipeline states, samplers, queues, and other captured objects.
- `performance`: embedded or newly collected replay-profiling results.

Explore rather than assuming paths:

```sh
gpudebug --json -s 412 -c 'go commands' -c 'list'
gpudebug --json -s 412 -c 'find decode'
gpudebug --json -s 412 \
  -c 'go commands/cb0/ce0/disp0' \
  -c 'info --all' \
  -c 'info pipeline'
```

Static navigation and much of `info` can work while replay is still loading or even when the replay device is incompatible. Commands that fetch replayed resources require a ready compatible replayer; check `status`.

Fetch a texture, buffer, shader source, heatmap, or counter series using the actions and names advertised at the current node:

```sh
gpudebug -s 412 \
  -c 'go commands/cb0/re0/draw0' \
  -c 'fetch color0 --out ~/gpu-runs/run-001/debug-output/color0.png'

gpudebug -s 412 \
  -c 'fetch @buf4 --out ~/gpu-runs/run-001/debug-output/buf4.bin'
```

Terminate sessions when finished:

```sh
gpudebug --terminate 412
```

Use `--oneshot` only for isolated queries where repeated trace startup is acceptable:

```sh
gpudebug --oneshot --json \
  -t workload.gputrace \
  -c 'go commands/cb0/ce0/disp0' \
  -c 'info pipeline'
```

## 4. Parse `gpudebug --json` defensively

In gpudebug 1.0, `--json` is not a single clean JSON document or guaranteed NDJSON stream. Observed behavior includes:

- a plain-text `Session N created.` preamble;
- one JSON object per `-c` command;
- occasional duplicate objects;
- structured values whose units may be omitted.

A consumer should:

1. Read stdout/stderr as a stream rather than calling a single-document JSON parser.
2. Extract and preserve non-JSON diagnostic lines separately.
3. Incrementally detect complete JSON objects or arrays.
4. Associate responses with the submitted command sequence.
5. Deduplicate only exact repeated objects and retain a warning that duplication occurred.
6. Store the raw stream beside normalized output.
7. Preserve unknown properties and counter names instead of discarding them.

Do not infer units silently. For example, observed values were:

```text
gpu_bandwidth        88.55
gpu_read_bandwidth   78.37
gpu_write_bandwidth  10.19
```

The output omitted units, while occupancy values explicitly included `%`. Read plus write approximately equaled total, and comparison with a known traffic model supported a percent-of-peak interpretation. Treat that as an empirical hypothesis with provenance, not a guaranteed GB/s conversion.

One interface discrepancy in gpudebug 1.0: the `performance/timeline` node advertises an `info` action but has no functioning handler. Profiling-session summary fields such as GPU time, execution mode, core count, overlap, and performance state are available from the parent `performance` node via `info --all`.

## 5. Profile captured GPU work

On M3/A17-class hardware or newer, `gpudebug` collects a GPU replay profile. **Verified working recipe (2026-07-16, gpudebug 1.0, M3 Pro), with two corrections to the earlier notes:**

1. **`profile run` only COLLECTS — you must then `profile load 0` (explicit index) before the `performance/*` tree is navigable.** Bare `profile load` returns `loaded_session_index: -1` and the tree stays EMPTY (every node `totalCount: 0`), which looks like a broken capture but is just the un-loaded state. `profile run --embed` alone is NOT enough.
2. **`go performance` + `info --all` errors (`'performance' has no info handler`)** — ignore the "inspect the performance root via info" step from older notes; go straight to the counter groups.

Also: after opening a session, `status` shows `replayer.state: loading` → `ready`; **wait for `ready` before `profile run`** (poll `status`). Use a persistent session for a multi-step investigation (open once, reuse `-s N`); `--oneshot` DOES work for a self-contained run **as long as the same invocation chains `profile run` → `profile load 0` → the `go` queries** (state does not survive across separate `--oneshot` processes).

**DEADLOCK WARNING (2026-07-17, MLX qmm trace, M3 Pro) — do NOT `--embed`, and do NOT re-`profile run` a dirty session.** On a ~1.2 GB external `gpucapture` trace, `profile run --gpu-state default --exec overlapping --embed` **hung indefinitely** and held the session's command-queue lock. Symptom to recognize: the `profile run` client sits at 0% CPU, a *separate* `gpudebug -s N -c status` client **also blocks** (status normally answers during load — a blocked status is the tell that the session command-queue lock is held), and `GPUToolsReplayService.xpc` is idle at 0% CPU (a deadlock in the embed write-back, not slow I/O). Two triggers, both avoidable:
1. `--embed` serializes the collected profile back INTO the trace bundle; on a multi-hundred-MB-buffer trace this write-back deadlocked. The §5 recipe does **not** use `--embed` — the "`--embed` alone is NOT enough" note above means *don't reach for it*, not *add it*. Use plain `profile run`, then `profile load <idx>`.
2. If the first `profile load 0` fails, **open a FRESH session on the trace — never fire a second `profile run` at a session that already collected.** The second run on the dirty session is what wedged.
- **`profile load 0` can legitimately fail with `no profiling sessions in trace`** on a fresh external capture (no *embedded* profile). `profile load` reads embedded sessions; a just-collected `profile run` result may be at a different index — run `profile list` (or `profile` with no arg) to find the real index, then `profile load <idx>`.
- **Recovery from a wedged session:** `kill -9` the blocked `gpudebug -s N` client(s) AND `kill -9` the `GPUToolsReplayService.xpc` process (it leaks the locked session; launchd respawns it on the next open). The trace bundle is unharmed — verify with `find <trace> -iname '*lock*' -o -iname '*.tmp'` (empty) and all files still at capture time.
- **Never `run_in_background` a `profile run`.** Run it foreground with a bounded timeout so a hang is immediately visible instead of masquerading as "still working."

```sh
# 1. open (slow: ~1 GB/trace); grab "Session N created." and wait for ready
gpudebug --json -t workload.gputrace -o out -c 'status'      # -> Session 28, replayer loading
gpudebug --json -s 28 -c 'status'                            # repeat until replayer.state=ready
# 2. collect + LOAD (load 0 is mandatory) — ~4 s run + ~2 s load
gpudebug --json -s 28 \
  -c 'profile run --gpu-state default --exec overlapping' \
  -c 'profile load 0'
# 3. counters live at performance/timeline/counters/<group>; `go <group>`
#    returns its sub-counters WITH values (no separate `info` needed).
gpudebug --json -s 28 -c 'go performance/timeline/counters' -c 'list'   # 30 groups
gpudebug --json -s 28 -c 'go performance/timeline/counters/occupancy'   # values inline
```

The 30 counter groups (each `go <group>` yields named sub-counters as `%` or a bare number):

- **occupancy**: `vs_/fs_/kernel_/total_occupancy` — % of max resident threads.
- **occupancy_manager**: `occupancy_manager_target` (the occupancy the GPU is *willing* to run; <100% ⇒ GPU is deliberately capping) + `l1_eviction_rate`.
- **alu**: `alu_utilization`. **f32**/**f16**: `_limiter` + `_utilization`. **instruction_throughput**: `_limiter` + `_utilization`.
- **bandwidth**: `gpu_bandwidth` / `gpu_read_bandwidth` / `gpu_write_bandwidth` (unitless ≈ % of peak; read+write≈total).
- **shader_launch_limiter**: `vertex_/fragment_/compute_shader_launch_limiter` (>80% ⇒ threads launch fine, not launch-bound).
- **last_level_cache**, **mmu**, **active_cores**, plus texture/control_flow/address_generation/integer_* groups.

**Reading the ladder** (per the WWDC M3 talk): a `_limiter` is the fraction of GPU-active time that unit gated issue; the highest `_limiter` across units is your bottleneck. Low occupancy with `occupancy_manager_target ≈ 100%` and low `l1_eviction_rate` ⇒ occupancy is capped by **register pressure** (not the manager, not cache) → 16-bit types / fewer live registers raise it.

**CRITICAL method lesson (2026-07-16, mm2d_q2_0): a `_limiter` names a CANDIDATE; validate by MOVING THE NUMBER.** The occupancy ladder is necessary but not sufficient — a low occupancy that *looks* binding may not be. Case: mm2d_q2_0 at 39% occupancy (vs a healthy mv at 81%) looked occupancy-limited. Right-sizing its threadgroup array (8 KB → 2 KB) and dropping a `max_total_threads` attribute raised occupancy to **51% — with ZERO change in ms/call or gpu_bandwidth.** That directly *disproved* occupancy as the bottleneck; the real limiter was `instruction_throughput_limiter=73%` (a scalar epilogue), which the occupancy fix never touched. Also note: `maxTotalThreadsPerThreadgroup` is a reported *ceiling* (set by the `[[max_total_threads_per_threadgroup]]` attribute), NOT actual registers-per-thread — don't infer register usage from it. So: apply the candidate fix, re-capture, and only believe the diagnosis if the *speed* moves. Never mass-apply an occupancy fix across kernels off the diagnosis alone.

Older ad-hoc explore (per-encoder/per-shader COSTS, still valid, but the rich COUNTERS only exist under `performance/timeline/counters` after `profile load 0`):

```sh
gpudebug --json -s 28 -c 'go performance/timeline' -c 'list'   # encoders / counters / shaders
```

Typical results include:

- per-encoder cost rankings;
- per-draw or per-dispatch command costs;
- per-shader costs and invocation counts;
- a GPU execution timeline;
- hardware counter series;
- performance heatmaps where supported.

### Headline recipe: default versus fixed GPU state

Profile the identical capture at the default operating behavior and at a fixed high state:

```sh
gpudebug --json -s 412 \
  -c 'profile run --gpu-state default --exec overlapping'

gpudebug --json -s 412 \
  -c 'profile run --gpu-state high --exec overlapping'
```

This comparison separates two effects:

- DVFS/operating-point sensitivity: the reduction in replay GPU busy time at the fixed high state.
- Scheduling or queueing gap: wall time not explained by GPU busy execution at the chosen state.

On the validated LLM decode workload, this decomposed an approximately 1.3 ms/token wall-versus-busy discrepancy into approximately 0.7 ms attributable to GPU operating point and approximately 0.6 ms attributable to a true scheduling gap.

Use `overlapping` for realistic GPU execution. A `serial` profile can improve per-command attribution but changes scheduling and must not be reported as representative wall-clock performance.

Fixed-state replay is diagnostic, not production behavior. Record device model, OS, gpudebug version, thermal state, selected GPU state, execution mode, and whether the profile was embedded.

## 6. Use `metalperftrace` for drawable/frame workloads

`metalperftrace` is intended for applications with `CAMetalLayer` sessions. It records frame pacing and related per-layer resource statistics, not captured command state.

Live NDJSON:

```sh
metalperftrace listen \
  --json \
  --pid "$TARGET_PID" \
  --interval 1 \
  --output ~/gpu-runs/run-001/live.json
```

The output file is rewritten as a JSON array when the process receives SIGINT, SIGTERM, or SIGHUP. If consuming stdout directly, treat it as NDJSON updates.

Look back over system-recorded sessions:

```sh
mkdir -p ~/gpu-runs/run-001/perf

metalperftrace collect \
  --last 5m \
  --json \
  ~/gpu-runs/run-001/perf
```

Analyze each resulting `.atrc`:

```sh
metalperftrace overview \
  --json \
  TRACE.atrc \
  > overview.json

metalperftrace overview \
  --json-include-timeline \
  TRACE.atrc \
  > timeline.json
```

Layer-session metrics may include FPS, frame timing, on-GPU time, drawable wait, skipped frames, CPU/resource use, memory, and shader compiler activity.

Optional runtime features:

```sh
metalperftrace setup --enable per-frame-metrics --pid "$TARGET_PID"
metalperftrace setup --enable shader-compiler-metrics --pid "$TARGET_PID"
metalperftrace setup --enable hud --pid "$TARGET_PID"
```

`per-frame-metrics` means per-drawable signposting and may increase trace size.

### StateReporting

StateReporting lets an application add relatively low-frequency semantic states—level, model, quality setting, workload phase, request class—to Metal performance traces. `metalperftrace overview` can display transitions and aggregate metrics by domain and label:

```sh
metalperftrace overview \
  --include-state-transitions \
  TRACE.atrc

metalperftrace overview \
  --aggregate \
  --domain com.example.workload \
  --state-label Decode \
  TRACE.atrc
```

StateReporting is contextual metadata, not a high-frequency event stream. Do not emit one transition per dispatch or token if that cadence triggers throttling.

## 7. Compute-only limitation in `metalperftrace`

macOS 27.0 does not create `metalperftrace` sessions for a pure `MTLCommandQueue` process with no `CAMetalLayer`.

Verified result on a compute-only candle/Metal LLM decode engine:

```text
metalperftrace listen --pid ...       -> zero-byte NDJSON
metalperftrace collect                -> approximately 1.7 KB .atrc
metalperftrace overview --json        -> {"error":"No session found"}
```

This is a tool limitation, not a missing flag:

- the installed manual defines collection for Metal layers;
- reports and filters are per-layer and expose `layerName`;
- `per-frame-metrics` is explicitly per-drawable;
- enabling it cannot synthesize a compute-queue session.

For live headless compute, use the next section.

## 8. Measure live compute workloads correctly

### Exact in-process command-buffer timing

After a command buffer completes, Metal exposes:

- `GPUStartTime`: host-domain time when GPU execution began;
- `GPUEndTime`: host-domain time when GPU execution ended;
- `kernelStartTime` and `kernelEndTime`: CPU-side scheduling intervals where available.

Record an application commit timestamp and completion-handler timestamp as well:

```text
GPU busy       = GPUEndTime - GPUStartTime
queue delay    = GPUStartTime - application commit timestamp
completion lag = completion-handler timestamp - GPUEndTime
```

Read GPU timestamps after `waitUntilCompleted` or inside an added completion handler; they remain zero until completion.

If one token/request spans multiple command buffers, compute the union of all `[GPUStartTime, GPUEndTime]` intervals. Summing intervals can overcount overlapping execution. Sort intervals by start time, merge overlaps, and separately measure gaps between merged intervals.

This instrumentation is not an unfortunate workaround—it is the precise supported API for application-attributed live command-buffer timing. What is missing is an OS-level `metalperftrace` aggregation/export path for the same data.

### External live observation with Metal System Trace

Use `xctrace` when the application cannot be modified or when driver scheduling, performance state, resource activity, display/system interactions, or thermal context matters:

```sh
xcrun xctrace record \
  --template 'Metal System Trace' \
  --attach "$TARGET_PID" \
  --time-limit 30s \
  --no-prompt \
  --output ~/gpu-runs/run-001/compute-live.trace
```

Alternatively, launch under the trace:

```sh
xcrun xctrace record \
  --template 'Metal System Trace' \
  --time-limit 30s \
  --no-prompt \
  --output ~/gpu-runs/run-001/compute-live.trace \
  --env MTL_CAPTURE_ENABLED=1 \
  --launch -- ./target [arguments]
```

Discover exportable tables dynamically:

```sh
xcrun xctrace export \
  --input ~/gpu-runs/run-001/compute-live.trace \
  --toc \
  --output ~/gpu-runs/run-001/toc.xml
```

Useful schemas commonly include:

- `metal-gpu-intervals`
- `metal-application-command-buffer-submissions`
- `metal-application-encoders-list`
- `metal-gpu-execution-points`
- `metal-gpu-counter-intervals`
- `gpu-performance-state-intervals`
- `metal-gpu-state-intervals`
- `metal-resource-allocations`
- shader-profiler tables
- process, thread, signpost, display, and thermal tables

Export only schemas advertised by the trace TOC:

```sh
xcrun xctrace export \
  --input ~/gpu-runs/run-001/compute-live.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' \
  --output ~/gpu-runs/run-001/metal-gpu-intervals.xml

xcrun xctrace export \
  --input ~/gpu-runs/run-001/compute-live.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="gpu-performance-state-intervals"]' \
  --output ~/gpu-runs/run-001/gpu-performance-state-intervals.xml
```

`xctrace` XML is typed but compacted with reusable `id` and `ref` values plus sentinel elements. A parser needs streaming reference resolution and must preserve schema/column metadata. Schema availability varies with Xcode, hardware, template settings, and recording conditions.

### Coarse corroboration with `powermetrics`

`powermetrics` can report sample-window per-process GPU time and system GPU frequency/power data:

```sh
sudo powermetrics \
  --samplers tasks,gpu_power \
  --show-process-gpu \
  --sample-rate 250 \
  --sample-count 120 \
  --format plist \
  --output-file ~/gpu-runs/run-001/powermetrics.plist
```

Limitations:

- requires privilege;
- sample-window rather than command-buffer attribution;
- system GPU readings can include unrelated processes;
- plist output is NUL-separated;
- estimated power is not appropriate for comparison between different devices;
- short token-level behavior may be below its useful resolution.

Use it as corroboration, not as the sole source of truth.

### Application counter sampling

For pass-level timestamps or supported statistics, query `MTLDevice.counterSets`, create an `MTLCounterSampleBuffer`, sample at supported encoder/pass boundaries, and resolve results. This provides targeted live counter data but increases implementation complexity and may affect or constrain measurement. Capability-detect every counter set and retain units and raw metadata.

## 9. Understand what each tool measures

| Question | Best source |
|---|---|
| Exact application-attributed command-buffer GPU time | `GPUStartTime` / `GPUEndTime` |
| Token/request queueing and scheduling gaps | Application boundaries plus merged command-buffer intervals |
| External live driver/GPU scheduling | `xctrace` Metal System Trace |
| Live GPU performance-state residency | `xctrace` performance-state schemas |
| Per-process/system coarse GPU usage and frequency | `powermetrics` |
| Captured shader, pipeline, binding, and resource state | `gpudebug` |
| Replay per-shader, encoder, command, and counter costs | `gpudebug profile` |
| Long-running drawable/frame behavior | `metalperftrace` |
| Exact semantic capture range in code you control | `MTLCaptureManager` |
| External replayable capture | `gpucapture` |

A replay profile is not an end-to-end benchmark. A live wall-clock measurement is not a shader attribution profile. Use both and state which clock and execution mode each number represents.

## 10. Reproducible optimization protocol

For every before/after comparison:

1. Record hardware model, OS build, Xcode/tool versions, application commit, build profile, compiler flags, model/workload identity, input sizes, and environment variables.
2. Eliminate first-run compilation and allocation effects unless cold-start behavior is the subject. Record warm-up policy.
3. Use stable semantic capture boundaries. Prefer `MTLCaptureManager` for controlled workloads.
4. Run repeated live measurements and report distribution statistics, not one best run.
5. Record thermal state and background activity. Avoid comparing traces collected under materially different conditions.
6. Capture one representative unit of work and profile it at default and fixed GPU states.
7. Keep `overlapping` replay for realistic scheduling; use `serial` only as an explicitly labeled attribution experiment.
8. Compare live wall time, merged GPU busy intervals, queueing gaps, replay GPU time, per-shader cost, dispatch counts, encoder costs, occupancy, and counters.
9. Verify semantic equivalence: same outputs, dispatch dimensions, pipelines, resource formats, and workload counts.
10. Preserve raw outputs and normalized summaries. Never retain only the agent's prose conclusion.

For LLM decode specifically, useful normalized metrics include:

- wall milliseconds per token;
- merged GPU-busy milliseconds per token;
- queue/scheduling gap per token;
- command buffers, encoders, and dispatches per token;
- shader cost and invocation count;
- default-versus-high-state replay delta;
- memory traffic model versus observed bandwidth percentage;
- CPU submission time and completion latency;
- model dimensions, context length, batch size, precision, and kernel configuration.

## 11. Artifact layout

Suggested structure:

```text
gpu-runs/run-001/
  environment/
    sw-vers.txt
    xcode-version.txt
    tool-versions.txt
    gpucapture.man.txt
    gpudebug.man.txt
    metalperftrace.man.txt
    git-revision.txt
    workload.json
  capture/
    workload.gputrace/
  gpudebug/
    open.raw-stream
    trace-tree.jsonl
    profile-default.raw-stream
    profile-high.raw-stream
    encoders.jsonl
    commands.jsonl
    shaders.jsonl
    counters.jsonl
    fetched/
  live/
    command-buffer-timings.jsonl
    compute-live.trace/
    toc.xml
    metal-gpu-intervals.xml
    gpu-performance-state-intervals.xml
    powermetrics.plist
  target/
    stdout.txt
    stderr.txt
  report/
    normalized.json
    comparison.md
```

GPU trace and Instruments bundles are directories, can be large, and may contain shader sources, buffers, labels, process names, and paths. Treat them as potentially sensitive. Archive deliberately and do not publish captures by default.

## 12. Troubleshooting

### `gpucapture list` does not show the target

- Verify the target inherited `MTL_CAPTURE_ENABLED=1`.
- Ensure it creates an `MTLDevice` and remains alive long enough to connect.
- For short CLI jobs, add `MTLCAPTURE_WAIT_FOR_SIGNAL=1`.
- Execute the real binary directly if a launcher may discard the environment.
- Check selected device/PID when more than one target exists.

### Capture boundary is ambiguous

- Run `gpucapture boundaries --pid PID`.
- Prefer a unique application label.
- Fall back to the numeric boundary ID when duplicate labels exist.
- Add a custom `MTLCaptureScope` if you control the application.

### `gpudebug fetch` fails

- Run `status` and wait for the replayer.
- Confirm a compatible replay device is available.
- Static navigation may work even when replay-dependent fetch does not.
- Use a device with compatible GPU/OS characteristics when possible.

### `gpudebug --json` fails strict parsing

- Expect the session preamble and multiple JSON objects.
- Parse incrementally and save the raw stream.
- Deduplicate exact repeats with a warning.
- Do not assume the output is one document.

### `performance/timeline info` fails

- Navigate to `performance` and run `info --all` there.
- Treat the child node's advertised `info` action as a gpudebug 1.0 defect.

### `metalperftrace` returns no session

- Confirm the process owns an active `CAMetalLayer` and presents drawables.
- For a compute-only target, stop troubleshooting: macOS 27.0 has no supported command-queue session. Use command-buffer timestamps and/or Metal System Trace.

### Performance numbers disagree

- Identify whether each number is wall time, GPU busy time, serialized replay cost, overlapping replay time, frame interval, or sample-window utilization.
- Check GPU state and thermal conditions.
- Merge overlapping command-buffer intervals.
- Separate startup/compilation from steady state.
- Confirm identical workload counts and outputs.

## 13. Recommended Apple feature requests

1. Add PID/`MTLCommandQueue` sessions to `metalperftrace`, initially aggregating command-buffer GPU start/end intervals, queueing gaps, and GPU performance-state residency while retaining NDJSON and `.atrc` output.
2. Add a strict machine-output mode to `gpudebug`: no text preamble, one documented response envelope per submitted command, stable command correlation, and no duplicate objects.
3. Include explicit units and semantic definitions for every performance counter, especially GPU read/write/total bandwidth.
4. Correct the advertised `performance/timeline` `info` action or implement its handler.
5. Publish a versioned JSON schema or schema identifier for agent integrations.

## 14. Metal 4.x tensor / `matmul2d` quantized-weight support (compute capability)

Separate from capture/profiling: what the M3's Metal stack can actually *run* for a weight-bound quantized GEMM, established by inspecting the shipped toolchain headers + the MSL spec (`Metal-Shading-Language-Specification`, 2026-06-04) + on-device test-compiles. This gates the ternary DSpark verify GEMM (see `docs/research/dspark-verify-weightbound-gemm.md`).

- **The hardware matmul is `tensor_ops::matmul2d`** (framework `<MetalPerformancePrimitives/…>`, namespace `mpp::tensor_ops`), consuming `tensor<address_space T, dextents, …>` operands. It decodes packed weight formats **in hardware** — this is why candle's q4_K mm2d (`kernel_mul_mm2d_q4k`, `tensor<device uint4b_format>`) reaches 142 GB/s while a hand-rolled `simdgroup_matrix` + software unpack of the same weight plateaus ~38 GB/s. It CANNOT be JIT-compiled by the runtime Metal compiler — build offline into a `.metallib` (`xcrun metal -std=metal4.x -c … ; xcrun metallib …`, per `candle-metal-kernels/scripts/build_mm2d_q4k.sh`) and `new_library_with_data` it.
- **`matmul2d` operand element types** (`MPPTensorOpsMatMul2dImpl.h` whitelist): A (activation) = half/bfloat/uint8/int8/float; **B (weight)** = uint8, int8, `uint4b_format`, `int4b_format`, and — **only in Metal 4.1** — `uint2b_format`/`int2b_format` (2-bit). The sub-byte packed operand is **always B**, i.e. this is intrinsically a *weight-bound* layout (activations dense, weights packed). Accumulate to float/int32. Ternary {−1,0,+1} ⇒ `int2b_format` (3 of 4 codes).
- **Version gating** (MSL §2.21–2.22, §7.2): tensor types + `matmul2d` = Metal 4.0; `int4b_format`/`uint4b_format` = Metal 4.0 + SDK 26.4; **`int2b_format`/`uint2b_format` = Metal 4.1**. Packed tensors: K axis in dimension 0, extent[0] a multiple of 32 (Fblock), **128-byte row-stride alignment**, per-block scales via `tensor_blockwise<tensor_plane_scales>`, scope `execution_simdgroups<simdgroups_per_threadgroup>`, C as a `cooperative_tensor` (no device barrier; read per-thread via `get_multidimensional_index`). No `pack`/`unpack` helper for 2-bit — rely on the tensor/matmul2d op to consume it.
- **M3 timeline — 2-bit was gated on Xcode 26.6, UNBLOCKED on Xcode 27 beta 3 (2026-07-16).** Same M3, macOS 27.0 beta, same hardware — the only variable was the toolchain version:
  - *Xcode 26.6 (stable), Metal toolchain `32023.883`:* 2-bit unreachable. (a) `-std=metal4.0` compiles `uint4b_format` but **`uint2b_format` fails** ("unknown type name"; only 4-bit in `metal_packed_numeric` lines 15/20); (b) **`-std=metal4.1` rejected** ("invalid value 'metal4.1' in '-std='"). `xcodebuild -downloadComponent MetalToolchain` re-downloaded the SAME `32023.883` — the *stable* line's toolchain has no 4.1.
  - *Xcode 27.0 beta 3 (build `27A5218g`) + its Metal Toolchain component (`27A5218h`, compiler **`metalfe-32023.918.1`**, target `air64-apple-darwin27.0.0`):* all three checks pass. `-std=metal4.1` valid; `uint2b_format` compiles; and `tensor_ops::matmul2d` **accepts `uint2b_format` as its B operand** — a minimal kernel mirroring `mm2d_q4k.metal` with a 2-bit B tensor compiled AND linked to an 11 KB `.metallib`. The MPP whitelist accepts 2-bit on the M3's air64 target (no M5 needed).
  - *Install path (the Metal Toolchain is decoupled from Xcode since 26):* the beta must come from the **Xcode _beta_** train (27.x) — the 26.6 stable toolchain lacks 4.1. Grab Xcode-beta via the developer-downloads web page (visual; xcodes CLI auth is flaky), then `DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer xcodebuild -downloadComponent MetalToolchain` (838 MB; no sudo). License accept needs one `sudo xcodebuild -license accept`.
  - **Lesson:** a toolchain-capability "wall" is a *dated* fact tied to the installed compiler build, not a permanent limit. Re-test on every Xcode beta with `xcrun metal -std=metal4.1 -c` on a `uint2b_format` matmul2d kernel before treating a platform gap as a dead end. The DSpark verify GEMM is now the near-mechanical q4_K-mm2d mirror (docs/research/dspark-verify-weightbound-gemm.md).

## 15. Kernel perf-diagnosis playbook + the ternary-verify utilization case study (2026-07-16)

A multi-day investigation into *why the Q2_0 verify matmul (M=5–8) under-utilizes the M3* produced a reusable diagnosis process and a settled architectural answer. Full case study + measurements: `docs/research/dspark-verify-weightbound-gemm.md`. This section is the durable methodology + the "don't retry these" record.

### A. The confounds that bit us — control them BEFORE believing any number

Every one of these silently inflated a comparison by 30–130% and produced a *false* conclusion until controlled. Check all three at the start of any A/B, not the end.

1. **DVFS / thermal clock droop across a sweep.** Running N variants back-to-back (200 iters each) heats the GPU; *later* variants run at a drooped clock and read 2–3× slower — a pure position artifact, not a kernel difference (`pmset -g therm` showed no *warning*, but the clock still droops). **Fix: interleave.** Round-robin small batches per variant (`for round { for variant { sync; time B dispatches; sync } }`) so the drift hits every variant equally. A sequential per-variant sweep is invalid for relative comparison. This flipped an apparent "NR=4 is faster" into the true "NR is monotonically slower."
2. **Harness / dispatch-path difference.** A kernel benched via a hand-rolled `command_encoder()` closure vs via the production `lin.forward` (MixedLinear) path can differ. **Control: run the SAME (reference) kernel through BOTH harnesses.** If ref-via-your-harness == ref-via-production, the harness is neutral; the remaining gap is the kernel. (Ours was neutral — the cost view confirmed both are compute-dominated.)
3. **Threadgroup 2D shape.** Identical thread *count* in a different 2D shape (`(8,8)` vs `(32,2)`) changes Metal's physical thread adjacency / memory scheduling and moved a number ~10%. **Match the reference kernel's `threadsPerThreadgroup` shape exactly**, not just the count.

### B. The loop, in order

1. **Validate the ruler first.** A known-equal pair MUST read equal (e.g. `mc`-via-lin.forward ≈ `mc`-via-your-harness ≈ your-templatized-`mc`). If they don't, you're measuring a confound — stop and fix it before interpreting anything.
2. **Attribute cost, don't assume wall==kernel.** Capture the isolated kernel; `performance/timeline/encoders` → per-encoder GPU time. If the compute encoders dominate (~92% for us) the wall number is real kernel time; the blit/zeros/setup encoders are the harness and were ~1%. (Per-shader/per-command *cost* trees are empty in gpudebug 1.0; per-*encoder* GPU time works.)
3. **Counters name a CANDIDATE; validate by MOVING THE NUMBER** (see §5). We "fixed occupancy" 39→51% with zero speed change — a *disproof*, not a null result.
4. **Isolate ONE variable per experiment.** A strip-probe (e.g. `probe_nofold`, `probe_fullk`) proves only that the *stripped* part is small — it does not prove any specific remainder is the bottleneck. State what's still unattributed. (See memory `after-each-result-what-does-it-prove`.)

### C. Interpreting occupancy on M3 (Dynamic Caching era)

- `occupancy_manager_target` is the GPU's **dynamic** occupancy decision, **not a hard resource cap** — confirm with pipeline-info (`maxTotalThreadsPerThreadgroup` and `staticThreadgroupMemoryLength`; ours were 1024 / 0 for every GEMV variant, so *no* static resource limited them).
- The target is a **readout of memory-access efficiency**: healthier access → higher target (isolated captures: mv 90% > mc 41% > transposed-`mct` 31%; the worse the coalescing, the lower the manager sets it, *and you meet it*). Raise it by improving transaction efficiency (coalescing, fewer/wider loads), NOT by adding threads.
- **Apple GPUs hide latency via OCCUPANCY, not per-thread ILP.** Adding accumulators (ILP, row-amortization) costs registers → lowers occupancy → loses. This is why `mc` deliberately uses `nr=2` and why every "more work per thread" variant we tried was slower.

### D. The settled result — two utilization walls, neither reaches the ideal

Roofline says M=8 verify *could* be weight-bound at ~0.22 ms / ~106 GB/s. It is not reachable on M3:

- **Tensor path** (`mm2d_q2_0`, hardware `uint2b_format matmul2d`): **compute-bound** at M=8 ≈ **0.55 ms / 43 GB/s**. Pre-M5 GPUs have **no matrix unit** — `matmul2d` lowers onto `simdgroup_matrix`+ALU (Rigel arXiv 2606.12765 on M4-Max: fp8 at 0.94× fp16; Apple WWDC26 330: "falls back to optimized shader implementations"). The fixed 8×8 fragment + dequant overhead at small M is the ~2× penalty. **This is the best verify kernel** and is 2.1× over the shipped `mc`.
- **GEMV path** (`mc`, weight-shared-columns): **occupancy-bound** ~41% ≈ **1.18 ms / 20 GB/s**. `nr=2` is already optimal.

**Do NOT retry these (all measured-refuted):** occupancy right-sizing (moved occupancy, not speed); ILP/independent-accumulators (`mc2`, slower); transposed `[K][M]` activation (`mct`, slower — breaks cross-thread coalescing); NR row-amortization (monotonically slower — register cost); int8 integer matmul2d (no integer datapath before M5); K-tile > 128 or `relaxed_precision` (no/marginal). Every production Apple LLM stack (llama.cpp/MLX/BaseRT) uses a GEMV at small M and reserves `matmul2d` for large-M prefill — our result reproduces that consensus.

**Consequence:** the DSpark verify per-matmul cannot go below mm2d's 0.55 ms on M3, so the speculative-decode multiplier is NOT in the verify kernel — it comes from **wiring mm2d into the verify** (bank the 2.1×), **`block_size`** (fill mm2d's flat-to-M=8 tile), and **CUDA/M5** (where int8 + the tensor unit engage). The utilization limit itself is architectural on this GPU.

### E. Cross-check against MLX's qmm — the same wall, reached a different way (2026-07-17)

Independent corroboration that mm2d is the right verify kernel at M≤8, from profiling MLX's own 2-bit path on the same M3 (debuggable-python capture per §2). Two settled facts:

1. **MLX doesn't even use its dequant-once QMM kernel at our width.** MLX dispatches the memory-bound `qmv` GEMV until `M ≥ get_qmv_batch_limit == 10` (M3 tier 'g', big mats); the compute-bound `qmm` simdgroup path engages only at M≥10. So m=5–8 (our verify) runs `qmv`, which is **~2× slower than our mm2d** on every shape (gate_up m=5: 23.8 vs 42.9 eff GB/s). MLX's `qmv` barely amortizes weight reuse across rows — ms/call grows ~linearly (0.56→3.05 ms, m=1→8) vs our mm2d flat at 1.10 ms to m=8. This *refuted* a research-agent prediction that MLX `qmv` would stay ~memory-bound at 100–125 GB/s; measurement said the opposite.
2. **When forced (m=32), MLX's `qmm` is f32-datapath-bound, and that floor is fundamental.** Clean qmm-only gpudebug profile (`affine_qmm_t_bfloat16_t_gs_128_b_2`, gate_up 34816×5120, m=32), limiters clock-independent (identical at `--gpu-state default` and `high`): **`f32_limiter 91.55%`** (util 84.13%), `instruction_throughput_limiter 85.08%`, `alu_util 48.28%`, occupancy 32.23% (manager_target 39.43%, L1-evict 0.13), `gpu_bandwidth 12.93` (read 11.17 / write 1.76 — nowhere near DRAM). qmm dequants the 2-bit weights to a **dense bf16** tile in threadgroup memory, then runs a dense bf16 simdgroup GEMM; bf16 executes on the **f32 pipe** (the f16 counters are empty), so it pays full dense-GEMM FLOPs and saturates f32 at ~91%. That is why its wall floor (~3.1 ms gate_up, flat m=10→32) is 2.8× our mm2d's 1.10 ms and cannot drop without abandoning the dequant-to-dense structure.

**The two kernels hit DIFFERENT walls, and ours is the cheaper one at low M:** our mm2d (`matmul2d` on the packed `uint2b` operand) is **instruction-throughput-bound at 73%** (§D — the 2-bit unpack), consuming the compressed operand directly; MLX qmm is **f32/bf16-datapath-bound at 91%** (a genuine dense GEMM after dequant). MLX has no fixable headroom that would beat mm2d at M≤8 — already ~91% f32-limited at M=32, and it doesn't use qmm below M=10 at all. Independently confirms §D: at the M=5–8 verify width the mm2d tensor-op route is the best available on M3; the spec-decode multiplier is not in the verify kernel.

**Method correction recorded:** my first MLX capture's counters were RNG-polluted (§2 two-phase lesson) and I *mis*-read them as "not saturated, has headroom." The clean isolated capture inverted that to "f32-saturated at 91%." Believe only the isolated-kernel capture; a `write≈read` bandwidth split is the tell that setup/RNG is in-frame.

### F. Live whole-loop triangulation + the strip-probe closure (2026-07-17)

Two additions that finish the §D/§E arc at the SYSTEM level (details: `docs/research/dspark-verify-weightbound-gemm.md` §strip-probe; ledger: ticket `eval-harness-validity-fixes` comments):

1. **The in-loop spec verify wall (192 ms/round at m=5 = 84% of the 229 ms round) is pure kernel time.** Eliminated by direct measurement on the LIVE loop (not replay): host CPU 1.5% during steady decode (ps probe); tap-layer capture cost nil (A/B/A profile, 265.7 vs 266.3 vs 266.0 ms/step); xctrace 15 s attach → depth-0 Compute intervals: **GPU 98.6% busy, gaps 213 ms/15.5 s (mostly 100 µs–1 ms), perf-state Maximum 100.0%**. No DVFS, no bubbles, no host — composing the per-shape bench-gemv kernel times reproduces the wall exactly. The xctrace XML parser for this lives at the M3's /tmp/parse_trace.py (id/ref-interned rows, positional columns).
2. **The §D "two instruction-reduction levers" are now MEASURED via the strip-probes** (built 2026-07-16, first run today): fold epilogue −7%, whole discrete K-loop −15–18% (probe_fullk t32 = 0.451 ms / 52.5 GB/s = the op's small-M ceiling; bigger M-tiles NEGATIVE). The "scalar epilogue is the limiter" hypothesis in the research doc is refuted; the cost is inside `matmul2d` (consistent with §D's no-matrix-unit account). `tensor_blockwise` remains worth ≤ ~15–18% on mm2d shapes (~+10–12% tok/s end-to-end); beyond that, only B3 bitplane/popcount (structure change) or M5.

Profiler caveat quantified in passing: the in-binary per-op profiler (`gguf profile`) reads ~253 ms/step for a loop whose true wall is 192 ms (and 235 vs 69 ms at m=1) — per-component sync bias swamps small ops (norms read 28 ms/step; they are ~µs kernels). Use it to enumerate calls, never for absolute walls; walls come from the in-loop timers + xctrace.

## Bottom line

macOS 27 removes the Xcode GUI from most GPU capture and replay-analysis loops. For applications you control, the strongest workflow is:

```text
MTLCaptureManager exact capture
  -> persistent gpudebug session
  -> default and fixed-state overlapping profiles
  -> per-shader/encoder/command/counter analysis
  -> live command-buffer timestamps
  -> xctrace Metal System Trace when scheduling/state context is needed
```

For drawable applications, add `metalperftrace` for long-session and per-frame telemetry. For compute-only applications, do not expect `metalperftrace` to produce a session on macOS 27.0; use public command-buffer timestamps and Metal System Trace instead.
