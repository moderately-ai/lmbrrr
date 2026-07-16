# Headless Metal capture and performance analysis on macOS 27

This guide describes the new macOS 27 GPU command-line workflow for coding agents, scripts, CI experiments, and human-driven performance investigations. It incorporates an end-to-end validation on an Apple M3 Pro using a candle/Metal LLM decode engine, including a one-token GPU capture analyzed entirely without the Xcode GPU debugger UI.

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

On M3/A17-class hardware or newer, `gpudebug` can collect a new GPU replay profile:

```sh
gpudebug --json -s 412 \
  -c 'profile run --gpu-state default --exec overlapping --embed'
```

After profiling, inspect the performance root first:

```sh
gpudebug --json -s 412 \
  -c 'go performance' \
  -c 'info --all' \
  -c 'list'
```

Then explore:

```sh
gpudebug --json -s 412 -c 'go performance/encoders' -c 'list'
gpudebug --json -s 412 -c 'go performance/commands' -c 'list'
gpudebug --json -s 412 -c 'go performance/shaders' -c 'list'
gpudebug --json -s 412 -c 'go performance/timeline' -c 'list'
gpudebug --json -s 412 -c 'go performance/timeline/counters' -c 'list'
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
