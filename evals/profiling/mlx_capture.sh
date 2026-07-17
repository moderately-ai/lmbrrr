#!/bin/bash
# Two-phase clean gpucapture of an MLX kernel (metal_notes.md §2 external flow).
# Phase 1 (prep) runs UNCAPTURED; phase 2 (kernel-only) runs under gpucapture, so the
# gpudebug timeline-global counters describe the kernel and not RNG/quantize setup.
#
# PREREQUISITE (one-time, no sudo): the Python interpreter must be DEBUGGABLE, or
# gpucapture rejects it ("invalid PID ... must be debuggable"). uv's cpython is
# adhoc/linker-signed with no hardened runtime, so re-signing is clean:
#   P=~/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/bin/python3.12
#   codesign -f -s - --entitlements evals/profiling/get-task-allow.plist "$P"
# See README.md. Runs on the M3 (needs macOS 27 gpucapture + MLX + a debuggable python).
#
# Usage: PY=<debuggable-python> ./mlx_capture.sh
set -u
PY=${PY:-/Users/tsanterre/lmbrrr-work/lmbrrr/.venv/bin/python}
OUT=${OUT:-/tmp/mlx_qmm_clean.gputrace}
HERE="$(cd "$(dirname "$0")" && pwd)"
rm -rf "$OUT"

echo "=== phase 1: prep (uncaptured) ==="
"$PY" "$HERE/mlx_prep.py" || exit 1

echo "=== phase 2: capture kernel-only ==="
export MTL_CAPTURE_ENABLED=1 MTLCAPTURE_WAIT_FOR_SIGNAL=1
"$PY" "$HERE/mlx_qmm_only.py" > /tmp/mlx_qmm_only.out 2>&1 &
PID=$!
echo "target pid=$PID (paused at MTLDevice creation until gpucapture attaches)"
for i in $(seq 1 60); do
  if gpucapture list 2>/dev/null | grep -q " $PID "; then echo "registered after ${i} polls"; break; fi
  /bin/sleep 0.5
done
gpucapture start --pid "$PID" --until-exit --output "$OUT"   # blocks until target exits
wait "$PID" 2>/dev/null
echo "=== target stdout ==="; cat /tmp/mlx_qmm_only.out
echo "=== trace ==="; ls "$OUT" 2>&1 | head
echo
echo "Next (metal_notes §5): open + profile (do NOT use --embed; it deadlocks a large trace):"
echo "  gpudebug --json -t $OUT -o /tmp/mlxprof -c status   # wait replayer.state=ready"
echo "  gpudebug --json -s <N> -c 'profile run --gpu-state default --exec overlapping'"
echo "  gpudebug --json -s <N> -c 'go performance/timeline/counters/f32'   # etc"
