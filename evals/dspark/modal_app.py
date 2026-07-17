"""Modal app for the DSpark training lane (MiniCPM-V-4.6 target).

Pipeline stages, each a separately runnable function so every stage's output
can be inspected before spending on the next:

    uv run modal run evals/dspark/modal_app.py::smoke
    uv run modal run evals/dspark/modal_app.py::download_prompts --sample-size 2000
    uv run modal run evals/dspark/modal_app.py::regenerate --num-samples 500
    uv run modal run evals/dspark/modal_app.py::inspect_data --path data/regen-smoke.jsonl
    uv run modal run evals/dspark/modal_app.py::prepare_cache --cache-name target-cache-smoke
    uv run modal run evals/dspark/modal_app.py::train --cache-name target-cache-smoke

State lives on the `lmbrrr-dspark` volume (mounted at /vol):
    /vol/hf                 HF hub cache (models, datasets)
    /vol/data               prompt + regenerated JSONL
    /vol/cache/<name>       target caches (DeepSpec v2 format)
    /vol/runs               checkpoints + tensorboard (via DSPARK_OUTPUT_ROOT)

Secrets: `huggingface` (HF_TOKEN) — created via `modal secret create
huggingface --from-dotenv ...`; never bake tokens into images or code.
"""

from __future__ import annotations

import json
import os
import subprocess

import modal

DEEPSPEC_LOCAL_PATH = os.path.expanduser(
    "~/workspace/github.com/deepseek-ai/DeepSpec"
)
LMBRRR_DSPARK_LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))

TARGET_MODEL = "openbmb/MiniCPM-V-4.6"
TRAIN_CONFIG = "/deepspec/config/dspark/dspark_minicpm_v46.py"

app = modal.App("lmbrrr-dspark")

volume = modal.Volume.from_name("lmbrrr-dspark", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

# Pinned to DeepSpec's requirements.txt; flash-linear-attention enables the
# fast triton DeltaNet path for target forwards (causal-conv1d needs nvcc at
# build time and only speeds the small conv — skipped for now, torch fallback
# is fine).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.9.1",
        "transformers==5.10.2",
        "numpy==2.4.4",
        "PyYAML==6.0.3",
        "tqdm==4.67.3",
        "tensorboard==2.20.0",
        "matplotlib==3.10.9",
        "triton==3.5.1",
        "typing_extensions==4.15.0",
        "sentencepiece==0.2.1",
        "safetensors==0.7.0",
        "prettytable==3.17.0",
        "datasets==4.8.5",
        "openai==2.6.1",
        "flash-linear-attention",
        "gguf==0.17.1",
    )
    .env(
        {
            "PYTHONPATH": "/deepspec",
            "HF_HOME": "/vol/hf",
            "TOKENIZERS_PARALLELISM": "false",
            "DSPARK_OUTPUT_ROOT": "/vol/runs",
        }
    )
    .add_local_dir(DEEPSPEC_LOCAL_PATH, remote_path="/deepspec")
    .add_local_dir(LMBRRR_DSPARK_LOCAL_PATH, remote_path="/lmbrrr-dspark")
)

VOLUMES = {"/vol": volume}


def _run(cmd: list[str], cwd: str | None = None, env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    subprocess.run(cmd, cwd=cwd, env=merged_env, check=True)


class GpuMonitor:
    """Background nvidia-smi sampler: prints live load/memory lines every
    `interval` seconds (so logs show headroom over time) and returns summary
    stats on stop(). Cheap enough to leave on for every GPU stage."""

    def __init__(self, tag: str = "gpu", interval: float = 10.0):
        import threading

        self.tag = tag
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.utils: list[float] = []
        self.mems: list[float] = []
        self.mem_total: float = float("nan")

    def _sample(self):
        text = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        util, used, total = (float(v) for v in text.splitlines()[0].split(","))
        return util, used, total

    def _loop(self):
        import time

        while not self._stop.is_set():
            try:
                util, used, total = self._sample()
                self.utils.append(util)
                self.mems.append(used)
                self.mem_total = total
                print(
                    f"[gpu:{self.tag}] util {util:5.1f}%  mem {used/1024:6.2f}/"
                    f"{total/1024:.2f} GiB  headroom {(total-used)/1024:6.2f} GiB",
                    flush=True,
                )
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=self.interval + 5)
        if not self.utils:
            return {}
        return {
            "mean_gpu_util_pct": round(sum(self.utils) / len(self.utils), 1),
            "max_gpu_util_pct": round(max(self.utils), 1),
            "peak_mem_gib": round(max(self.mems) / 1024, 2),
            "mem_total_gib": round(self.mem_total / 1024, 2),
            "min_headroom_gib": round((self.mem_total - max(self.mems)) / 1024, 2),
        }


# head_dim 256 flex-attention kernels need >100KB shared memory per SM, which
# rules out L4/Ada for training; H100 matches the training environment.
@app.function(image=image, gpu="H100", volumes=VOLUMES, secrets=[hf_secret], timeout=1800)
def smoke() -> None:
    """Synthetic forward+backward in the pinned env — the gate CPU could not run."""
    _run(
        [
            "python",
            "/deepspec/scripts/smoke_minicpm_dspark.py",
            "--config-path",
            TARGET_MODEL,
            "--device",
            "cuda",
        ]
    )
    volume.commit()


@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret], timeout=3600)
def download_prompts(
    sample_size: int = 2000,
    test_size: float = 0.05,
    train_name: str = "perfectblend_train.jsonl",
    test_name: str = "perfectblend.jsonl",
) -> None:
    """Open-PerfectBlend prompts -> /vol/data via DeepSpec's own splitter."""
    os.makedirs("/vol/data", exist_ok=True)
    _run(
        [
            "python",
            "/deepspec/scripts/data/download_and_split.py",
            "--sample-size",
            str(sample_size),
            "--test-size",
            str(test_size),
            "--train-output-path",
            f"/vol/data/{train_name}",
            "--test-output-dir",
            "/vol/data/eval_datasets",
            "--test-output-name",
            test_name,
            "--skip-existing",
        ]
    )
    volume.commit()


@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret], timeout=3 * 3600)
def build_blend(
    output_name: str = "blend-v1-500k.jsonl",
    scale: float = 1.0,
    seed: int = 42,
    per_source_cap: int = 0,
) -> None:
    """Build the task-balanced PROMPT corpus (build_blend.py) on the volume:
    streams weak-class sources (WMT19/opus translation, XSum/BillSum summ,
    NQ-open qa, SQuAD rag, WildChat/smoltalk writing) + downsampled math/code,
    decontaminates each prompt against the Spec-Bench eval prompts + GSM8K test,
    and writes a single ShareGPT-schema prompt JSONL. CPU-only; the target
    regenerates every answer downstream (self-distillation). scale multiplies
    every per-source count; per_source_cap>0 overrides all counts (smoke)."""
    os.makedirs("/vol/data", exist_ok=True)
    cmd = [
        "python",
        "/lmbrrr-dspark/build_blend.py",
        "--output",
        f"/vol/data/{output_name}",
        "--spec-bench",
        "/vol/data/spec_bench_question.jsonl",
        "--scale",
        str(scale),
        "--seed",
        str(seed),
    ]
    if per_source_cap:
        cmd.extend(["--per-source-cap", str(per_source_cap)])
    _run(cmd)
    volume.commit()


@app.function(image=image, gpu="H100", volumes=VOLUMES, secrets=[hf_secret], timeout=8 * 3600)
def regenerate(
    num_samples: int = 500,
    max_new_tokens: int = 1024,
    batch_size: int = 128,
    input_name: str = "perfectblend_train.jsonl",
    output_name: str = "regen-smoke.jsonl",
    skip_samples: int = 0,
    model: str = TARGET_MODEL,
    done_marker: str | None = None,
    unique: bool = False,
) -> None:
    """Replace assistant turns with target-model generations (non-thinking).

    Defaults from the 2026-07-11 rightsizing sweep: batch 128 + length-sorted
    admission = 3.65x the old batch-16 config (192 OOMs on prefill logits at
    this vocab). Pass model=/vol/models/minicpm-v46-fakequant-q4kft for
    deployment-config traces.

    Idempotency (used by regenerate_sharded; standalone calls leave both off):
    - `done_marker` set  -> skip entirely if the marker already exists (a prior
      attempt of this shard completed), and write it on success. Makes a Modal
      infra RESCHEDULE (which restarts a lost call regardless of retries=) a
      no-op instead of a full re-run.
    - `unique=True`      -> append the container task id to the output filename
      so a rescheduled duplicate writes a DIFFERENT file and can never truncate
      a sibling's already-complete output (the 2026-07-17 corruption). The
      merge dedups by conversation id.
    """
    import os
    import uuid

    if done_marker is not None:
        volume.reload()
        if os.path.exists(f"/vol/data/{done_marker}"):
            print(f"SKIP {output_name}: marker {done_marker} already present", flush=True)
            return

    out = output_name
    resume_glob = None
    if unique:
        task = os.environ.get("MODAL_TASK_ID") or uuid.uuid4().hex
        base, ext = os.path.splitext(output_name)
        out = f"{base}-{task}{ext}"
        # Prior attempts of THIS shard (base-<othertask>.jsonl) — resume from
        # them so a reschedule regenerates only the remainder. The current
        # file (base-<task>.jsonl) doesn't exist yet at the writer's resume
        # read (it opens --output after the resume scan), so it is not
        # re-read here.
        resume_glob = f"/vol/data/{base}-*{ext}"

    monitor = GpuMonitor(tag=f"regen-b{batch_size}")
    monitor.start()
    cmd = [
        "python",
        "/lmbrrr-dspark/regenerate_answers.py",
        "--sort-by-length",
        "--model",
        model,
        "--input",
        f"/vol/data/{input_name}",
        "--output",
        f"/vol/data/{out}",
        "--num-samples",
        str(num_samples),
        "--skip-samples",
        str(skip_samples),
        "--max-new-tokens",
        str(max_new_tokens),
        "--batch-size",
        str(batch_size),
    ]
    if resume_glob is not None:
        cmd += ["--resume-glob", resume_glob]
    _run(cmd, env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    print("REGEN_GPU", json.dumps(monitor.stop()), flush=True)
    # Write the completion marker only after the subprocess wrote a full file,
    # so its existence authoritatively means "this shard is done".
    if done_marker is not None:
        n = sum(1 for _ in open(f"/vol/data/{out}", "r", encoding="utf-8"))
        with open(f"/vol/data/{done_marker}", "w", encoding="utf-8") as m:
            m.write(json.dumps({"output": out, "count": n}))
    volume.commit()


@app.function(image=image, volumes=VOLUMES, timeout=10 * 3600)
def regenerate_sharded(
    total_samples: int = 20000,
    shards: int = 8,
    max_new_tokens: int = 1024,
    batch_size: int = 32,
    input_name: str = "perfectblend_train.jsonl",
    output_prefix: str = "regen-round1",
    model: str = TARGET_MODEL,
) -> None:
    """Fan regeneration across parallel GPU containers, then merge.

    Idempotent + corruption-proof (2026-07-17 duplicate-run postmortem):
    Modal restarts a lost container's call on an infra reschedule regardless
    of retries=, so a naive orchestrator that holds N handles for hours can be
    restarted and re-drive every shard — re-running completed work AND (with a
    shared output filename opened "w") truncating a sibling's finished file.

    Guards:
      * each shard writes a task-UNIQUE file, so no two attempts ever touch the
        same path (truncation structurally impossible);
      * each completed shard drops a per-index `.done` marker; this orchestrator
        spawns only shards lacking a marker, and a re-spawned shard self-skips
        if its marker exists (no 2x compute on restart);
      * the merge trusts only marked shards and dedups by conversation id, so
        duplicate files from any reschedule collapse harmlessly.

    Correctness depends only on the dedup (never on commit timing or atomic
    rename); the markers only save money by avoiding re-runs.
    """
    import glob
    import json
    import os

    per_shard = (total_samples + shards - 1) // shards

    def marker_name(nn: int) -> str:
        return f"{output_prefix}-shard{nn:02d}.done"

    volume.reload()
    pending = [
        nn for nn in range(shards)
        if not os.path.exists(f"/vol/data/{marker_name(nn)}")
    ]
    print(
        f"spawning {len(pending)}/{shards} shards "
        f"({shards - len(pending)} already complete): {pending}",
        flush=True,
    )
    handles = {
        nn: regenerate.spawn(
            num_samples=per_shard,
            skip_samples=nn * per_shard,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            input_name=input_name,
            output_name=f"{output_prefix}-shard{nn:02d}.jsonl",
            model=model,
            done_marker=marker_name(nn),
            unique=True,
        )
        for nn in pending
    }
    failures = []
    for nn, handle in handles.items():
        try:
            handle.get()
        except Exception as exc:  # a poison shard must not abort the merge
            failures.append((nn, repr(exc)))
            print(f"SHARD {nn:02d} FAILED: {exc!r}", flush=True)

    volume.reload()
    done = [nn for nn in range(shards) if os.path.exists(f"/vol/data/{marker_name(nn)}")]
    missing = [nn for nn in range(shards) if nn not in done]

    merged = f"/vol/data/{output_prefix}.jsonl"
    seen: set = set()
    total = 0
    with open(merged, "w", encoding="utf-8") as out_handle:
        for nn in done:
            for path in sorted(glob.glob(f"/vol/data/{output_prefix}-shard{nn:02d}-*.jsonl")):
                with open(path, "r", encoding="utf-8") as in_handle:
                    for line in in_handle:
                        if not line.strip():
                            continue
                        try:
                            rid = json.loads(line).get("id")
                        except Exception:
                            continue
                        if rid in seen:
                            continue
                        seen.add(rid)
                        out_handle.write(line)
                        total += 1
    print(
        f"merged {len(done)}/{shards} shards -> {merged} "
        f"({total} conversations); missing={missing} failures={failures}",
        flush=True,
    )
    volume.commit()


@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret], timeout=1800)
def drafter_fixture(
    checkpoint: str = "runs/checkpoints/lmbrrr/dspark_block8_minicpm_v46/step_24",
    output: str = "fixtures/drafter-parity.safetensors",
) -> None:
    """Pinned-env parity fixture for the Candle drafter port (CPU is enough)."""
    os.makedirs(os.path.dirname(f"/vol/{output}"), exist_ok=True)
    _run(
        [
            "python",
            "/lmbrrr-dspark/make_drafter_fixture.py",
            "--checkpoint",
            f"/vol/{checkpoint}",
            "--output",
            f"/vol/{output}",
        ]
    )
    volume.commit()


@app.function(image=image, volumes=VOLUMES, timeout=600)
def inspect_data(path: str = "data/regen-smoke.jsonl", samples: int = 3) -> str:
    """Print head/count of a volume JSONL so stages can be reviewed cheaply."""
    import json

    full = f"/vol/{path}"
    count = 0
    shown = []
    with open(full, "r", encoding="utf-8") as handle:
        for line in handle:
            count += 1
            if len(shown) < samples:
                shown.append(json.loads(line))
    report = json.dumps({"path": full, "rows": count, "head": shown}, ensure_ascii=False, indent=2)
    print(report, flush=True)
    return report


@app.function(image=image, gpu="H100:4", volumes=VOLUMES, secrets=[hf_secret], timeout=6 * 3600, ephemeral_disk=1024 * 1024)
def prepare_cache(
    train_data: str = "data/regen-smoke.jsonl",
    cache_name: str = "target-cache-smoke",
    local_batch_size: int = 8,
    target_model: str | None = "/vol/models/minicpm-v46-fakequant-q4kft",
    config: str = TRAIN_CONFIG,
) -> None:
    """DeepSpec target cache (hidden states via hooks) for the given JSONL.

    Builds on container-local disk and copies final artifacts to the volume
    once: the cache writer stages gigabytes through _tmp and renames them,
    which a commit-based volume reconciles painfully slowly (measured ~15+ min
    for a 3.6 GB smoke cache written directly to /vol).
    """
    import shutil

    build_dir = f"/tmp/cache-build/{cache_name}"
    output_dir = f"/vol/cache/{cache_name}"
    _run(
        [
            "python",
            "/deepspec/scripts/data/prepare_target_cache.py",
            "--config",
            config,
            "--train-data-path",
            f"/vol/{train_data}",
            "--output-dir",
            build_dir,
            "--local-batch-size",
            str(local_batch_size),
            "--num-workers",
            "2",
        ]
        + (
            # Deployment-config consistency: capture hidden states under the
            # same fake-quant weights that generated the traces.
            ["--opts", f"model.target_model_name_or_path={target_model}"]
            if target_model
            else []
        ),
        cwd="/deepspec",
    )
    print(f"copying cache {build_dir} -> {output_dir}", flush=True)
    shutil.copytree(build_dir, output_dir, dirs_exist_ok=False)
    volume.commit()


def _train_impl(
    cache_name: str,
    global_batch_size: int | None,
    num_train_epochs: int | None,
    logging_steps: int | None,
    torch_compile: bool,
    exp_name: str | None,
    local_batch_size: int,
    stage_local: bool,
    draft_init_checkpoint: str | None,
    lr: float | None,
    target_model: str,
    config: str = TRAIN_CONFIG,
    cache_path: str | None = None,
) -> None:
    """Shared trainer invocation for the H100:4 and H100:8 entrypoints.

    Rightsizing (2026-07-11): volume random reads starve the GPU (180 s/step
    at gb=64); staging the cache to container NVMe (694 MB/s copy) plus
    micro-batching gives ~5 s/step. lbs=4 peaked at 76.7/79.7 GiB, so 2 is
    the safe default for 4k-token tails.

    cache_path overrides the /vol/cache/{cache_name} + staging path for fused
    prep+train callers whose cache already lives on container NVMe."""
    if cache_path is None:
        cache_path = f"/vol/cache/{cache_name}"
        if stage_local:
            import time

            started = time.monotonic()
            _run(["cp", "-r", cache_path, "/tmp/cache-staged"])
            print(f"staged cache in {time.monotonic()-started:.0f}s", flush=True)
            cache_path = "/tmp/cache-staged"
    opts = [
        f"data.target_cache_path={cache_path}",
        f"train.local_batch_size={local_batch_size}",
        # Must match the cache's recorded target (validated at trainer init);
        # also sources the frozen embed/lm_head copy from the same weights.
        f"model.target_model_name_or_path={target_model}",
    ]
    if draft_init_checkpoint:
        # Weights-only warm start (DeepSpec db36013); fresh schedule.
        opts.append(f"model.draft_init_checkpoint={draft_init_checkpoint}")
    if lr is not None:
        opts.append(f"train.lr={lr}")
    if global_batch_size is not None:
        opts.append(f"train.global_batch_size={global_batch_size}")
    if num_train_epochs is not None:
        opts.append(f"train.num_train_epochs={num_train_epochs}")
    if logging_steps is not None:
        opts.append(f"logging.logging_steps={logging_steps}")
    opts.append(f"train.torch_compile={'true' if torch_compile else 'false'}")
    cmd = ["python", "/deepspec/train.py", "--config", config]
    for opt in opts:
        cmd.extend(["--opts", opt])
    env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    if exp_name is not None:
        # exp_name is a top-level config key; --opts requires existing keys, so
        # pass through the config module contract instead of inventing one.
        cmd.extend(["--opts", f"exp_name={exp_name}"])
    _run(cmd, cwd="/deepspec", env=env)
    volume.commit()


@app.function(image=image, gpu="H100", volumes=VOLUMES, secrets=[hf_secret], timeout=23 * 3600)
def mtp_distill(
    input_name: str = "regen-r4-400k.jsonl",
    num_samples: int = 40000,
    epochs: int = 2,
    lr: float = 5e-5,
    max_tokens: int = 1536,
    batch_size: int = 8,
    grad_accum: int = 2,
    exp_name: str = "mtp-distill-r1",
    qat_q4k: bool = False,
    init_from: str | None = None,
    span_mask: bool = False,
) -> None:
    """Align the Qwen3.5-0.8B vendor MTP head to the fakequant target
    (mtp_distill.py): (final hidden, next token) -> next-next token over the
    round-4 regen corpus. Single H100; the head is ~30M params, the teacher
    forward dominates and is cheap at 0.8B. Prints the vendor-baseline
    holdout top-1 (the position-1 acceptance proxy) before training and at
    every eval; best checkpoint lands at /vol/runs/<exp_name>/mtp.safetensors
    (vendor tensor names — drop-in for lmbrrr --drafter-mtp)."""
    monitor = GpuMonitor(tag=f"mtp-distill-{exp_name}")
    monitor.start()
    cmd = [
        "python",
        "/lmbrrr-dspark/mtp_distill.py",
        "--model",
        "/vol/models/minicpm-v46-fakequant-q4kft",
        "--input",
        f"/vol/data/{input_name}",
        "--output-dir",
        f"/vol/runs/{exp_name}",
        "--num-samples",
        str(num_samples),
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--max-tokens",
        str(max_tokens),
        "--batch-size",
        str(batch_size),
        "--grad-accum",
        str(grad_accum),
    ]
    if qat_q4k:
        cmd.append("--qat-q4k")
    if init_from:
        cmd.extend(["--init-from", init_from])
    if span_mask:
        cmd.append("--span-mask")
    _run(cmd, env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    print("MTP_DISTILL_GPU", json.dumps(monitor.stop()), flush=True)
    volume.commit()


@app.function(image=image, gpu="H100:4", volumes=VOLUMES, secrets=[hf_secret], timeout=23 * 3600, ephemeral_disk=1024 * 1024)
def train(
    cache_name: str = "target-cache-smoke",
    global_batch_size: int | None = None,
    num_train_epochs: int | None = None,
    logging_steps: int | None = 1,
    torch_compile: bool = True,
    exp_name: str | None = None,
    local_batch_size: int = 2,
    stage_local: bool = True,
    draft_init_checkpoint: str | None = None,
    lr: float | None = None,
    target_model: str = "/vol/models/minicpm-v46-fakequant-q4kft",
    config: str = TRAIN_CONFIG,
) -> None:
    """Train the drafter on a prepared cache (4x H100); see _train_impl."""
    _train_impl(
        cache_name,
        global_batch_size,
        num_train_epochs,
        logging_steps,
        torch_compile,
        exp_name,
        local_batch_size,
        stage_local,
        draft_init_checkpoint,
        lr,
        target_model,
        config,
    )


@app.function(image=image, gpu="H100:8", volumes=VOLUMES, secrets=[hf_secret], timeout=23 * 3600, ephemeral_disk=1024 * 1024)
def train8(
    cache_name: str = "target-cache-smoke",
    global_batch_size: int | None = None,
    num_train_epochs: int | None = None,
    logging_steps: int | None = 1,
    torch_compile: bool = True,
    exp_name: str | None = None,
    local_batch_size: int = 2,
    stage_local: bool = True,
    draft_init_checkpoint: str | None = None,
    lr: float | None = None,
    target_model: str = "/vol/models/minicpm-v46-fakequant-q4kft",
    config: str = TRAIN_CONFIG,
) -> None:
    """8x H100 variant of train (same recipe, fixed global batch: pure DDP
    speedup). First use: a 1-epoch probe on the 40k cache to measure the
    real s/step before the round-3 500k run commits to the scaling math."""
    _train_impl(
        cache_name,
        global_batch_size,
        num_train_epochs,
        logging_steps,
        torch_compile,
        exp_name,
        local_batch_size,
        stage_local,
        draft_init_checkpoint,
        lr,
        target_model,
        config,
    )


@app.function(image=image, gpu="H100:4", volumes=VOLUMES, secrets=[hf_secret], timeout=23 * 3600, ephemeral_disk=3 * 1024 * 1024)
def prep_and_train(
    train_data: str = "data/regen-bonsai-r1b.jsonl",
    cache_name: str = "target-cache-bonsai-r1b",
    config: str = "/deepspec/config/dspark/dspark_bonsai.py",
    target_model: str = "prism-ml/Ternary-Bonsai-27B-unpacked",
    num_train_epochs: int = 6,
    lr: float | None = None,
    exp_name: str = "dspark_block7_bonsai_r1",
    local_batch_size: int = 2,
    cache_batch_size: int = 8,
) -> None:
    """Fused prep+train at the 3.0 TiB ephemeral ceiling, for caches that bust
    prepare_cache's 1 TiB single-stage disk (the 38k Bonsai cache filled it at
    ~60% of the build): build the target cache on container NVMe and train
    directly from it — the volume receives only checkpoints. Spawns the
    evaluator on the final checkpoint. Same fused design as the round-3/4
    chains, parameterized for any lane."""
    final_ckpt = f"/vol/runs/checkpoints/lmbrrr/{exp_name}/step_latest"
    if os.path.exists(f"{final_ckpt}/model.safetensors"):
        print(f"{exp_name} final checkpoint exists; skipping prep+train", flush=True)
    else:
        build_dir = f"/tmp/cache-build/{cache_name}"
        _run(
            [
                "python",
                "/deepspec/scripts/data/prepare_target_cache.py",
                "--config",
                config,
                "--train-data-path",
                f"/vol/{train_data}",
                "--output-dir",
                build_dir,
                "--local-batch-size",
                str(cache_batch_size),
                "--num-workers",
                "2",
                "--opts",
                f"model.target_model_name_or_path={target_model}",
            ],
            cwd="/deepspec",
        )
        _train_impl(
            cache_name=cache_name,
            global_batch_size=None,
            num_train_epochs=num_train_epochs,
            logging_steps=1,
            torch_compile=True,
            exp_name=exp_name,
            local_batch_size=local_batch_size,
            stage_local=False,
            draft_init_checkpoint=None,
            lr=lr,
            target_model=target_model,
            config=config,
            cache_path=build_dir,
        )
    evaluate.spawn(
        checkpoint=f"runs/checkpoints/lmbrrr/{exp_name}/step_latest",
        target_model=target_model,
    )


@app.function(image=image, gpu="H100", volumes=VOLUMES, secrets=[hf_secret], timeout=4 * 3600)
def evaluate(
    checkpoint: str = "runs/checkpoints/lmbrrr/dspark_block8_minicpm_v46_round1/step_380",
    tasks: str = "gsm8k:20,mt-bench:10",
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    confidence_threshold: float = 0.0,
    target_model: str = TARGET_MODEL,
) -> None:
    """Run the DeepSpec DSpark evaluator (tau, accept_rate@k, confidence
    reliability artifacts) against a checkpoint on the volume.

    The MiniCPM evaluator re-forwards the accepted prefix per round (the
    hybrid text decoder has no croppable HF cache), so counts are kept small
    by default; artifacts land under the checkpoint's tensorboard dir.
    """
    ckpt = f"/vol/{checkpoint}"
    tb_dir = f"{ckpt}/eval-tensorboard"
    cmd = [
        "python",
        "/deepspec/eval.py",
        "--target_name_or_path",
        target_model,
        "--draft_name_or_path",
        ckpt,
        "--evaluator",
        "minicpm",
        "--tasks",
        tasks,
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--confidence-threshold",
        str(confidence_threshold),
        "--tensorboard-dir",
        tb_dir,
        "--step",
        "0",
    ]
    _run(cmd, cwd="/deepspec")
    volume.commit()


@app.function(image=image, gpu="H100", volumes=VOLUMES, secrets=[hf_secret], timeout=3 * 3600)
def regen_bench(
    samples: int = 96,
    batches: str = "16,64,128",
    max_new_tokens: int = 1024,
    input_name: str = "perfectblend_train.jsonl",
) -> None:
    """Data-gen rightsizing probe: sweep generate batch size (plain vs
    length-sorted admission) on a fixed sample slice, recording wall time,
    padded tokens/s, and mean GPU utilization per configuration."""
    import re
    import time

    rows = []
    for batch in [int(b) for b in batches.split(",")]:
        for sorted_mode in (False, True):
            monitor = GpuMonitor(tag=f"b{batch}-{'sorted' if sorted_mode else 'plain'}")
            monitor.start()
            started = time.monotonic()
            cmd = [
                "python",
                "/lmbrrr-dspark/regenerate_answers.py",
                "--model",
                TARGET_MODEL,
                "--input",
                f"/vol/data/{input_name}",
                "--output",
                f"/tmp/regen-bench-b{batch}-{'sorted' if sorted_mode else 'plain'}.jsonl",
                "--num-samples",
                str(samples),
                "--max-new-tokens",
                str(max_new_tokens),
                "--batch-size",
                str(batch),
            ]
            if sorted_mode:
                cmd.append("--sort-by-length")
            print("+", " ".join(cmd), flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            stats = monitor.stop()
            wall = time.monotonic() - started
            tail = (proc.stdout or "").strip().splitlines()
            done_line = next(
                (line for line in reversed(tail) if line.startswith("done:")), ""
            )
            tokens = 0
            match = re.search(r"~(\d+) generated tokens", done_line)
            if match:
                tokens = int(match.group(1))
            if proc.returncode != 0:
                print(proc.stdout[-2000:], flush=True)
                print(proc.stderr[-4000:], flush=True)
                raise RuntimeError(f"bench run failed (batch={batch})")
            rows.append(
                {
                    "batch": batch,
                    "sorted": sorted_mode,
                    "wall_s": round(wall, 1),
                    "padded_tokens": tokens,
                    "padded_tok_per_s": round(tokens / max(wall, 1e-9), 1),
                    **stats,
                    "done_line": done_line,
                }
            )
            print("ROW", json.dumps(rows[-1]), flush=True)
    print("SUMMARY", json.dumps(rows, indent=2), flush=True)


@app.function(image=image, gpu="H100", volumes=VOLUMES, secrets=[hf_secret], timeout=2 * 3600, ephemeral_disk=512 * 1024)
def train_probe(
    cache_name: str = "target-cache-round1",
    stage_local: bool = False,
    local_batch_size: int = 1,
    global_batch_size: int = 64,
    steps: int = 10,
    num_workers: int = 4,
    torch_compile: bool = False,
) -> None:
    """Training rightsizing probe on ONE H100: a few optimizer steps with the
    GPU monitor live, optionally staging the target cache onto container-local
    NVMe first — attributes step time between volume IO and compute, and
    tests micro-batch sizes > 1."""
    import time

    cache_path = f"/vol/cache/{cache_name}"
    if stage_local:
        started = time.monotonic()
        _run(["cp", "-r", cache_path, "/tmp/cache-staged"])
        elapsed = time.monotonic() - started
        size_bytes = int(
            subprocess.run(
                ["du", "-sb", "/tmp/cache-staged"], capture_output=True, text=True
            ).stdout.split()[0]
        )
        print(
            f"staged {size_bytes/2**30:.1f} GiB in {elapsed:.0f}s "
            f"({size_bytes/2**20/elapsed:.0f} MB/s)",
            flush=True,
        )
        cache_path = "/tmp/cache-staged"

    monitor = GpuMonitor(tag=f"train-lbs{local_batch_size}-{'nvme' if stage_local else 'vol'}")
    monitor.start()
    started = time.monotonic()
    cmd = ["python", "/deepspec/train.py", "--config", TRAIN_CONFIG]
    for opt in [
        f"data.target_cache_path={cache_path}",
        f"data.num_workers={num_workers}",
        f"train.local_batch_size={local_batch_size}",
        f"train.global_batch_size={global_batch_size}",
        f"train.max_train_steps={steps}",
        f"train.torch_compile={'true' if torch_compile else 'false'}",
        "logging.logging_steps=1",
        "exp_name=train_probe_scratch",
    ]:
        cmd.extend(["--opts", opt])
    _run(cmd, cwd="/deepspec")
    stats = monitor.stop()
    wall = time.monotonic() - started
    print(
        "PROBE",
        json.dumps(
            {
                "cache": cache_path,
                "stage_local": stage_local,
                "local_batch_size": local_batch_size,
                "global_batch_size": global_batch_size,
                "steps": steps,
                "wall_s": round(wall, 1),
                "s_per_step_incl_startup": round(wall / max(steps, 1), 2),
                **stats,
            }
        ),
        flush=True,
    )


# Fake-quant checkpoint generation moved to the Rust side (`lmbrrr
# fakequant-export`): gguf-py implements k-quant DEquantize only, while
# candle carries the deployment's own Q4_K quantizer — strictly more
# faithful. Build locally, then `modal volume put` the directory to
# /vol/models/.


# Continuous-batching probe: HF generate measured ~1.4k tok/s at 43% util on
# a 2.6 GB model whose decode bandwidth ceiling is >100k tok/s — the limiter
# is the framework. SGLang supports the Qwen3-Next-class GatedDeltaNet
# hybrid; this probe checks whether it loads the MiniCPM-V-4.6 composite
# (fakequant checkpoint) and what a concurrent chat workload sustains.
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libnuma1", "numactl")
    .uv_pip_install("vllm", "openai==2.6.1")
    .env(
        {
            "HF_HOME": "/vol/hf",
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_HOME": "/usr/local/cuda",
        }
    )
    .add_local_dir(DEEPSPEC_LOCAL_PATH, remote_path="/deepspec")
    .add_local_dir(LMBRRR_DSPARK_LOCAL_PATH, remote_path="/lmbrrr-dspark")
)


@app.function(image=vllm_image, gpu="H100", volumes=VOLUMES, secrets=[hf_secret], timeout=2 * 3600)
def vllm_probe(
    model_path: str = "/vol/models/minicpm-v46-fakequant-q4kft",
    concurrency: int = 64,
    samples: int = 192,
    max_tokens: int = 512,
) -> None:
    """vLLM fallback probe (SGLang 0.5.7 verdict: no Qwen3.5-hybrid impl and
    the transformers fallback rejects hybrids). Same load shape as
    sglang_probe."""
    import threading
    import time
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    from openai import OpenAI

    cmd = [
        "vllm",
        "serve",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        "30000",
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        "0.85",
        # Text-only serving: the qwen3_5 registration is multimodal-aware,
        # but we never send images; zero the budget so mm profiling skips.
        "--limit-mm-per-prompt",
        '{"image": 0, "video": 0}',
    ]
    print("+", " ".join(cmd), flush=True)
    server = subprocess.Popen(cmd)
    try:
        deadline = time.monotonic() + 1200
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"vllm server exited early: {server.returncode}")
            try:
                urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=5)
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise RuntimeError("vllm server never became healthy")
                time.sleep(5)
        print("server healthy", flush=True)

        prompts = []
        with open("/vol/data/perfectblend_train.jsonl", "r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                users = [m["content"] for m in row.get("conversations", []) if m.get("role") == "user"]
                if users:
                    prompts.append(users[0])
                if len(prompts) >= samples:
                    break

        client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="none")
        monitor = GpuMonitor(tag="vllm")
        monitor.start()
        completed = 0
        total_tokens = 0
        lock = threading.Lock()

        def one(prompt: str):
            nonlocal completed, total_tokens
            response = client.chat.completions.create(
                model=model_path,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            with lock:
                completed += 1
                total_tokens += response.usage.completion_tokens

        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(one, prompts))
        wall = time.monotonic() - started
        stats = monitor.stop()
        print(
            "VLLM_PROBE",
            json.dumps(
                {
                    "samples": completed,
                    "completion_tokens": total_tokens,
                    "wall_s": round(wall, 1),
                    "tok_per_s": round(total_tokens / max(wall, 1e-9), 1),
                    "concurrency": concurrency,
                    **stats,
                }
            ),
            flush=True,
        )
        sample = client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": "Explain how tides work in two sentences."}],
            max_tokens=80,
            temperature=0.0,
        )
        print("SAMPLE", sample.choices[0].message.content[:400], flush=True)
    finally:
        server.terminate()


sglang_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libnuma1", "numactl")
    .uv_pip_install("sglang[all]==0.5.7", "openai==2.6.1")
    # SGLang 0.5.7 pins an older transformers that predates qwen3_5_text;
    # force the pin we use everywhere else and let the probe judge whether
    # sglang tolerates it.
    .uv_pip_install("transformers==5.10.2")
    .env(
        {
            "HF_HOME": "/vol/hf",
            "TOKENIZERS_PARALLELISM": "false",
            # deep_gemm asserts on CUDA_HOME (needs the devel toolkit).
            "CUDA_HOME": "/usr/local/cuda",
        }
    )
)


@app.function(image=sglang_image, gpu="H100", volumes=VOLUMES, secrets=[hf_secret], timeout=2 * 3600)
def sglang_probe(
    model_path: str = "/vol/models/minicpm-v46-textonly-fakequant",
    concurrency: int = 64,
    samples: int = 192,
    max_tokens: int = 512,
    extra_args: str = "",
) -> None:
    import threading
    import time
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    from openai import OpenAI

    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        "30000",
        "--dtype",
        "bfloat16",
        "--mem-fraction-static",
        "0.85",
    ] + ([a for a in extra_args.split() if a])
    print("+", " ".join(cmd), flush=True)
    server = subprocess.Popen(cmd)
    try:
        deadline = time.monotonic() + 900
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"sglang server exited early: {server.returncode}")
            try:
                urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=5)
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise RuntimeError("sglang server never became healthy")
                time.sleep(5)
        print("server healthy", flush=True)

        prompts = []
        with open("/vol/data/perfectblend_train.jsonl", "r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                users = [m["content"] for m in row.get("conversations", []) if m.get("role") == "user"]
                if users:
                    prompts.append(users[0])
                if len(prompts) >= samples:
                    break

        client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="none")
        monitor = GpuMonitor(tag="sglang")
        monitor.start()
        completed = 0
        total_tokens = 0
        lock = threading.Lock()

        def one(prompt: str):
            nonlocal completed, total_tokens
            response = client.chat.completions.create(
                model=model_path,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            with lock:
                completed += 1
                total_tokens += response.usage.completion_tokens

        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(one, prompts))
        wall = time.monotonic() - started
        stats = monitor.stop()
        print(
            "SGLANG_PROBE",
            json.dumps(
                {
                    "samples": completed,
                    "completion_tokens": total_tokens,
                    "wall_s": round(wall, 1),
                    "tok_per_s": round(total_tokens / max(wall, 1e-9), 1),
                    "concurrency": concurrency,
                    **stats,
                }
            ),
            flush=True,
        )
        # Qualitative sanity: one greedy sample printed for coherence review.
        sample = client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": "Explain how tides work in two sentences."}],
            max_tokens=80,
            temperature=0.0,
        )
        print("SAMPLE", sample.choices[0].message.content[:400], flush=True)
    finally:
        server.terminate()


@app.function(image=vllm_image, timeout=600)
def vllm_introspect() -> None:
    """Ground truth on vLLM's qwen3_5 support: registered architectures,
    config class shape, and whether a text-only entry exists."""
    import inspect

    from vllm.model_executor.models.registry import ModelRegistry

    names = [n for n in ModelRegistry.get_supported_archs() if "3_5" in n or "3_next" in n.lower() or "Next" in n]
    print("ARCHS", json.dumps(sorted(names)), flush=True)
    import vllm.model_executor.models.qwen3_next as qn

    src = inspect.getsource(qn)
    i = src.index("No Qwen3Next layer found")
    print("ERR_CONTEXT", src[max(0, i - 1500) : i + 200], flush=True)
    j = src.index("linear_attention") if "linear_attention" in src else -1
    print("HAS_linear_attention_literal", j != -1, flush=True)
    for needle in ("layer_types", "full_attention_interval", "layers_block_type"):
        print(needle, src.count(needle), flush=True)


@app.function(image=vllm_image, gpu="H100", volumes=VOLUMES, secrets=[hf_secret], timeout=8 * 3600)
def vllm_regenerate(
    num_samples: int = 500,
    max_tokens: int = 1024,
    concurrency: int = 96,
    temperature: float = 0.0,
    model_path: str = "/vol/models/minicpm-v46-fakequant-q4kft",
    input_name: str = "perfectblend_train.jsonl",
    output_name: str = "regen-vllm-500.jsonl",
) -> None:
    """Deployment-config trace generation via vLLM's native
    MiniCPMV4_6ForConditionalGeneration (measured 6.6k tok/s vs 1.25k on the
    HF path). Drives DeepSpec's generate_train_data.py against a local
    server; greedy by default per the round-2 plan."""
    import time
    import urllib.request

    server = subprocess.Popen(
        [
            "vllm",
            "serve",
            model_path,
            "--host",
            "127.0.0.1",
            "--port",
            "30000",
            "--dtype",
            "bfloat16",
            "--gpu-memory-utilization",
            "0.85",
            "--limit-mm-per-prompt",
            '{"image": 0, "video": 0}',
        ]
    )
    try:
        # vLLM 0.24's CUDA-graph capture (50+ sizes) + FlashInfer GDN JIT push
        # first-token readiness past the old 1200s deadline on the hybrid model.
        # Fix is PATIENCE, not --enforce-eager (which disables CUDA graphs and
        # cut regen throughput ~8x: 817 vs round-4's ~6600 tok/s). Keep graphs;
        # just wait out the one-time cold start (max wait; breaks when healthy).
        deadline = time.monotonic() + 3600
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"vllm server exited early: {server.returncode}")
            try:
                urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=5)
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise RuntimeError("vllm server never became healthy")
                time.sleep(5)
        print("server healthy", flush=True)
        monitor = GpuMonitor(tag="vllm-regen", interval=30)
        monitor.start()
        _run(
            [
                "python",
                "/deepspec/scripts/data/generate_train_data.py",
                "--model",
                model_path,
                "--server-address",
                "127.0.0.1:30000",
                "--input-file-path",
                f"/vol/data/{input_name}",
                "--output-file-path",
                f"/vol/data/{output_name}",
                "--concurrency",
                str(concurrency),
                "--temperature",
                str(temperature),
                "--max-tokens",
                str(max_tokens),
                "--num-samples",
                str(num_samples),
                "--disable-thinking",
            ],
            cwd="/deepspec",
        )
        print("REGEN_GPU", json.dumps(monitor.stop()), flush=True)
    finally:
        server.terminate()
    volume.commit()


@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret], timeout=3600)
def token_frequency(
    input_name: str = "regen-r2-40k.jsonl",
    top_k: int = 65536,
    output_name: str = "frspec/assistant_ranked_ids.json",
) -> None:
    """FR-Spec vocabulary ranking per the literature review on the ticket:
    unigram frequency over ASSISTANT SPANS ONLY of the target-regenerated
    conversations (VocabTrim's calibration-source ablation: target
    generations beat raw/all text), emitting a long rank-ordered id list
    (smaller profiles are prefixes) plus the cumulative coverage curve for
    principled size selection. Chat-template/EOS/control tokens are pinned
    to the front regardless of corpus rank (turn-boundary tokens missing
    from the set would collapse tau at every end-of-turn)."""
    import json as _json
    import os
    from collections import Counter

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TARGET_MODEL, trust_remote_code=True)
    counts: Counter = Counter()
    n = 0
    with open(f"/vol/data/{input_name}") as f:
        for line in f:
            row = _json.loads(line)
            # regenerate_answers.py writes {"id", "conversations": [{"role",
            # "content"}, ...], "status"} per row; error rows carry no
            # conversations key. Only assistant spans are counted: the
            # drafter only ever proposes continuation tokens.
            texts = [
                m.get("content", "")
                for m in row.get("conversations", [])
                if m.get("role") == "assistant"
            ]
            for t in texts:
                counts.update(tok(t, add_special_tokens=False)["input_ids"])
            n += 1
            if n % 5000 == 0:
                print(f"{n} rows, {len(counts)} distinct tokens")

    total = sum(counts.values())
    # Pinned control tokens first: tokenizer specials plus every added/special
    # vocab entry (chat template, tool, vision placeholders live there).
    pinned: list[int] = []
    seen = set()
    for special in list(tok.all_special_ids) + [
        tok.convert_tokens_to_ids(t) for t in tok.get_added_vocab()
    ]:
        if isinstance(special, int) and special >= 0 and special not in seen:
            pinned.append(special)
            seen.add(special)
    ranked = pinned + [tid for tid, _ in counts.most_common() if tid not in seen]
    ranked = ranked[:top_k]

    # Cumulative occurrence coverage at candidate profile sizes.
    curve = {}
    running = 0
    checkpoints = {4096, 8192, 16384, 24576, 32768, 49152, 65536}
    for i, tid in enumerate(ranked, 1):
        running += counts.get(tid, 0)
        if i in checkpoints:
            curve[str(i)] = round(running / max(1, total), 5)

    os.makedirs("/vol/data/frspec", exist_ok=True)
    out = {
        "source": input_name,
        "counting": "assistant_spans_only",
        "rows": n,
        "distinct": len(counts),
        "total_occurrences": total,
        "pinned_control_tokens": len(pinned),
        "top_k": top_k,
        "coverage_curve": curve,
        "ids": ranked,
    }
    with open(f"/vol/data/{output_name}", "w") as f:
        _json.dump(out, f)
    print(f"wrote {output_name}: pinned {len(pinned)}, curve {curve}")
    volume.commit()


@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret], timeout=3600)
def contamination_scan(
    corpus_names: str = "perfectblend_train_600k.jsonl,regen-r2-40k.jsonl",
    ngram_words: int = 13,
) -> None:
    """Eval-set decontamination check: word-13-gram overlap between the
    training corpora and (a) the gsm8k test split, (b) the vendored
    Spec-Bench questions. Reports hit counts and example row ids; a nonzero
    gsm8k hit rate means the measured tau slope could be partly
    memorization and the corpus needs filtering before round 3."""
    import json as _json
    import re

    from datasets import load_dataset

    def norm_ngrams(text: str) -> set:
        words = re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
        return {
            " ".join(words[i : i + ngram_words])
            for i in range(len(words) - ngram_words + 1)
        }

    eval_grams: set = set()
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    for row in gsm:
        eval_grams |= norm_ngrams(row["question"])
    n_gsm = len(eval_grams)
    with open("/vol/data/spec_bench_question.jsonl") as f:
        for line in f:
            d = _json.loads(line)
            for turn in d["turns"]:
                eval_grams |= norm_ngrams(turn)
    print(f"eval n-grams: gsm8k {n_gsm}, +spec-bench -> {len(eval_grams)}")

    for name in corpus_names.split(","):
        hits = 0
        rows = 0
        examples = []
        with open(f"/vol/data/{name}") as f:
            for i, line in enumerate(f):
                row = _json.loads(line)
                texts = [
                    m.get("content", "") for m in row.get("conversations", [])
                ] or [str(row.get("prompt", "")) + " " + str(row.get("text", ""))]
                rows += 1
                if any(norm_ngrams(t) & eval_grams for t in texts if t):
                    hits += 1
                    if len(examples) < 5:
                        examples.append(i)
        print(f"{name}: rows {rows}, contaminated {hits} ({hits/max(1,rows)*100:.3f}%), first hits {examples}")


# --------------------------- round-3 detached chain ---------------------------
# Laptop-free execution: each stage is idempotent (skips finished work) and
# .spawn()s its successor, so `modal run --detach ::round3_stage0_pool` drives
# pool-filter -> regen -> cache prep -> train -> eval end to end in the cloud.
# Any failure resumes by re-running the failed stage; completed outputs skip.

R3_POOL = "perfectblend_train_120k_clean.jsonl"
R3_REGEN = "regen-r3-120k.jsonl"
R3_CACHE = "target-cache-r3-120k"
R3_EXP = "dspark_r3_fresh120k"
R3_SAMPLES = 120000


@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret], timeout=3600)
def round3_stage0_pool() -> None:
    """Filtered round-3 pool: first R3_SAMPLES rows of the 600k pool that
    share no word-13-gram with gsm8k test or the Spec-Bench questions
    (measured contamination 0.030% — filtering is hygiene, not correction)."""
    import json as _json
    import re

    from datasets import load_dataset

    out_path = f"/vol/data/{R3_POOL}"
    if os.path.exists(out_path):
        print(f"{R3_POOL} exists; skipping filter", flush=True)
    else:
        def norm_ngrams(text: str, n: int = 13) -> set:
            words = re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
            return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}

        eval_grams: set = set()
        for row in load_dataset("openai/gsm8k", "main", split="test"):
            eval_grams |= norm_ngrams(row["question"])
        with open("/vol/data/spec_bench_question.jsonl") as f:
            for line in f:
                for turn in _json.loads(line)["turns"]:
                    eval_grams |= norm_ngrams(turn)

        kept = 0
        dropped = 0
        with open("/vol/data/perfectblend_train_600k.jsonl") as src, open(
            out_path + ".tmp", "w"
        ) as dst:
            for line in src:
                if kept >= R3_SAMPLES:
                    break
                row = _json.loads(line)
                texts = [m.get("content", "") for m in row.get("conversations", [])]
                if any(norm_ngrams(t) & eval_grams for t in texts if t):
                    dropped += 1
                    continue
                dst.write(line)
                kept += 1
        os.rename(out_path + ".tmp", out_path)
        print(f"pool: kept {kept}, dropped {dropped} contaminated", flush=True)
        volume.commit()
    round3_stage1_regen.spawn()


@app.function(image=vllm_image, gpu="H100", volumes=VOLUMES, secrets=[hf_secret], timeout=8 * 3600)
def round3_stage1_regen() -> None:
    """Regenerate assistant turns for the round-3 pool (vLLM, greedy,
    deployment-config target), then spawn cache prep."""
    if os.path.exists(f"/vol/data/{R3_REGEN}"):
        print(f"{R3_REGEN} exists; skipping regen", flush=True)
    else:
        vllm_regenerate.local(
            num_samples=R3_SAMPLES,
            input_name=R3_POOL,
            output_name=R3_REGEN,
        )
    round3_stage2_prep.spawn()


@app.function(image=image, gpu="H100:4", volumes=VOLUMES, secrets=[hf_secret], timeout=20 * 3600, ephemeral_disk=1024 * 1024)
def round3_stage2_prep() -> None:
    """Target cache for the round-3 corpus (same body as prepare_cache with a
    20h window: 60k took well under 6h, 120k gets 3x headroom), then spawns
    training.

    Completion is judged by a _COMPLETE sentinel written only after the
    whole volume copy succeeds — manifest.json existence is NOT proof (a
    partial copytree can include it; the 2026-07-11 ENOSPC failure would
    have made a manifest-based skip hand the trainer a truncated cache).
    Any directory without the sentinel is a partial copy and is removed
    before rebuilding."""
    import shutil

    cache_dir = f"/vol/cache/{R3_CACHE}"
    sentinel = f"{cache_dir}/_COMPLETE"
    if os.path.exists(sentinel):
        print(f"cache {R3_CACHE} complete; skipping prep", flush=True)
    else:
        if os.path.exists(cache_dir):
            print(f"removing partial cache {cache_dir}", flush=True)
            shutil.rmtree(cache_dir)
            volume.commit()
        prepare_cache.local(
            train_data=f"data/{R3_REGEN}",
            cache_name=R3_CACHE,
        )
        with open(sentinel, "w") as f:
            f.write("ok\n")
        volume.commit()
    round3_stage3_train.spawn()


@app.function(image=image, gpu="H100:8", volumes=VOLUMES, secrets=[hf_secret], timeout=23 * 3600, ephemeral_disk=1024 * 1024)
def round3_stage3_train() -> None:
    """Fresh 120k train at the validated B recipe (lr 6e-4, lbs 2, fixed
    global batch; 10 epochs with per-epoch checkpoints for the plateau
    stop), then spawns the evaluator on the final checkpoint."""
    final_ckpt = f"/vol/runs/checkpoints/lmbrrr/{R3_EXP}/step_latest"
    if os.path.exists(f"{final_ckpt}/model.safetensors"):
        print(f"{R3_EXP} final checkpoint exists; skipping train", flush=True)
    else:
        _train_impl(
            cache_name=R3_CACHE,
            global_batch_size=None,
            num_train_epochs=10,
            logging_steps=1,
            torch_compile=True,
            exp_name=R3_EXP,
            local_batch_size=2,
            stage_local=True,
            draft_init_checkpoint=None,
            lr=6e-4,
            target_model="/vol/models/minicpm-v46-fakequant-q4kft",
        )
    evaluate.spawn(
        checkpoint=f"runs/checkpoints/lmbrrr/{R3_EXP}/step_latest",
        target_model="/vol/models/minicpm-v46-fakequant-q4kft",
    )


@app.function(image=image, gpu="H100:8", volumes=VOLUMES, secrets=[hf_secret], timeout=23 * 3600, ephemeral_disk=1536 * 1024)
def round3_prep_and_train() -> None:
    """Fused stage 2+3: build the target cache on container-local NVMe and
    train directly from it, skipping the 810 GiB volume round-trip entirely
    (the single-stream copytree measured ~100 MB/s and stalled twice; a
    rebuild is ~15 min of compute, cheaper than storing the cache). The
    volume receives only checkpoints. Spawns the evaluator when done."""
    build_dir = f"/tmp/cache-build/{R3_CACHE}"
    _run(
        [
            "python",
            "/deepspec/scripts/data/prepare_target_cache.py",
            "--config",
            TRAIN_CONFIG,
            "--train-data-path",
            f"/vol/data/{R3_REGEN}",
            "--output-dir",
            build_dir,
            "--local-batch-size",
            "8",
            "--num-workers",
            "2",
            "--opts",
            "model.target_model_name_or_path=/vol/models/minicpm-v46-fakequant-q4kft",
        ],
        cwd="/deepspec",
    )
    opts = [
        f"data.target_cache_path={build_dir}",
        "train.local_batch_size=2",
        "model.target_model_name_or_path=/vol/models/minicpm-v46-fakequant-q4kft",
        "train.lr=0.0006",
        "train.num_train_epochs=10",
        "logging.logging_steps=1",
        "train.torch_compile=true",
        f"exp_name={R3_EXP}",
    ]
    cmd = ["python", "/deepspec/train.py", "--config", TRAIN_CONFIG]
    for opt in opts:
        cmd.extend(["--opts", opt])
    _run(cmd, cwd="/deepspec", env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    volume.commit()
    evaluate.spawn(
        checkpoint=f"runs/checkpoints/lmbrrr/{R3_EXP}/step_latest",
        target_model="/vol/models/minicpm-v46-fakequant-q4kft",
    )


# --------------------------- round-4 detached chain ---------------------------
# 400k fresh: sized by Modal's measured ephemeral-disk ceiling (3.0 TiB; a
# 500k cache at ~3.3 TiB does not fit single-container). Same fused design as
# round 3 (cache on NVMe, volume gets checkpoints only); 6-epoch budget from
# round-3's epoch-7 plateau, with per-epoch checkpoints for the plateau stop.

R4_POOL = "perfectblend_train_400k_clean.jsonl"
R4_REGEN = "regen-r4-400k.jsonl"
R4_CACHE = "target-cache-r4-400k"
R4_EXP = "dspark_r4_fresh400k"
R4_SAMPLES = 400000


@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret], timeout=7200)
def round4_stage0_pool() -> None:
    """First R4_SAMPLES contamination-free rows of the 600k pool (word-13-gram
    screen vs gsm8k test + Spec-Bench), then spawns regen."""
    import json as _json
    import re

    from datasets import load_dataset

    out_path = f"/vol/data/{R4_POOL}"
    if os.path.exists(out_path):
        print(f"{R4_POOL} exists; skipping filter", flush=True)
    else:
        def norm_ngrams(text: str, n: int = 13) -> set:
            words = re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
            return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}

        eval_grams: set = set()
        for row in load_dataset("openai/gsm8k", "main", split="test"):
            eval_grams |= norm_ngrams(row["question"])
        with open("/vol/data/spec_bench_question.jsonl") as f:
            for line in f:
                for turn in _json.loads(line)["turns"]:
                    eval_grams |= norm_ngrams(turn)

        kept = 0
        dropped = 0
        with open("/vol/data/perfectblend_train_600k.jsonl") as src, open(
            out_path + ".tmp", "w"
        ) as dst:
            for line in src:
                if kept >= R4_SAMPLES:
                    break
                row = _json.loads(line)
                texts = [m.get("content", "") for m in row.get("conversations", [])]
                if any(norm_ngrams(t) & eval_grams for t in texts if t):
                    dropped += 1
                    continue
                dst.write(line)
                kept += 1
        os.rename(out_path + ".tmp", out_path)
        print(f"pool: kept {kept}, dropped {dropped} contaminated", flush=True)
        volume.commit()
    round4_stage1_regen.spawn()


@app.function(image=vllm_image, gpu="H100", volumes=VOLUMES, secrets=[hf_secret], timeout=8 * 3600)
def round4_stage1_regen() -> None:
    """400k regen (vLLM, greedy, deployment-config target; ~3.6 h at the
    measured 40k-in-21:38 pace), then spawns the fused prep+train."""
    if os.path.exists(f"/vol/data/{R4_REGEN}"):
        print(f"{R4_REGEN} exists; skipping regen", flush=True)
    else:
        vllm_regenerate.local(
            num_samples=R4_SAMPLES,
            input_name=R4_POOL,
            output_name=R4_REGEN,
        )
    round4_prep_and_train.spawn()


@app.function(image=image, gpu="H100:8", volumes=VOLUMES, secrets=[hf_secret], timeout=23 * 3600, ephemeral_disk=3 * 1024 * 1024)
def round4_prep_and_train() -> None:
    """Fused prep+train at the 3.0 TiB ephemeral ceiling: ~2.64 TiB cache on
    NVMe, 6 epochs (round-3 plateaued by 7 on a 3x smaller corpus), per-epoch
    checkpoints; spawns the evaluator on completion."""
    build_dir = f"/tmp/cache-build/{R4_CACHE}"
    _run(
        [
            "python",
            "/deepspec/scripts/data/prepare_target_cache.py",
            "--config",
            TRAIN_CONFIG,
            "--train-data-path",
            f"/vol/data/{R4_REGEN}",
            "--output-dir",
            build_dir,
            "--local-batch-size",
            "8",
            "--num-workers",
            "2",
            "--opts",
            "model.target_model_name_or_path=/vol/models/minicpm-v46-fakequant-q4kft",
        ],
        cwd="/deepspec",
    )
    opts = [
        f"data.target_cache_path={build_dir}",
        "train.local_batch_size=2",
        "model.target_model_name_or_path=/vol/models/minicpm-v46-fakequant-q4kft",
        "train.lr=0.0006",
        "train.num_train_epochs=6",
        "logging.logging_steps=1",
        "train.torch_compile=true",
        f"exp_name={R4_EXP}",
    ]
    cmd = ["python", "/deepspec/train.py", "--config", TRAIN_CONFIG]
    for opt in opts:
        cmd.extend(["--opts", opt])
    _run(cmd, cwd="/deepspec", env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    volume.commit()
    evaluate.spawn(
        checkpoint=f"runs/checkpoints/lmbrrr/{R4_EXP}/step_latest",
        target_model="/vol/models/minicpm-v46-fakequant-q4kft",
    )


@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret], timeout=7200)
def rank_tokens(
    input_name: str = "regen-r4-400k.jsonl",
    output_name: str = "frspec-assistant-ranked.json",
    keep_top: int = 65536,
) -> None:
    """FR-Spec frequency ranking: token counts over the regenerated ASSISTANT
    text (exactly the distribution the drafter must predict), special/control
    tokens pinned at the ranking front so EOS and template tokens stay
    draftable at any slice size. Emits /vol/artifacts/<output_name> with an
    "ids" array (most-frequent first, keep_top long) + count metadata."""
    from collections import Counter

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, trust_remote_code=True)
    counts: Counter = Counter()
    rows = 0
    toks = 0
    with open(f"/vol/data/{input_name}", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") != "success":
                continue
            for msg in row.get("conversations", []):
                if msg.get("role") != "assistant":
                    continue
                ids = tokenizer.encode(msg["content"], add_special_tokens=False)
                counts.update(ids)
                toks += len(ids)
            rows += 1
            if rows % 20000 == 0:
                print(f"progress: rows={rows} tokens={toks}", flush=True)

    special = [i for i in sorted(set(tokenizer.all_special_ids)) if i is not None]
    ranked = special + [
        t for t, _ in counts.most_common() if t not in set(special)
    ]
    seen = set(ranked)
    # Back-fill by id order so any slice size up to the full vocab is valid.
    vocab_size = len(tokenizer)
    ranked.extend(i for i in range(vocab_size) if i not in seen)
    ranked = ranked[:keep_top]

    coverage = sum(counts[t] for t in ranked[: 32768]) / max(toks, 1)
    os.makedirs("/vol/artifacts", exist_ok=True)
    payload = {
        "ids": ranked,
        "source": input_name,
        "rows": rows,
        "assistant_tokens": toks,
        "distinct_tokens": len(counts),
        "top32k_coverage": coverage,
        "special_pinned": len(special),
    }
    with open(f"/vol/artifacts/{output_name}", "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    volume.commit()
    print(
        f"done: rows={rows} tokens={toks} distinct={len(counts)} "
        f"top32k_coverage={coverage:.4f} -> /vol/artifacts/{output_name}",
        flush=True,
    )


@app.function(image=image, volumes=VOLUMES, timeout=600)
def inspect_checkpoint(checkpoint: str = "runs/checkpoints/lmbrrr/dspark_block7_bonsai_smoke1853/step_6") -> None:
    """List a draft checkpoint's tensor keys + config — tells the GGUF
    converter exactly which tensors the checkpoint carries (frozen
    embed/head may be omitted)."""
    import os, json
    from safetensors import safe_open

    ckpt = f"/vol/{checkpoint}"
    cfg_path = os.path.join(ckpt, "config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        print("CONFIG_KEYS", json.dumps({k: cfg[k] for k in sorted(cfg) if not isinstance(cfg[k], (list, dict))}, default=str))
        for k in ("target_layer_ids", "layer_types"):
            if k in cfg:
                print(f"CONFIG_{k}", json.dumps(cfg[k]))
    st = os.path.join(ckpt, "model.safetensors")
    with safe_open(st, framework="pt") as f:
        keys = sorted(f.keys())
        print(f"TENSOR_COUNT {len(keys)}")
        for k in keys:
            print("T", k, list(f.get_slice(k).get_shape()), f.get_slice(k).get_dtype())


@app.function(image=image, volumes=VOLUMES, timeout=1800)
def convert_gguf(
    checkpoint: str = "runs/checkpoints/lmbrrr/dspark_block7_bonsai_smoke1853/step_6",
    out_name: str | None = None,
) -> None:
    """Convert a trained draft checkpoint to a dspark GGUF for lmbrrr (bf16).
    Writes to /vol/models/<out_name>; download it, then `lmbrrr gguf requant`
    to Q8_0/Q4_1 for deployment."""
    import os

    name = out_name or (os.path.basename(checkpoint.rstrip("/")) + "-dspark-bf16.gguf")
    os.makedirs("/vol/models", exist_ok=True)
    out_path = f"/vol/models/{name}"
    _run(
        [
            "python",
            "/lmbrrr-dspark/convert_dspark_gguf.py",
            "--checkpoint",
            f"/vol/{checkpoint}",
            "--out",
            out_path,
        ]
    )
    print(f"CONVERTED -> {out_path} ({os.path.getsize(out_path)/1e9:.2f} GB)", flush=True)
    volume.commit()


@app.function(image=image, volumes=VOLUMES, timeout=600)
def inventory_shards(prefix: str = "regen-bonsai-r1b", shards: int = 12) -> None:
    """Line-count each shard file on the volume — distinguish complete from
    truncated shards after a partial/duplicated regen run."""
    import os
    total = 0
    for i in range(shards):
        p = f"/vol/data/{prefix}-shard{i:02d}.jsonl"
        n = sum(1 for _ in open(p)) if os.path.exists(p) else -1
        print(f"shard{i:02d} {n}", flush=True)
        if n > 0:
            total += n
    print(f"TOTAL {total}", flush=True)


@app.function(image=image, volumes=VOLUMES, timeout=1200)
def merge_shards(
    prefix: str = "regen-bonsai-r1b",
    shards: int = 12,
    output_name: str = "regen-bonsai-r1b.jsonl",
) -> None:
    """Concatenate existing non-empty shard files into one JSONL. Idempotent
    recovery for a sharded regen whose orchestrator didn't reach its own
    merge (e.g. a preempted/duplicated parent, stopped manually)."""
    import os

    merged = f"/vol/data/{output_name}"
    total, used = 0, []
    with open(merged, "w", encoding="utf-8") as out:
        for i in range(shards):
            p = f"/vol/data/{prefix}-shard{i:02d}.jsonl"
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                print(f"skip shard{i:02d} (missing/empty)", flush=True)
                continue
            n = 0
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line)
                        n += 1
            total += n
            used.append(i)
    print(f"merged shards {used} -> {merged}  ({total} conversations)", flush=True)
    volume.commit()
