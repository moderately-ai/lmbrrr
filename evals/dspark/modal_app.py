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


@app.function(image=image, gpu="H100", volumes=VOLUMES, secrets=[hf_secret], timeout=8 * 3600)
def regenerate(
    num_samples: int = 500,
    max_new_tokens: int = 1024,
    batch_size: int = 128,
    input_name: str = "perfectblend_train.jsonl",
    output_name: str = "regen-smoke.jsonl",
    skip_samples: int = 0,
    model: str = TARGET_MODEL,
) -> None:
    """Replace assistant turns with target-model generations (non-thinking).

    Defaults from the 2026-07-11 rightsizing sweep: batch 128 + length-sorted
    admission = 3.65x the old batch-16 config (192 OOMs on prefill logits at
    this vocab). Pass model=/vol/models/minicpm-v46-fakequant-q4kft for
    deployment-config traces."""
    monitor = GpuMonitor(tag=f"regen-b{batch_size}")
    monitor.start()
    _run(
        [
            "python",
            "/lmbrrr-dspark/regenerate_answers.py",
            "--sort-by-length",
            "--model",
            model,
            "--input",
            f"/vol/data/{input_name}",
            "--output",
            f"/vol/data/{output_name}",
            "--num-samples",
            str(num_samples),
            "--skip-samples",
            str(skip_samples),
            "--max-new-tokens",
            str(max_new_tokens),
            "--batch-size",
            str(batch_size),
        ],
        env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    print("REGEN_GPU", json.dumps(monitor.stop()), flush=True)
    volume.commit()


@app.function(image=image, volumes=VOLUMES, timeout=10 * 3600)
def regenerate_sharded(
    total_samples: int = 20000,
    shards: int = 8,
    max_new_tokens: int = 1024,
    batch_size: int = 32,
    input_name: str = "perfectblend_train.jsonl",
    output_prefix: str = "regen-round1",
) -> None:
    """Fan regeneration across parallel GPU containers, then merge shards."""
    per_shard = (total_samples + shards - 1) // shards
    calls = [
        {
            "num_samples": per_shard,
            "skip_samples": shard * per_shard,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "input_name": input_name,
            "output_name": f"{output_prefix}-shard{shard:02d}.jsonl",
        }
        for shard in range(shards)
    ]
    handles = [regenerate.spawn(**call) for call in calls]
    for handle in handles:
        handle.get()
    volume.reload()
    merged = f"/vol/data/{output_prefix}.jsonl"
    with open(merged, "w", encoding="utf-8") as out_handle:
        for shard in range(shards):
            shard_path = f"/vol/data/{output_prefix}-shard{shard:02d}.jsonl"
            with open(shard_path, "r", encoding="utf-8") as in_handle:
                for line in in_handle:
                    out_handle.write(line)
    print(f"merged {shards} shards -> {merged}", flush=True)
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


@app.function(image=image, gpu="H100:4", volumes=VOLUMES, secrets=[hf_secret], timeout=6 * 3600)
def prepare_cache(
    train_data: str = "data/regen-smoke.jsonl",
    cache_name: str = "target-cache-smoke",
    local_batch_size: int = 8,
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
            TRAIN_CONFIG,
            "--train-data-path",
            f"/vol/{train_data}",
            "--output-dir",
            build_dir,
            "--local-batch-size",
            str(local_batch_size),
            "--num-workers",
            "2",
        ],
        cwd="/deepspec",
    )
    print(f"copying cache {build_dir} -> {output_dir}", flush=True)
    shutil.copytree(build_dir, output_dir, dirs_exist_ok=False)
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
) -> None:
    """Train the drafter on a prepared cache; checkpoints land in /vol/runs.

    Rightsizing (2026-07-11): volume random reads starve the GPU (180 s/step
    at gb=64); staging the cache to container NVMe (694 MB/s copy) plus
    micro-batching gives ~5 s/step. lbs=4 peaked at 76.7/79.7 GiB, so 2 is
    the safe default for 4k-token tails."""
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
    ]
    if global_batch_size is not None:
        opts.append(f"train.global_batch_size={global_batch_size}")
    if num_train_epochs is not None:
        opts.append(f"train.num_train_epochs={num_train_epochs}")
    if logging_steps is not None:
        opts.append(f"logging.logging_steps={logging_steps}")
    opts.append(f"train.torch_compile={'true' if torch_compile else 'false'}")
    cmd = ["python", "/deepspec/train.py", "--config", TRAIN_CONFIG]
    for opt in opts:
        cmd.extend(["--opts", opt])
    env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    if exp_name is not None:
        # exp_name is a top-level config key; --opts requires existing keys, so
        # pass through the config module contract instead of inventing one.
        cmd.extend(["--opts", f"exp_name={exp_name}"])
    _run(cmd, cwd="/deepspec", env=env)
    volume.commit()


@app.function(image=image, gpu="H100", volumes=VOLUMES, secrets=[hf_secret], timeout=4 * 3600)
def evaluate(
    checkpoint: str = "runs/checkpoints/lmbrrr/dspark_block8_minicpm_v46_round1/step_380",
    tasks: str = "gsm8k:20,mt-bench:10",
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    confidence_threshold: float = 0.0,
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
        TARGET_MODEL,
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
sglang_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("sglang[all]==0.5.7", "openai==2.6.1")
    .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false"})
)


@app.function(image=sglang_image, gpu="H100", volumes=VOLUMES, secrets=[hf_secret], timeout=2 * 3600)
def sglang_probe(
    model_path: str = "/vol/models/minicpm-v46-fakequant-q4kft",
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
