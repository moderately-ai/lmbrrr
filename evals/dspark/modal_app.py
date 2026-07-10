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
def download_prompts(sample_size: int = 2000, test_size: float = 0.05) -> None:
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
            "/vol/data/perfectblend_train.jsonl",
            "--test-output-dir",
            "/vol/data/eval_datasets",
            "--skip-existing",
        ]
    )
    volume.commit()


@app.function(image=image, gpu="H100", volumes=VOLUMES, secrets=[hf_secret], timeout=6 * 3600)
def regenerate(
    num_samples: int = 500,
    max_new_tokens: int = 1024,
    batch_size: int = 16,
    input_name: str = "perfectblend_train.jsonl",
    output_name: str = "regen-smoke.jsonl",
) -> None:
    """Replace assistant turns with target-model generations (non-thinking)."""
    _run(
        [
            "python",
            "/lmbrrr-dspark/regenerate_answers.py",
            "--model",
            TARGET_MODEL,
            "--input",
            f"/vol/data/{input_name}",
            "--output",
            f"/vol/data/{output_name}",
            "--num-samples",
            str(num_samples),
            "--max-new-tokens",
            str(max_new_tokens),
            "--batch-size",
            str(batch_size),
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


@app.function(image=image, gpu="H100", volumes=VOLUMES, secrets=[hf_secret], timeout=6 * 3600)
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


@app.function(image=image, gpu="H100", volumes=VOLUMES, secrets=[hf_secret], timeout=23 * 3600)
def train(
    cache_name: str = "target-cache-smoke",
    global_batch_size: int | None = None,
    num_train_epochs: int | None = None,
    logging_steps: int | None = 1,
    exp_name: str | None = None,
) -> None:
    """Train the drafter on a prepared cache; checkpoints land in /vol/runs."""
    opts = [f"data.target_cache_path=/vol/cache/{cache_name}"]
    if global_batch_size is not None:
        opts.append(f"train.global_batch_size={global_batch_size}")
    if num_train_epochs is not None:
        opts.append(f"train.num_train_epochs={num_train_epochs}")
    if logging_steps is not None:
        opts.append(f"logging.logging_steps={logging_steps}")
    cmd = ["python", "/deepspec/train.py", "--config", TRAIN_CONFIG]
    for opt in opts:
        cmd.extend(["--opts", opt])
    env = {}
    if exp_name is not None:
        # exp_name is a top-level config key; --opts requires existing keys, so
        # pass through the config module contract instead of inventing one.
        cmd.extend(["--opts", f"exp_name={exp_name}"])
    _run(cmd, cwd="/deepspec", env=env)
    volume.commit()
