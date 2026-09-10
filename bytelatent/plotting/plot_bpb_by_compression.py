# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Plot training BPB and FLORES-English eval BPB vs. step (and vs. bytes
consumed) for BLT runs at different compression ratios.

Reads, for each run under /scratch/gsa/train/fineweb_data_optimal_143M_tok_{T1,T2,...}:
  - metrics.jsonl for bpb/interval_across_gpus (training BPB) and
    n_bytes/interval_across_gpus (bytes processed since the last logged row,
    summed into a running total to get cumulative bytes consumed), sampled
    every 20k steps
  - evals/<step>/results.json for the flores_plus_eng_Latn bits_per_byte (eval BPB),
    one folder per evaluated checkpoint

Different compression ratios pack a different number of raw bytes into the
same token/step budget, so plotting against cumulative bytes consumed (rather
than step) puts runs on equal footing in terms of how much raw text they've
actually seen.

Cumulative FLOPs uses a BLT-aware estimate (bytelatent/plotting/blt_flops.py)
rather than the "speed/FLOPS" field logged in metrics.jsonl: that logged value
applies a generic dense-transformer formula to the whole model using the raw
byte seq_len, which doesn't account for the global/latent transformer running
on n_bytes/patch_size patches instead of n_bytes -- so it barely moves across
compression ratios even though the global transformer's real per-step cost
should shrink roughly linearly with the patch size. See blt_flops.py's module
docstring for the full explanation. Since patch size is static within a run,
FLOPs/step is constant, so cumulative FLOPs is just flops_per_step * step.

Saves three figures to results/:
  - bpb_vs_step.pdf   (x-axis: training step)
  - bpb_vs_bytes.pdf  (x-axis: cumulative bytes consumed)
  - bpb_vs_flops.pdf  (x-axis: cumulative FLOPs, BLT-aware estimate)

Usage:
python bytelatent/plotting/plot_bpb_by_compression.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

from bytelatent.plotting.blt_flops import (
    build_meta_model,
    corrected_flops_per_step,
    load_model_args,
)

TRAIN_DIR = Path("/scratch/gsa/train")
RUN_NAME_TEMPLATE = "fineweb_data_optimal_143M_tok_{}"
COMPRESSION_RATIOS = ["T1", "T2", "T4", "T6", "T8", "T12", "T18"]
STEP_INTERVAL = 20_000
TRAIN_FIELD = "bpb/interval_across_gpus"
BYTES_FIELD = "n_bytes/interval_across_gpus"
EVAL_TASK = "flores_plus_eng_Latn"
EVAL_FIELD = "bits_per_byte,none"
RESULTS_DIR = Path("results")

CONFIG_DIR = Path("bytelatent/configs/toklens/data_optimal_fineweb")
BASE_CONFIG = CONFIG_DIR / "data_optimal_base_fineweb.yaml"
CONFIG_TEMPLATE = "data_optimal_143M_{}.yaml"


def flops_per_step_for_ratio(ratio: str, n_bytes_per_step: int) -> float:
    """BLT-aware FLOPs/step for a compression ratio's config, at the given
    (observed) raw bytes processed per step."""
    model_args = load_model_args([str(BASE_CONFIG), str(CONFIG_DIR / CONFIG_TEMPLATE.format(ratio))])
    model = build_meta_model(model_args)
    return corrected_flops_per_step(model_args, model, n_bytes_per_step)["flops_total"]


def load_metrics(path: Path) -> list[dict]:
    """Load metrics rows, sorted by step, each annotated with a running
    cumulative byte count under the "_cum_bytes" key."""
    required = {TRAIN_FIELD, BYTES_FIELD}
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "global_step" in row and required.issubset(row):
                rows.append(row)
    rows.sort(key=lambda r: r["global_step"])

    cum_bytes = 0.0
    for row in rows:
        cum_bytes += row[BYTES_FIELD]
        row["_cum_bytes"] = cum_bytes
    return rows


def nearest_row(rows: list[dict], target_step: int) -> dict:
    return min(rows, key=lambda r: abs(r["global_step"] - target_step))


def load_eval_bpb_by_step(evals_dir: Path) -> dict[int, float]:
    bpb_by_step = {}
    if not evals_dir.is_dir():
        return bpb_by_step
    for step_dir in sorted(evals_dir.iterdir()):
        results_path = step_dir / "results.json"
        if not results_path.is_file():
            continue
        try:
            step = int(step_dir.name)
        except ValueError:
            continue
        with open(results_path) as f:
            results = json.load(f)
        bpb = (
            results.get("tasks", {})
            .get("results", {})
            .get(EVAL_TASK, {})
            .get(EVAL_FIELD)
        )
        if bpb is not None:
            bpb_by_step[step] = bpb
    return bpb_by_step


def collect_run_data(ratio: str) -> dict:
    run_dir = TRAIN_DIR / RUN_NAME_TEMPLATE.format(ratio)
    rows = load_metrics(run_dir / "metrics.jsonl")
    max_step = max(r["global_step"] for r in rows)
    checkpoint_steps = list(range(0, max_step + 1, STEP_INTERVAL))

    # Bytes/step is constant within a run (static patching, fixed batch/seq_len),
    # so derive it from the observed data rather than re-deriving batch_size *
    # seq_len * world_size from config, and use it to get an exact FLOPs/step.
    bytes_per_step = rows[-1]["_cum_bytes"] / rows[-1]["global_step"]
    flops_per_step = flops_per_step_for_ratio(ratio, round(bytes_per_step))

    train_rows = [nearest_row(rows, s) for s in checkpoint_steps]
    train_steps = [r["global_step"] for r in train_rows]
    train_bytes = [r["_cum_bytes"] for r in train_rows]
    train_flops = [s * flops_per_step for s in train_steps]
    train_bpb = [r[TRAIN_FIELD] for r in train_rows]

    eval_bpb_by_step = load_eval_bpb_by_step(run_dir / "evals")
    eval_steps = [s for s in checkpoint_steps if s in eval_bpb_by_step]
    eval_bpb = [eval_bpb_by_step[s] for s in eval_steps]
    eval_rows = [nearest_row(rows, s) for s in eval_steps]
    eval_bytes = [r["_cum_bytes"] for r in eval_rows]
    eval_flops = [s * flops_per_step for s in eval_steps]

    return {
        "train_steps": train_steps,
        "train_bytes": train_bytes,
        "train_flops": train_flops,
        "train_bpb": train_bpb,
        "eval_steps": eval_steps,
        "eval_bytes": eval_bytes,
        "eval_flops": eval_flops,
        "eval_bpb": eval_bpb,
    }


def plot_figure(run_data: dict[str, dict], x_key: str, x_label: str, output_path: Path):
    fig, (ax_train, ax_eval) = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ratio, color in zip(COMPRESSION_RATIOS, colors):
        data = run_data[ratio]
        ax_train.plot(
            data[f"train_{x_key}"],
            data["train_bpb"],
            marker="o",
            markersize=3,
            label=ratio,
            color=color,
        )
        ax_eval.plot(
            data[f"eval_{x_key}"],
            data["eval_bpb"],
            marker="o",
            markersize=3,
            label=ratio,
            color=color,
        )

    ax_train.set_xlabel(x_label)
    ax_train.set_ylabel("BPB")
    ax_train.set_yscale("log")
    ax_train.set_title("Training BPB (FineWeb)")
    ax_train.legend(title="Compression ratio")
    ax_train.grid(True, which="both", alpha=0.3)

    ax_eval.set_xlabel(x_label)
    ax_eval.set_ylabel("BPB")
    ax_eval.set_title("Eval BPB (FLORES English)")
    ax_eval.legend(title="Compression ratio")
    ax_eval.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    run_data = {ratio: collect_run_data(ratio) for ratio in COMPRESSION_RATIOS}

    plot_figure(run_data, "steps", "Training step", RESULTS_DIR / "bpb_vs_step.pdf")
    plot_figure(
        run_data, "bytes", "Cumulative bytes consumed", RESULTS_DIR / "bpb_vs_bytes.pdf"
    )
    plot_figure(
        run_data, "flops", "Cumulative FLOPs", RESULTS_DIR / "bpb_vs_flops.pdf"
    )


if __name__ == "__main__":
    main()
