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
actually seen. Cumulative FLOPs (from speed/FLOPS * speed/curr_iter_time,
integrated over steps) normalizes by compute spent instead.

Saves three figures to results/:
  - bpb_vs_step.pdf   (x-axis: training step)
  - bpb_vs_bytes.pdf  (x-axis: cumulative bytes consumed)
  - bpb_vs_flops.pdf  (x-axis: cumulative FLOPs)

Usage:
    python bytelatent/plotting/plot_bpb_by_compression.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

TRAIN_DIR = Path("/scratch/gsa/train")
RUN_NAME_TEMPLATE = "fineweb_data_optimal_143M_tok_{}"
COMPRESSION_RATIOS = ["T1", "T2", "T4", "T6", "T8", "T12", "T18"]
STEP_INTERVAL = 20_000
TRAIN_FIELD = "bpb/interval_across_gpus"
BYTES_FIELD = "n_bytes/interval_across_gpus"
FLOPS_RATE_FIELD = "speed/FLOPS"
ITER_TIME_FIELD = "speed/curr_iter_time"
EVAL_TASK = "flores_plus_eng_Latn"
EVAL_FIELD = "bits_per_byte,none"
RESULTS_DIR = Path("results")


def load_metrics(path: Path) -> list[dict]:
    """Load metrics rows, sorted by step, each annotated with running
    cumulative totals under "_cum_bytes" and "_cum_flops"."""
    required = {TRAIN_FIELD, BYTES_FIELD, FLOPS_RATE_FIELD, ITER_TIME_FIELD}
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
    cum_flops = 0.0
    prev_step = 0
    for row in rows:
        cum_bytes += row[BYTES_FIELD]
        # FLOPS is an instantaneous rate (flops/sec); multiplying by the
        # per-step time and the number of steps since the last log gives the
        # FLOPs spent over that interval.
        step_delta = row["global_step"] - prev_step
        cum_flops += row[FLOPS_RATE_FIELD] * row[ITER_TIME_FIELD] * step_delta
        prev_step = row["global_step"]
        row["_cum_bytes"] = cum_bytes
        row["_cum_flops"] = cum_flops
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

    train_rows = [nearest_row(rows, s) for s in checkpoint_steps]
    train_steps = [r["global_step"] for r in train_rows]
    train_bytes = [r["_cum_bytes"] for r in train_rows]
    train_flops = [r["_cum_flops"] for r in train_rows]
    train_bpb = [r[TRAIN_FIELD] for r in train_rows]

    eval_bpb_by_step = load_eval_bpb_by_step(run_dir / "evals")
    eval_steps = [s for s in checkpoint_steps if s in eval_bpb_by_step]
    eval_bpb = [eval_bpb_by_step[s] for s in eval_steps]
    eval_rows = [nearest_row(rows, s) for s in eval_steps]
    eval_bytes = [r["_cum_bytes"] for r in eval_rows]
    eval_flops = [r["_cum_flops"] for r in eval_rows]

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
