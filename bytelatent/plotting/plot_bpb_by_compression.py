# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Plot training BPB vs. step for BLT runs at different compression ratios.

Reads metrics.jsonl files under /scratch/gsa/train/fineweb_data_optimal_143M_tok_{T1,T2,...}
and plots bpb/interval_across_gpus at every 20k training steps.

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
FIELD = "bpb/interval_across_gpus"
OUTPUT_PATH = Path("bpb_by_compression_ratio.png")


def load_metrics(path: Path) -> list[dict]:
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
            if "global_step" in row and FIELD in row:
                rows.append(row)
    return rows


def nearest_at_steps(rows: list[dict], steps: list[int]) -> list[float]:
    values = []
    for target in steps:
        closest = min(rows, key=lambda r: abs(r["global_step"] - target))
        values.append(closest[FIELD])
    return values


def main():
    fig, ax = plt.subplots(figsize=(8, 5))

    for ratio in COMPRESSION_RATIOS:
        metrics_path = TRAIN_DIR / RUN_NAME_TEMPLATE.format(ratio) / "metrics.jsonl"
        rows = load_metrics(metrics_path)
        max_step = max(r["global_step"] for r in rows)
        steps = list(range(0, max_step + 1, STEP_INTERVAL))
        bpb = nearest_at_steps(rows, steps)
        ax.plot(steps, bpb, marker="o", markersize=3, label=ratio)

    ax.set_xlabel("Training step")
    ax.set_ylabel("BPB")
    ax.set_yscale("log")
    ax.set_title("Training BPB by compression ratio")
    ax.legend(title="Compression ratio")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
