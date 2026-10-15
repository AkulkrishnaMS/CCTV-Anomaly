"""
Parse MNAD's Train.py log.txt into a clean per-epoch CSV + loss curve plot.

Train.py already logs every epoch's loss inside log.txt (Epoch / Loss:
Prediction / Compactness / Separateness) — this script just extracts that
into a structured, plottable form. It doesn't touch or re-run training.

Usage:
    python parse_training_log.py

Run this any time — even while training is still in progress, to check
progress so far — or after it finishes for the final curve.
"""

import re
import csv
from pathlib import Path

import matplotlib.pyplot as plt

# ---- Edit this to match your actual log path ----
LOG_PATH = Path("exp/household/pred/household_run60/log.txt")
OUTPUT_CSV = Path("household_run60_epoch_log.csv")
OUTPUT_PLOT = Path("household_run60_loss_curve.png")
# ---------------------------------------------------

EPOCH_PATTERN = re.compile(
    r"Epoch:\s*(\d+)\s*\n\s*"
    r"Loss:\s*Prediction\s+([\d.]+)/\s*Compactness\s+([\d.]+)/\s*Separateness\s+([\d.]+)"
)


def parse_log(log_path: Path) -> list[tuple[int, float, float, float]]:
    if not log_path.exists():
        raise FileNotFoundError(f"log.txt not found at {log_path} — check the path/exp_dir name.")

    text = log_path.read_text(errors="ignore")
    matches = EPOCH_PATTERN.findall(text)
    if not matches:
        raise ValueError("No epoch entries found — log format may differ from what's expected.")

    return [
        (int(epoch), float(pred), float(compact), float(sep))
        for epoch, pred, compact, sep in matches
    ]


def save_csv(rows: list[tuple[int, float, float, float]], out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss_prediction", "compactness", "separateness"])
        writer.writerows(rows)
    print(f"Saved CSV -> {out_path}  ({len(rows)} epochs)")


def save_plot(rows: list[tuple[int, float, float, float]], out_path: Path) -> None:
    epochs = [r[0] for r in rows]
    pred_loss = [r[1] for r in rows]
    compact_loss = [r[2] for r in rows]
    sep_loss = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, pred_loss, label="Prediction loss", color="crimson")
    ax.plot(epochs, compact_loss, label="Compactness loss", color="steelblue")
    ax.plot(epochs, sep_loss, label="Separateness loss", color="seagreen")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("MNAD Training Loss — Household Dataset")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot -> {out_path}")


def main() -> None:
    rows = parse_log(LOG_PATH)
    save_csv(rows, OUTPUT_CSV)
    save_plot(rows, OUTPUT_PLOT)

    print(f"\nLatest epoch logged: {rows[-1][0]}")
    print(f"Latest prediction loss: {rows[-1][1]:.6f}")
    if len(rows) >= 10:
        first_10_avg = sum(r[1] for r in rows[:10]) / 10
        last_10_avg = sum(r[1] for r in rows[-10:]) / 10
        print(f"Avg prediction loss, first 10 epochs: {first_10_avg:.6f}")
        print(f"Avg prediction loss, last 10 epochs:  {last_10_avg:.6f}")
        if last_10_avg < first_10_avg * 0.5:
            print("-> Clear improvement across training.")
        elif abs(last_10_avg - first_10_avg) / first_10_avg < 0.05:
            print("-> Loss has mostly plateaued in the later epochs.")


if __name__ == "__main__":
    main()
