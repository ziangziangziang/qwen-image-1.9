#!/usr/bin/env python3
"""Parse a training log and plot the loss curve.

Reads lines of the form:
    [train] step=N/MAX loss=X.XXXXXX

Produces:
  - A PNG with two series: raw loss (faint) and rolling mean (bold)
  - Edit steps (loss >= 0.5) and gen steps (loss < 0.5) plotted in
    different colours so the bimodal pattern is visible

Usage:
    python scripts/plot_loss.py [--log <path>] [--out <path>] [--window <int>]
"""
import argparse
import re
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="/scratch/training/slerp-selective-20260408/post_merge_train/train.log",
                   help="Path to train.log")
    p.add_argument("--out", default=None,
                   help="Output PNG path (default: <log_dir>/loss_curve.png)")
    p.add_argument("--window", type=int, default=100,
                   help="Rolling-mean window size (default: 100)")
    p.add_argument("--edit-only", action="store_true",
                   help="Plot only edit steps (loss >= 0.5) rolling mean")
    return p.parse_args()


def rolling_mean(values, window):
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start:i+1]) / (i - start + 1))
    return out


def main():
    args = parse_args()
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[plot] log not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else log_path.parent / "loss_curve.png"

    pattern = re.compile(r"\[train\] step=(\d+)/\d+ loss=([0-9.]+)")
    steps, losses = [], []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))

    if not steps:
        print("[plot] no step/loss lines found in log", file=sys.stderr)
        sys.exit(1)

    print(f"[plot] {len(steps)} steps parsed  max_step={steps[-1]}  "
          f"loss range=[{min(losses):.4f}, {max(losses):.4f}]")

    # Separate edit (loss >= 0.5) and gen (loss < 0.5) steps
    edit_steps = [(s, l) for s, l in zip(steps, losses) if l >= 0.5]
    gen_steps  = [(s, l) for s, l in zip(steps, losses) if l < 0.5]

    # Print bucketed edit-loss averages (diagnostic table)
    if edit_steps:
        bucket_size = max(steps) // 8 or 1
        buckets: dict[int, list] = {}
        for s, l in edit_steps:
            if l < 15:  # exclude extreme spikes from averages
                b = s // bucket_size
                buckets.setdefault(b, []).append(l)
        print("\n  Step range        avg edit loss   samples")
        print("  ─────────────────────────────────────────")
        for b in sorted(buckets):
            vals = buckets[b]
            lo, hi = b * bucket_size, (b + 1) * bucket_size
            print(f"  {lo:>6}–{hi:<6}    {sum(vals)/len(vals):.4f}          {len(vals)}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[plot] matplotlib not available — skipping PNG output")
        return

    # Clip extreme spikes for display (keep in raw but cap y-axis)
    cap = sorted(losses)[int(len(losses) * 0.97)]  # 97th percentile cap
    display_losses = [min(l, cap) for l in losses]

    rm_all   = rolling_mean(display_losses, args.window)
    edit_rm  = rolling_mean([min(l, cap) for _, l in edit_steps], args.window) if edit_steps else []

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})
    ax, ax2 = axes

    # Upper panel: raw + rolling mean
    ax.scatter([s for s, l in gen_steps],  [min(l, cap) for _, l in gen_steps],
               s=4, alpha=0.25, color="#4fc3f7", label="Gen steps (raw)")
    ax.scatter([s for s, l in edit_steps], [min(l, cap) for _, l in edit_steps],
               s=4, alpha=0.25, color="#ef9a9a", label="Edit steps (raw)")
    ax.plot(steps, rm_all, lw=1.8, color="#29b6f6", alpha=0.7, label=f"All rolling mean (w={args.window})")
    if edit_steps:
        ax.plot([s for s, _ in edit_steps], edit_rm, lw=2.2, color="#e53935",
                label=f"Edit rolling mean (w={args.window})")

    # Annotate best edit region
    if edit_steps:
        best_idx  = edit_rm.index(min(edit_rm))
        best_step = edit_steps[best_idx][0]
        best_val  = min(edit_rm)
        ax.axvline(best_step, color="#ffb300", lw=1.5, ls="--", alpha=0.8)
        ax.annotate(f"Best edit\nstep~{best_step}\n{best_val:.3f}",
                    xy=(best_step, best_val), xytext=(best_step + max(steps)*0.03, best_val + cap*0.1),
                    fontsize=8, color="#ffb300",
                    arrowprops=dict(arrowstyle="->", color="#ffb300", lw=1))

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve — Qwen-Image 1.9 LoRA (MagicBrush)", fontsize=13)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, cap * 1.1)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, alpha=0.2)

    # Lower panel: edit rolling mean only (zoomed)
    if edit_steps:
        ax2.plot([s for s, _ in edit_steps], edit_rm, lw=2, color="#e53935")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Edit loss\n(rolling mean)")
        ax2.set_ylim(0.8, max(edit_rm[:len(edit_rm)//4 + 1]) * 1.1 if edit_rm else 3)
        ax2.grid(True, alpha=0.2)
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    print(f"[plot] saved → {out_path}")


if __name__ == "__main__":
    main()
