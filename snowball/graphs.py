"""
Learning graphs — visualise benchmark results with matplotlib.

Produces:
  1. Top-1 & Top-3 accuracy vs training round  (learning curve)
  2. Entropy vs training round                  (convergence)
  3. Accuracy vs number of questions asked      (efficiency)
  4. Hyper-parameter comparison                 (optimisation)

Usage:
    uv run snowball-graphs              # generate all graphs
    uv run snowball-graphs --out ./plots  # custom output directory
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from snowball.benchmark import run_benchmark, BenchmarkResult
from snowball.engine import EngineParams


def plot_learning_curve(result: BenchmarkResult, ax: plt.Axes) -> None:
    rounds = np.arange(1, len(result.top1_per_round) + 1)
    ax.plot(rounds, result.top1_per_round, label="Top-1", linewidth=2)
    ax.plot(rounds, result.top3_per_round, label="Top-3", linewidth=2, linestyle="--")
    ax.set_xlabel("Training Round")
    ax.set_ylabel("Accuracy (rolling window)")
    ax.set_title("Learning Curve: Accuracy vs Training")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)


def plot_entropy(result: BenchmarkResult, ax: plt.Axes) -> None:
    rounds = np.arange(1, len(result.entropy_per_round) + 1)
    ax.plot(rounds, result.entropy_per_round, color="tab:red", linewidth=2)
    ax.set_xlabel("Training Round")
    ax.set_ylabel("Avg Entropy (bits)")
    ax.set_title("Convergence: Posterior Entropy vs Training")
    ax.grid(True, alpha=0.3)


def plot_accuracy_vs_questions(result: BenchmarkResult, ax: plt.Axes) -> None:
    nqs = sorted(result.accuracy_vs_questions.keys())
    accs = [result.accuracy_vs_questions[n] for n in nqs]
    ax.bar(nqs, accs, width=1.5, color="tab:green", alpha=0.8)
    ax.set_xlabel("Questions Asked per Session")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Accuracy vs Questions Asked (after training)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")


def plot_hyperparam_comparison(results: dict[str, BenchmarkResult], ax: plt.Axes) -> None:
    for label, res in results.items():
        rounds = np.arange(1, len(res.top1_per_round) + 1)
        ax.plot(rounds, res.top1_per_round, label=label, linewidth=1.5)
    ax.set_xlabel("Training Round")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Hyper-parameter Comparison")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)


def generate_all_graphs(out_dir: Path, n_rounds: int = 300) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running baseline benchmark…")
    baseline = run_benchmark(n_rounds=n_rounds)
    print(f"  Baseline done: top-1={baseline.top1_per_round[-1]:.1%}")

    # --- Graph 1: Learning curve ---
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_learning_curve(baseline, ax)
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curve.png", dpi=150)
    plt.close(fig)
    print(f"  Saved learning_curve.png")

    # --- Graph 2: Entropy ---
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_entropy(baseline, ax)
    fig.tight_layout()
    fig.savefig(out_dir / "entropy_convergence.png", dpi=150)
    plt.close(fig)
    print(f"  Saved entropy_convergence.png")

    # --- Graph 3: Accuracy vs questions ---
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_accuracy_vs_questions(baseline, ax)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_vs_questions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved accuracy_vs_questions.png")

    # --- Graph 4: Hyper-parameter sweep ---
    print("Running hyper-parameter sweep…")
    configs = {
        "baseline (v=9, e=-2)": baseline,
    }

    sweep = [
        ("entropy-only (v=0, e=-4)", EngineParams(velocity_exp=0.0, entropy_exp=-4.0, lack_exp=0.0)),
        ("high-velocity (v=15, e=-1)", EngineParams(velocity_exp=15.0, entropy_exp=-1.0)),
        ("low-velocity (v=3, e=-3)", EngineParams(velocity_exp=3.0, entropy_exp=-3.0)),
        ("balanced (v=6, e=-3, l=2)", EngineParams(velocity_exp=6.0, entropy_exp=-3.0, lack_exp=2.0)),
        ("fast-train (amt=2.0)", EngineParams(train_amount=2.0)),
    ]
    for label, p in sweep:
        print(f"  Running: {label}")
        r = run_benchmark(n_rounds=n_rounds, params=p)
        configs[label] = r
        print(f"    → top-1={r.top1_per_round[-1]:.1%}")

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_hyperparam_comparison(configs, ax)
    fig.tight_layout()
    fig.savefig(out_dir / "hyperparam_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved hyperparam_comparison.png")

    # --- Summary figure (all 4 in one) ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plot_learning_curve(baseline, axes[0, 0])
    plot_entropy(baseline, axes[0, 1])
    plot_accuracy_vs_questions(baseline, axes[1, 0])
    plot_hyperparam_comparison(configs, axes[1, 1])
    fig.suptitle("Snowball-PY: Learning & Optimisation Dashboard", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "dashboard.png", dpi=150)
    plt.close(fig)
    print(f"  Saved dashboard.png")

    # Print best config
    best_label = max(configs, key=lambda k: configs[k].top1_per_round[-1])
    best = configs[best_label]
    print(f"\nBest config: {best_label}")
    print(f"  Final top-1: {best.top1_per_round[-1]:.1%}")
    print(f"  Final top-3: {best.top3_per_round[-1]:.1%}")
    print(f"  Accuracy vs questions: {best.accuracy_vs_questions}")


def main():
    parser = argparse.ArgumentParser(description="Generate learning graphs")
    parser.add_argument("--out", type=str, default="./plots", help="Output directory")
    parser.add_argument("--rounds", type=int, default=300, help="Training rounds")
    args = parser.parse_args()

    generate_all_graphs(Path(args.out), n_rounds=args.rounds)


if __name__ == "__main__":
    main()
