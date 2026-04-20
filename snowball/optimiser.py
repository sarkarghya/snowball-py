"""
Hyper-parameter optimiser for Snowball-PY.

Objective
---------
Maximise a composite score that rewards:
  1. High top-1 accuracy (weight 0.6)
  2. Fast convergence — high accuracy after only 50 training rounds (weight 0.2)
  3. High accuracy with fewer questions — accuracy at 5 questions (weight 0.2)

The search uses a simple grid sweep followed by a local refinement step.

Usage:
    uv run python -m snowball.optimiser          # run full optimisation
    uv run python -m snowball.optimiser --quick   # fast mode (fewer evals)
"""

from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from snowball.benchmark import run_benchmark, BenchmarkResult
from snowball.engine import EngineParams


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def objective(result: BenchmarkResult) -> float:
    """Composite score in [0, 1]. Higher is better.

    Components:
      - final_acc:  rolling top-1 accuracy at the end             (60%)
      - early_acc:  rolling top-1 accuracy at round 50            (20%)
      - few_q_acc:  top-1 accuracy using only 5 questions         (20%)
    """
    final_acc = result.top1_per_round[-1] if result.top1_per_round else 0
    early_acc = result.top1_per_round[min(49, len(result.top1_per_round) - 1)]
    few_q_acc = result.accuracy_vs_questions.get(5, 0)

    return 0.6 * final_acc + 0.2 * early_acc + 0.2 * few_q_acc


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    params: EngineParams
    score: float
    result: BenchmarkResult


def grid_search(
    n_rounds: int = 200,
    seed: int = 42,
) -> list[SearchResult]:
    """Sweep key hyper-parameters and return all results sorted by score."""

    velocity_exps = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0]
    entropy_exps = [-1.0, -2.0, -3.0, -4.0]
    lack_exps = [0.0, 1.0, 2.0]
    train_amounts = [0.5, 1.0, 2.0]

    # Full grid is large; use a Latin-hypercube-style subset
    combos = list(itertools.product(velocity_exps, entropy_exps, lack_exps, train_amounts))
    rng = np.random.default_rng(seed)
    # Sample a manageable subset
    max_evals = min(len(combos), 40)
    indices = rng.choice(len(combos), size=max_evals, replace=False)
    selected = [combos[i] for i in sorted(indices)]

    results: list[SearchResult] = []
    print(f"Grid search: {len(selected)} configurations, {n_rounds} rounds each")

    for i, (v, e, l, ta) in enumerate(selected):
        params = EngineParams(
            velocity_exp=v,
            entropy_exp=e,
            lack_exp=l,
            train_amount=ta,
        )
        bench = run_benchmark(n_rounds=n_rounds, params=params, seed=seed)
        score = objective(bench)
        results.append(SearchResult(params=params, score=score, result=bench))
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(selected)}] best so far: {max(r.score for r in results):.3f}")

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def local_refine(
    base: EngineParams,
    n_rounds: int = 200,
    seed: int = 42,
) -> list[SearchResult]:
    """Refine around a good configuration with small perturbations."""
    results: list[SearchResult] = []
    deltas = [-0.5, 0, 0.5]

    # Perturb velocity_exp and entropy_exp around the base
    for dv in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        for de in [-0.5, 0.0, 0.5]:
            params = EngineParams(
                velocity_exp=max(0, base.velocity_exp + dv),
                entropy_exp=min(-0.1, base.entropy_exp + de),
                lack_exp=base.lack_exp,
                train_amount=base.train_amount,
            )
            bench = run_benchmark(n_rounds=n_rounds, params=params, seed=seed)
            score = objective(bench)
            results.append(SearchResult(params=params, score=score, result=bench))

    results.sort(key=lambda r: r.score, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_optimisation_results(
    grid_results: list[SearchResult],
    refined_results: list[SearchResult],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Score distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    scores = [r.score for r in grid_results]
    axes[0].hist(scores, bins=20, color="steelblue", edgecolor="white")
    axes[0].axvline(max(scores), color="red", linestyle="--", label=f"Best: {max(scores):.3f}")
    axes[0].set_xlabel("Objective Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Grid Search Score Distribution")
    axes[0].legend()

    # 2. Velocity exp vs score
    v_exps = [r.params.velocity_exp for r in grid_results]
    axes[1].scatter(v_exps, scores, alpha=0.6, c=scores, cmap="viridis")
    axes[1].set_xlabel("velocity_exp")
    axes[1].set_ylabel("Objective Score")
    axes[1].set_title("Score vs Velocity Exponent")

    # 3. Refinement comparison
    if refined_results:
        ref_scores = [r.score for r in refined_results]
        axes[2].hist(ref_scores, bins=15, color="orange", edgecolor="white", alpha=0.8, label="Refined")
        axes[2].hist(scores, bins=15, color="steelblue", edgecolor="white", alpha=0.5, label="Grid")
        axes[2].set_xlabel("Objective Score")
        axes[2].set_title("Grid vs Refined Scores")
        axes[2].legend()

    fig.suptitle("Hyper-parameter Optimisation", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "optimisation.png", dpi=150)
    plt.close(fig)
    print(f"Saved optimisation.png")

    # 4. Best config learning curve
    best = max(grid_results + refined_results, key=lambda r: r.score)
    fig, ax = plt.subplots(figsize=(10, 5))
    rounds = np.arange(1, len(best.result.top1_per_round) + 1)
    ax.plot(rounds, best.result.top1_per_round, label="Top-1", linewidth=2)
    ax.plot(rounds, best.result.top3_per_round, label="Top-3", linewidth=2, linestyle="--")
    ax.set_xlabel("Training Round")
    ax.set_ylabel("Accuracy")
    p = best.params
    ax.set_title(f"Best Config: v={p.velocity_exp}, e={p.entropy_exp}, l={p.lack_exp}, ta={p.train_amount}  (score={best.score:.3f})")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "best_config_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved best_config_curve.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optimise Snowball hyper-parameters")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer evals)")
    parser.add_argument("--out", type=str, default="./plots", help="Output directory")
    args = parser.parse_args()

    n_rounds = 100 if args.quick else 200
    out_dir = Path(args.out)

    t0 = time.perf_counter()

    # Phase 1: Grid search
    print("=== Phase 1: Grid Search ===")
    grid_results = grid_search(n_rounds=n_rounds)
    best_grid = grid_results[0]
    print(f"\nBest grid config (score={best_grid.score:.3f}):")
    print(f"  velocity_exp={best_grid.params.velocity_exp}")
    print(f"  entropy_exp={best_grid.params.entropy_exp}")
    print(f"  lack_exp={best_grid.params.lack_exp}")
    print(f"  train_amount={best_grid.params.train_amount}")

    # Phase 2: Local refinement around best
    print("\n=== Phase 2: Local Refinement ===")
    refined = local_refine(best_grid.params, n_rounds=n_rounds)
    best_refined = refined[0]
    print(f"\nBest refined config (score={best_refined.score:.3f}):")
    print(f"  velocity_exp={best_refined.params.velocity_exp}")
    print(f"  entropy_exp={best_refined.params.entropy_exp}")
    print(f"  lack_exp={best_refined.params.lack_exp}")
    print(f"  train_amount={best_refined.params.train_amount}")

    # Overall best
    all_results = grid_results + refined
    best = max(all_results, key=lambda r: r.score)
    elapsed = time.perf_counter() - t0

    print(f"\n{'='*50}")
    print(f"OVERALL BEST (score={best.score:.3f}, {elapsed:.0f}s total):")
    print(f"  velocity_exp  = {best.params.velocity_exp}")
    print(f"  entropy_exp   = {best.params.entropy_exp}")
    print(f"  lack_exp      = {best.params.lack_exp}")
    print(f"  train_amount  = {best.params.train_amount}")
    print(f"  final top-1   = {best.result.top1_per_round[-1]:.1%}")
    print(f"  final top-3   = {best.result.top3_per_round[-1]:.1%}")
    print(f"  5q accuracy   = {best.result.accuracy_vs_questions.get(5, 0):.1%}")
    print(f"{'='*50}")

    # Generate plots
    plot_optimisation_results(grid_results, refined, out_dir)


if __name__ == "__main__":
    main()
