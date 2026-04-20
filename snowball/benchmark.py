"""
Benchmark — runs many simulated survey sessions, measures accuracy over time,
and collects data for learning graphs.

Objective
---------
Maximise **top-1 accuracy** (correct target in the #1 position) while
minimising the **number of questions asked per session**.

We measure:
  - top-1, top-3 accuracy vs. training round
  - average entropy after N questions
  - accuracy vs. number of questions asked
  - convergence speed across hyper-parameter settings
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from snowball.data import SurveyData
from snowball.engine import PqaEngine, EngineParams
from snowball.simulator import SimulatedRespondent


@dataclass
class BenchmarkResult:
    """Collected metrics from a benchmark run."""
    top1_per_round: list[float] = field(default_factory=list)  # rolling accuracy
    top3_per_round: list[float] = field(default_factory=list)
    entropy_per_round: list[float] = field(default_factory=list)
    accuracy_vs_questions: dict[int, float] = field(default_factory=dict)
    total_time: float = 0.0
    params: EngineParams = field(default_factory=EngineParams)


def run_benchmark(
    n_rounds: int = 300,
    questions_per_session: int = 10,
    params: EngineParams | None = None,
    seed: int = 42,
    window: int = 50,
) -> BenchmarkResult:
    """Run a full benchmark and return collected metrics."""
    params = params or EngineParams()
    data = SurveyData
    rng = np.random.default_rng(seed)
    engine = PqaEngine(
        data.n_questions(), data.n_answers(), data.n_targets(),
        params=params, rng=np.random.default_rng(seed + 1),
    )

    result = BenchmarkResult(params=params)
    top1_hits: list[bool] = []
    top3_hits: list[bool] = []
    entropies: list[float] = []

    t0 = time.perf_counter()

    for r in range(n_rounds):
        target = int(rng.integers(0, data.n_targets()))
        respondent = SimulatedRespondent(target, rng=rng)
        quiz = engine.start_quiz()

        for step in range(questions_per_session):
            q = engine.next_question(quiz)
            a = respondent.answer(q)
            engine.record_answer(quiz, q, a)

        top = engine.list_top_targets(quiz, k=3)
        top1_hits.append(top[0][0] == target)
        top3_hits.append(any(idx == target for idx, _ in top))
        entropies.append(quiz.entropy())

        engine.record_quiz_target(quiz, target)

        # Rolling accuracy over last `window` rounds
        recent1 = top1_hits[-window:]
        recent3 = top3_hits[-window:]
        result.top1_per_round.append(sum(recent1) / len(recent1))
        result.top3_per_round.append(sum(recent3) / len(recent3))
        result.entropy_per_round.append(np.mean(entropies[-window:]))

    result.total_time = time.perf_counter() - t0

    # Accuracy vs number of questions asked (evaluate at different depths)
    for n_q in [3, 5, 7, 10, 15]:
        if n_q > data.n_questions():
            continue
        correct = 0
        n_eval = min(200, n_rounds)
        rng2 = np.random.default_rng(seed + 999)
        for _ in range(n_eval):
            target = int(rng2.integers(0, data.n_targets()))
            resp = SimulatedRespondent(target, rng=rng2)
            quiz = engine.start_quiz()
            for step in range(n_q):
                q = engine.next_question(quiz)
                a = resp.answer(q)
                engine.record_answer(quiz, q, a)
            top = engine.list_top_targets(quiz, k=1)
            if top[0][0] == target:
                correct += 1
        result.accuracy_vs_questions[n_q] = correct / n_eval

    return result


def main():
    print("Running benchmark with default params…")
    result = run_benchmark(n_rounds=300)
    print(f"Done in {result.total_time:.1f}s")
    print(f"Final top-1 accuracy: {result.top1_per_round[-1]:.1%}")
    print(f"Final top-3 accuracy: {result.top3_per_round[-1]:.1%}")
    print(f"Final avg entropy: {result.entropy_per_round[-1]:.2f} bits")
    print("\nAccuracy vs questions asked:")
    for nq, acc in sorted(result.accuracy_vs_questions.items()):
        print(f"  {nq:2d} questions → {acc:.1%}")


if __name__ == "__main__":
    main()
