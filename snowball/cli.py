"""
Interactive CLI for running survey sessions.

Usage:
    uv run snowball          # interactive mode
    uv run snowball --sim    # simulated mode (auto-answers)
"""

from __future__ import annotations

import argparse
import sys

from snowball.data import SurveyData
from snowball.engine import PqaEngine, EngineParams
from snowball.simulator import SimulatedRespondent


def run_interactive(engine: PqaEngine) -> None:
    data = SurveyData
    quiz = engine.start_quiz()
    n_asked = 0
    max_questions = min(10, engine.Q)

    print("\n=== Product/Service Discovery Survey ===\n")
    print("Answer each question to help us find the best product/service for you.\n")

    while n_asked < max_questions:
        q = engine.next_question(quiz)
        qdef = data.QUESTIONS[q]

        print(f"Q{n_asked + 1}: {qdef.text}")
        for i, ans in enumerate(qdef.answers):
            print(f"  [{i + 1}] {ans}")

        while True:
            try:
                choice = input("Your choice (number, or 'q' to quit): ").strip()
                if choice.lower() == "q":
                    print("\nQuitting early.\n")
                    _show_results(engine, quiz, data)
                    return
                ans_idx = int(choice) - 1
                if 0 <= ans_idx < len(qdef.answers):
                    break
                print(f"  Please enter 1–{len(qdef.answers)}")
            except (ValueError, EOFError):
                print(f"  Please enter 1–{len(qdef.answers)}")

        engine.record_answer(quiz, q, ans_idx)
        n_asked += 1

        # Show top guesses after each answer
        top = engine.list_top_targets(quiz, k=3)
        print(f"  → Top match: {data.TARGETS[top[0][0]].name} ({top[0][1]:.1%})\n")

    _show_results(engine, quiz, data)


def _show_results(engine: PqaEngine, quiz, data: type[SurveyData]) -> None:
    print("=== Results ===")
    for rank, (idx, prob) in enumerate(engine.list_top_targets(quiz, k=5), 1):
        t = data.TARGETS[idx]
        print(f"  {rank}. {t.name} ({prob:.1%}) — {t.description}")
    print()


def run_simulated(engine: PqaEngine, n_rounds: int = 50) -> None:
    """Run simulated sessions and train the engine."""
    import numpy as np

    data = SurveyData
    rng = np.random.default_rng(42)

    for r in range(n_rounds):
        target = int(rng.integers(0, data.n_targets()))
        respondent = SimulatedRespondent(target, rng=rng)
        quiz = engine.start_quiz()

        for _ in range(10):
            q = engine.next_question(quiz)
            a = respondent.answer(q)
            engine.record_answer(quiz, q, a)

        top = engine.list_top_targets(quiz, k=1)
        correct = top[0][0] == target
        engine.record_quiz_target(quiz, target)

        if (r + 1) % 10 == 0:
            print(f"Round {r + 1}/{n_rounds}  last_correct={correct}  target={data.TARGETS[target].name}")

    print(f"\nTraining complete. {engine.total_quizzes_trained} quizzes trained.")


def main():
    parser = argparse.ArgumentParser(description="Snowball survey engine")
    parser.add_argument("--sim", action="store_true", help="Run simulated sessions")
    parser.add_argument("--rounds", type=int, default=50, help="Simulation rounds")
    args = parser.parse_args()

    data = SurveyData
    engine = PqaEngine(data.n_questions(), data.n_answers(), data.n_targets())

    if args.sim:
        run_simulated(engine, args.rounds)
    else:
        run_interactive(engine)


if __name__ == "__main__":
    main()
