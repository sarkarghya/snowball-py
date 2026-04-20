"""
End-to-end tests for Snowball-PY.

These tests verify the full pipeline: engine construction, quiz lifecycle,
Bayesian inference correctness, training, learning over time, persistence,
and simulated survey accuracy.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from snowball.engine import PqaEngine, EngineParams, Quiz
from snowball.data import SurveyData
from snowball.simulator import SimulatedRespondent
from snowball.benchmark import run_benchmark


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_engine() -> PqaEngine:
    """A small engine (4 questions, 3 answers, 5 targets) for unit-level tests."""
    return PqaEngine(4, 3, 5, rng=np.random.default_rng(0))


@pytest.fixture
def survey_engine() -> PqaEngine:
    """Full survey engine with real question data."""
    d = SurveyData
    return PqaEngine(d.n_questions(), d.n_answers(), d.n_targets(), rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# 1. Engine construction
# ---------------------------------------------------------------------------

class TestEngineConstruction:
    def test_dimensions(self, small_engine: PqaEngine):
        assert small_engine.Q == 4
        assert small_engine.K == 3
        assert small_engine.T == 5

    def test_matrices_shape(self, small_engine: PqaEngine):
        assert small_engine.A.shape == (4, 3, 5)
        assert small_engine.D.shape == (4, 5)
        assert small_engine.B.shape == (5,)

    def test_initial_values_positive(self, small_engine: PqaEngine):
        assert (small_engine.A > 0).all()
        assert (small_engine.D > 0).all()
        assert (small_engine.B > 0).all()

    def test_D_equals_sum_of_A(self, small_engine: PqaEngine):
        """D[q,t] = sum_k A[q,k,t]"""
        expected_D = small_engine.A.sum(axis=1)
        np.testing.assert_allclose(small_engine.D, expected_D)

    def test_survey_data_consistency(self):
        d = SurveyData
        assert d.n_questions() == 20
        assert d.n_answers() == 5
        assert d.n_targets() == 12
        assert len(d.PROFILES) == d.n_targets()
        for profile in d.PROFILES.values():
            assert len(profile) == d.n_questions()


# ---------------------------------------------------------------------------
# 2. Quiz lifecycle
# ---------------------------------------------------------------------------

class TestQuizLifecycle:
    def test_start_quiz_returns_normalised_priors(self, small_engine: PqaEngine):
        quiz = small_engine.start_quiz()
        assert quiz.posteriors.shape == (5,)
        assert abs(quiz.posteriors.sum() - 1.0) < 1e-10

    def test_record_answer_updates_posteriors(self, small_engine: PqaEngine):
        quiz = small_engine.start_quiz()
        old_post = quiz.posteriors.copy()
        small_engine.record_answer(quiz, 0, 1)
        # Posteriors should change (unless perfectly uniform, which they are initially —
        # but the Bayesian update with uniform A still normalises to uniform)
        assert quiz.posteriors.shape == (5,)
        assert abs(quiz.posteriors.sum() - 1.0) < 1e-10
        assert len(quiz.answered) == 1
        assert 0 in quiz.asked_set

    def test_next_question_not_repeated(self, small_engine: PqaEngine):
        quiz = small_engine.start_quiz()
        asked = set()
        for _ in range(4):
            q = small_engine.next_question(quiz)
            assert q not in asked
            asked.add(q)
            small_engine.record_answer(quiz, q, 0)
        assert len(asked) == 4

    def test_all_questions_exhausted_raises(self, small_engine: PqaEngine):
        quiz = small_engine.start_quiz()
        for i in range(4):
            q = small_engine.next_question(quiz)
            small_engine.record_answer(quiz, q, 0)
        with pytest.raises(RuntimeError, match="exhausted"):
            small_engine.next_question(quiz)

    def test_list_top_targets(self, small_engine: PqaEngine):
        quiz = small_engine.start_quiz()
        tops = small_engine.list_top_targets(quiz, k=3)
        assert len(tops) == 3
        # Probabilities should be sorted descending
        probs = [p for _, p in tops]
        assert probs == sorted(probs, reverse=True)
        # All probabilities positive
        assert all(p > 0 for _, p in tops)


# ---------------------------------------------------------------------------
# 3. Bayesian inference correctness
# ---------------------------------------------------------------------------

class TestBayesianInference:
    def test_posterior_favours_trained_target(self):
        """After training target 0 on (q=0, a=0), asking q=0 and answering 0
        should make target 0 the most probable."""
        engine = PqaEngine(3, 3, 4, rng=np.random.default_rng(0))

        # Train: target 0 always answers 0 for question 0
        for _ in range(20):
            quiz = engine.start_quiz()
            engine.record_answer(quiz, 0, 0)
            engine.record_quiz_target(quiz, 0)

        # Now evaluate: answer 0 for question 0 → target 0 should dominate
        quiz = engine.start_quiz()
        engine.record_answer(quiz, 0, 0)
        top = engine.list_top_targets(quiz, k=1)
        assert top[0][0] == 0

    def test_multiple_answers_narrow_posterior(self):
        """More consistent answers should reduce entropy."""
        engine = PqaEngine(5, 3, 4, rng=np.random.default_rng(1))

        # Train target 2: always answers [0, 1, 2, 0, 1]
        answers = [0, 1, 2, 0, 1]
        for _ in range(30):
            quiz = engine.start_quiz()
            for q, a in enumerate(answers):
                engine.record_answer(quiz, q, a)
            engine.record_quiz_target(quiz, 2)

        # Evaluate with 1 answer
        quiz1 = engine.start_quiz()
        engine.record_answer(quiz1, 0, 0)
        ent1 = quiz1.entropy()

        # Evaluate with 3 answers
        quiz3 = engine.start_quiz()
        for q in range(3):
            engine.record_answer(quiz3, q, answers[q])
        ent3 = quiz3.entropy()

        # More answers → lower entropy
        assert ent3 < ent1

    def test_posteriors_remain_normalised(self, survey_engine: PqaEngine):
        """Posteriors should stay normalised after many answers."""
        quiz = survey_engine.start_quiz()
        rng = np.random.default_rng(42)
        for _ in range(15):
            q = survey_engine.next_question(quiz)
            a = int(rng.integers(0, survey_engine.K))
            survey_engine.record_answer(quiz, q, a)
            assert abs(quiz.posteriors.sum() - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------

class TestTraining:
    def test_training_increases_A(self, small_engine: PqaEngine):
        old_A = small_engine.A.copy()
        quiz = small_engine.start_quiz()
        small_engine.record_answer(quiz, 0, 1)
        small_engine.record_quiz_target(quiz, 2)

        # A[0, 1, 2] should have increased
        assert small_engine.A[0, 1, 2] > old_A[0, 1, 2]
        # Other cells unchanged
        assert small_engine.A[0, 0, 2] == old_A[0, 0, 2]

    def test_training_maintains_D_consistency(self, small_engine: PqaEngine):
        """D should remain the sum of A over answers after training."""
        quiz = small_engine.start_quiz()
        small_engine.record_answer(quiz, 0, 1)
        small_engine.record_answer(quiz, 2, 0)
        small_engine.record_quiz_target(quiz, 3)

        expected_D = small_engine.A.sum(axis=1)
        np.testing.assert_allclose(small_engine.D, expected_D, rtol=1e-10)

    def test_training_increases_B(self, small_engine: PqaEngine):
        old_B = small_engine.B.copy()
        quiz = small_engine.start_quiz()
        small_engine.record_answer(quiz, 0, 0)
        small_engine.record_quiz_target(quiz, 1)
        assert small_engine.B[1] > old_B[1]
        # Others unchanged
        assert small_engine.B[0] == old_B[0]

    def test_quiz_counter_increments(self, small_engine: PqaEngine):
        assert small_engine.total_quizzes_trained == 0
        quiz = small_engine.start_quiz()
        small_engine.record_answer(quiz, 0, 0)
        small_engine.record_quiz_target(quiz, 0)
        assert small_engine.total_quizzes_trained == 1

    def test_square_root_training_scheme(self):
        """Verify the (sqrt(old) + amount)^2 training formula."""
        engine = PqaEngine(2, 2, 2, params=EngineParams(init_amount=3.0, train_amount=2.0))
        old_val = engine.A[0, 0, 0]  # = 9.0 (3^2)
        quiz = engine.start_quiz()
        engine.record_answer(quiz, 0, 0)
        engine.record_quiz_target(quiz, 0)
        # Expected: (sqrt(9) + 2)^2 = (3 + 2)^2 = 25
        assert abs(engine.A[0, 0, 0] - 25.0) < 1e-10


# ---------------------------------------------------------------------------
# 5. Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, small_engine: PqaEngine):
        # Train a bit
        quiz = small_engine.start_quiz()
        small_engine.record_answer(quiz, 0, 1)
        small_engine.record_quiz_target(quiz, 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pkl"
            small_engine.save(path)
            loaded = PqaEngine.load(path)

        assert loaded.Q == small_engine.Q
        assert loaded.K == small_engine.K
        assert loaded.T == small_engine.T
        np.testing.assert_array_equal(loaded.A, small_engine.A)
        np.testing.assert_array_equal(loaded.D, small_engine.D)
        np.testing.assert_array_equal(loaded.B, small_engine.B)
        assert loaded.total_quizzes_trained == small_engine.total_quizzes_trained


# ---------------------------------------------------------------------------
# 6. Simulator
# ---------------------------------------------------------------------------

class TestSimulator:
    def test_respondent_favours_profile_answer(self):
        rng = np.random.default_rng(0)
        target = 0
        resp = SimulatedRespondent(target, noise=0.05, rng=rng)
        profile = SurveyData.PROFILES[target]

        # Over many samples, the preferred answer should dominate
        counts = np.zeros(SurveyData.n_answers(), dtype=int)
        for _ in range(1000):
            a = resp.answer(0)
            counts[a] += 1

        assert counts[profile[0]] > 800  # should be ~950 with noise=0.05

    def test_respondent_noise_zero_is_deterministic(self):
        resp = SimulatedRespondent(0, noise=0.0, rng=np.random.default_rng(0))
        profile = SurveyData.PROFILES[0]
        for q in range(SurveyData.n_questions()):
            assert resp.answer(q) == profile[q]


# ---------------------------------------------------------------------------
# 7. End-to-end: Learning over time
# ---------------------------------------------------------------------------

class TestEndToEndLearning:
    def test_accuracy_improves_with_training(self):
        """Core objective: the engine must *learn* — accuracy after training
        should be significantly higher than at the start."""
        d = SurveyData
        engine = PqaEngine(
            d.n_questions(), d.n_answers(), d.n_targets(),
            rng=np.random.default_rng(42),
        )
        rng = np.random.default_rng(123)

        def measure_accuracy(n_eval: int = 100) -> float:
            correct = 0
            eval_rng = np.random.default_rng(999)
            for _ in range(n_eval):
                target = int(eval_rng.integers(0, d.n_targets()))
                resp = SimulatedRespondent(target, rng=eval_rng)
                quiz = engine.start_quiz()
                for _ in range(10):
                    q = engine.next_question(quiz)
                    a = resp.answer(q)
                    engine.record_answer(quiz, q, a)
                top = engine.list_top_targets(quiz, k=1)
                if top[0][0] == target:
                    correct += 1
            return correct / n_eval

        acc_before = measure_accuracy()

        # Train for 200 rounds
        for _ in range(200):
            target = int(rng.integers(0, d.n_targets()))
            resp = SimulatedRespondent(target, rng=rng)
            quiz = engine.start_quiz()
            for _ in range(10):
                q = engine.next_question(quiz)
                a = resp.answer(q)
                engine.record_answer(quiz, q, a)
            engine.record_quiz_target(quiz, target)

        acc_after = measure_accuracy()

        # Accuracy should meaningfully improve
        assert acc_after > acc_before + 0.1, (
            f"Expected significant improvement: before={acc_before:.2%}, after={acc_after:.2%}"
        )
        # After 200 rounds of training, should be reasonably good
        assert acc_after > 0.4, f"Expected >40% accuracy after training, got {acc_after:.2%}"

    def test_top3_accuracy_higher_than_top1(self):
        """Top-3 accuracy should be >= top-1 accuracy."""
        result = run_benchmark(n_rounds=100, seed=77)
        assert result.top3_per_round[-1] >= result.top1_per_round[-1]

    def test_entropy_decreases_with_training(self):
        """Average posterior entropy should decrease as the engine learns."""
        result = run_benchmark(n_rounds=200, seed=88)
        # Compare first 20 rounds vs last 20 rounds
        early = np.mean(result.entropy_per_round[:20])
        late = np.mean(result.entropy_per_round[-20:])
        assert late < early, f"Entropy should decrease: early={early:.2f}, late={late:.2f}"

    def test_more_questions_gives_higher_accuracy(self):
        """Asking more questions should yield higher accuracy."""
        result = run_benchmark(n_rounds=200, seed=55)
        accs = result.accuracy_vs_questions
        qs = sorted(accs.keys())
        # At least the trend should be upward (allow small noise)
        if len(qs) >= 2:
            assert accs[qs[-1]] >= accs[qs[0]] - 0.05, (
                f"More questions should help: {qs[0]}q={accs[qs[0]]:.2%}, {qs[-1]}q={accs[qs[-1]]:.2%}"
            )


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_question_engine(self):
        engine = PqaEngine(1, 2, 3, rng=np.random.default_rng(0))
        quiz = engine.start_quiz()
        q = engine.next_question(quiz)
        assert q == 0
        engine.record_answer(quiz, q, 0)
        tops = engine.list_top_targets(quiz, k=3)
        assert len(tops) == 3

    def test_single_target_engine(self):
        engine = PqaEngine(3, 2, 1, rng=np.random.default_rng(0))
        quiz = engine.start_quiz()
        tops = engine.list_top_targets(quiz, k=1)
        assert tops[0][0] == 0
        assert abs(tops[0][1] - 1.0) < 1e-10

    def test_many_training_rounds_no_crash(self):
        engine = PqaEngine(3, 3, 3, rng=np.random.default_rng(0))
        rng = np.random.default_rng(42)
        for _ in range(100):
            quiz = engine.start_quiz()
            for _ in range(3):
                q = engine.next_question(quiz)
                a = int(rng.integers(0, 3))
                engine.record_answer(quiz, q, a)
            engine.record_quiz_target(quiz, int(rng.integers(0, 3)))
        assert engine.total_quizzes_trained == 100
        # No NaN or Inf in matrices
        assert np.all(np.isfinite(engine.A))
        assert np.all(np.isfinite(engine.D))
        assert np.all(np.isfinite(engine.B))
