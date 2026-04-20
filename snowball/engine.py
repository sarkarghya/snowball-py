"""
Core PQA Engine

Notation
--------
- Q  = number of questions
- K  = number of answer options per question (fixed across questions)
- T  = number of targets (products/services to discover)

Knowledge Base (KB)
-------------------
A[q, k, t]  — frequency evidence that target t produces answer k for question q.
              Stored as *squared* underlying counts (see training section).
D[q, t]     — row sum of A over answers:  D[q,t] = sum_k A[q,k,t]
B[t]        — prior weight of target t (unnormalised).

Bayesian update (RecordAnswer)
------------------------------
When the user answers question q with answer k:
    posterior[t] ∝ prior[t] * A[q,k,t] / D[q,t]    (likelihood ratio)

Question selection (NextQuestion)
---------------------------------
For each candidate question, score a priority that combines:
  1. Expected entropy reduction
  2. Posterior velocity (how much posteriors shift)
  3. Information lack (prefer under-trained questions)
Select proportionally to priority (softmax-weighted random).

Training
--------
After a quiz ends with confirmed target t, for every (q, k) answered:
    underlying = sqrt(A[q,k,t])
    A[q,k,t]  = (underlying + amount)^2
    D[q,t]   += delta
    B[t]      = (sqrt(B[t]) + amount)^2
"""

from __future__ import annotations

import math
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPS = 1e-300  # floor to avoid log(0)
_LOG2 = math.log(2.0)


def _safe_log2(x: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(x, _EPS)) / _LOG2


def _normalise(v: np.ndarray) -> np.ndarray:
    s = v.sum()
    if s <= 0:
        return np.full_like(v, 1.0 / len(v))
    return v / s


# ---------------------------------------------------------------------------
# Quiz (per-session state)
# ---------------------------------------------------------------------------

@dataclass
class Quiz:
    """Mutable state for one survey session."""
    posteriors: np.ndarray              # shape (T,)
    answered: list[tuple[int, int]] = field(default_factory=list)  # (q_idx, ans_idx)
    asked_set: set[int] = field(default_factory=set)

    def entropy(self) -> float:
        p = self.posteriors
        return -float(np.sum(p * _safe_log2(p)))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

@dataclass
class EngineParams:
    """Tuneable hyper-parameters."""
    init_amount: float = 1.0       # initial KB fill per cell
    train_amount: float = 1.0      # training increment
    entropy_exp: float = -2.0      # exponent for nExpectedTargets in priority
    velocity_exp: float = 9.0      # exponent for velocity component
    lack_exp: float = 1.0          # exponent for lack component
    selection_softmax_temp: float = 1.0  # temperature for question selection


class PqaEngine:
    """Probabilistic Question-Answering engine."""

    # ---- construction ------------------------------------------------------

    def __init__(
        self,
        n_questions: int,
        n_answers: int,
        n_targets: int,
        *,
        params: EngineParams | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.Q = n_questions
        self.K = n_answers
        self.T = n_targets
        self.params = params or EngineParams()
        self._rng = rng or np.random.default_rng()

        a0 = self.params.init_amount
        # Store squared counts  (underlying count = a0)
        self.A = np.full((self.Q, self.K, self.T), a0 * a0, dtype=np.float64)
        self.D = np.full((self.Q, self.T), a0 * a0 * self.K, dtype=np.float64)
        self.B = np.full(self.T, a0 * a0, dtype=np.float64)

        self.total_quizzes_trained: int = 0

    # ---- quiz lifecycle ----------------------------------------------------

    def start_quiz(self) -> Quiz:
        priors = _normalise(self.B.copy())
        return Quiz(posteriors=priors)

    def record_answer(self, quiz: Quiz, question: int, answer: int) -> None:
        """Bayesian update: multiply posteriors by likelihood."""
        A_qa = self.A[question, answer, :]   # (T,)
        D_q = self.D[question, :]            # (T,)
        likelihood = A_qa / np.maximum(D_q, _EPS)
        quiz.posteriors *= likelihood
        quiz.posteriors = _normalise(quiz.posteriors)
        quiz.answered.append((question, answer))
        quiz.asked_set.add(question)

    def next_question(self, quiz: Quiz) -> int:
        """Select the most informative unanswered question."""
        candidates = [q for q in range(self.Q) if q not in quiz.asked_set]
        if not candidates:
            raise RuntimeError("All questions exhausted")

        priorities = np.zeros(len(candidates), dtype=np.float64)
        prior = quiz.posteriors  # (T,)

        for idx, q in enumerate(candidates):
            priorities[idx] = self._score_question(q, prior)

        # Softmax-weighted random selection
        temp = self.params.selection_softmax_temp
        log_p = np.log(np.maximum(priorities, _EPS)) / max(temp, 1e-12)
        log_p -= log_p.max()
        weights = np.exp(log_p)
        weights /= weights.sum()
        chosen_idx = self._rng.choice(len(candidates), p=weights)
        return candidates[chosen_idx]

    def list_top_targets(self, quiz: Quiz, k: int = 5) -> list[tuple[int, float]]:
        """Return top-k (target_idx, probability) pairs."""
        k = min(k, self.T)
        indices = np.argpartition(quiz.posteriors, -k)[-k:]
        indices = indices[np.argsort(-quiz.posteriors[indices])]
        return [(int(i), float(quiz.posteriors[i])) for i in indices]

    def record_quiz_target(self, quiz: Quiz, target: int) -> None:
        """Train the KB with a completed quiz."""
        amount = self.params.train_amount
        for q, a in quiz.answered:
            old = self.A[q, a, target]
            underlying = math.sqrt(max(old, 0))
            new_val = (underlying + amount) ** 2
            delta = new_val - old
            self.A[q, a, target] = new_val
            self.D[q, target] += delta

        old_b = self.B[target]
        self.B[target] = (math.sqrt(max(old_b, 0)) + amount) ** 2
        self.total_quizzes_trained += 1

    # ---- question scoring --------------------------------------------------

    def _score_question(self, q: int, prior: np.ndarray) -> float:
        """Compute priority for a candidate question (entropy + velocity + lack)."""
        D_q = self.D[q, :]  # (T,)

        total_weight = 0.0
        weighted_entropy = 0.0
        weighted_velocity = 0.0
        lack_sum = 0.0

        for k in range(self.K):
            A_qk = self.A[q, k, :]  # (T,)
            likelihood = A_qk / np.maximum(D_q, _EPS)
            posterior = prior * likelihood
            w = posterior.sum()
            if w <= _EPS:
                continue
            posterior /= w

            # Entropy of this hypothetical posterior
            H = -float(np.sum(posterior * _safe_log2(posterior)))

            # Velocity: Euclidean distance squared between posterior and prior
            diff = posterior - prior
            V = float(np.dot(diff, diff))

            # Lack: how under-trained is this question for each target
            inv_D = 1.0 / np.maximum(D_q, _EPS)
            lack = float(np.sum(inv_D * inv_D))

            total_weight += w
            weighted_entropy += w * H
            weighted_velocity += w * math.sqrt(max(V, 0))
            lack_sum += w * lack

        if total_weight <= _EPS:
            return _EPS

        avg_H = weighted_entropy / total_weight
        avg_V = weighted_velocity / total_weight
        avg_lack = lack_sum / total_weight
        n_expected = 2.0 ** avg_H

        # Priority combines the three signals
        p = self.params
        v_comp = max(avg_V, _EPS)
        lack_comp = max(avg_lack, _EPS)
        n_comp = max(n_expected, _EPS)

        priority = (lack_comp ** p.lack_exp) * (v_comp ** p.velocity_exp) * (n_comp ** p.entropy_exp)
        return max(priority, _EPS)

    # ---- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        state = {
            "Q": self.Q, "K": self.K, "T": self.T,
            "params": self.params,
            "A": self.A, "D": self.D, "B": self.B,
            "total_quizzes_trained": self.total_quizzes_trained,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> "PqaEngine":
        with open(path, "rb") as f:
            state = pickle.load(f)
        eng = cls.__new__(cls)
        eng.Q = state["Q"]
        eng.K = state["K"]
        eng.T = state["T"]
        eng.params = state["params"]
        eng.A = state["A"]
        eng.D = state["D"]
        eng.B = state["B"]
        eng.total_quizzes_trained = state["total_quizzes_trained"]
        eng._rng = np.random.default_rng()
        return eng
