"""
Simulated respondent — generates synthetic survey answers for benchmarking.

Each simulated respondent has a hidden target. Their answers are drawn from a
noisy distribution centred on the target's profile (from SurveyData.PROFILES).
"""

from __future__ import annotations

import numpy as np

from snowball.data import SurveyData


class SimulatedRespondent:
    """Generates answers as if a human with a known target preference."""

    def __init__(
        self,
        target_idx: int,
        *,
        noise: float = 0.15,
        rng: np.random.Generator | None = None,
    ):
        self.target_idx = target_idx
        self.noise = noise
        self._rng = rng or np.random.default_rng()
        self._profile = SurveyData.PROFILES[target_idx]

    def answer(self, question_idx: int) -> int:
        """Return an answer index for the given question."""
        n_answers = SurveyData.n_answers()
        preferred = self._profile[question_idx]

        # Build probability distribution: high weight on preferred answer, noise on rest
        probs = np.full(n_answers, self.noise / (n_answers - 1))
        probs[preferred] = 1.0 - self.noise
        probs /= probs.sum()

        return int(self._rng.choice(n_answers, p=probs))
