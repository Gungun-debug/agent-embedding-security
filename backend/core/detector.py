"""
Shared detector core -- the research contribution.

Both sensors reduce to the same question:

    "Is this new embedding consistent with what came before,
     or does it represent a harmful shift?"

DriftScorer   -- for a SEQUENCE of embeddings of the same evolving item
                 (versions of one tool description). Catches gradual,
                 cumulative escalation, not just large single jumps.

DensityScorer -- for a POPULATION of embeddings that grows over time
                 (a memory store). Catches new entries sitting in
                 anomalously dense neighbourhoods, the signature of
                 retrieval-space poisoning.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(1.0 - np.dot(a, b))


@dataclass
class DriftEvent:
    step: int
    step_distance: float        # distance from the running baseline
    cumulative_distance: float  # distance from the ORIGINAL version
    flagged: bool
    reason: str = ""


@dataclass
class DriftScorer:
    """
    Tracks one evolving item (e.g. one tool's description history).

    step_threshold       : flag a single step that jumps further than this
    cumulative_threshold : flag when total distance from the very first
                           version exceeds this, even if every individual
                           step was small -- this is what catches slow,
                           salami-sliced escalation
    ema_alpha            : smoothing for the running baseline
    """
    step_threshold: float = 0.25
    cumulative_threshold: float = 0.40
    ema_alpha: float = 0.3

    _baseline: np.ndarray | None = field(default=None, repr=False)
    _running: np.ndarray | None = field(default=None, repr=False)
    _step: int = 0
    history: list[DriftEvent] = field(default_factory=list)

    def observe(self, embedding: np.ndarray) -> DriftEvent:
        self._step += 1

        if self._baseline is None:
            self._baseline = embedding.copy()
            self._running = embedding.copy()
            ev = DriftEvent(self._step, 0.0, 0.0, False, "baseline")
            self.history.append(ev)
            return ev

        step_d = cosine_distance(self._running, embedding)
        cum_d = cosine_distance(self._baseline, embedding)

        flagged, reason = False, ""
        if step_d > self.step_threshold:
            flagged, reason = True, "large single-step jump"
        elif cum_d > self.cumulative_threshold:
            flagged, reason = True, "cumulative drift from baseline"

        self._running = self.ema_alpha * embedding + (1 - self.ema_alpha) * self._running

        ev = DriftEvent(self._step, step_d, cum_d, flagged, reason)
        self.history.append(ev)
        return ev


@dataclass
class DensityEvent:
    index: int
    mean_knn_distance: float
    z_score: float | None
    flagged: bool
    reason: str = ""


@dataclass
class DensityScorer:
    """
    Tracks a growing population of embeddings (e.g. a memory store).

    Legitimate entries sit at a typical average distance from their k
    nearest existing neighbours. Poison entries crafted to crowd around
    common queries show up as anomalously LOW mean k-NN distance
    relative to the calibrated clean baseline.
    """
    k: int = 5
    z_threshold: float = 2.0
    min_population: int = 5

    _pool: list[np.ndarray] = field(default_factory=list)
    _baseline_mean: float | None = field(default=None, repr=False)
    _baseline_std: float | None = field(default=None, repr=False)
    history: list[DensityEvent] = field(default_factory=list)

    def _knn_mean_distance(self, embedding: np.ndarray) -> float:
        if not self._pool:
            return 0.0
        d = sorted(cosine_distance(embedding, p) for p in self._pool)
        return float(np.mean(d[: min(self.k, len(d))]))

    def calibrate(self, clean_embeddings: list[np.ndarray]) -> None:
        """Establish the baseline density distribution from known-clean data."""
        self._pool = list(clean_embeddings)
        dists = []
        for i, e in enumerate(self._pool):
            others = self._pool[:i] + self._pool[i + 1:]
            if not others:
                continue
            d = sorted(cosine_distance(e, o) for o in others)
            dists.append(float(np.mean(d[: min(self.k, len(d))])))
        if dists:
            self._baseline_mean = float(np.mean(dists))
            self._baseline_std = float(np.std(dists)) + 1e-6

    def observe(self, embedding: np.ndarray) -> DensityEvent:
        idx = len(self.history)
        mean_d = self._knn_mean_distance(embedding)

        z = None
        flagged, reason = False, ""
        if self._baseline_mean is not None and len(self._pool) >= self.min_population:
            z = (mean_d - self._baseline_mean) / self._baseline_std
            if z < -self.z_threshold:
                flagged = True
                reason = f"anomalously dense (z={z:.2f}) - possible poisoning"

        self._pool.append(embedding)
        ev = DensityEvent(idx, mean_d, z, flagged, reason)
        self.history.append(ev)
        return ev
