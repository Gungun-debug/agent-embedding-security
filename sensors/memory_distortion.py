"""Sensor B -- Memory Distortion Detector."""

from __future__ import annotations
from dataclasses import dataclass, field
from core.detector import DensityScorer, DensityEvent


@dataclass
class MemoryDistortionMonitor:
    embedder: object
    k: int = 5
    z_threshold: float = 2.0
    min_population: int = 5

    _scorer: DensityScorer = field(default=None)
    _entries: list = field(default_factory=list)

    def __post_init__(self):
        self._scorer = DensityScorer(
            k=self.k, z_threshold=self.z_threshold, min_population=self.min_population
        )

    def calibrate(self, clean_entries: list[str]) -> None:
        self._entries = list(clean_entries)
        self._scorer.calibrate(list(self.embedder.embed(clean_entries)))

    def observe_entry(self, text: str) -> DensityEvent:
        vec = self.embedder.embed([text])[0]
        self._entries.append(text)
        return self._scorer.observe(vec)
