"""Sensor A -- Tool Description Drift Detector."""

from __future__ import annotations
from dataclasses import dataclass, field
from core.detector import DriftScorer, DriftEvent


@dataclass
class ToolDriftMonitor:
    embedder: object
    step_threshold: float = 0.25
    cumulative_threshold: float = 0.40

    _scorers: dict = field(default_factory=dict)
    _texts: dict = field(default_factory=dict)

    def observe_version(self, tool_name: str, description: str) -> DriftEvent:
        if tool_name not in self._scorers:
            self._scorers[tool_name] = DriftScorer(
                step_threshold=self.step_threshold,
                cumulative_threshold=self.cumulative_threshold,
            )
            self._texts[tool_name] = []
        self._texts[tool_name].append(description)
        vec = self.embedder.embed([description])[0]
        return self._scorers[tool_name].observe(vec)

    def history(self, tool_name: str) -> list[DriftEvent]:
        return self._scorers[tool_name].history if tool_name in self._scorers else []
