"""
FastAPI backend for the Temporal Embedding-Space Monitoring system.

Design choice: STATELESS endpoints. Rather than keeping per-user session
state on the server (which gets messy with multiple concurrent demo
viewers on a public Space), each request carries the full context it
needs, and the server recomputes the real DriftScorer / DensityScorer
from scratch each call. This is slightly more data over the wire, but
avoids any session/concurrency bugs and matches exactly how
eval/run_experiments.py already uses the same core classes.

Endpoints:
  GET  /health
  POST /drift/evaluate    -- given a tool's version history, score every step
  POST /memory/evaluate   -- given clean + already-injected entries and one
                             candidate, score the candidate
"""

from __future__ import annotations
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.embedder import get_embedder
from core.detector import DriftScorer, DensityScorer

app = FastAPI(title="Agent Embedding Security API")

# Public demo: dashboard is hosted on a different origin (claude.ai artifact),
# so the browser needs CORS permission to call this API from there.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup, reused across requests (this is the expensive part).
_embedder = get_embedder()


@app.get("/health")
def health():
    return {"status": "ok", "embedder": getattr(_embedder, "name", "unknown")}


# ---------------------------------------------------------------------------
# Drift sensor
# ---------------------------------------------------------------------------

class DriftRequest(BaseModel):
    versions: List[str]                # full history, including the new one, in order
    step_threshold: float = 0.9        # generous by default; cumulative is the real signal
    cumulative_threshold: float = 0.23  # your calibrated value


class DriftStepResult(BaseModel):
    step: int
    text: str
    step_distance: float
    cumulative_distance: float
    flagged: bool
    reason: str


class DriftResponse(BaseModel):
    embedder: str
    steps: List[DriftStepResult]
    first_flagged_step: Optional[int]


@app.post("/drift/evaluate", response_model=DriftResponse)
def evaluate_drift(req: DriftRequest):
    if not req.versions:
        return DriftResponse(embedder=getattr(_embedder, "name", "unknown"), steps=[], first_flagged_step=None)

    # Fit the dev TF-IDF fallback on this corpus if that's what's active;
    # sentence-transformers ignores fit() entirely (pretrained).
    _embedder.fit(req.versions)

    scorer = DriftScorer(
        step_threshold=req.step_threshold,
        cumulative_threshold=req.cumulative_threshold,
    )

    steps = []
    first_flagged = None
    for i, text in enumerate(req.versions):
        vec = _embedder.embed([text])[0]
        ev = scorer.observe(vec)
        steps.append(DriftStepResult(
            step=i, text=text,
            step_distance=ev.step_distance,
            cumulative_distance=ev.cumulative_distance,
            flagged=ev.flagged, reason=ev.reason,
        ))
        if ev.flagged and first_flagged is None:
            first_flagged = i

    return DriftResponse(
        embedder=getattr(_embedder, "name", "unknown"),
        steps=steps,
        first_flagged_step=first_flagged,
    )


# ---------------------------------------------------------------------------
# Memory distortion sensor
# ---------------------------------------------------------------------------

class MemoryRequest(BaseModel):
    clean_entries: List[str]           # the calibration baseline (known-clean memory)
    already_injected: List[str] = []   # entries added in this session so far
    candidate: str                     # the new entry to evaluate
    k: int = 5
    z_threshold: float = 1.5           # your calibrated value


class MemoryResponse(BaseModel):
    embedder: str
    mean_knn_distance: float
    z_score: Optional[float]
    flagged: bool
    reason: str


@app.post("/memory/evaluate", response_model=MemoryResponse)
def evaluate_memory(req: MemoryRequest):
    corpus = req.clean_entries + req.already_injected + [req.candidate]
    _embedder.fit(corpus)

    clean_vecs = list(_embedder.embed(req.clean_entries))
    scorer = DensityScorer(k=req.k, z_threshold=req.z_threshold, min_population=1)
    scorer.calibrate(clean_vecs)

    # Replay already-injected entries so the population/density reflects
    # the current state of the store, then score the real candidate.
    for text in req.already_injected:
        scorer.observe(_embedder.embed([text])[0])

    ev = scorer.observe(_embedder.embed([req.candidate])[0])

    return MemoryResponse(
        embedder=getattr(_embedder, "name", "unknown"),
        mean_knn_distance=ev.mean_knn_distance,
        z_score=ev.z_score,
        flagged=ev.flagged,
        reason=ev.reason,
    )
