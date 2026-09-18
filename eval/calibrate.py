"""
Threshold calibration.

Sweeps candidate thresholds for both sensors against the malicious AND
benign sets, and reports the value that maximises Youden's J
(detection rate - false positive rate). Run this BEFORE run_experiments
and copy the chosen thresholds into core/detector.py.

Usage:
    python -m eval.calibrate
"""

from __future__ import annotations
import json
import os
import numpy as np

from core.embedder import get_embedder
from core.detector import DriftScorer, DensityScorer
from attacks.drift_attack import MALICIOUS_ESCALATIONS, BENIGN_SEQUENCES
from attacks.poison_attack import (
    CLEAN_MEMORY, POISON_TEMPLATES, BENIGN_NEW_ENTRIES, FLOOD_FACTOR,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def sweep_drift(embedder):
    """Sweep cumulative_threshold; step_threshold is set generously so the
    cumulative signal (the actual contribution) is what's being measured."""
    corpus = [v for _, vs in MALICIOUS_ESCALATIONS + BENIGN_SEQUENCES for v in vs]
    embedder.fit(corpus)

    def max_cumulative(sequences):
        """Largest cumulative drift each sequence reaches from its baseline."""
        out = []
        for _, versions in sequences:
            s = DriftScorer(step_threshold=99, cumulative_threshold=99)
            peak = 0.0
            for v in versions:
                ev = s.observe(embedder.embed([v])[0])
                peak = max(peak, ev.cumulative_distance)
            out.append(peak)
        return np.array(out)

    mal = max_cumulative(MALICIOUS_ESCALATIONS)
    ben = max_cumulative(BENIGN_SEQUENCES)

    best = None
    for t in np.arange(0.02, 1.00, 0.01):
        tpr = float((mal > t).mean())
        fpr = float((ben > t).mean())
        j = tpr - fpr
        if best is None or j > best["youden_j"]:
            best = {"threshold": round(float(t), 3), "detection_rate": tpr,
                    "false_positive_rate": fpr, "youden_j": round(j, 3)}

    return {
        "malicious_peak_drift": {"min": float(mal.min()), "mean": float(mal.mean()),
                                 "max": float(mal.max())},
        "benign_peak_drift": {"min": float(ben.min()), "mean": float(ben.mean()),
                              "max": float(ben.max())},
        "separable": float(ben.max()) < float(mal.min()),
        "best": best,
    }


def sweep_density(embedder):
    clean_texts = [t for _, t in CLEAN_MEMORY]
    poison_texts = [t for _, t in POISON_TEMPLATES for _ in range(FLOOD_FACTOR)]
    benign_texts = [t for _, t in BENIGN_NEW_ENTRIES]

    embedder.fit(clean_texts + poison_texts + benign_texts)
    clean_vecs = list(embedder.embed(clean_texts))

    def z_scores(texts):
        s = DensityScorer(z_threshold=99)  # observe without flagging
        s.calibrate(clean_vecs)
        out = []
        for t in texts:
            ev = s.observe(embedder.embed([t])[0])
            if ev.z_score is not None:
                out.append(ev.z_score)
        return np.array(out)

    mal_z = z_scores(poison_texts)
    ben_z = z_scores(benign_texts)

    best = None
    for t in np.arange(0.2, 5.0, 0.1):
        tpr = float((mal_z < -t).mean())
        fpr = float((ben_z < -t).mean())
        j = tpr - fpr
        if best is None or j > best["youden_j"]:
            best = {"z_threshold": round(float(t), 2), "detection_rate": tpr,
                    "false_positive_rate": fpr, "youden_j": round(j, 3)}

    return {
        "poison_z": {"min": float(mal_z.min()), "mean": float(mal_z.mean())},
        "benign_z": {"min": float(ben_z.min()), "mean": float(ben_z.mean())},
        "best": best,
    }


def main():
    embedder = get_embedder()
    name = getattr(embedder, "name", "unknown")
    print(f"[embedder] using: {name}\n")

    drift = sweep_drift(embedder)
    density = sweep_density(embedder)

    print("=== Drift sensor (cumulative_threshold) ===")
    print(json.dumps(drift, indent=2))
    print("\n=== Density sensor (z_threshold) ===")
    print(json.dumps(density, indent=2))

    print("\n--- Copy into core/detector.py ---")
    print(f"DriftScorer.cumulative_threshold = {drift['best']['threshold']}")
    print(f"DensityScorer.z_threshold        = {density['best']['z_threshold']}")

    if not drift["separable"]:
        print("\nNOTE: benign and malicious drift ranges overlap, so no single "
              "threshold separates them cleanly. Report this honestly -- it is "
              "a real limit of the signal, and the overlap region is exactly "
              "where a stealthy attacker would operate.")

    out = os.path.join(HERE, "calibration.json")
    with open(out, "w") as f:
        json.dump({"embedder": name, "drift": drift, "density": density}, f, indent=2)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
