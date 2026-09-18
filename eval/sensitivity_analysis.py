"""
Adaptive-attacker sensitivity analysis.

Two questions a real evaluation needs to answer, not just "does the
detector catch our hand-picked attack":

  1. GRANULARITY INVARIANCE
     The cumulative-drift check compares the CURRENT version against the
     ORIGINAL baseline, not step-to-step. That means, in principle, it
     shouldn't matter whether an attacker reaches a given final
     description in 2 big jumps or 50 tiny ones -- the check is a
     function of the endpoint, not the path. This experiment verifies
     that empirically: if true, salami-slicing (the exact evasion
     tactic this project is named for) gives the attacker no advantage
     against THIS signal. That is a real, reportable robustness result
     -- and if it turns out NOT to hold in practice (e.g. because the
     EMA-smoothed running baseline used for the step-jump check behaves
     differently), that's equally worth reporting honestly.

  2. SAFE ESCALATION CEILING
     Every detector has a blind spot: some change small enough to stay
     under threshold. This measures it directly -- for each malicious
     tool, how far up the escalation ladder can an attacker go before
     getting flagged, and what does that residual, undetected scope
     expansion actually grant them? This is the honest "here is what our
     method does NOT catch" section every reviewer looks for.

Usage:
    python -m eval.sensitivity_analysis
"""

from __future__ import annotations
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.embedder import get_embedder
from core.detector import DriftScorer
from attacks.drift_attack import MALICIOUS_ESCALATIONS, BENIGN_SEQUENCES

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# The calibrated value from eval/calibrate.py -- update if you recalibrate.
CUMULATIVE_THRESHOLD = 0.23


# ---------------------------------------------------------------------------
# 1. Granularity invariance
# ---------------------------------------------------------------------------

def granularity_invariance(embedder):
    """
    For each malicious tool, embed the baseline (v0) and final malicious
    version (vN), then walk a straight line between them in embedding
    space using different numbers of interpolation steps. Record, as a
    FRACTION of the way along that line, where the cumulative-drift
    check first fires. If the check is truly granularity-invariant,
    that fraction should be ~identical across step counts.
    """
    corpus = [v for _, vs in MALICIOUS_ESCALATIONS for v in vs]
    embedder.fit(corpus)

    step_counts = [2, 5, 10, 25, 50]
    results = {}

    for name, versions in MALICIOUS_ESCALATIONS:
        baseline = embedder.embed([versions[0]])[0]
        target = embedder.embed([versions[-1]])[0]

        per_granularity = {}
        for n in step_counts:
            scorer = DriftScorer(step_threshold=99, cumulative_threshold=99)  # observe only
            fire_fraction = None
            for i in range(1, n + 1):
                frac = i / n
                point = (1 - frac) * baseline + frac * target
                ev = scorer.observe(point)
                if fire_fraction is None and ev.cumulative_distance > CUMULATIVE_THRESHOLD:
                    fire_fraction = frac
            per_granularity[n] = fire_fraction

        fired_fractions = [f for f in per_granularity.values() if f is not None]
        results[name] = {
            "fire_fraction_by_steps": per_granularity,
            "spread": (max(fired_fractions) - min(fired_fractions)
                       if len(fired_fractions) > 1 else 0.0),
        }

    max_spread = max(r["spread"] for r in results.values())
    return {
        "per_tool": results,
        "max_spread_across_granularities": max_spread,
        "granularity_invariant": max_spread < 0.05,  # fires within +/-5% regardless of step count
    }


# ---------------------------------------------------------------------------
# 2. Safe escalation ceiling
# ---------------------------------------------------------------------------

def safe_escalation_ceiling(embedder):
    """
    For each malicious tool's real (hand-written) version sequence,
    find the last version an attacker could reach WITHOUT crossing the
    cumulative threshold -- i.e. the largest undetected escalation this
    detector permits, in the attacker's own words.
    """
    corpus = [v for _, vs in MALICIOUS_ESCALATIONS for v in vs]
    embedder.fit(corpus)

    results = {}
    for name, versions in MALICIOUS_ESCALATIONS:
        scorer = DriftScorer(step_threshold=99, cumulative_threshold=99)
        last_safe_index = 0
        last_safe_text = versions[0]
        for i, v in enumerate(versions):
            ev = scorer.observe(embedder.embed([v])[0])
            if ev.cumulative_distance <= CUMULATIVE_THRESHOLD:
                last_safe_index = i
                last_safe_text = v
            else:
                break
        results[name] = {
            "total_versions": len(versions),
            "max_safe_index": last_safe_index,
            "max_safe_description": last_safe_text,
            "fully_evades_detection": last_safe_index == len(versions) - 1,
        }
    return results


def plot_granularity(inv_results):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, r in inv_results["per_tool"].items():
        xs = sorted(r["fire_fraction_by_steps"].keys())
        ys = [r["fire_fraction_by_steps"][x] for x in xs]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("Number of interpolation steps (attack granularity)")
    ax.set_ylabel("Fraction of path traveled\nbefore detection fires")
    ax.set_ylim(0, 1.05)
    ax.set_title("Does salami-slicing help evade the cumulative check?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "granularity_invariance.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main():
    embedder = get_embedder()
    print(f"[embedder] using: {getattr(embedder, 'name', 'unknown')}\n")

    inv = granularity_invariance(embedder)
    print("=== Granularity invariance ===")
    print(json.dumps({k: v for k, v in inv.items() if k != "per_tool"}, indent=2))
    for name, r in inv["per_tool"].items():
        print(f"  {name}: fires at {r['fire_fraction_by_steps']}")

    ceiling = safe_escalation_ceiling(embedder)
    print("\n=== Safe escalation ceiling (what slips through undetected) ===")
    for name, r in ceiling.items():
        print(f"  {name}: safe up to version {r['max_safe_index']}/{r['total_versions']-1}"
              f" -> \"{r['max_safe_description'][:80]}...\"")

    p = plot_granularity(inv)
    print(f"\nfigure: {p}")

    out = os.path.join(HERE, "sensitivity_results.json")
    with open(out, "w") as f:
        json.dump({"embedder": getattr(embedder, "name", "unknown"),
                   "granularity_invariance": inv, "safe_escalation_ceiling": ceiling}, f, indent=2)
    print(f"results: {out}")


if __name__ == "__main__":
    main()
