"""
Runs both attacks with the detector OFF (passive logging) and ON (blocking),
against BOTH malicious and benign inputs, and reports:

  - attack success rate  (did the escalation / poisoning land?)
  - detection rate       (did we catch the malicious ones?)
  - false positive rate  (did we wrongly flag the benign ones?)
  - retrieval quality    (precision@3 before vs after)

A detector that blocks everything scores perfectly on detection and
terribly on false positives, which is why both are reported together.

Usage:
    python -m eval.run_experiments
"""

from __future__ import annotations
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.embedder import get_embedder
from sensors.tool_drift import ToolDriftMonitor
from sensors.memory_distortion import MemoryDistortionMonitor
from attacks.drift_attack import MALICIOUS_ESCALATIONS, BENIGN_SEQUENCES
from attacks.poison_attack import (
    CLEAN_MEMORY, TEST_QUERIES, POISON_TEMPLATES, BENIGN_NEW_ENTRIES,
    FLOOD_FACTOR, retrieve, precision_at_k,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# Experiment 1 -- tool description drift
# --------------------------------------------------------------------------

def experiment_drift(embedder):
    corpus = [v for _, vs in MALICIOUS_ESCALATIONS + BENIGN_SEQUENCES for v in vs]
    embedder.fit(corpus)

    def run(sequences, blocking: bool):
        out = {}
        monitor = ToolDriftMonitor(embedder=embedder)
        for name, versions in sequences:
            accepted, blocked_at = [], None
            for i, v in enumerate(versions):
                ev = monitor.observe_version(name, v)
                if blocking and ev.flagged:
                    blocked_at = i
                    break  # update rejected; agent keeps last accepted version
                accepted.append(v)
            out[name] = {
                "versions_total": len(versions),
                "last_accepted_index": len(accepted) - 1,
                "reached_final_version": len(accepted) == len(versions),
                "blocked_at_step": blocked_at,
            }
        return out

    mal_off = run(MALICIOUS_ESCALATIONS, blocking=False)
    mal_on = run(MALICIOUS_ESCALATIONS, blocking=True)
    ben_on = run(BENIGN_SEQUENCES, blocking=True)

    n_mal = len(MALICIOUS_ESCALATIONS)
    n_ben = len(BENIGN_SEQUENCES)
    detected = sum(1 for r in mal_on.values() if r["blocked_at_step"] is not None)
    false_pos = sum(1 for r in ben_on.values() if r["blocked_at_step"] is not None)

    return {
        "malicious_detector_off": mal_off,
        "malicious_detector_on": mal_on,
        "benign_detector_on": ben_on,
        "summary": {
            "attack_success_rate_off": sum(
                1 for r in mal_off.values() if r["reached_final_version"]) / n_mal,
            "attack_success_rate_on": sum(
                1 for r in mal_on.values() if r["reached_final_version"]) / n_mal,
            "detection_rate": detected / n_mal,
            "false_positive_rate": false_pos / n_ben,
        },
    }


# --------------------------------------------------------------------------
# Experiment 2 -- memory poisoning
# --------------------------------------------------------------------------

def experiment_poison(embedder):
    all_text = ([t for _, t in CLEAN_MEMORY]
                + [t for _, t in POISON_TEMPLATES]
                + [t for _, t in BENIGN_NEW_ENTRIES]
                + [q for q, _ in TEST_QUERIES])
    embedder.fit(all_text)

    clean_ids = [i for i, _ in CLEAN_MEMORY]
    clean_texts = [t for _, t in CLEAN_MEMORY]
    clean_vecs = list(embedder.embed(clean_texts))

    def measure(ids, vecs):
        precisions, times = [], []
        arr = np.array(vecs)
        for q, expected in TEST_QUERIES:
            qv = embedder.embed([q])[0]
            runs = []
            retrieved = None
            for _ in range(200):  # average out sub-ms timing noise
                retrieved, el = retrieve(qv, ids, arr, k=3)
                runs.append(el)
            precisions.append(precision_at_k(retrieved, expected, k=3))
            times.append(float(np.mean(runs)))
        return {
            "precision_at_3": float(np.mean(precisions)),
            "avg_latency_ms": float(np.mean(times) * 1000),
        }

    baseline = measure(clean_ids, clean_vecs)

    # attacker floods each target region with near-duplicate poison entries
    flood = [(f"{pid}_{r}", txt)
             for pid, txt in POISON_TEMPLATES
             for r in range(FLOOD_FACTOR)]

    results = {"baseline_clean": baseline}
    for mode in ("off", "on"):
        monitor = MemoryDistortionMonitor(embedder=embedder)
        monitor.calibrate(clean_texts)

        ids, vecs = list(clean_ids), list(clean_vecs)
        poison_blocked = 0
        for pid, txt in flood:
            ev = monitor.observe_entry(txt)
            if mode == "on" and ev.flagged:
                poison_blocked += 1
                continue
            ids.append(pid)
            vecs.append(embedder.embed([txt])[0])

        results[mode] = {
            "poison_admitted": len(ids) - len(clean_ids),
            "poison_blocked": poison_blocked,
            **measure(ids, vecs),
        }

    # false positives: legitimate entries added to a clean store
    fp_monitor = MemoryDistortionMonitor(embedder=embedder)
    fp_monitor.calibrate(clean_texts)
    benign_flagged = sum(
        1 for _, t in BENIGN_NEW_ENTRIES if fp_monitor.observe_entry(t).flagged)

    results["summary"] = {
        "poison_detection_rate": results["on"]["poison_blocked"] / len(flood),
        "false_positive_rate": benign_flagged / len(BENIGN_NEW_ENTRIES),
        "precision_drop_without_detector":
            baseline["precision_at_3"] - results["off"]["precision_at_3"],
    }
    return results


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def plot_all(drift, poison):
    paths = []

    names = [n for n, _ in MALICIOUS_ESCALATIONS]
    off = [drift["malicious_detector_off"][n]["last_accepted_index"] for n in names]
    on = [drift["malicious_detector_on"][n]["last_accepted_index"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, off, 0.4, label="Detector OFF", color="#d9534f")
    ax.bar(x + 0.2, on, 0.4, label="Detector ON", color="#5cb85c")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Last accepted version\n(higher = more escalation)")
    ax.set_title("Tool description drift: how far the escalation gets")
    ax.legend(); fig.tight_layout()
    p = os.path.join(FIG_DIR, "drift_comparison.png"); fig.savefig(p, dpi=150)
    plt.close(fig); paths.append(p)

    labels = ["Clean\n(baseline)", "Poisoned\ndetector OFF", "Poisoned\ndetector ON"]
    vals = [poison["baseline_clean"]["precision_at_3"],
            poison["off"]["precision_at_3"],
            poison["on"]["precision_at_3"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals, color=["#5bc0de", "#d9534f", "#5cb85c"])
    ax.set_ylim(0, 1.08); ax.set_ylabel("Avg precision@3")
    ax.set_title("Retrieval quality under memory poisoning")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "poison_comparison.png"); fig.savefig(p, dpi=150)
    plt.close(fig); paths.append(p)

    fig, ax = plt.subplots(figsize=(7, 4))
    groups = ["Tool drift", "Memory poisoning"]
    det = [drift["summary"]["detection_rate"], poison["summary"]["poison_detection_rate"]]
    fpr = [drift["summary"]["false_positive_rate"], poison["summary"]["false_positive_rate"]]
    x = np.arange(len(groups))
    ax.bar(x - 0.2, det, 0.4, label="Detection rate", color="#5cb85c")
    ax.bar(x + 0.2, fpr, 0.4, label="False positive rate", color="#f0ad4e")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Rate")
    ax.set_title("Detection vs false positives (both sensors)")
    ax.legend(); fig.tight_layout()
    p = os.path.join(FIG_DIR, "detection_vs_fpr.png"); fig.savefig(p, dpi=150)
    plt.close(fig); paths.append(p)

    return paths


def main():
    embedder = get_embedder()
    print(f"[embedder] using: {getattr(embedder, 'name', 'unknown')}\n")

    drift = experiment_drift(embedder)
    poison = experiment_poison(embedder)

    print("=== Tool Drift ===")
    print(json.dumps(drift["summary"], indent=2))
    print("\n=== Memory Poisoning ===")
    print(json.dumps({"baseline": poison["baseline_clean"],
                      "off": poison["off"], "on": poison["on"],
                      "summary": poison["summary"]}, indent=2))

    for p in plot_all(drift, poison):
        print(f"figure: {p}")

    out = os.path.join(HERE, "results.json")
    with open(out, "w") as f:
        json.dump({"embedder": getattr(embedder, "name", "unknown"),
                   "drift": drift, "poison": poison}, f, indent=2)
    print(f"results: {out}")


if __name__ == "__main__":
    main()
