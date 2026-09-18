# Temporal Embedding-Space Monitoring for AI Agent Security

One lightweight detector core, applied to two structurally different
attacks on AI agents.

**Sensor A — Tool Description Drift.** Tracks a tool's description across
sessions and flags when its *meaning* gradually shifts toward broader,
more permissive scope, even when each individual edit looks harmless.

**Sensor B — Memory Distortion.** Tracks new entries added to an agent's
long-term memory and flags entries positioned to crowd the embedding
space and degrade future retrieval.

Both share one pipeline: **embed → compare against history/population →
score → flag.** That shared core is the contribution: a single
generalisable consistency check catching two different attack classes.

---

## Setup (Windows / PowerShell)

```powershell
cd C:\Users\gungu\Projects\agent-embedding-security
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Note: use `python -m pip`, not bare `pip`.

---

## Run order

**1. Calibrate thresholds** (do this first, and again any time you change
the embedder or add data):

```powershell
python -m eval.calibrate
```

It sweeps thresholds against both malicious and benign sets, picks the
value maximising detection-minus-false-positives, and prints two numbers
to copy into `core/detector.py`.

**2. Run the experiments:**

```powershell
python -m eval.run_experiments
```

Writes metrics to `eval/results.json` and figures to `figures/`.

---

## What gets measured

Both sensors are evaluated against a **malicious** set and a **benign
control** set, so the output reports detection rate *and* false positive
rate together. A detector that blocks everything scores 100% detection
and 100% false positives — reporting only the first number would hide
that, which is why both appear side by side.

| Metric | Meaning |
|---|---|
| `attack_success_rate_off/on` | Did the attack land, without/with the detector |
| `detection_rate` | Fraction of malicious cases caught |
| `false_positive_rate` | Fraction of *benign* cases wrongly flagged |
| `precision_at_3` | Retrieval quality, clean vs poisoned vs defended |

---

## Layout

```
core/embedder.py        pluggable embedding backend (ST default, TF-IDF fallback)
core/detector.py        shared scoring logic — DriftScorer + DensityScorer
sensors/tool_drift.py         Sensor A
sensors/memory_distortion.py  Sensor B
attacks/drift_attack.py       malicious escalations + benign control sequences
attacks/poison_attack.py      clean memory, queries, poison, benign new entries
eval/calibrate.py             threshold sweep (run first)
eval/run_experiments.py       full evaluation + figures
figures/                      output plots
```

---

## Status

Working: full pipeline runs end to end, both attacks succeed when
unopposed, both sensors detect, calibration sweep functional.

Still to do:
- Re-run everything on sentence-transformers and use *those* numbers in
  the paper (TF-IDF is a fallback and is not semantically meaningful).
- Scale the data up — currently 5 malicious + 5 benign tool sequences,
  15 memory entries. Fine for a prototype, thin for an evaluation section.
- Add an adaptive attacker: how slowly must an attacker drift to stay
  under the calibrated threshold? That sensitivity analysis is the
  honest limit of the method and belongs in the paper.
- Build the Streamlit demo last, once thresholds are stable.

## Known result worth reporting

The memory poisoning attack only succeeds when the attacker injects
**multiple near-duplicate entries per target** (see `FLOOD_FACTOR`); a
single poison entry rarely outranks a genuine one. That requirement is a
finding in its own right, not an implementation detail.
