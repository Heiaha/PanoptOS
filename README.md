# PanoptOS: public research artifact

PanoptOS investigates early detection of automated Old School RuneScape
accounts using only public Hiscores time series and delayed labels obtained
from account disappearance after Jagex enforcement. This repository is the
public companion to the [technical report](docs/report.pdf); a less formal
[adaptation](docs/informal.md) is also included.

**This is a research artifact, not a running system.** PanoptOS *does* run
as a live continuous-crawl pipeline with model inference deployed on a
Raspberry Pi 5, but that production system is not what this repository
contains. What is published here is the research subset: the methodology,
model architecture, feature construction, training code, and a reference
implementation of the active-discovery policy against a synthetic
environment.

What is **deliberately not published**, even though it exists and works:

- The production orchestration: live HTTP client, async worker loops,
  rate-limit tuning, proxy configuration, database schema and writes,
  Docker deployment scaffolding, health/restart logic.
- Trained model weights and ONNX artifacts.
- Raw data, account names, and per-account scores.
- Account inference tooling.

Nothing in this repository will, by itself, score named players or crawl
the live Hiscores.

Two framing notes for readers and downstream users:

- The model produces a **statistical risk score**, not an accusation about
  any individual account. The numbers reported in the technical report are
  aggregated.
- This work is **complementary** to detection systems Jagex operates
  internally: Jagex has access to client telemetry, payment metadata, IP
  history, and behavioral signals at a scale that no external observer can
  approach. PanoptOS demonstrates only that *some* signal is recoverable
  from public Hiscores alone.

## What's here

```
panoptos/
  discovery/        # Active-discovery policy + offline simulation
    policy.py       #   Probability matching over Bayesian utility estimates
    simulation.py   #   Synthetic environment for offline evaluation
  training/         # Model, features, and training pipeline
    constants.py    #   Skill/activity inventory + character encoding
    features.py     #   Per-snapshot feature construction
    model.py        #   Dual-branch transformer + name encoder + Platt head
    dataset.py      #   Streaming Parquet IterableDataset
    train.py        #   Training loop with online Welford scaler + Platt fit
docs/
  report.pdf        # Technical report
  informal.md       # Informal adaptation
```

## Quick start: discovery simulation

The active-discovery policy can be exercised against a synthetic environment
with no external dependencies and no network access:

```bash
python -m panoptos.discovery.simulation
```

This runs the policy against a small toy population of tables with varying
bot rates and saturation, and prints the per-table pull allocation that the
bandit converges to.

## Citation

If you build on this work, please cite the technical report (see `docs/`).
