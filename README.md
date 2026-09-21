# 🛡️ Smart Response Shield

**AI-driven predictive impact assessment and self-healing infrastructure — a simulation.**

Graduation (Senior) Project · Information Systems
Submitted to the Prince Sultan Defense Studies & Research Center (PSDSARC) graduation project programme.
Topic area: *Artificial Intelligence — Intelligent Support Systems*.

---

## ⚠️ Scope and data provenance

This project is an **academic simulation built entirely on synthetic data**.

* Every coordinate, facility, asset name and track in this repository is **fictional**. The "Area of Operations" is an arbitrary rectangle used only to give the simulation a consistent coordinate frame. No facility here corresponds to any real installation.
* There is **no live sensor feed, no radar integration and no restricted or classified data** of any kind.
* All training data is generated locally by `scripts/generate_dataset.py` from the physics model in `src/ballistics.py`.
* The system produces **advisory text only**. No component actuates, controls or commands anything.
* The physics used is standard undergraduate projectile motion — the same equations that describe any falling object — applied to the **defensive** question of *where will this land, and how long do we have to evacuate*.

---

## What it does

| # | Capability | Module |
|---|---|---|
| 1 | Predicts the flight path and impact point of an incoming unpowered object | `src/ballistics.py`, `src/geo.py` |
| 2 | Identifies which protected facility is at risk | `src/facilities.py` |
| 3 | Learns which sector is historically targeted, with calibrated confidence | `src/pattern_model.py` |
| 4 | Issues graded, **explainable** protective recommendations | `src/response_rules.py` |
| 5 | Reroutes utility supply automatically when nodes are lost | `src/self_healing.py` |

---

## Quick start

```bash
# 1. Environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Generate the synthetic asset register and labelled dataset
python scripts/generate_dataset.py

# 3. Train and evaluate the classifier
python scripts/train_model.py

# 4. Launch the dashboard
streamlit run app.py

# 5. Run the test suite
pytest -v
```

---

## Results

```
Test accuracy       :  86.00%
Majority baseline   :  37.83%
Lift over baseline  : +48.17 points
Macro F1            :   0.842
5-fold CV accuracy  :  83.83% ± 1.95
```

**Why the baseline column matters more than the accuracy column.** An earlier
iteration of this project reported a working model that scored **40.0%** — against
a majority-class baseline of **40.3%**. It was performing *fractionally worse than
always answering "North Sector"*, because the dataset generator drew its labels
with `np.random.choice`, independently of any input feature. No model can learn a
mapping that does not exist.

The fix was in the data, not the model: labels are now **derived from the physics**
(launch parameters → integrated trajectory → impact point → nearest facility →
that facility's sector), so the target is a genuine non-linear function of the
inputs. `tests/test_core.py::TestModel::test_model_beats_baseline` is the
permanent regression guard against that defect returning.

Accuracy is deliberately capped below 100% by an 8% label-noise rate representing
tracking error and evasive manoeuvring. **A result near 100% here would indicate
leakage between features and labels, not a good model.**

---

## Architecture

```
smart_response_shield/
├── app.py                      # Streamlit dashboard (4 tabs)
├── config.py                   # All constants and thresholds in one place
├── data/
│   ├── facilities.csv          # Synthetic asset register (generated)
│   └── synthetic_threat_data.csv
├── models/                     # Trained artefacts (gitignored, regenerable)
├── src/
│   ├── geo.py                  # Haversine, bearing, destination point
│   ├── ballistics.py           # RK2 trajectory integration with drag
│   ├── facilities.py           # Asset register + nearest-neighbour matching
│   ├── data_generation.py      # Physics-consistent dataset generator
│   ├── pattern_model.py        # Random Forest: train / evaluate / persist
│   ├── response_rules.py       # Explainable rule engine
│   └── self_healing.py         # Graph rerouting with N-1 redundancy
├── scripts/
│   ├── generate_dataset.py
│   └── train_model.py
└── tests/test_core.py          # 44 tests
```

### Design decision: why the recommendation layer is *not* machine learning

Prediction is statistical and belongs in a model. Deciding whether to order an
evacuation is a consequential decision that must be **auditable after the fact** —
a reviewing officer has to be able to ask *"why did it recommend that?"* and get an
answer in plain language. Every assessment therefore returns the exact rules that
fired, and every threshold is declared in `config.py` rather than buried in learned
weights. This is a defensible engineering position, not a shortcut.

### Design decision: N-1 redundancy

Only criticality-4-and-above facilities were given a backup feeder. This asymmetry
is intentional, and it produces the project's most useful finding: losing
`Substation_North` leaves service availability at **91.7%** — the two criticality-5
assets behind it self-heal onto their backup paths, while `NTH-03` (criticality 3,
no redundancy) goes dark. The simulation therefore **quantifies what a redundancy
investment is worth**, which is a more interesting result than simply showing that
rerouting works.

---

## Known limitations

These are stated plainly because a capstone that overclaims is easier to challenge
than one that scopes itself honestly.

* Trajectories assume a non-manoeuvring object, a constant drag coefficient and a
  stationary atmosphere. No wind, no thrust, no terminal guidance.
* The spherical-earth approximation carries ~0.5% error at these ranges — negligible
  beside the kinematic uncertainty, but it is an approximation.
* Labels derive from geometry, so the classifier learns a **physical relationship**,
  not adversarial intent. Real targeting behaviour would not be recoverable from
  launch kinematics alone.
* The network model is **topological**: it does not model load flow, generation
  capacity or transfer limits. "Rerouted" means a path exists, not that the path
  could carry the load.
* Reported accuracy is bounded above by the label-noise rate.

---

## Changes from the first prototype

| Issue | Status |
|---|---|
| Classifier trained on randomly-generated labels (learned nothing) | **Fixed** — labels now derived from physics; +48.2 pt lift |
| Nearest-facility matching (Objective 2) absent | **Implemented** — vectorised haversine |
| Protective recommendations (Objective 4) absent | **Implemented** — explainable rule engine |
| Model retrained on every UI interaction | **Fixed** — persisted artefact + `st.cache_resource` |
| Modules executed demos and printed at import time | **Fixed** — no side effects on import |
| Reroute used a hard-coded source/target pair | **Fixed** — per-facility routing from surviving sources |
| Directed graph blocked legitimate reroutes | **Fixed** — undirected, matching bidirectional links |
| Flat `/111000` coordinate conversion | **Fixed** — proper geodesic destination-point |
| No atmospheric drag; no time-to-impact | **Added** — RK2 integration |
| `venv/` (676 MB) committed to the repository | **Removed** — `.gitignore` added |
| No tests, no README, no requirements.txt | **Added** — 44 tests |

---

## Licence

Academic coursework. Not for operational use.
