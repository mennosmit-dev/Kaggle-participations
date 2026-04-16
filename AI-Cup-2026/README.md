# AI Cup 2026 (Birds)

Bird-species classification and decision support for wind-farm operations.

This project contains our work for **AI Cup 2026**, a Dutch student AI competition focused on building meaningful AI for a real societal challenge. The 2026 challenge asked participants to model bird movement in wind-farm airspace and support smarter operational decisions that reduce collisions while preserving energy production.

We approached the competition in **two parts**:

1. **Performance Track** – predictive modeling on the competition dataset
2. **Implementation Track** – **RadarBird**, a deployment proposal for a real-world radar-based bird classification system

## Result

- **Public leaderboard:** **Top 9.5%**
- You can present this as **9 / 95 teams** if that is how you want to report it in your main repo table.

## Competition overview

AI Cup 2026 combined a classic modeling challenge with an implementation component. Besides model accuracy, teams could also compete on how responsibly and realistically their system could be deployed in practice.

### Performance Track
The modeling side focused on predicting bird classes from track-level and contextual data.

### Implementation Track
The implementation side asked teams to design an AI system that could actually be used in the field, with attention to reliability, uncertainty, operational impact, and responsible deployment.

## Approaches

## 1) Tabular ensemble baseline

File: `src/performance_tabular_ensemble.py`

This script is the clean export of the first notebook. It builds a **tabular ensemble** around boosted trees and contextual feature engineering.

### Main ideas
- Trajectory-level feature extraction from track data
- Temporal and geometric features
- Environmental/context features such as:
  - weather
  - weekly priors
  - coastal proximity
  - sun/moon related information
- Label encoding and strict evaluation
- Ensemble of **LightGBM** and **CatBoost**

### Why this approach
Tabular boosting models are usually a strong starting point for structured competition data. They train quickly, handle heterogeneous features well, and give a reliable baseline before moving to more complex sequence models.

## 2) Hard-negative baseline + sequence blend

Files:
- `src/performance_hardneg_baseline.py`
- `src/performance_xgb_sequence_blend.py`

This part contains the more advanced iteration from the second notebook.

### Hard-negative baseline
The baseline script adds stronger evaluation and ensembling choices, including:
- blocked time-based folds
- AP-oriented blending
- hard-negative handling
- contextual priors and environmental covariates

### Extended blend
The wrapper script builds on the baseline and adds:
- **XGBoost**
- sequence modeling components
- a blended prediction stage combining tabular and sequential signals

In short, this version moves from “strong tabular baseline” toward a **hybrid ensemble** that tries to capture both static and temporal patterns in bird tracks.

## RadarBird – Implementation Track

File:
- `docs/RadarBird.pdf`

**RadarBird** is our implementation-track proposal: a radar-based bird species classification and operational decision-support system for wind farms.

### Core idea
RadarBird uses existing radar infrastructure to classify bird groups in real time and convert those predictions into confidence-aware recommendations for wind-farm operators.

### What makes it different
- **Reactive mode:** responds when birds are currently detected
- **Predictive mode:** pre-alerts operators when migration risk is high before the peak event starts

### Proposed system components
- radar track ingestion
- track-level feature extraction
- bird-group classification
- uncertainty scoring and abstention
- tiered operational recommendations
- active-learning feedback loop for future retraining

### Modeling direction in the proposal
The proposal describes a system centered on:
- gradient-boosted tree ensembles
- a logistic meta-stacker
- future Temporal Convolutional Network integration
- calibrated confidence thresholds
- abstention for uncertain predictions

### Responsible-AI design choices
A major focus of RadarBird is that predictions should not automatically trigger actions unless they are both ecologically important and reliable enough in validation. The proposal therefore includes:
- confidence tiers
- explicit uncertainty handling
- phased rollout
- human-in-the-loop validation
- conservative fallback behavior

## Project structure

```text
ai-cup-2026-birds/
├── README.md
├── requirements.txt
├── docs/
│   └── RadarBird.pdf
└── src/
    ├── performance_tabular_ensemble.py
    ├── performance_hardneg_baseline.py
    └── performance_xgb_sequence_blend.py
```

## Notes

- The scripts were cleaned from Kaggle notebook format and converted into standalone `.py` files for GitHub.
- Some paths still reflect Kaggle defaults and may need small local path changes before running outside Kaggle.
- The implementation-track proposal is included separately because it is conceptually different from the leaderboard modeling work.

## Links

- Competition: https://www.teamepoch.ai/ai-cup-2026/
- Performance track: Kaggle competition page
