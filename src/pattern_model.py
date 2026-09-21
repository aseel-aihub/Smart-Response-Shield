"""
Pattern-analysis model (Objective 3).

Learns, from historical launch parameters alone, which sector a threat is most
likely to be heading for — and, crucially, with what confidence.

DESIGN NOTES
------------
1. Nothing trains at import time. The first prototype executed
   `model.fit(...)` at module level, so every Streamlit rerun retrained the
   forest from scratch and printed to the console. Training now lives behind
   an explicit `train()` call driven by scripts/train_model.py, and the app
   loads a persisted artefact.

2. The model is evaluated against an explicit majority-class baseline. For a
   classifier whose whole purpose is to justify the word "AI" in the project
   title, showing that it beats a constant guess is the only meaningful claim,
   and it is exactly the check that exposed the original bug.

3. Stratified splitting keeps the sector proportions identical in train and
   test, which matters because the classes are imbalanced by construction
   (the geometry sends more threats into some sectors than others).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

from config import (
    MODEL_METRICS_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    THREAT_DATA_CSV,
)
from src.data_generation import FEATURE_COLUMNS

__all__ = [
    "ModelMetrics",
    "train",
    "load_model",
    "load_metrics",
    "predict_sector",
]


@dataclass
class ModelMetrics:
    accuracy: float
    baseline_accuracy: float
    macro_f1: float
    cv_mean: float
    cv_std: float
    n_train: int
    n_test: int
    labels: list
    confusion: list
    report: dict
    feature_importances: dict

    @property
    def lift_over_baseline(self) -> float:
        """Percentage points gained over always predicting the commonest class."""
        return self.accuracy - self.baseline_accuracy


def train(
    data_path=THREAT_DATA_CSV,
    model_path=MODEL_PATH,
    metrics_path=MODEL_METRICS_PATH,
    n_estimators: int = 200,
    # Depth 14 matches the unbounded forest on cross-validation (83.8% vs 84.0%)
    # while cutting the serialised artefact by a third.
    max_depth: int | None = 14,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
    save: bool = True,
) -> tuple[RandomForestClassifier, ModelMetrics]:
    """Fit the classifier, evaluate it, and optionally persist both."""
    df = pd.read_csv(data_path)

    X = df[FEATURE_COLUMNS]
    y = df["target_sector"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # The honest yardstick: what would predicting the commonest class achieve?
    majority_class = y_train.value_counts().idxmax()
    baseline = accuracy_score(y_test, np.full(len(y_test), majority_class))

    cv = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    labels = sorted(y.unique().tolist())

    metrics = ModelMetrics(
        accuracy=float(accuracy_score(y_test, y_pred)),
        baseline_accuracy=float(baseline),
        macro_f1=float(f1_score(y_test, y_pred, average="macro")),
        cv_mean=float(cv.mean()),
        cv_std=float(cv.std()),
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
        labels=labels,
        confusion=confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        report=classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        feature_importances=dict(
            zip(FEATURE_COLUMNS, [float(v) for v in model.feature_importances_])
        ),
    )

    if save:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        metrics_path.write_text(json.dumps(asdict(metrics), indent=2))

    return model, metrics


def load_model(model_path=MODEL_PATH) -> RandomForestClassifier:
    """Load the persisted classifier, training one first if none exists."""
    if not model_path.exists():
        model, _ = train()
        return model
    return joblib.load(model_path)


def load_metrics(metrics_path=MODEL_METRICS_PATH) -> ModelMetrics | None:
    if not metrics_path.exists():
        return None
    return ModelMetrics(**json.loads(metrics_path.read_text()))


def predict_sector(model, launch_lat, launch_lon, speed, angle_deg, heading_deg):
    """
    Predict the likely target sector and return the full probability vector.

    The probabilities matter more than the label here: a 41% top prediction and
    a 95% top prediction call for very different responses, and the dashboard
    surfaces that distinction instead of hiding it behind a single answer.
    """
    row = pd.DataFrame(
        [[launch_lat, launch_lon, speed, angle_deg, heading_deg]],
        columns=FEATURE_COLUMNS,
    )

    label = model.predict(row)[0]
    proba = dict(zip(model.classes_, model.predict_proba(row)[0].astype(float)))

    return label, proba
