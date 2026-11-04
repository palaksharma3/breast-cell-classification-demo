"""Deterministic educational baseline; no patient inputs are written to disk."""

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

DATA = load_breast_cancer()
FEATURE_NAMES = tuple(str(name) for name in DATA.feature_names)
CLASS_NAMES = {0: "Malignant", 1: "Benign"}
LOWER_BOUNDS = DATA.data.min(axis=0)
UPPER_BOUNDS = DATA.data.max(axis=0)
EXAMPLES = {"Public example A (dataset row 0)": 0, "Public example B (dataset row 19)": 19}


@dataclass(frozen=True)
class ModelBundle:
    estimator: GaussianNB
    accuracy: float
    malignant_precision: float
    malignant_recall: float
    confusion: np.ndarray
    train_size: int
    test_size: int


def train_model() -> ModelBundle:
    """Keep the original split and algorithm; evaluate actual held-out predictions."""
    x_train, x_test, y_train, y_test = train_test_split(
        DATA.data, DATA.target, test_size=0.33, random_state=42
    )
    estimator = GaussianNB().fit(x_train, y_train)
    predicted = estimator.predict(x_test)
    return ModelBundle(
        estimator=estimator,
        accuracy=float(accuracy_score(y_test, predicted)),
        malignant_precision=float(precision_score(y_test, predicted, pos_label=0, zero_division=0)),
        malignant_recall=float(recall_score(y_test, predicted, pos_label=0, zero_division=0)),
        confusion=confusion_matrix(y_test, predicted, labels=[0, 1]),
        train_size=len(y_train),
        test_size=len(y_test),
    )


def validate_measurements(values: Mapping[str, object]) -> np.ndarray:
    """Require every named feature and reorder explicitly, never by dict order."""
    if set(values) != set(FEATURE_NAMES):
        raise ValueError("Provide exactly the 30 named features; no missing or extra fields.")
    ordered = []
    for index, name in enumerate(FEATURE_NAMES):
        value = values[name]
        if value is None or isinstance(value, (bool, str, bytes)):
            raise ValueError(f"{name}: enter a numeric measurement; no blank fields.")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name}: enter a finite numeric measurement.") from exc
        if not np.isfinite(number):
            raise ValueError(f"{name}: enter a finite numeric measurement.")
        lower, upper = float(LOWER_BOUNDS[index]), float(UPPER_BOUNDS[index])
        if not lower <= number <= upper:
            raise ValueError(f"{name}: this demo accepts {lower:g} to {upper:g} (observed dataset range).")
        ordered.append(number)
    return np.asarray([ordered], dtype=np.float64)


def predict_measurements(bundle: ModelBundle, values: Mapping[str, object]) -> tuple[int, str]:
    vector = validate_measurements(values)
    predicted_class = int(bundle.estimator.predict(vector)[0])
    return predicted_class, CLASS_NAMES[predicted_class]


def example_values(label: str) -> dict[str, float]:
    row = DATA.data[EXAMPLES[label]]
    return dict(zip(FEATURE_NAMES, (float(value) for value in row)))
