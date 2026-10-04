"""Single registry of the twelve classifiers and their hyperparameter grids."""

from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import RANDOM_STATE

ModelSpec = tuple[BaseEstimator, dict[str, list]]

SCALERS = {
    "minmax": MinMaxScaler,
    "standard": StandardScaler,
}


def get_classifiers(names: list[str] | None = None, random_state: int = RANDOM_STATE) -> dict[str, ModelSpec]:
    """Return ``{name: (estimator, param_grid)}``; grid keys target the ``clf`` pipeline step."""
    rs = random_state
    # Every model trains single-threaded: parallelism comes from running many
    # fits at once (GridSearchCV / joblib), and nested threading would oversubscribe.
    registry: dict[str, ModelSpec] = {
        # liblinear is binary-only in scikit-learn >= 1.8, so it is not in the grid
        "Logistic Regression": (LogisticRegression(max_iter=1000), {
            "clf__C": [0.01, 0.1, 1, 10],
        }),
        "Naive Bayes": (GaussianNB(), {}),
        "K-Nearest Neighbors": (KNeighborsClassifier(), {
            "clf__n_neighbors": [3, 5, 7, 11],
            "clf__weights": ["uniform", "distance"],
        }),
        "Decision Tree": (DecisionTreeClassifier(random_state=rs), {
            "clf__max_depth": [15, 25, 35],
            "clf__min_samples_split": [2, 5, 10],
            "clf__criterion": ["gini", "entropy"],
        }),
        "Random Forest": (RandomForestClassifier(random_state=rs), {
            "clf__n_estimators": [500, 1000],
            "clf__max_depth": [50, None],
            "clf__max_features": ["sqrt", "log2"],
        }),
        "Extra Trees": (ExtraTreesClassifier(random_state=rs), {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [None, 50],
        }),
        "AdaBoost": (AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), random_state=rs), {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.01, 0.1],
        }),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=rs), {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [3, 4],
        }),
        "XGBoost": (XGBClassifier(random_state=rs, n_jobs=1), {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [3, 4],
        }),
        "LightGBM": (LGBMClassifier(random_state=rs, verbose=-1, n_jobs=1), {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.05, 0.1],
            "clf__num_leaves": [15, 31],
        }),
        # Platt-calibrated SVC; replaces SVC(probability=True), deprecated in scikit-learn 1.9
        "Support Vector Machine": (CalibratedClassifierCV(SVC(random_state=rs), ensemble=False), {
            "clf__estimator__C": [0.1, 1],
            "clf__estimator__gamma": ["scale", "auto"],
        }),
        "Neural Network": (MLPClassifier(max_iter=5000, random_state=rs), {
            "clf__hidden_layer_sizes": [(50, 10), (100, 50)],
            "clf__alpha": [0.005, 0.001],
            "clf__learning_rate": ["constant", "adaptive"],
        }),
    }
    if names is None:
        return registry
    unknown = set(names) - set(registry)
    if unknown:
        raise KeyError(f"Unknown classifiers: {sorted(unknown)}. Available: {list(registry)}")
    return {name: registry[name] for name in names}


def build_pipeline(estimator: BaseEstimator, scaler: str | None = "minmax") -> Pipeline:
    """Scaler (fit on training data only) followed by a fresh clone of ``estimator``."""
    step = SCALERS[scaler]() if scaler else "passthrough"
    return Pipeline([("scaler", step), ("clf", clone(estimator))])
