"""SHAP feature importance for the tree ensembles and feature-set reduction."""

import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from .config import N_JOBS, RANDOM_STATE, TOP_MODELS
from .datasets import Split
from .models import get_classifiers


@dataclass
class ShapResult:
    model: BaseEstimator
    accuracy: float
    importance: pd.Series
    shap_values: np.ndarray | list
    X_explained: pd.DataFrame
    class_names: list[str]
    elapsed: float


def per_class_list(shap_values) -> list[np.ndarray] | np.ndarray:
    """Normalise SHAP output to a list of (n_samples, n_features) arrays, one per class."""
    if isinstance(shap_values, list):
        return shap_values
    if shap_values.ndim == 3:
        return [shap_values[:, :, c] for c in range(shap_values.shape[2])]
    return shap_values


def mean_abs_shap(shap_values) -> np.ndarray:
    """Mean |SHAP| per feature, averaged over samples and classes."""
    values = per_class_list(shap_values)
    if isinstance(values, list):
        return np.mean([np.abs(v).mean(axis=0) for v in values], axis=0)
    return np.abs(values).mean(axis=0)


def explain_models(
    split: Split,
    models: dict[str, BaseEstimator] | None = None,
    max_samples: int | None = None,
    seed: int = RANDOM_STATE,
    n_jobs: int = N_JOBS,
    verbose: bool = True,
) -> dict[str, ShapResult]:
    """Fit each tree model on the training split and compute TreeSHAP on the test split.

    Trees are scale-invariant, so models are fit on unscaled features. Models are
    processed one at a time, each using ``n_jobs`` threads.
    """
    if models is None:
        models = {name: est for name, (est, _) in get_classifiers(TOP_MODELS).items()}

    encoder = LabelEncoder().fit(pd.concat([split.y_train, split.y_test]))
    y_train, y_test = encoder.transform(split.y_train), encoder.transform(split.y_test)
    class_names = list(encoder.classes_)

    X_explain = split.X_test
    if max_samples and len(X_explain) > max_samples:
        X_explain = X_explain.sample(max_samples, random_state=seed)

    results = {}
    for name, estimator in models.items():
        start = time.perf_counter()
        model = clone(estimator)
        if "n_jobs" in model.get_params():
            model.set_params(n_jobs=n_jobs)
        model.fit(split.X_train, y_train)
        acc = accuracy_score(y_test, model.predict(split.X_test))

        values = shap.TreeExplainer(model).shap_values(X_explain)
        importance = pd.Series(mean_abs_shap(values), index=X_explain.columns, name=name)
        elapsed = time.perf_counter() - start

        results[name] = ShapResult(model, acc, importance.sort_values(ascending=False),
                                   values, X_explain, class_names, elapsed)
        if verbose:
            print(f"{name:<16} acc={acc:.4f}  ({elapsed:.1f}s)")
    return results


def combine_importances(results: dict[str, ShapResult], normalize: bool = True) -> pd.DataFrame:
    """One column per model plus ``Mean Importance``.

    With ``normalize`` each model's importances are rescaled to sum to 1 so models
    with different SHAP scales (e.g. LightGBM log-odds vs. forest probabilities)
    contribute equally to the mean.
    """
    table = pd.DataFrame({name: r.importance for name, r in results.items()})
    if normalize:
        table = table / table.sum()
    table["Mean Importance"] = table.mean(axis=1)
    return table.sort_values("Mean Importance", ascending=False).rename_axis("Feature").reset_index()


def top_k_features(combined: pd.DataFrame, k: int = 10) -> list[str]:
    return combined.nlargest(k, "Mean Importance")["Feature"].tolist()


def plot_shap_summary(result: ShapResult, top_k: int = 10, title: str | None = None) -> plt.Figure:
    """SHAP bar summary of the ``top_k`` most important features for one model."""
    top = result.importance.head(top_k).index.tolist()
    idx = [result.X_explained.columns.get_loc(f) for f in top]
    values = per_class_list(result.shap_values)
    values = [v[:, idx] for v in values] if isinstance(values, list) else values[:, idx]

    plt.figure()
    shap.summary_plot(values, result.X_explained[top], plot_type="bar",
                      class_names=result.class_names, show=False)
    fig = plt.gcf()
    ax = plt.gca()
    if title:
        ax.set_title(title)
    ax.set_xlabel("")
    fig.tight_layout()
    return fig
