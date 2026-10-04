"""Model fitting, metrics, repeated experiments and statistical comparison."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .config import CV_FOLDS, N_RUNS, RANDOM_STATE, TOP_MODELS
from .datasets import BinarySplits, Split
from .models import ModelSpec, build_pipeline

METRICS = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc"]


def multiclass_metrics(y_true, y_pred, y_prob=None, n_classes: int | None = None) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": np.nan,
    }
    if y_prob is not None:
        try:
            if y_prob.shape[1] == 2:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                labels = np.arange(n_classes or y_prob.shape[1])
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", labels=labels)
        except ValueError:
            pass
    return metrics


def multilabel_metrics(Y_true, Y_pred) -> dict[str, float]:
    Y_true, Y_pred = np.asarray(Y_true), np.asarray(Y_pred)
    return {
        "subset_accuracy": accuracy_score(Y_true, Y_pred),
        "hamming_loss": hamming_loss(Y_true, Y_pred),
        "label_accuracy": 1 - hamming_loss(Y_true, Y_pred),
        "jaccard_samples": jaccard_score(Y_true, Y_pred, average="samples", zero_division=0),
        "f1_micro": f1_score(Y_true, Y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(Y_true, Y_pred, average="macro", zero_division=0),
    }


def fit_model(
    estimator: BaseEstimator,
    param_grid: dict,
    X,
    y,
    *,
    scaler: str | None = "minmax",
    cv: int = CV_FOLDS,
    best_params: dict | None = None,
    scoring: str = "accuracy",
    n_jobs: int = -1,
    seed: int = RANDOM_STATE,
) -> tuple[Pipeline, dict]:
    """Fit a scaler+classifier pipeline, grid-searching unless ``best_params`` is given."""
    pipe = build_pipeline(estimator, scaler)
    if best_params is not None:
        pipe.set_params(**best_params)
        return pipe.fit(X, y), best_params
    if not param_grid:
        return pipe.fit(X, y), {}

    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed),
        scoring=scoring,
        n_jobs=n_jobs,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def evaluate_models(
    split: Split,
    classifiers: dict[str, ModelSpec],
    *,
    scaler: str | None = "minmax",
    cv: int = CV_FOLDS,
    best_params: dict[str, dict] | None = None,
    n_jobs: int = -1,
    seed: int = RANDOM_STATE,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict], dict[str, Pipeline]]:
    """Fit every classifier on ``split`` and score it on the (real) test set."""
    encoder = LabelEncoder().fit(pd.concat([split.y_train, split.y_test]))
    y_train, y_test = encoder.transform(split.y_train), encoder.transform(split.y_test)
    n_classes = len(encoder.classes_)

    rows, params, fitted = [], {}, {}
    for name, (estimator, grid) in classifiers.items():
        start = time.perf_counter()
        model, params[name] = fit_model(
            estimator, grid, split.X_train, y_train,
            scaler=scaler, cv=cv, best_params=(best_params or {}).get(name),
            n_jobs=n_jobs, seed=seed,
        )
        fit_time = time.perf_counter() - start

        start = time.perf_counter()
        y_pred = model.predict(split.X_test)
        y_prob = model.predict_proba(split.X_test) if hasattr(model, "predict_proba") else None
        predict_time = time.perf_counter() - start

        scores = multiclass_metrics(y_test, y_pred, y_prob, n_classes)
        rows.append({"model": name, **scores, "fit_time": fit_time, "predict_time": predict_time})
        fitted[name] = model
        if verbose:
            print(f"  {name:<24} acc={scores['accuracy']:.4f}  f1={scores['f1_macro']:.4f}  "
                  f"auc={scores['roc_auc']:.4f}  ({fit_time:.1f}s)")
    return pd.DataFrame(rows), params, fitted


@dataclass
class RepeatedResult:
    runs: pd.DataFrame
    best_params: dict[str, dict]


def run_repeated(
    df: pd.DataFrame,
    split_fn: Callable[..., Split],
    classifiers: dict[str, ModelSpec],
    *,
    n_runs: int = N_RUNS,
    seed: int = RANDOM_STATE,
    cv: int = CV_FOLDS,
    scaler: str | None = "minmax",
    retune_each_run: bool = False,
    n_jobs: int = -1,
    verbose: bool = True,
) -> RepeatedResult:
    """Repeat split -> fit -> score with seeds ``seed + i``.

    Hyperparameters are tuned on the first run and reused afterwards unless
    ``retune_each_run`` is set.
    """
    frames, best_params = [], None
    for i in range(n_runs):
        run_seed = seed + i
        if verbose:
            print(f"Run {i + 1}/{n_runs} (seed={run_seed})")
        split = split_fn(df, seed=run_seed)
        scores, params, _ = evaluate_models(
            split, classifiers, scaler=scaler, cv=cv,
            best_params=None if retune_each_run else best_params,
            n_jobs=n_jobs, seed=run_seed, verbose=verbose,
        )
        if best_params is None:
            best_params = params
        frames.append(scores.assign(run=i, seed=run_seed))
    return RepeatedResult(pd.concat(frames, ignore_index=True), best_params or {})


def summarize(runs: pd.DataFrame, metrics: list[str] | None = None, as_text: bool = False) -> pd.DataFrame:
    """Mean and standard deviation per model, sorted by mean accuracy."""
    metrics = metrics or METRICS + ["fit_time", "predict_time"]
    grouped = runs.groupby("model")[metrics]
    mean, std = grouped.mean(), grouped.std(ddof=1)
    order = mean.sort_values(metrics[0], ascending=False).index

    if as_text:
        text = {m: mean[m].map("{:.4f}".format) + " ± " + std[m].map("{:.4f}".format) for m in metrics}
        return pd.DataFrame(text).loc[order]

    out = pd.concat({m: pd.DataFrame({"mean": mean[m], "std": std[m]}) for m in metrics}, axis=1)
    out.columns = [f"{m}_{stat}" for m, stat in out.columns]
    return out.loc[order]


def wilcoxon_pairwise(
    runs: pd.DataFrame, models: list[str] | None = None, metric: str = "accuracy"
) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank tests on per-run scores."""
    models = models or TOP_MODELS
    wide = runs.pivot(index="run", columns="model", values=metric)
    rows = []
    for a, b in combinations(models, 2):
        try:
            stat, p = wilcoxon(wide[a], wide[b])
        except ValueError:
            stat, p = np.nan, np.nan
        rows.append({"model_a": a, "model_b": b, "W": stat, "p_value": p})
    return pd.DataFrame(rows)


def evaluate_binary(
    splits: BinarySplits,
    classifiers: dict[str, ModelSpec],
    *,
    scaler: str | None = "minmax",
    tune: bool = False,
    cv: int = CV_FOLDS,
    n_jobs: int = -1,
    seed: int = RANDOM_STATE,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """One-vs-all classifiers per crystal system, scored on the real held-out test set.

    Returns the accuracy table (models x systems, in %) and, for each model, the
    0/1 multilabel prediction matrix on the test set.
    """
    systems = list(splits.train)
    accuracy = pd.DataFrame(index=list(classifiers), columns=systems, dtype=float)
    predictions = {name: pd.DataFrame(0, index=splits.Y_test.index, columns=systems) for name in classifiers}

    for system in systems:
        X_bal, y_bal = splits.train[system]
        if verbose:
            print(f"{system}:")
        for name, (estimator, grid) in classifiers.items():
            model, _ = fit_model(
                estimator, grid if tune else {}, X_bal, y_bal,
                scaler=scaler, cv=cv, n_jobs=n_jobs, seed=seed,
            )
            y_pred = model.predict(splits.X_test)
            predictions[name][system] = np.asarray(y_pred).astype(int)
            accuracy.loc[name, system] = accuracy_score(splits.Y_test[system], y_pred) * 100
            if verbose:
                print(f"  {name:<24} acc={accuracy.loc[name, system]:.2f}%")
    return accuracy, predictions


def multilabel_report(
    Y_test: pd.DataFrame, predictions: dict[str, pd.DataFrame], models: list[str] | None = None
) -> pd.DataFrame:
    models = models or [m for m in TOP_MODELS if m in predictions]
    return pd.DataFrame(
        {name: multilabel_metrics(Y_test, predictions[name][Y_test.columns]) for name in models}
    ).T
