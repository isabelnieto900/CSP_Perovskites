"""Model fitting, metrics, repeated experiments and statistical comparison."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
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

from .config import CV_FOLDS, N_JOBS, N_RUNS, RANDOM_STATE, TOP_MODELS
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
    n_jobs: int = N_JOBS,
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


def _encode_labels(split: Split) -> tuple[np.ndarray, np.ndarray, int]:
    encoder = LabelEncoder().fit(pd.concat([split.y_train, split.y_test]))
    return encoder.transform(split.y_train), encoder.transform(split.y_test), len(encoder.classes_)


def _score_fitted(name: str, model: Pipeline, split: Split, y_test, n_classes: int, fit_time: float) -> dict:
    start = time.perf_counter()
    y_pred = model.predict(split.X_test)
    y_prob = model.predict_proba(split.X_test) if hasattr(model, "predict_proba") else None
    predict_time = time.perf_counter() - start
    scores = multiclass_metrics(y_test, y_pred, y_prob, n_classes)
    return {"model": name, **scores, "fit_time": fit_time, "predict_time": predict_time}


def _fit_and_score(
    name: str, estimator: BaseEstimator, params: dict, split: Split, scaler: str | None, **extra
) -> dict:
    """Worker task: fit with fixed hyperparameters and return only the scores."""
    y_train, y_test, n_classes = _encode_labels(split)
    start = time.perf_counter()
    model, _ = fit_model(estimator, {}, split.X_train, y_train, scaler=scaler, best_params=params)
    row = _score_fitted(name, model, split, y_test, n_classes, time.perf_counter() - start)
    return {**row, **extra}


def _print_scores(row: dict, prefix: str = "") -> None:
    print(f"  {prefix}{row['model']:<24} acc={row['accuracy']:.4f}  f1={row['f1_macro']:.4f}  "
          f"auc={row['roc_auc']:.4f}  ({row['fit_time']:.1f}s)", flush=True)


def evaluate_models(
    split: Split,
    classifiers: dict[str, ModelSpec],
    *,
    scaler: str | None = "minmax",
    cv: int = CV_FOLDS,
    best_params: dict[str, dict] | None = None,
    n_jobs: int = N_JOBS,
    seed: int = RANDOM_STATE,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict], dict[str, Pipeline]]:
    """Fit every classifier on ``split`` and score it on the (real) test set.

    Models are processed one after another; ``n_jobs`` parallelises each grid search.
    """
    y_train, y_test, n_classes = _encode_labels(split)

    rows, params, fitted = [], {}, {}
    for name, (estimator, grid) in classifiers.items():
        start = time.perf_counter()
        model, params[name] = fit_model(
            estimator, grid, split.X_train, y_train,
            scaler=scaler, cv=cv, best_params=(best_params or {}).get(name),
            n_jobs=n_jobs, seed=seed,
        )
        row = _score_fitted(name, model, split, y_test, n_classes, time.perf_counter() - start)
        rows.append(row)
        fitted[name] = model
        if verbose:
            _print_scores(row)
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
    n_jobs: int = N_JOBS,
    verbose: bool = True,
) -> RepeatedResult:
    """Repeat split -> fit -> score with seeds ``seed + i``.

    Hyperparameters are tuned on the first run (grid search parallelised over
    ``n_jobs``) and reused afterwards; the remaining runs x models fits are then
    executed in parallel. With ``retune_each_run`` every run is tuned sequentially.
    Results do not depend on ``n_jobs``.
    """
    if verbose:
        print(f"Run 1/{n_runs} (seed={seed}): tuning hyperparameters", flush=True)
    scores, best_params, _ = evaluate_models(
        split_fn(df, seed=seed), classifiers, scaler=scaler, cv=cv,
        n_jobs=n_jobs, seed=seed, verbose=verbose,
    )
    frames = [scores.assign(run=0, seed=seed)]
    remaining = range(1, n_runs)

    if retune_each_run:
        for i in remaining:
            if verbose:
                print(f"Run {i + 1}/{n_runs} (seed={seed + i})", flush=True)
            scores, _, _ = evaluate_models(
                split_fn(df, seed=seed + i), classifiers, scaler=scaler, cv=cv,
                n_jobs=n_jobs, seed=seed + i, verbose=verbose,
            )
            frames.append(scores.assign(run=i, seed=seed + i))
    elif len(remaining):
        splits = {i: split_fn(df, seed=seed + i) for i in remaining}
        tasks = [
            delayed(_fit_and_score)(name, estimator, best_params[name], splits[i], scaler, run=i, seed=seed + i)
            for i in remaining
            for name, (estimator, _) in classifiers.items()
        ]
        if verbose:
            print(f"Runs 2-{n_runs}: {len(tasks)} fits in parallel (n_jobs={n_jobs})", flush=True)
        rows = []
        for row in Parallel(n_jobs=n_jobs, return_as="generator_unordered")(tasks):
            rows.append(row)
            if verbose:
                _print_scores(row, prefix=f"run {row['run'] + 1:>2}  ")
        frames.append(pd.DataFrame(rows))

    order = {name: k for k, name in enumerate(classifiers)}
    runs = pd.concat(frames, ignore_index=True)
    runs = runs.sort_values(["run", "model"], key=lambda s: s.map(order) if s.name == "model" else s)
    return RepeatedResult(runs.reset_index(drop=True), best_params)


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
    n_jobs: int = N_JOBS,
    seed: int = RANDOM_STATE,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """One-vs-all classifiers per crystal system, scored on the real held-out test set.

    Without ``tune`` all systems x models fits run in parallel over ``n_jobs``;
    with ``tune`` they run one after another, each grid search parallelised.
    Returns the accuracy table (models x systems, in %) and, for each model, the
    0/1 multilabel prediction matrix on the test set.
    """
    systems = list(splits.train)
    accuracy = pd.DataFrame(index=list(classifiers), columns=systems, dtype=float)
    predictions = {name: pd.DataFrame(0, index=splits.Y_test.index, columns=systems) for name in classifiers}

    jobs = [
        (system, name, estimator, grid if tune else {})
        for system in systems
        for name, (estimator, grid) in classifiers.items()
    ]
    if tune:
        results = (
            _binary_task(system, name, est, grid, *splits.train[system], splits.X_test, scaler, cv, n_jobs, seed)
            for system, name, est, grid in jobs
        )
    else:
        if verbose:
            print(f"{len(jobs)} fits in parallel (n_jobs={n_jobs})", flush=True)
        results = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(_binary_task)(system, name, est, grid, *splits.train[system], splits.X_test, scaler, cv, 1, seed)
            for system, name, est, grid in jobs
        )

    for system, name, y_pred in results:
        predictions[name][system] = y_pred
        accuracy.loc[name, system] = accuracy_score(splits.Y_test[system], y_pred) * 100
        if verbose:
            print(f"  {system:<13} {name:<24} acc={accuracy.loc[name, system]:.2f}%", flush=True)
    return accuracy, predictions


def _binary_task(system, name, estimator, grid, X, y, X_test, scaler, cv, n_jobs, seed):
    model, _ = fit_model(estimator, grid, X, y, scaler=scaler, cv=cv, n_jobs=n_jobs, seed=seed)
    return system, name, np.asarray(model.predict(X_test)).astype(int)


def multilabel_report(
    Y_test: pd.DataFrame, predictions: dict[str, pd.DataFrame], models: list[str] | None = None
) -> pd.DataFrame:
    models = models or [m for m in TOP_MODELS if m in predictions]
    return pd.DataFrame(
        {name: multilabel_metrics(Y_test, predictions[name][Y_test.columns]) for name in models}
    ).T
