"""Cleaning, decorrelation and label encoding of the master dataset."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from .config import CORR_THRESHOLD, CRYSTAL_SYSTEMS, ENGINEERED_FEATURES, META_COLS, TARGET


def clean_master(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Formula", TARGET]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric descriptor columns, excluding identifiers and (one-hot) targets."""
    excluded = set(META_COLS) | {TARGET} | set(CRYSTAL_SYSTEMS)
    return [c for c in df.select_dtypes(include="number").columns if c not in excluded]


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[feature_columns(df)].corr()


def decorrelate(
    df: pd.DataFrame,
    threshold: float = CORR_THRESHOLD,
    keep: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop one feature of every pair with |corr| > threshold.

    Features in ``keep`` are never dropped; when a kept feature is correlated with
    another one, the other one is dropped. Otherwise the later column is dropped.
    Returns the reduced frame (non-feature columns preserved) and the dropped list.
    """
    keep = set(ENGINEERED_FEATURES if keep is None else keep)
    feats = feature_columns(df)
    corr = df[feats].corr().abs().to_numpy()

    to_drop: set[str] = set()
    for j in range(len(feats)):
        for i in range(j):
            if corr[i, j] <= threshold:
                continue
            a, b = feats[i], feats[j]
            if a in keep and b in keep:
                continue
            to_drop.add(a if b in keep else b)

    dropped = [f for f in feats if f in to_drop]
    return df.drop(columns=dropped), dropped


def class_count_per_formula(df: pd.DataFrame) -> pd.Series:
    """Number of distinct crystal systems reported for each formula."""
    return df.groupby("Formula")[TARGET].nunique().rename("class_count")


def class_count_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compounds per (crystal system, number of polymorphs): rows systems, cols class counts."""
    counts = class_count_per_formula(df)
    unique = df[["Formula", TARGET]].drop_duplicates().join(counts, on="Formula")
    table = unique.groupby([TARGET, "class_count"]).size().unstack(fill_value=0)
    systems = [s for s in CRYSTAL_SYSTEMS if s in table.index]
    return table.reindex(index=systems, columns=range(1, counts.max() + 1), fill_value=0)


def multilabel_mask(df: pd.DataFrame, features: list[str] | None = None) -> pd.Series:
    """True for rows whose feature vector appears with more than one label."""
    features = features or feature_columns(df)
    n_labels = df.groupby(features, dropna=False)[TARGET].transform("nunique")
    return n_labels > 1


def remove_multilabel_rows(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    features = features or feature_columns(df)
    single = df.loc[~multilabel_mask(df, features)]
    return single.drop_duplicates(subset=features + [TARGET]).reset_index(drop=True)


def encode_multilabel(df: pd.DataFrame) -> pd.DataFrame:
    """One row per formula with one binary column per crystal system."""
    labels = df.groupby("Formula")[TARGET].apply(list)
    features = df.drop(columns=TARGET).drop_duplicates(subset="Formula").set_index("Formula")

    mlb = MultiLabelBinarizer(classes=sorted(CRYSTAL_SYSTEMS))
    encoded = pd.DataFrame(mlb.fit_transform(labels), index=labels.index, columns=mlb.classes_)
    return pd.concat([features, encoded], axis=1).reset_index()


def label_cardinality(Y: pd.DataFrame | np.ndarray) -> pd.Series:
    """Distribution of the number of labels per sample."""
    return pd.Series(np.asarray(Y).sum(axis=1)).value_counts().sort_index()
