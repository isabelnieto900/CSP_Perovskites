"""Leakage-free construction of the M7, M4 and binary-relevance datasets.

Every builder splits the real data first and oversamples only the training part,
so test sets never contain synthetic SMOTE samples.
"""

from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from .config import CRYSTAL_SYSTEMS, RANDOM_STATE, TARGET, TEST_SIZE
from .preprocessing import feature_columns, multilabel_mask, remove_multilabel_rows


class Split(NamedTuple):
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass
class BinarySplits:
    """Seven one-vs-all balanced training sets sharing one real multilabel test set."""

    train: dict[str, tuple[pd.DataFrame, pd.Series]]
    X_test: pd.DataFrame
    Y_test: pd.DataFrame
    formulas_test: pd.Series


def xy(df: pd.DataFrame, features: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    features = list(features) if features is not None else feature_columns(df)
    return df[features], df[TARGET]


def balance_smote(
    X: pd.DataFrame, y: pd.Series, seed: int = RANDOM_STATE, k_neighbors: int = 5
) -> tuple[pd.DataFrame, pd.Series]:
    k = min(k_neighbors, int(y.value_counts().min()) - 1)
    X_res, y_res = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X, y)
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)


def drop_conflicting_labels(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Remove feature vectors mapped to more than one label, then exact duplicates."""
    features = list(X.columns)
    tmp = X.reset_index(drop=True).assign(**{TARGET: y.to_numpy()})
    n_labels = tmp.groupby(features, dropna=False)[TARGET].transform("nunique")
    tmp = tmp.loc[n_labels == 1].drop_duplicates().reset_index(drop=True)
    return tmp[features], tmp[TARGET].rename(y.name)


def make_m7_split(
    df: pd.DataFrame,
    seed: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
    features: list[str] | None = None,
    drop_polymorphs_from_test: bool = False,
) -> Split:
    """Seven classes: stratified split, SMOTE on train, then drop conflicting labels in train.

    Polymorphic compounds share one feature vector across several labels, so a
    single-label classifier cannot get all their test rows right. Set
    ``drop_polymorphs_from_test`` to evaluate on single-label compounds only.
    """
    X, y = xy(df, features)
    polymorph = multilabel_mask(df, feature_columns(df)).to_numpy()
    X_tr, X_te, y_tr, y_te, _, poly_te = train_test_split(
        X, y, polymorph, test_size=test_size, stratify=y, random_state=seed
    )
    X_tr, y_tr = balance_smote(X_tr, y_tr, seed)
    X_tr, y_tr = drop_conflicting_labels(X_tr, y_tr)
    if drop_polymorphs_from_test:
        X_te, y_te = X_te[~poly_te], y_te[~poly_te]
    return Split(X_tr, X_te.reset_index(drop=True), y_tr, y_te.reset_index(drop=True))


def make_m4_split(
    df: pd.DataFrame,
    seed: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
    features: list[str] | None = None,
) -> Split:
    """Single-label compounds only (four classes): stratified split, SMOTE on train."""
    single = remove_multilabel_rows(df, feature_columns(df))
    X, y = xy(single, features)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_tr, y_tr = balance_smote(X_tr, y_tr, seed)
    return Split(X_tr, X_te.reset_index(drop=True), y_tr, y_te.reset_index(drop=True))


def make_binary_splits(
    df_encoded: pd.DataFrame,
    seed: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
    features: list[str] | None = None,
    systems: list[str] = CRYSTAL_SYSTEMS,
) -> BinarySplits:
    """Hold out a real multilabel test set; balance each one-vs-all training set with SMOTE."""
    features = list(features) if features is not None else feature_columns(df_encoded)
    train_df, test_df = train_test_split(df_encoded, test_size=test_size, random_state=seed)

    train = {
        system: balance_smote(train_df[features], train_df[system], seed)
        for system in systems
    }
    return BinarySplits(
        train=train,
        X_test=test_df[features].reset_index(drop=True),
        Y_test=test_df[systems].reset_index(drop=True),
        formulas_test=test_df["Formula"].reset_index(drop=True),
    )


def split_to_frames(split: Split) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and test frames with the target column appended, ready to save."""
    train = split.X_train.assign(**{TARGET: split.y_train.to_numpy()})
    test = split.X_test.assign(**{TARGET: split.y_test.to_numpy()})
    return train, test
