import pandas as pd
import pytest

from csp_perovskites import config, datasets as ds, io, preprocessing as pp
from csp_perovskites.models import build_pipeline, get_classifiers


@pytest.fixture(scope="module")
def master():
    return pp.clean_master(io.load_master())


@pytest.fixture(scope="module")
def decorrelated(master):
    return pp.decorrelate(master)[0]


def _rows(X: pd.DataFrame) -> set[tuple]:
    return set(map(tuple, X.round(10).to_numpy()))


def test_decorrelation_keeps_engineered_features(decorrelated):
    feats = pp.feature_columns(decorrelated)
    assert len(feats) == 27
    assert set(config.ENGINEERED_FEATURES) <= set(feats)


def test_multilabel_encoding_matches_polymorph_counts(master):
    enc = pp.encode_multilabel(master)
    assert len(enc) == master["Formula"].nunique() == 5279
    assert pp.label_cardinality(enc[config.CRYSTAL_SYSTEMS])[1] == 4502


@pytest.mark.parametrize("make_split", [ds.make_m7_split, ds.make_m4_split])
def test_test_set_contains_only_real_samples(decorrelated, make_split):
    split = make_split(decorrelated, seed=0)
    real = _rows(decorrelated[split.X_test.columns])
    assert _rows(split.X_test) <= real


@pytest.mark.parametrize("make_split", [ds.make_m7_split, ds.make_m4_split])
def test_training_set_is_balanced(decorrelated, make_split):
    split = make_split(decorrelated, seed=0)
    counts = split.y_train.value_counts()
    assert counts.max() / counts.min() < 1.2
    assert len(split.X_train) == len(split.y_train)
    assert len(split.X_test) == len(split.y_test)


def test_m4_has_four_classes_and_paper_size(decorrelated):
    split = ds.make_m4_split(decorrelated, seed=config.RANDOM_STATE)
    assert set(split.y_train) == {"Cubic", "Orthorhombic", "Trigonal", "Tetragonal"}
    assert (split.y_train.value_counts() == 2428).all()


def test_m7_train_has_no_conflicting_labels(decorrelated):
    split = ds.make_m7_split(decorrelated, seed=0)
    tmp = split.X_train.assign(y=split.y_train.to_numpy())
    assert (tmp.groupby(list(split.X_train.columns))["y"].nunique() == 1).all()


def test_m7_can_exclude_polymorphs_from_test(decorrelated):
    split = ds.make_m7_split(decorrelated, seed=0, drop_polymorphs_from_test=True)
    assert set(split.y_test) <= {"Cubic", "Orthorhombic", "Trigonal", "Tetragonal"}


def test_reduced_features(decorrelated):
    split = ds.make_m7_split(decorrelated, seed=0, features=config.TOP10_M7)
    assert list(split.X_train.columns) == config.TOP10_M7


def test_binary_splits_share_real_test_set(master):
    enc = pp.encode_multilabel(master)
    splits = ds.make_binary_splits(enc, seed=0)
    assert set(splits.train) == set(config.CRYSTAL_SYSTEMS)
    assert list(splits.Y_test.columns) == config.CRYSTAL_SYSTEMS
    assert _rows(splits.X_test) <= _rows(enc[splits.X_test.columns])
    for X_bal, y_bal in splits.train.values():
        assert y_bal.value_counts().nunique() == 1


def test_registry_has_twelve_classifiers_and_rejects_unknown():
    assert len(get_classifiers()) == 12
    with pytest.raises(KeyError):
        get_classifiers(["Not a model"])


def test_pipeline_scaler_is_configurable():
    est, _ = get_classifiers(["Naive Bayes"])["Naive Bayes"]
    assert build_pipeline(est, "minmax").named_steps["scaler"].__class__.__name__ == "MinMaxScaler"
    assert build_pipeline(est, None).named_steps["scaler"] == "passthrough"
