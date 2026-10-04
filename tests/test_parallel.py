from functools import partial

import pandas as pd
import pytest

from csp_perovskites import config, datasets as ds, evaluation as ev, io, models, preprocessing as pp

FAST = ["Naive Bayes", "Decision Tree", "LightGBM"]
SCORES = ["model", "run", "seed", *ev.METRICS]


@pytest.fixture(scope="module")
def master():
    return pp.clean_master(io.load_master())


def _fast_classifiers():
    return {name: (est, {}) for name, (est, _) in models.get_classifiers(FAST).items()}


def test_run_repeated_is_identical_in_parallel(master):
    df = pp.decorrelate(master)[0]
    split_fn = partial(ds.make_m4_split, features=config.TOP10_M4)
    kwargs = dict(n_runs=3, cv=2, verbose=False)

    sequential = ev.run_repeated(df, split_fn, _fast_classifiers(), n_jobs=1, **kwargs)
    parallel = ev.run_repeated(df, split_fn, _fast_classifiers(), n_jobs=2, **kwargs)

    assert len(parallel.runs) == 3 * len(FAST)
    pd.testing.assert_frame_equal(sequential.runs[SCORES], parallel.runs[SCORES])
    assert list(parallel.runs["model"].iloc[: len(FAST)]) == FAST


def test_evaluate_binary_is_identical_in_parallel(master):
    splits = ds.make_binary_splits(pp.encode_multilabel(master), seed=0)
    acc_seq, pred_seq = ev.evaluate_binary(splits, _fast_classifiers(), n_jobs=1, verbose=False)
    acc_par, pred_par = ev.evaluate_binary(splits, _fast_classifiers(), n_jobs=2, verbose=False)

    pd.testing.assert_frame_equal(acc_seq, acc_par)
    for name in FAST:
        pd.testing.assert_frame_equal(pred_seq[name], pred_par[name])
