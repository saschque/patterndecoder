"""
Tests for the evaluation helpers, the validation split and the data cache.

Run from the repository root with: python -m pytest tests
"""

import warnings

import numpy as np
import pandas as pd

from patterndecoder.data import data_checksum, get_stock_data
from patterndecoder.evaluation import (
    MedianForecast,
    diebold_mariano,
    forecast_diagnostics,
    newey_west_test,
)
from patterndecoder.training import split_train_validation
from patterndecoder.utils import WindowedDataset, load_config


def returns(days=120, seed=1):
    """Random daily log returns on business days."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2023-01-02", periods=days, name="Date")
    return pd.DataFrame({"Close": rng.normal(0.0005, 0.01, days)}, index=index)


def test_median_forecast_is_constant_and_aligned():
    """One forecast per window, all equal to the training median."""
    params = load_config("config/config.yaml")
    train, test = returns(seed=2), returns(seed=3)
    model = MedianForecast(train["Close"])
    predictions = model.predict(WindowedDataset(params)(test))
    assert predictions.shape == (len(test) - params["window_size"], 1)
    assert np.all(predictions == np.median(train["Close"]))


def test_diebold_mariano_detects_a_better_forecast():
    """A forecast with clearly smaller errors gets a negative statistic and small p."""
    rng = np.random.default_rng(4)
    small, large = rng.normal(0, 1, 400), rng.normal(0, 2, 400)
    result = diebold_mariano(small, large)
    assert result["mean_diff"] < 0 and result["statistic"] < 0
    assert result["p_value"] < 0.01
    swapped = diebold_mariano(large, small)
    assert np.isclose(swapped["statistic"], -result["statistic"])
    assert np.isclose(swapped["p_value"], result["p_value"])


def test_diebold_mariano_identical_forecasts():
    """Identical errors give no test result instead of a division by zero."""
    errors = np.random.default_rng(5).normal(0, 1, 100)
    result = diebold_mariano(errors, errors)
    assert result["mean_diff"] == 0 and np.isnan(result["p_value"])


def test_newey_west_test():
    """A series with a clear positive mean is significant; autocorrelation widens the test."""
    rng = np.random.default_rng(6)
    noise = rng.normal(0, 1, 2000)
    result = newey_west_test(0.15 + noise)
    assert result["mean"] > 0 and result["p_value"] < 0.001 and result["n"] == 2000
    persistent = np.convolve(noise, np.ones(10), mode="same") / 10
    naive = persistent.mean() / (persistent.std() / np.sqrt(persistent.size))
    assert abs(newey_west_test(persistent, max_lag=20)["statistic"]) < abs(naive)


def test_forecast_diagnostics():
    """Constant and zero forecasts are recognised as such."""
    actuals = np.array([0.01, -0.02, 0.03, -0.01, 0.0])
    constant = forecast_diagnostics(np.full(5, 0.001), actuals)
    assert constant["pred_std"] == 0 and np.isnan(constant["correlation"])
    assert constant["hit_rate"] == 0.5
    assert np.isnan(forecast_diagnostics(np.zeros(5), actuals)["hit_rate"])
    perfect = forecast_diagnostics(actuals, actuals)
    assert np.isclose(perfect["correlation"], 1) and perfect["hit_rate"] == 1


def test_validation_split_is_chronological_and_complete():
    """Fit and validation targets are disjoint, ordered and cover all days."""
    params = load_config("config/config.yaml")
    data = returns(days=200)
    fit, val = split_train_validation(data, params)
    n_val = int(len(data) * params["validation_split"])
    assert len(fit) == len(data) - n_val
    assert len(val) == n_val + params["window_size"]
    assert val.index[params["window_size"]] > fit.index[-1]
    assert val.index[params["window_size"] - 1] == fit.index[-1]


def test_cached_data_are_read_and_checked(tmp_path):
    """An existing CSV is read without download; a wrong checksum warns."""
    data = returns()
    data.to_csv(tmp_path / "TEST.csv", date_format="%Y-%m-%d")
    params = {
        "data_path": f"{tmp_path}/",
        "ticker": "TEST",
        "type": "csv",
        "columns": ["Close"],
        "data_checksum": data_checksum(data),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = get_stock_data(params)
    assert len(loaded) == len(data)
    params["data_checksum"] = "0" * 64
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        get_stock_data(params)
    assert len(caught) == 1
