"""
Tests for the volatility use case.

Run from the repository root with: python -m pytest tests
"""

import numpy as np
import pandas as pd

from patterndecoder import study, volatility
from patterndecoder.robustness import benchmark_forecasts, split_window
from patterndecoder.utils import load_config


def har_series(days=900, seed=3):
    """A positive series that follows a HAR process with known coefficients."""
    rng = np.random.default_rng(seed)
    values = np.full(days, 0.008)
    for day in range(22, days):
        values[day] = (
            0.001
            + 0.4 * values[day - 1]
            + 0.3 * values[day - 5 : day].mean()
            + 0.2 * values[day - 22 : day].mean()
            + rng.laplace(0, 0.0005)
        )
    index = pd.bdate_range("2021-02-05", periods=days, name="Date")
    return pd.Series(np.abs(values), index=index)


def test_parkinson_volatility():
    """ln(High / Low) / sqrt(4 ln 2), and zero when high equals low."""
    prices = pd.DataFrame({"High": [101.0, 100.0], "Low": [99.0, 100.0]})
    expected = np.log(101 / 99) / np.sqrt(4 * np.log(2))
    np.testing.assert_allclose(volatility.parkinson_volatility(prices), [expected, 0.0])


def test_har_fit_recovers_the_coefficients():
    """The least-absolute-deviations fit recovers the coefficients of a HAR process."""
    series = har_series(days=3000)
    coefficients = volatility.fit_har(series, series.index[-1])
    np.testing.assert_allclose(coefficients, [0.001, 0.4, 0.3, 0.2], atol=0.08)


def test_volatility_benchmarks_use_only_the_past():
    """Persistence and EWMA forecast each day from the days before it."""
    params = load_config("config/config.yaml")
    data = har_series().to_frame("Close")
    train, test = split_window(data, params, 0.8)
    forecasts, val_targets, actuals = volatility.volatility_benchmarks(train, test, params)
    assert set(forecasts) == {"Persistence", "EWMA", "HAR", "Median"}
    series = data["Close"]
    np.testing.assert_allclose(forecasts["Persistence"][0], series.shift(1)[actuals.index])
    day = actuals.index[5]
    previous = series[series.index < day]
    ewma = previous.ewm(alpha=1 - volatility.EWMA_DECAY, adjust=False).mean().iloc[-1]
    assert np.isclose(forecasts["EWMA"][0][day], ewma)
    for test_forecast, val_forecast in forecasts.values():
        assert test_forecast.index.equals(actuals.index)
        assert val_forecast.index.equals(val_targets.index)
        assert np.isfinite(test_forecast).all() and np.isfinite(val_forecast).all()
    har_mae = np.mean(np.abs(forecasts["HAR"][0] - actuals))
    assert har_mae < np.mean(np.abs(forecasts["Median"][0] - actuals))


def test_series_setting_selects_the_volatility(monkeypatch):
    """With series: parkinson, load_returns and the benchmarks switch to volatility."""
    params = load_config("config/config.yaml")
    params["series"] = "parkinson"
    index = pd.bdate_range("2021-02-05", periods=400, name="Date")
    rng = np.random.default_rng(4)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index))))
    prices = pd.DataFrame(
        {"Close": close, "High": close * 1.01, "Low": close * 0.99}, index=index
    )
    monkeypatch.setattr(study, "get_stock_data", lambda _params, _download=False: prices)
    data = study.load_returns(params)
    assert list(data.columns) == ["Close"]
    np.testing.assert_allclose(data["Close"], np.log(1.01 / 0.99) / np.sqrt(4 * np.log(2)))
    forecasts, _, _ = benchmark_forecasts(*split_window(data, params, 0.8), params)
    assert "HAR" in forecasts and "Naive" not in forecasts
