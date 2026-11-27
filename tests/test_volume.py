"""
Tests for the trading volume use case.

Run from the repository root with: python -m pytest tests
"""

import numpy as np
import pandas as pd

from patterndecoder import robustness, study, utils, volume
from patterndecoder.utils import load_config
from patterndecoder.volatility import fit_har


def test_calendar_dummies():
    """Weekdays and third Fridays, with the quarterly expiry months."""
    dates = pd.DatetimeIndex(
        ["2024-03-15", "2024-04-19", "2024-04-12", "2024-04-15", "2024-04-18"]
    )
    dummies = volume.calendar_dummies(dates)
    np.testing.assert_array_equal(dummies["third_friday"], [1, 1, 0, 0, 0])
    np.testing.assert_array_equal(dummies["quarterly_expiry"], [1, 0, 0, 0, 0])
    np.testing.assert_array_equal(dummies["mon"], [0, 0, 0, 1, 0])
    np.testing.assert_array_equal(dummies["thu"], [0, 0, 0, 0, 1])


def test_log_volume_fills_gaps_and_centres():
    """Days without volume take the previous value; the first half has median zero."""
    index = pd.bdate_range("2021-02-05", periods=6)
    prices = pd.DataFrame({"Volume": [100.0, 0.0, 200.0, 400.0, np.nan, 800.0]}, index=index)
    raw = np.log([100, 100, 200, 400, 400, 800])
    np.testing.assert_allclose(volume.log_volume(prices), raw - np.median(raw[:3]))


def calendar_series(days=1500, seed=5):
    """Log volume with HAR dynamics, a dip on Mondays and spikes on third Fridays."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2019-01-02", periods=days, name="Date")
    calendar = volume.calendar_dummies(index)
    values = np.zeros(days)
    for day in range(22, days):
        values[day] = (
            0.5 * values[day - 1]
            + 0.2 * values[day - 5 : day].mean()
            + 0.2 * values[day - 22 : day].mean()
            - 0.2 * calendar["mon"].iloc[day]
            + 0.5 * calendar["third_friday"].iloc[day]
            + rng.laplace(0, 0.05)
        )
    return pd.Series(values, index=index)


def test_harx_learns_the_calendar_effects():
    """HAR-X recovers the calendar effects of such a series and beats HAR on it."""
    series = calendar_series()
    coefficients = fit_har(series, series.index[-1], volume.calendar_dummies)
    # constant, three HAR terms, Monday to Thursday, third Friday, quarterly expiry
    np.testing.assert_allclose(coefficients[[4, 8]], [-0.2, 0.5], atol=0.05)
    params = load_config("config/config.yaml")
    params["series"] = "log_volume"
    forecasts, _, actuals = robustness.benchmark_forecasts(
        *robustness.split_window(series.to_frame("Close"), params, 0.7), params
    )
    assert set(forecasts) == {"HARX", "HAR", "Persistence", "EWMA", "Median"}
    assert robustness.tested_pair(forecasts) == ("HARX", "HAR")
    mae = {name: float(np.mean(np.abs(pair[0] - actuals))) for name, pair in forecasts.items()}
    assert mae["HARX"] < mae["HAR"] < mae["Median"]


def test_series_setting_selects_the_volume(monkeypatch):
    """With series: log_volume, load_returns returns the centred log volume."""
    params = load_config("config/config.yaml")
    params["series"] = "log_volume"
    index = pd.bdate_range("2021-02-05", periods=50, name="Date")
    prices = pd.DataFrame({"Close": 100.0, "Volume": np.arange(1, 51) * 1e6}, index=index)
    monkeypatch.setattr(study, "get_stock_data", lambda _params, _download=False: prices)
    data = study.load_returns(params)
    assert list(data.columns) == ["Close"]
    assert np.isclose(data["Close"].iloc[:25].median(), 0)


def test_next_day_calendar_inputs():
    """With calendar: next, each row carries the calendar of the following trading day."""
    index = pd.bdate_range("2024-03-11", periods=6, name="Date")  # Monday 11 to Monday 18 March
    current = utils.calendar_inputs(index)
    following = utils.calendar_inputs(index, "next", expiry=True)
    assert list(following.index) == list(index)
    np.testing.assert_allclose(following.iloc[:-1, :6].to_numpy(), current.iloc[1:].to_numpy())
    # the trading day after Thursday 14 March 2024 is a quarterly expiry Friday
    assert following.loc["2024-03-14", ["third_friday", "quarterly_expiry"]].tolist() == [1, 1]
    assert following.loc["2024-03-15", "third_friday"] == 0


def test_windows_carry_the_target_day_calendar():
    """The last input row of a window holds the calendar features of its target day."""
    params = load_config("config/config.yaml")
    params.update(calendar="next", expiry_features=True, input_scale=1)
    index = pd.bdate_range("2024-01-01", periods=80, name="Date")
    data = pd.DataFrame({"Close": np.linspace(0.0, 1.0, 80)}, index=index)
    inputs, _ = next(iter(utils.WindowedDataset(params)(data)))
    window = params["window_size"]
    expected = utils.calendar_inputs(index[window : window + 1], expiry=True).to_numpy()[0]
    np.testing.assert_allclose(inputs.numpy()[0, -1, 1:], expected, atol=1e-6)
