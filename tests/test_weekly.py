"""
Tests for forecasts of returns over several days (patterndecoder/weekly.py).

Run from the repository root with: python -m pytest tests
"""

import csv
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
from statsmodels.regression.quantile_regression import QuantReg

from patterndecoder import robustness, weekly
from patterndecoder.data import describe_returns
from patterndecoder.evaluation import MedianForecast
from patterndecoder.study import load_returns
from patterndecoder.training import forward_sums
from patterndecoder.utils import WindowedDataset, get_model_performance, load_config
from tests.test_robustness import read_rows, returns


def weekly_params(**changes):
    """The config with five-day targets."""
    params = load_config("config/config.yaml")
    params.update(target_days=5, **changes)
    return params


def test_forward_sums_of_arrays_and_series():
    """Each sum covers a day and the next days - 1 days and keeps the first date."""
    values = np.arange(10, dtype="float32")
    np.testing.assert_allclose(forward_sums(values, 3), [3, 6, 9, 12, 15, 18, 21, 24])
    assert forward_sums(values, 1) is values
    series = pd.Series(values, index=pd.bdate_range("2024-01-01", periods=10))
    sums = forward_sums(series, 3)
    assert list(sums.index) == list(series.index[:8])
    np.testing.assert_allclose(sums.to_numpy(), [3, 6, 9, 12, 15, 18, 21, 24])
    frame = forward_sums(series.to_frame("Close"), 3)
    np.testing.assert_allclose(frame["Close"].to_numpy(), sums.to_numpy())


def test_windowed_dataset_targets_sum_the_next_days():
    """Every input window is followed by the sum of the next five returns, nothing later."""
    params = weekly_params(window_size=10, input_scale=1, batch_size=64)
    data = returns(days=60)
    inputs, targets = (np.concatenate(part) for part in zip(*WindowedDataset(params)(data)))
    assert len(inputs) == len(targets) == 60 - 10 - 4
    close = data["Close"].to_numpy()
    for i, (window, target) in enumerate(zip(inputs, targets)):
        np.testing.assert_allclose(window[:, 0], close[i : i + 10], rtol=1e-6)
        np.testing.assert_allclose(target[0], close[i + 10 : i + 15].sum(), rtol=1e-5)


def test_predictions_are_dated_by_the_first_target_day():
    """Predictions and actuals of a five-day forecast are dated by the first day summed."""
    params = weekly_params(window_size=10)
    data = returns(days=60)
    result = get_model_performance(MedianForecast([0.0]), params, data)
    expected = forward_sums(data, 5).iloc[10:]
    assert list(result["predictions"].index) == list(data.index[10:56])
    np.testing.assert_allclose(result["mae"], expected["Close"].abs().mean(), atol=1e-6)


def test_last_week_regression_uses_only_earlier_returns():
    """The last-week forecast of a day uses the five returns before it, fit like a direct LAD."""
    close = returns(days=400)["Close"]
    values = close.to_numpy()
    rows = range(5, len(values) - 4)
    previous = np.array([values[j - 5 : j].sum() for j in rows])
    following = np.array([values[j : j + 5].sum() for j in rows])
    design = np.column_stack([np.ones(len(previous)), previous])
    intercept, slope = QuantReg(following, design).fit(q=0.5).params
    forecasts = weekly.last_week_forecasts(close, close, 5)
    assert forecasts.iloc[:5].isna().all()
    np.testing.assert_allclose(forecasts.iloc[5:-4], intercept + slope * previous, rtol=1e-6)
    # a persistent series gives a positive slope (momentum)
    rng = np.random.default_rng(3)
    shocks = rng.normal(0, 0.01, 2000)
    persistent = np.zeros(2000)
    for day in range(1, 2000):
        persistent[day] = 0.8 * persistent[day - 1] + shocks[day]
    series = pd.Series(persistent, index=pd.bdate_range("2015-01-01", periods=2000))
    forecast = weekly.last_week_forecasts(series, series, 5).iloc[5:-4]
    assert np.corrcoef(forecast, forward_sums(series, 5).iloc[5:])[0, 1] > 0.3


def test_benchmarks_forecast_the_five_day_targets():
    """Validation and test targets are five-day sums within their own part."""
    params = weekly_params(window_size=30)
    data = returns()
    train, test = robustness.split_window(data, params, 0.5)
    forecasts, val_targets, actuals = weekly.benchmark_forecasts(train, test, params)
    assert list(actuals.index) == list(data.index[150:176])
    np.testing.assert_allclose(actuals, forward_sums(data["Close"].iloc[150:180], 5), atol=1e-12)
    n_val = int(150 * params["validation_split"])
    assert list(val_targets.index) == list(data.index[150 - n_val : 146])
    assert set(forecasts) == {"Naive", "Median", "LastWeek"}
    median = forecasts["Median"][0].iloc[0]
    assert np.isclose(median, forward_sums(train["Close"], 5).median())
    for test_forecast, val_forecast in forecasts.values():
        assert test_forecast.notna().all() and val_forecast.notna().all()


def test_run_trains_each_window_length(tmp_path, monkeypatch):
    """run writes the benchmarks once and a model's forecasts for every window length."""
    params = weekly_params(epochs=1, patience=1, data_path=f"{tmp_path}/", ticker="SYNTH-AR")
    params.pop("data_checksum", None)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(params), encoding="utf-8")
    monkeypatch.setattr(weekly, "model_files", lambda: ["models/patterndecoder.yaml"])
    monkeypatch.setattr(weekly, "WINDOW_SIZES", (30, 40))
    out = tmp_path / "run.csv"
    weekly.run(3, 0.9, str(out), config_path=str(config_path))
    by_key = defaultdict(list)
    for row in read_rows(out):
        by_key[(row["model"], row["window_size"])].append(row)
    name = load_config("models/patterndecoder.yaml")["name"]
    benchmarks = {(bench, "0") for bench in ("Naive", "Median", "LastWeek")}
    assert set(by_key) == benchmarks | {(name, "30"), (name, "40")}
    close = load_returns(params)["Close"]
    start = int(len(close) * 0.9 + 1e-9)
    targets = forward_sums(close.iloc[start:], 5)
    for rows in by_key.values():
        assert [row["date"] for row in rows] == [f"{day:%Y-%m-%d}" for day in targets.index]
        np.testing.assert_allclose([float(row["actual"]) for row in rows], targets, atol=1e-7)
        assert all(np.isfinite(float(row["prediction"])) for row in rows)
        assert {row["target_days"] for row in rows} == {"5"}


def forecast_rows(days=100, seed=5):
    """Rows of two windows and two seeds: a model that is good with 60 days only."""
    rng = np.random.default_rng(seed)
    rows = []
    for start, first_day in ((0.5, "2020-01-01"), (0.6, "2021-01-01")):
        dates = pd.bdate_range(first_day, periods=days)
        actuals = rng.normal(0.001, 0.02, days)
        for run_seed in (1, 2):
            forecasts = {
                ("Naive", 0, 0.02): np.zeros(days),
                ("Median", 0, 0.02): np.full(days, 0.001),
                ("LastWeek", 0, 0.02): np.full(days, 0.002),
                ("Model", 30, 0.03): -actuals,
                ("Model", 60, 0.01): 0.6 * actuals + rng.normal(0, 0.002, days),
            }
            for (model, window, val_mae), predictions in forecasts.items():
                rows += [
                    {
                        "model": model,
                        "seed": run_seed,
                        "test_start": start,
                        "date": f"{date:%Y-%m-%d}",
                        "actual": actual,
                        "prediction": prediction,
                        "val_mae": val_mae,
                        "window_size": window,
                        "target_days": 5,
                    }
                    for date, actual, prediction in zip(dates, actuals, predictions)
                ]
    return rows


def test_summarize_chooses_windows_on_validation(tmp_path, monkeypatch):
    """The window length with the lower validation MAE is evaluated, on all days and weekly."""
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    path = tmp_path / "runs.csv"
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=weekly.FIELDS)
        writer.writeheader()
        writer.writerows(forecast_rows())
    out = str(tmp_path / "summary.csv")
    weekly.summarize([str(path)], out)
    pooled = {row["model"]: row for row in read_rows(out)}
    model = pooled["Model"]
    assert model["window_sizes"] == "60 60"
    assert int(model["days"]) == 200 and int(model["weeks"]) == 40
    for bench in weekly.BENCHMARKS:
        assert float(model[f"vs_{bench}_%"]) < 0 and float(model[f"vs_{bench}_weekly_%"]) < 0
        assert float(model[f"p_vs_{bench}"]) < 0.05 and float(model[f"p_vs_{bench}_weekly"]) < 0.05
    row = {key: float(value) if key.startswith(("vs_", "p_vs_")) else value
           for key, value in model.items()}
    assert weekly.beaten(row, weekly.ALPHA) == list(weekly.BENCHMARKS)
    windows = read_rows(out.replace(".csv", "_windows.csv"))
    assert {row["window_size"] for row in windows if row["model"] == "Model"} == {"60"}


def test_describe_returns_reports_data_problems():
    """The data check counts missing prices and zero returns and lists the largest moves."""
    index = pd.bdate_range("2019-12-30", periods=600)
    close = pd.Series(np.exp(np.cumsum(np.full(600, 0.001))), index=index)
    close.iloc[300] = close.iloc[299]
    close.iloc[400] = close.iloc[399] * 1.2
    close.iloc[401:] *= 1.2
    close.iloc[500] = np.nan
    text = describe_returns(close)
    assert text.startswith("1 missing prices, 1 zero returns")
    assert f"{index[400]:%Y-%m-%d} +18.2 %" in text
