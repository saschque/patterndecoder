"""
Tests for the input scaling and the walk-forward evaluation.

Run from the repository root with: python -m pytest tests
"""

import csv
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

from patterndecoder import robustness
from patterndecoder.synthetic import synthetic_data
from patterndecoder.utils import MovingAverage, WindowedDataset, load_config


def returns(days=300, seed=7):
    """Random daily log returns on business days."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2023-01-02", periods=days, name="Date")
    return pd.DataFrame({"Close": rng.normal(0.0004, 0.01, days)}, index=index)


def read_rows(path):
    """Reads a CSV file into a list of dicts."""
    with open(path, encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_input_scale_scales_inputs_not_targets():
    """input_scale multiplies the return inputs only; the moving average undoes it."""
    params = load_config("config/config.yaml")
    data = returns(days=80)
    plain = list(WindowedDataset(dict(params, input_scale=1))(data))
    scaled = list(WindowedDataset(dict(params, input_scale=100))(data))
    for (x_plain, y_plain), (x_scaled, y_scaled) in zip(plain, scaled):
        np.testing.assert_allclose(x_scaled[..., 0], 100 * x_plain[..., 0], rtol=1e-6)
        np.testing.assert_allclose(x_scaled[..., 1:], x_plain[..., 1:])
        np.testing.assert_allclose(y_scaled, y_plain)
    window = params["window_size"]
    moving_plain = MovingAverage(1, window).predict(
        WindowedDataset(dict(params, input_scale=1))(data)
    )
    moving_scaled = MovingAverage(1, window, 100).predict(
        WindowedDataset(dict(params, input_scale=100))(data)
    )
    np.testing.assert_allclose(moving_scaled, moving_plain, rtol=1e-5)


def test_split_window_gives_context_and_no_overlap():
    """The test data start window_size days early; consecutive windows tile the data."""
    params = load_config("config/config.yaml")
    data = returns()
    window = params["window_size"]
    train, test = robustness.split_window(data, params, 0.5)
    assert len(train) == 150
    assert test.index[window] == data.index[150]
    assert test.index[window - 1] == train.index[-1]
    days = []
    for start in (0.5, 0.6, 0.7, 0.8, 0.9):
        days += list(robustness.split_window(data, params, start)[1].index[window:])
    assert days == list(data.index[150:])
    try:
        robustness.split_window(data, params, 0.05)
    except ValueError:
        pass
    else:
        raise AssertionError("a window that leaves no training context must be rejected")


def synthetic_forecast_rows(seed=8):
    """Forecast rows for two windows, two seeds and four models of known quality."""
    rng = np.random.default_rng(seed)
    rows = []
    for start, first_day in ((0.5, "2024-01-01"), (0.6, "2024-07-01")):
        dates = pd.bdate_range(first_day, periods=40)
        actuals = rng.normal(0.0005, 0.01, len(dates))
        for run_seed in (1, 2):
            noise = rng.normal(0, 0.0005, len(dates))
            forecasts = {
                "Naive": np.zeros(len(dates)),
                "Median": np.full(len(dates), 0.0004),
                "Good": actuals * 0.5 + noise,
                "Bad": -actuals + noise,
            }
            for model, predictions in forecasts.items():
                rows += [
                    {
                        "model": model,
                        "seed": run_seed,
                        "test_start": start,
                        "date": f"{date:%Y-%m-%d}",
                        "actual": actual,
                        "prediction": prediction,
                        "val_mae": 0.005,
                    }
                    for date, actual, prediction in zip(dates, actuals, predictions)
                ]
    return rows


def test_summarize_ensembles_and_pools_windows(tmp_path, monkeypatch):
    """Forecasts are averaged over seeds and pooled over windows."""
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    rows = synthetic_forecast_rows()
    path = tmp_path / "runs.csv"
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=robustness.FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    out = str(tmp_path / "summary.csv")
    robustness.summarize([str(path)], out)
    pooled = {row["model"]: row for row in read_rows(out)}
    naive_mae = np.mean([abs(row["actual"]) for row in rows if row["model"] == "Naive"])
    assert np.isclose(float(pooled["Naive"]["ensemble_mae"]), naive_mae, atol=1e-7)
    assert int(pooled["Good"]["days"]) == 80
    assert float(pooled["Good"]["vs_Naive_%"]) < 0 < float(pooled["Bad"]["vs_Naive_%"])
    assert float(pooled["Good"]["p_vs_Naive"]) < 0.05
    # the MAE of averaged forecasts is never above the mean MAE of the single forecasts
    assert float(pooled["Good"]["ensemble_mae"]) <= float(pooled["Good"]["single_run_mae"]) + 1e-7
    windows = read_rows(out.replace(".csv", "_windows.csv"))
    assert {row["test_start"] for row in windows} == {"0.5", "0.6"}


def test_run_trains_and_writes_forecasts(tmp_path, monkeypatch):
    """
    run trains a model for one window of a synthetic series and writes its
    forecasts next to the zero, median and oracle forecasts.
    """
    params = load_config("config/config.yaml")
    params.update(epochs=1, patience=1, data_path=f"{tmp_path}/", ticker="SYNTH-AR")
    params.pop("data_checksum", None)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(params), encoding="utf-8")
    monkeypatch.setattr(robustness, "model_files", lambda: ["models/patterndecoder.yaml"])
    out = tmp_path / "run.csv"
    robustness.run(3, 0.9, str(out), config_path=str(config_path))
    by_model = defaultdict(list)
    for row in read_rows(out):
        by_model[row["model"]].append(row)
    name = load_config("models/patterndecoder.yaml")["name"]
    assert set(by_model) == {"Naive", "Median", "Oracle", name}
    data = synthetic_data(params)
    window_days = [f"{date:%Y-%m-%d}" for date in data.index[int(len(data) * 0.9 + 1e-9) :]]
    for rows in by_model.values():
        assert [row["date"] for row in rows] == window_days
        assert all(np.isfinite(float(row["prediction"])) for row in rows)
        assert all(float(row["val_mae"]) > 0 for row in rows)
    oracle = [float(row["prediction"]) for row in by_model["Oracle"]]
    np.testing.assert_allclose(oracle, data["Oracle"].iloc[-len(oracle) :], atol=1e-8)


def test_oracle_share_is_reported(tmp_path, monkeypatch):
    """With oracle forecasts, the summary reports how much of their advantage a model reaches."""
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    rows = synthetic_forecast_rows()
    rows += [dict(row, model="Oracle", prediction=row["actual"] * 0.9)
             for row in rows if row["model"] == "Naive"]
    path = tmp_path / "runs.csv"
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=robustness.FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    out = str(tmp_path / "summary.csv")
    robustness.summarize([str(path)], out)
    shares = {row["model"]: float(row["oracle_share_%"]) for row in read_rows(out)}
    assert shares["Oracle"] == 100 and shares["Naive"] == 0
    assert shares["Bad"] < 0 < shares["Good"] < 100
