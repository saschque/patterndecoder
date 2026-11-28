"""
Tests for the panel of stocks of the cross-section study.

Run from the repository root with: python -m pytest tests
"""

import numpy as np
import pandas as pd

from patterndecoder import panel

PARAMS = {
    "start": "2022-01-03",
    "end": "2023-01-01",
    "window_size": 30,
    "validation_split": 0.2,
    "ticker": panel.SYNTHETIC_PANEL,
}


def test_universes_have_distinct_tickers():
    """The universes list the 40 DAX and the 71 SDAX companies once each, without overlap."""
    assert len(panel.UNIVERSE) == len(set(panel.UNIVERSE)) == 40
    assert len(panel.SDAX_UNIVERSE) == len(set(panel.SDAX_UNIVERSE)) == 71
    assert not set(panel.UNIVERSE) & set(panel.SDAX_UNIVERSE)
    assert len(panel.FTSE250_UNIVERSE) == len(set(panel.FTSE250_UNIVERSE)) == 250


def test_synthetic_panel_has_a_known_reversal():
    """The oracle forecasts the synthetic excess returns with an IC near the reversal."""
    data = panel.panel_data(panel.synthetic_panel(PARAMS))
    assert data["features"].shape == (len(data["dates"]), 40, len(panel.FEATURES))
    rows = np.arange(2, len(data["dates"]))
    frame = pd.DataFrame(
        {
            "date": np.repeat(rows, 40),
            "prediction": data["oracle"][rows].ravel(),
            "actual": data["target"][rows].ravel(),
        }
    )
    assert 0.06 < panel.daily_ic(frame).mean() < 0.14
    oracle = -panel.REVERSAL * data["target"][rows - 1]
    np.testing.assert_allclose(data["oracle"][rows], oracle, atol=1e-9)


def test_panel_data_features():
    """Excess returns average zero each day; the panel return is the daily mean return."""
    data = panel.panel_data(panel.synthetic_panel(PARAMS), input_scale=100)
    target, features = data["target"][1:], data["features"][1:]
    np.testing.assert_allclose(np.nanmean(target, axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(features[..., 0], 100 * target, rtol=1e-5, atol=1e-5)
    assert np.allclose(features[:, :1, 1], features[:, -1:, 1])
    assert np.isnan(data["features"][0, :, 0]).all()


def test_panel_windows_use_only_the_days_before_the_target():
    """Each sample holds the window before its target day; incomplete stocks are left out."""
    data = panel.panel_data(panel.synthetic_panel(PARAMS))
    data["target"][100, 3] = np.nan
    data["features"][95, 5, 2] = np.nan
    inputs, targets, keys = panel.panel_windows(data, [100, 101], window=30)
    assert (keys[:, 0] == 100).sum() == 38
    assert {3, 5}.isdisjoint(keys[keys[:, 0] == 100, 1])
    for sample in (0, 50):
        row, stock = keys[sample]
        np.testing.assert_array_equal(inputs[sample], data["features"][row - 30 : row, stock])
        assert np.isclose(targets[sample], data["target"][row, stock])


def test_download_panel_retries_missing_tickers(monkeypatch):
    """Tickers that a download misses are requested again; lasting gaps are left out."""
    calls = []

    def fake_download(_, tickers):
        calls.append(list(tickers))
        found = [ticker for ticker in tickers if ticker == "A" or len(calls) > 1 and ticker == "B"]
        return pd.DataFrame({"Date": pd.Timestamp("2024-01-02"), "Ticker": found, "Close": 1.0})

    monkeypatch.setattr(panel, "download_prices", fake_download)
    result = panel.download_panel({}, tickers=("A", "B", "C"))
    assert sorted(result["Ticker"]) == ["A", "B"]
    assert calls == [["A", "B", "C"], ["B", "C"], ["C"]]


def test_split_rows():
    """Fit, validation and test days follow each other without overlap."""
    rows = panel.split_rows(1000, PARAMS, 0.5)
    assert rows["fit"][0] == 30 and rows["fit"][-1] + 1 == rows["val"][0] == 400
    assert rows["val"][-1] + 1 == rows["test"][0] == 500 and rows["test"][-1] == 599
    try:
        panel.split_rows(1000, PARAMS, 0.02)
    except ValueError:
        pass
    else:
        raise AssertionError("a window that leaves no training context must be rejected")


def test_daily_ic():
    """A forecast with the same ranking as the outcomes has IC 1, the reverse -1."""
    actual = np.tile([0.03, -0.01, 0.02, 0.0], 2)
    frame = pd.DataFrame({"date": np.repeat([1, 2], 4), "actual": actual})
    assert panel.daily_ic(frame.assign(prediction=10 * actual)).tolist() == [1.0, 1.0]
    assert panel.daily_ic(frame.assign(prediction=-actual)).tolist() == [-1.0, -1.0]
    assert panel.daily_ic(frame.assign(prediction=0.0)).isna().all()


def test_long_short():
    """The portfolio is long the top and short the bottom fifth; turnover counts the trades."""
    tickers = [f"S{number}" for number in range(10)]
    actual = np.linspace(-0.045, 0.045, 10)
    days = [
        pd.DataFrame({"date": day, "ticker": tickers, "prediction": prediction, "actual": actual})
        for day, prediction in ((1, actual), (2, actual), (3, actual[::-1]))
    ]
    result = panel.long_short(pd.concat(days), share=0.2, cost=0.001)
    best, worst = actual[-2:].mean(), actual[:2].mean()
    np.testing.assert_allclose(result["gross"], [best - worst, best - worst, worst - best])
    np.testing.assert_allclose(result["turnover"], [2.0, 0.0, 4.0])
    np.testing.assert_allclose(result["net"], result["gross"] - 0.001 * result["turnover"])
