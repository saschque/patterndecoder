"""
Tests for the synthetic series used as positive controls.

Run from the repository root with: python -m pytest tests
"""

import numpy as np
import pandas as pd

from patterndecoder import synthetic
from patterndecoder.data import get_stock_data
from patterndecoder.utils import load_config


def generate(ticker):
    """Generates one synthetic series with the dates of the project config."""
    params = load_config("config/config.yaml")
    return synthetic.synthetic_data(dict(params, ticker=ticker))


def log_returns(data):
    """Daily log returns from the generated prices (the first one is lost)."""
    return np.log(data["Close"]).diff().to_numpy()


def test_series_are_deterministic_and_return_like():
    """Each series is the same on every call and has the size of daily returns."""
    for ticker in synthetic.PROCESSES:
        first, second = generate(ticker), generate(ticker)
        pd.testing.assert_frame_equal(first, second)
        assert len(first) == len(pd.bdate_range("2021-02-05", "2026-02-05"))
        assert 0.008 < np.nanstd(log_returns(first)) < 0.013


def test_oracles_follow_the_design():
    """The oracle forecasts are the conditional means of the designed structure."""
    data = generate("SYNTH-AR")
    returns, oracle = log_returns(data), data["Oracle"].to_numpy()
    expected = synthetic.DRIFT + synthetic.AR_COEFFICIENT * (returns[1:-1] - synthetic.DRIFT)
    np.testing.assert_allclose(oracle[2:], expected, atol=1e-12)

    oracle = generate("SYNTH-CYCLE")["Oracle"].to_numpy()
    days = np.arange(len(oracle))
    wave = synthetic.CYCLE_AMPLITUDE * synthetic.SIGMA * np.sin(
        2 * np.pi * days / synthetic.CYCLE_PERIOD + synthetic.CYCLE_PHASE
    )
    np.testing.assert_allclose(oracle, synthetic.DRIFT + wave, atol=1e-15)

    pattern = (generate("SYNTH-MOTIF")["Oracle"].to_numpy() - synthetic.DRIFT) / synthetic.SIGMA
    running = np.flatnonzero(np.abs(pattern - synthetic.MOTIF[1]) < 1e-9)
    assert len(running) > 20
    for start in running[running < len(pattern) - 9]:
        np.testing.assert_allclose(pattern[start : start + 9], synthetic.MOTIF[1:], atol=1e-9)


def test_get_stock_data_generates_synthetic_series(tmp_path):
    """A SYNTH- ticker is generated and cached instead of downloaded."""
    params = load_config("config/config.yaml")
    params.update(data_path=f"{tmp_path}/", ticker="SYNTH-CYCLE", data_checksum=None)
    data = get_stock_data(params)
    assert (tmp_path / "SYNTH-CYCLE.csv").exists()
    assert list(data.columns) == ["Close", "Oracle"]
    np.testing.assert_allclose(data["Oracle"], generate("SYNTH-CYCLE")["Oracle"], atol=1e-15)
