"""
Tests for the cross-section study.

Run from the repository root with: python -m pytest tests
"""

import numpy as np
import pandas as pd

from patterndecoder import cross_section, panel
from tests.test_smoke import write_config


def test_reversal_forecasts_recover_the_slope():
    """The LAD regression of the reversal rule finds a known slope despite fat tails."""
    rng = np.random.default_rng(0)
    inputs = rng.normal(0, 1, (4000, 30, 4)).astype("float32")
    targets = (-0.2 * inputs[:, -1, 0] + rng.laplace(0, 0.5, 4000)).astype("float32")
    forecasts = cross_section.reversal_forecasts((inputs, targets), (inputs[:5], targets[:5]))
    np.testing.assert_allclose(forecasts, -0.2 * inputs[:5, -1, 0], atol=0.05)


def test_run_and_summarize_on_the_synthetic_panel(tmp_path, monkeypatch):
    """Two short runs on the synthetic panel give forecasts and a summary with the known signal."""
    _, config_path = write_config(
        tmp_path,
        ticker=panel.SYNTHETIC_PANEL,
        data_path=f"{tmp_path}/",
        start="2022-01-03",
        end="2023-01-01",
    )
    monkeypatch.setattr(cross_section, "model_files", lambda: ["models/patterndecoder.yaml"])
    monkeypatch.setattr(cross_section, "PASSES", 1)
    paths = [str(tmp_path / f"run{seed}.csv") for seed in (1, 2)]
    for seed, path in zip((1, 2), paths):
        cross_section.run(seed, 0.8, path, config_path=str(config_path))
    frame = pd.read_csv(paths[0])
    assert list(frame.columns) == cross_section.FIELDS
    assert set(frame["model"]) == {"Naive", "Reversal", "Oracle", "PatternDecoder", "Linear"}
    assert frame.groupby("model").size().nunique() == 1
    assert np.isfinite(frame["prediction"]).all()

    out = tmp_path / "summary.csv"
    cross_section.summarize(paths, str(out))
    pooled = pd.read_csv(out).set_index("model")
    assert pooled.loc["Oracle", "ic"] > 0.05
    assert abs(pooled.loc["Reversal", "ic"] - pooled.loc["Oracle", "ic"]) < 1e-3
    assert np.isnan(pooled.loc["Naive", "ic"])
    assert pooled.loc["Oracle", "long_short_gross_%"] > 0
    windows = pd.read_csv(tmp_path / "summary_windows.csv")
    assert set(windows["model"]) == set(pooled.index) and (windows["days"] == 26).all()
