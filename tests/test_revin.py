# pylint: disable=E1101
"""
Tests for the reversible instance normalization.

Run from the repository root with: python -m pytest tests
"""

import numpy as np
import tensorflow as tf

from patterndecoder.revin import RevIN
from patterndecoder.utils import WindowedDataset, compile_and_train, load_model
from tests.test_smoke import synthetic_returns, write_config


def test_revin_reverses_the_normalization():
    """A model that returns its normalized last input value forecasts the last value."""
    rng = np.random.default_rng(1)
    series = rng.normal(0.5, 0.2, (4, 30, 1)).astype("float32")
    calendar = rng.uniform(-1, 1, (4, 30, 6)).astype("float32")
    inputs = np.concatenate([100 * series, calendar], axis=-1)
    last_value = tf.keras.Sequential([tf.keras.layers.Lambda(lambda x: x[:, -1, :1])])
    forecasts = RevIN(last_value, input_scale=100)(inputs).numpy()
    np.testing.assert_allclose(forecasts, series[:, -1, :], atol=1e-5)


def test_revin_forecasts_follow_the_level(tmp_path):
    """Shifting the series by a constant shifts every forecast by that constant."""
    params, _ = write_config(tmp_path, revin=True, model_overrides=False)
    data = synthetic_returns()
    tf.keras.backend.clear_session()
    model = load_model(params, "models/patterndecoder.yaml")
    assert isinstance(model, RevIN)
    windows = WindowedDataset(params)
    forecasts = model.predict(windows(data), verbose=0)
    shifted = model.predict(windows(data + 0.5), verbose=0)
    np.testing.assert_allclose(shifted - forecasts, 0.5, atol=1e-4)


def test_revin_checkpoint_round_trip(tmp_path):
    """A model with RevIN is checkpointed and restored like the others."""
    data = synthetic_returns()
    params, config_path = write_config(tmp_path, revin=True)
    tf.keras.backend.clear_session()
    model = load_model(params, "models/patterndecoder.yaml")
    compile_and_train(model, data, config_path)
    forecasts = model.predict(WindowedDataset(params)(data), verbose=0)
    params, config_path = write_config(tmp_path, revin=True, training=False)
    tf.keras.backend.clear_session()
    reloaded = load_model(params, "models/patterndecoder.yaml")
    _, reloaded = compile_and_train(reloaded, data, config_path)
    np.testing.assert_allclose(
        reloaded.predict(WindowedDataset(params)(data), verbose=0), forecasts, atol=1e-6
    )
