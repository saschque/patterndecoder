# pylint: disable=E1101
"""
Smoke tests for the models and the training pipeline.

Every model builds, trains for one epoch and predicts deterministically (no
dropout at inference); compile_and_train checkpoints, reloads weights and
histories, and works without checkpointing. Synthetic data, no download.

Run from the repository root with: python -m pytest tests
"""

import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from patterndecoder.utils import (
    WindowedDataset,
    compile_and_train,
    load_config,
    load_model,
    string_to_object,
)

MODEL_FILES = sorted(f for f in glob.glob("models/*.yaml") if "example" not in f)


def synthetic_returns(days=160, seed=0):
    """Random daily log returns on business days, shaped like the real data."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2022-01-03", periods=days, name="Date")
    return pd.DataFrame({"Close": rng.normal(0.0003, 0.01, days)}, index=index)


def write_config(tmp_path, **overrides):
    """Writes the project config with test overrides; returns (params, path)."""
    params = load_config("config/config.yaml")
    params.update(
        epochs=2,
        patience=1,
        verbose=False,
        training=True,
        tmp_weights_file=f"{tmp_path}/",
        tmp_history_file=f"{tmp_path}/",
    )
    params.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(params), encoding="utf-8")
    return params, str(path)


def test_every_model_trains_and_predicts_deterministically(tmp_path):
    """Each model trains one epoch; two predictions on the same data are identical."""
    params, _ = write_config(tmp_path)
    data = synthetic_returns()
    windows = WindowedDataset(params)
    expected_rows = len(data) - params["window_size"]
    assert MODEL_FILES, "no model definitions found"
    for model_file in MODEL_FILES:
        tf.keras.backend.clear_session()
        model = load_model(params, model_file)
        model.compile(loss="mae", optimizer="adam")
        model.fit(windows(data, shuffle=True), epochs=1, verbose=0)
        first = model.predict(windows(data), verbose=0)
        second = model.predict(windows(data), verbose=0)
        assert first.shape == (expected_rows, 1), model_file
        assert np.isfinite(first).all(), model_file
        np.testing.assert_allclose(first, second, err_msg=model_file)


def test_compile_and_train_saves_and_reloads(tmp_path):
    """Training writes weights and history; training: False restores both."""
    data = synthetic_returns()
    params, config_path = write_config(tmp_path)
    tf.keras.backend.clear_session()
    model = load_model(params, "models/patterndecoder.yaml")
    history, model = compile_and_train(model, data, config_path)
    assert "val_mae" in history.history
    predictions = model.predict(WindowedDataset(params)(data), verbose=0)

    params, config_path = write_config(tmp_path, training=False)
    tf.keras.backend.clear_session()
    reloaded = load_model(params, "models/patterndecoder.yaml")
    loaded_history, reloaded = compile_and_train(reloaded, data, config_path)
    np.testing.assert_allclose(
        reloaded.predict(WindowedDataset(params)(data), verbose=0), predictions, rtol=1e-6
    )
    assert loaded_history.history["val_mae"] == history.history["val_mae"]
    assert loaded_history.epoch == list(range(len(history.history["loss"])))


def test_compile_and_train_without_checkpoint(tmp_path):
    """restore_best: False trains without writing or loading a checkpoint."""
    params, config_path = write_config(tmp_path, restore_best=False, early_stopping=False)
    tf.keras.backend.clear_session()
    model = load_model(params, "models/transformer.yaml")
    history, _ = compile_and_train(model, synthetic_returns(), config_path)
    assert len(history.history["loss"]) == params["epochs"]
    assert not list(tmp_path.glob("*.keras"))


def test_string_to_object():
    """Dotted paths resolve to objects; other strings give a clear error."""
    assert string_to_object("tensorflow.keras.optimizers.AdamW") is tf.keras.optimizers.AdamW
    try:
        string_to_object("AdamW")
    except ValueError as error:
        assert "dotted path" in str(error)
    else:
        raise AssertionError("expected a ValueError for a path without a dot")


def built_model(params, model_file, data):
    """A freshly built, untrained model, called once on the data."""
    tf.keras.backend.clear_session()
    model = load_model(dict(params, model_overrides=False), model_file)
    return model, model.predict(WindowedDataset(params)(data), verbose=0)


def test_models_build_only_the_layers_they_use(tmp_path):
    """
    Models without an encoder stack do not depend on n_layers_enc: with the same
    seed, one and three encoder layers give identical initial forecasts.
    """
    params, _ = write_config(tmp_path)
    data = synthetic_returns()
    for model_file in ("models/patterndecoder.yaml", "models/decoderonly.yaml"):
        _, one = built_model(dict(params, n_layers_enc=1), model_file, data)
        _, three = built_model(dict(params, n_layers_enc=3), model_file, data)
        np.testing.assert_array_equal(one, three, err_msg=model_file)


def test_seed_sets_the_output_layer(tmp_path):
    """The output layer has no fixed seed of its own; another seed gives other weights."""
    params, _ = write_config(tmp_path)
    data = synthetic_returns()
    first, _ = built_model(dict(params, seed=1), "models/transformer.yaml", data)
    first_kernel = first.output_dense.get_weights()[0]
    second, _ = built_model(dict(params, seed=2), "models/transformer.yaml", data)
    assert not np.allclose(first_kernel, second.output_dense.get_weights()[0])


def test_readout_and_target_scale(tmp_path):
    """The last-step readout and the output scale train and give forecasts of return size."""
    params, _ = write_config(tmp_path, readout="last", target_scale=100)
    data = synthetic_returns()
    model, _ = built_model(params, "models/patterndecoder.yaml", data)
    model.compile(loss="mae", optimizer="adam")
    model.fit(WindowedDataset(params)(data, shuffle=True), epochs=1, verbose=0)
    forecasts = model.predict(WindowedDataset(params)(data), verbose=0)
    assert np.isfinite(forecasts).all()
    assert np.abs(forecasts).max() < 0.1
