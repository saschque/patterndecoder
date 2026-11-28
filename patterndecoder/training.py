# pylint: disable=E1101
"""
Training helpers: random seeds, the chronological validation split and the
targets of forecasts over several days.

Used by compile_and_train, load_model and WindowedDataset in utils.py.
"""

import numpy as np
import pandas as pd
import tensorflow as tf


def set_seed(params):
    """
    Seeds Python, NumPy and TensorFlow if the config defines ``seed``.

    Call it before a model is built (weights and dropout seeds are drawn when
    the layers are created) and again before training (shuffling).

    Args:
        params (dict): Configuration; ``seed`` (int or None) is optional.
    """
    seed = params.get("seed")
    if seed is not None:
        tf.keras.utils.set_random_seed(int(seed))


def split_train_validation(data, params):
    """
    Splits a training DataFrame chronologically into a fit and a validation part.

    The last ``validation_split`` share of the rows is used for validation. The
    validation part starts ``window_size`` rows earlier so that its first target
    has a full input window; those context rows are inputs only, never targets
    of both parts.

    Args:
        data (pd.DataFrame): Training data in chronological order.
        params (dict): Configuration with ``validation_split`` (0 disables the
            split) and ``window_size``.

    Returns:
        tuple: (fit_data, validation_data); validation_data is None when
        ``validation_split`` is 0 or missing.
    """
    share = float(params.get("validation_split") or 0)
    if share <= 0:
        return data, None
    n_val = int(len(data) * share)
    if n_val <= params["forecast_horizon"]:
        return data, None
    fit_data = data.iloc[:-n_val]
    validation_data = data.iloc[-(n_val + params["window_size"]) :]
    return fit_data, validation_data


def forward_sums(values, days=1):
    """
    The sum of every value and the next ``days - 1`` values: the target of a
    forecast of the return over the next ``days`` days, made at the close of
    the day before the first of them.

    The last ``days - 1`` positions are dropped, so a sum never reaches past the
    data it is computed from: summed within each part (fit, validation, test),
    no target reaches into a later part. With ``days`` 1 the values are
    returned unchanged.

    Args:
        values (np.ndarray, pd.Series or pd.DataFrame): Daily values in
            chronological order; arrays are summed along the first axis.
        days (int): Number of days in each sum.

    Returns:
        Same type as ``values``, ``days - 1`` rows shorter; a pandas sum keeps
        the date of its first day.
    """
    days = int(days)
    if days == 1:
        return values
    if isinstance(values, (pd.Series, pd.DataFrame)):
        return values.rolling(days).sum().shift(1 - days).iloc[: len(values) - days + 1]
    return np.lib.stride_tricks.sliding_window_view(values, days, axis=0).sum(axis=-1)
