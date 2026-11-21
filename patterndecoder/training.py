"""
Training helpers: random seeds and the chronological validation split.

Used by compile_and_train and load_model in utils.py.
"""

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
