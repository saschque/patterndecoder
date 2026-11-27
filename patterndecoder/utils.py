# pylint: disable=E1101, R0913, R0903, R0917, R0902, R0801, R0914
"""
Utilities for dataset handling, preprocessing, and model training in machine learning projects.

This module provides functions and classes to facilitate data loading, transformation,
and preparation for model training and evaluation. It supports various formats and
frameworks, enabling efficient data management for time series forecasting and other tasks.

Key Features:
- Configuration loading from YAML (market data: see `patterndecoder.data`).
- Configuration management through YAML files.
- Preprocessing techniques such as calendar feature engineering (dummy and cyclical).
- Dataset splitting into train/test sets.
- Sliding window creation for time series data via `WindowedDataset`.
- Model compilation and training with callbacks (checkpoints, early stopping).
- Model evaluation metrics (RMSE, MAE).
- Training history persistence and loading.
- Visualization of model performance through plots and tables.

Classes:
- `WindowedDataset`: Generates sliding windows of time series data for model training.
- `Naive`: Implements a simple naive forecasting baseline model.
- `Forecasts`: Evaluates and visualizes model performance on train/test datasets.

Functions:
- `load_config()`: Loads configuration parameters from a YAML file.
- `compile_and_train()`: Compiles and trains a Keras model with callbacks.
- `get_model_performance()`: Computes metrics (MAE, RMSE) and predictions for a model.
- `get_rmse()`: Computes Root Mean Squared Error (RMSE).
- `get_mae()`: Computes Mean Absolute Error (MAE).
- `add_calendar_dummies()`: Generates one-hot encoded calendar features.
- `add_cyclical_calendar_features()`: Encodes calendar features as cyclical sine/cosine.
- `save_training_history()`: Persists training history to disk as JSON.
- `load_training_history()`: Loads persisted training history from disk.
"""

import importlib
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from prettytable import PrettyTable
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from patterndecoder.training import set_seed, split_train_validation


def string_to_object(path):
    """
    Converts a string representation of a class name to the actual class object.

    Args:
        path (str): String representation of the class name, including the module path.
            Example: "tensorflow.keras.optimizers.Adam"
    Returns:
        object: The class object corresponding to the provided string path.
    """
    # Split the string into module path and class name
    if "." not in path:
        raise ValueError(
            f"Expected a dotted path such as 'tensorflow.keras.optimizers.Adam', got '{path}'"
        )
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_model(params, path="models/example.yaml"):
    """
    Loads the specified model

    Args:
        path (str): String representation of the yaml containing model specs.
            An optional ``params`` mapping in it overrides config entries for
            this model unless ``params["model_overrides"]`` is False.
    Returns:
        object: The class object corresponding to the provided yaml.
    Note: attention is passed as a (class, kwargs) spec, not an instance,
        so each layer/role builds its own via build_attention().
    """
    with open(path, encoding="utf-8") as file:
        model_params = yaml.safe_load(file)
        name = model_params["name"]
        # Optional per-model settings (e.g. layer counts) override the config
        if params.get("model_overrides", True):
            params = {**params, **model_params.get("params", {})}
        attn_kwargs = {"n_heads": params["n_heads"], "d_model": params["d_model"]}
        attn_spec = (string_to_object(model_params["attention"]), attn_kwargs)
        set_seed(params)
        model = string_to_object(model_params["architecture"])(params, attn_spec, name)
    return model


def load_config(path="config/config.yaml"):
    """
    Loads configuration parameters from a YAML file.

    Args:
        path (str): Path to the configuration file. Default is 'config/config.yaml'.

    Returns:
        dict: All configuration parameters.
    """

    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


# Settings stored with every training history, so a run can be reproduced
HISTORY_PARAMS = [
    "window_size", "forecast_horizon", "batch_size", "optimizer", "learning_rate",
    "loss", "metrics", "epochs", "early_stopping", "patience", "validation_split",
    "seed", "dropout", "l1_reg", "l2_reg", "d_model", "n_heads", "units",
    "n_layers_enc", "n_layers_dec", "input_scale", "target_scale", "readout", "series",
    "calendar", "expiry_features",
]


def save_training_history(
    history,
    model_name,
    params,
    out_dir="training_histories",
    suffix=None,
):
    """
    Persists a Keras History object to disk as JSON.

    Args:
        history (tf.keras.callbacks.History): History returned by model.fit
        model_name (str): Name of the model
        params (dict): Training/config parameters (stored for reproducibility)
        out_dir (str): Directory where histories are stored
        suffix (str): Optional suffix (e.g. run id)
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{suffix}" if suffix else ""

    filename = f"{model_name}_history_{suffix}.json"
    path = Path(out_dir) / filename

    payload = {
        "model": model_name,
        "timestamp": timestamp,
        "params": {key: params.get(key) for key in HISTORY_PARAMS},
        "history": history.history,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return path


def load_training_history(
    model_name,
    out_dir="training_histories",
    suffix=None,
):
    """
    Loads a persisted training history saved via `save_training_history`.

    Args:
        model_name (str): Name of the model
        out_dir (str): Directory containing saved histories
        suffix (str | None): Optional suffix used during saving

    Returns:
        dict: Loaded history payload (params + history)
    """
    out_dir = Path(out_dir)

    if suffix is None:
        pattern = f"{model_name}_history_*.json"
    else:
        pattern = f"{model_name}_history__{suffix}.json"

    candidates = sorted(
        out_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No training history found for model='{model_name}', suffix='{suffix}'"
        )

    path = candidates[0]

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return dict_to_history(payload)


def dict_to_history(history_dict):
    """
    Convert a persisted history dict into a tf.keras.callbacks.History object.

    Args:
        history_dict (dict): Payload written by `save_training_history`; its
            ``history`` field maps each metric to a list of per-epoch values.

    Returns:
        tf.keras.callbacks.History
    """
    history = tf.keras.callbacks.History()

    # Attach history
    history.history = history_dict["history"]

    # Infer epochs from metric length
    first_metric = next(iter(history.history.values()))
    history.epoch = list(range(len(first_metric)))

    return history


def compile_and_train(model, data, config_path="config/config.yaml"):
    """
    Compiles and trains a Keras model with the settings in a config file.

    The last ``validation_split`` share of ``data`` is held out (chronologically)
    as validation set. The checkpoint keeps the epoch with the lowest validation
    metric, early stopping watches the same metric, and the best weights are
    loaded at the end. With ``training: False`` the model is only built and the
    stored weights and history are loaded.

    Args:
        model (tf.keras.Model): The Keras model to compile and train.
        data (pd.DataFrame): Training data (log returns) in chronological order.
        config_path (str): Path of the config file (loss, optimizer, learning
            rate, epochs, seed, validation_split, early stopping, ...).

    Returns:
        tuple: The training history and the model with the best weights.
    """
    # Load configuration parameters
    params = load_config(config_path)

    # Chronological validation split (last validation_split share of data)
    fit_data, val_data = split_train_validation(data, params)
    windows = WindowedDataset(params)
    train_ds = windows(fit_data, shuffle=True)
    val_ds = windows(val_data) if val_data is not None else None
    monitor = ("val_" if val_ds is not None else "") + params["metrics"][0]
    set_seed(params)

    # Model checkpoint to save the best model
    file_path = params["tmp_weights_file"] + model.name + ".keras"

    callbacks = []
    if params["restore_best"]:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            file_path,
            monitor=monitor,
            mode="min",
            save_best_only=True,
            verbose=params["verbose"],
        )
        callbacks.append(checkpoint)

    # Early stopping
    if params["early_stopping"]:
        earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="min",
            restore_best_weights=False,
            patience=params["patience"],
        )
        callbacks.append(earlystopping)

    # Compile the model
    optimizer = string_to_object(params["optimizer"])(
        learning_rate=params["learning_rate"]
    )

    model.compile(loss=params["loss"], optimizer=optimizer, metrics=params["metrics"])

    if params["training"] is False:
        model.fit(train_ds, epochs=1, verbose=0)
        model.load_weights(file_path)
        history = load_training_history(
            model_name=model.name, out_dir=params["tmp_history_file"]
        )
        return history, model

    # Train the model
    history = model.fit(
        train_ds,
        epochs=params["epochs"],
        callbacks=callbacks,
        validation_data=val_ds,
        verbose=params["verbose"],
    )

    # Persist training history in training mode
    if params["training"] is True:
        save_training_history(
            history=history,
            out_dir=params["tmp_history_file"],
            model_name=model.name,
            params=params,
        )

    # Load best weights (only written when restore_best is enabled)
    if params["restore_best"]:
        model.load_weights(file_path)

    return history, model


def add_calendar_dummies(df):
    """
    Generate calendar dummy variables from a DataFrame's DatetimeIndex.

    Creates one-hot encoded features for day of week, day of month, and month
    from the DatetimeIndex of the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        DataFrame with one-hot encoded calendar features (dayofweek, dayofmonth, month).

    Raises
    ------
    ValueError
        If the DataFrame index is not a DatetimeIndex.

    Examples
    --------
    >>> dates = pd.date_range('2023-01-01', periods=5)
    >>> df = pd.DataFrame({'value': range(5)}, index=dates)
    >>> cal_dummies = add_calendar_dummies(df)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    cal = pd.DataFrame(index=df.index)

    cal["dayofweek"] = df.index.dayofweek  # 0–6
    cal["dayofmonth"] = df.index.day  # 1–31
    cal["month"] = df.index.month  # 1–12

    cal = pd.get_dummies(
        cal, columns=["dayofweek", "dayofmonth", "month"], drop_first=False
    )

    return cal


def add_cyclical_calendar_features(df):
    """
    Add cyclical calendar features to a DataFrame with DatetimeIndex.

    Encodes temporal information (day of week, day of month, and month of year)
    as cyclical sine and cosine features to capture the periodic nature of calendar
    patterns while maintaining mathematical continuity.

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex.

    Returns:
        pd.DataFrame: DataFrame with cyclical calendar features:
            - dow_sin, dow_cos: Day of week (0-6) encoded cyclically
            - dom_sin, dom_cos: Day of month (1-31) encoded cyclically
            - month_sin, month_cos: Month of year (1-12) encoded cyclically

    Raises:
        ValueError: If the DataFrame index is not a DatetimeIndex.

    Example:
        >>> df = pd.DataFrame(index=pd.date_range('2023-01-01', periods=3))
        >>> features = add_cyclical_calendar_features(df)
        >>> features.shape
        (3, 6)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    cal = pd.DataFrame(index=df.index)

    # ---- day of week (0–6) ----
    dow = df.index.dayofweek
    cal["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    cal["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # ---- day of month (1–31) ----
    dom = df.index.day
    cal["dom_sin"] = np.sin(2 * np.pi * (dom - 1) / 31)
    cal["dom_cos"] = np.cos(2 * np.pi * (dom - 1) / 31)

    # ---- month of year (1–12) ----
    month = df.index.month
    cal["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    cal["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    return cal


def add_expiry_dummies(dates):
    """
    Dummies for the third Friday of a month and for the third Friday of March,
    June, September and December, when options and futures expire.
    """
    # pylint cannot see the date attributes of a DatetimeIndex
    # pylint: disable=no-member
    dates = pd.DatetimeIndex(dates)
    third_friday = (dates.dayofweek == 4) & (dates.day >= 15) & (dates.day <= 21)
    quarterly = third_friday & dates.month.isin((3, 6, 9, 12))
    columns = {"third_friday": third_friday, "quarterly_expiry": quarterly}
    return pd.DataFrame(columns, index=dates).astype(float)


def calendar_inputs(index, calendar="current", expiry=False):
    """
    Calendar features of each input row: of the row's own day ("current") or of
    the next trading day ("next"), which is known in advance, so that the last
    row of a window carries the calendar of the target day. With ``expiry`` the
    expiry dummies of add_expiry_dummies are added.
    """
    dates = pd.DatetimeIndex(index)
    if calendar == "next":
        dates = dates[1:].append(pd.DatetimeIndex([dates[-1] + pd.offsets.BDay(1)]))
    elif calendar != "current":
        raise ValueError(f"calendar must be 'current' or 'next', not {calendar!r}")
    features = add_cyclical_calendar_features(pd.DataFrame(index=dates))
    if expiry:
        features = pd.concat([features, add_expiry_dummies(dates)], axis=1)
    features.index = index
    return features


class WindowedDataset:
    """
    Generates sliding windows of time series data for model training.

    This class creates TensorFlow datasets from pandas DataFrames, enabling models
    to train on sequences of historical data to predict future values based on
    the specified forecast horizon.

    Args:
        params (dict): Configuration with window_size, forecast_horizon,
            batch_size, stride, shuffle_buffer, columns, target_column and
            optionally input_scale (factor for the return inputs, default 1),
            calendar ("current" or "next") and expiry_features (see calendar_inputs).

    Methods:
        __call__(dataframe, shuffle=False):
            Generates a TensorFlow dataset of sliding windows from a pandas DataFrame.
    """

    def __init__(self, params):
        """
        Initializes the `WindowedDataset` class for generating sliding windows of time series data.

        Args:
            params (dict): Dictionary containing parameters for window generation:
                - window_size (int): Number of time steps in each input window.
                - forecast_horizon (int): Number of future time steps to predict.
                - batch_size (int): Batch size for the dataset.
                - stride (int): Stride between consecutive windows.
                - shuffle_buffer (int): Buffer size for shuffling windows.
                - columns (list[str]): List of column names to use as features.

        Returns:
            None
        """

        self.window_size = params["window_size"]
        self.forecast_horizon = params["forecast_horizon"]
        self.batch_size = params["batch_size"]
        self.stride = params["stride"]
        self.shuffle_buffer = params["shuffle_buffer"]
        self.columns = params["columns"]
        self.target_column = params["target_column"]
        # Factor for the return inputs (not the targets): 1 keeps raw log returns
        self.input_scale = params.get("input_scale", 1)
        # Calendar of each row's own day or of the next trading day (see calendar_inputs)
        self.calendar = params.get("calendar", "current")
        self.expiry = bool(params.get("expiry_features", False))

    def __call__(
        self, dataframe, shuffle=False, training_mode=True, multi_horizon=False
    ):
        dataframe = dataframe.sort_index()

        # -------- X (inputs) --------
        calendar_features = calendar_inputs(dataframe.index, self.calendar, self.expiry)
        base_features = dataframe[self.columns] * self.input_scale

        x = pd.concat([base_features, calendar_features], axis=1)
        x_array = x.values.astype("float32")

        # -------- y (target) --------
        if self.target_column in dataframe.columns:
            y_array = dataframe[self.target_column].values.astype("float32")
        elif len(dataframe.columns) == 1:
            y_array = dataframe.iloc[:, 0].values.astype("float32")
        else:
            raise KeyError(
                f"Target column '{self.target_column}' not found in DataFrame"
            )

        min_length = self.window_size + self.forecast_horizon
        if len(x_array) < min_length:
            raise ValueError(f"Need at least {min_length} samples, got {len(x_array)}")

        inputs = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=x_array[: -self.forecast_horizon],
            targets=None,
            sequence_length=self.window_size,
            sequence_stride=self.stride,
            shuffle=False,
            batch_size=self.batch_size,
        )

        if multi_horizon:
            targets = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=y_array[self.window_size :],
                targets=None,
                sequence_length=self.forecast_horizon,
                sequence_stride=self.stride,
                shuffle=False,
                batch_size=self.batch_size,
            )
        else:
            targets = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=y_array[self.window_size + self.forecast_horizon - 1 :],
                targets=None,
                sequence_length=1,
                sequence_stride=self.stride,
                shuffle=False,
                batch_size=self.batch_size,
            )

        dataset = tf.data.Dataset.zip((inputs, targets))

        if training_mode and shuffle:
            dataset = dataset.shuffle(self.shuffle_buffer)

        return dataset.prefetch(tf.data.AUTOTUNE)


def get_rmse(test_data, predicted_data):
    """
    Computes the Root Mean Squared Error (RMSE) between actual and predicted values.

    Args:
        test_data (np.array or list): Ground truth values.
        predicted_data (np.array or list): Predicted values.

    Returns:
        float: RMSE value indicating prediction accuracy. Lower values indicate better performance.
    """

    mse = np.mean(np.square(test_data - predicted_data))
    return np.sqrt(mse)


def get_mae(test_data, predicted_data):
    """
    Computes the Mean Absolute Error (MAE) between actual and predicted values.

    Args:
        test_data (np.array or list): Ground truth values.
        predicted_data (np.array or list): Predicted values.

    Returns:
        float: MAE value indicating prediction accuracy. Lower values indicate better performance.
    """
    mae_value = np.mean(np.abs(test_data - predicted_data))
    return mae_value


class MovingAverage:
    """
    Implements a moving average forecasting model that predicts the average
    of the last `window_size` values of the target column.

    Args:
        forecast_horizon (int): Number of predictions per sample.
        window_size (int): Number of historical steps to average over.
        input_scale (float): input_scale of the windowed dataset, undone here.

    Methods:
        predict(data, verbose=False):
            Generates moving average forecasts for all samples in the dataset.
    """

    def __init__(self, forecast_horizon, window_size, input_scale=1):
        self.name = "MovingAverage"
        self.forecast_horizon = forecast_horizon
        self.window_size = window_size
        self.input_scale = input_scale

    def predict(self, data, verbose=False):
        """
        Generates moving average forecasts for all samples in the dataset.

        Args:
            data (tf.data.Dataset): Windowed dataset of (X, y) tuples.
            verbose (bool): Whether to display progress.

        Returns:
            np.array: Shape (num_samples, forecast_horizon), matching the target.
        """
        if verbose:
            print("MovingAverage Model: Predicting...")

        predictions = []

        for sample in data.unbatch():
            # Unpack tuple
            x, _ = sample  # ignore target

            # Convert to numpy if Tensor
            if hasattr(x, "numpy"):
                x = x.numpy()

            # log returns are in the first column (scaled by input_scale)
            target_series = x[-self.window_size :, 0] / self.input_scale

            # Compute mean over the window
            window_mean = np.mean(target_series)

            # Repeat for forecast horizon
            forecast = np.full((self.forecast_horizon,), window_mean)
            predictions.append(forecast)

        return np.array(predictions)


class Naive:
    """
    Implements a simple naive forecasting model that returns zeros as predictions.

    Args:
        forecast_horizon (int): Number of predictions per sample.
        window_size (int): Size of input windows used in forecasting.

    Methods:
        predict(data, verbose=False):
            Generates naive predictions by returning zeros for all samples in the dataset.
    """

    def __init__(self, forecast_horizon, window_size):
        """
        Initializes the `Naive` class, a simple forecasting model that returns zeros as predictions.

        Args:
            forecast_horizon (int): Number of predictions per sample.
            window_size (int): Size of input windows used in forecasting.

        Returns:
            None
        """
        self.name = "Naive"

        self.forecast_horizon = forecast_horizon
        self.window_size = window_size

    def predict(self, data, verbose=False):
        """
        Generates naive predictions by returning zeros for all samples in the dataset.

        Args:
            data (tf.data.Dataset): Input dataset containing samples for prediction.
            verbose (bool): Whether to display progress during prediction. Default is False.

        Returns:
            np.array: Array filled with zeros matching the shape `(num_samples, forecast_horizon)`.
        """

        if verbose:
            print("Naive Model: Predicting...")
        unbatched_dataset = data.unbatch()
        count = 0
        for _ in unbatched_dataset:
            count += 1

        return np.zeros(shape=(count, self.forecast_horizon))


def get_model_performance(model, params, dataset):
    """
    Computes MAE and RMSE of a model's one-step forecasts on one dataset.

    Args:
        model: Trained model or baseline with a ``predict`` method.
        params (dict): Configuration (window_size and windowing settings).
        dataset (pd.DataFrame): Log returns to forecast (train or test set).

    Returns:
        dict: MAE, RMSE, the predictions and the price-path curve.
    """

    # Create windowed dataset (non-shuffled for temporal integrity)
    window = WindowedDataset(params)
    windowed_dataset = window(dataset, shuffle=False, training_mode=False)

    # Make predictions
    if isinstance(model, ARIMAResultsWrapper):
        predictions = model.predict(start=params["window_size"] + 1, end=len(dataset))
        # ARIMA predictions are already aligned with the correct indices
    else:
        predictions = model.predict(windowed_dataset, verbose=0)
        # predictions shape: (num_windows, forecast_horizon, num_features)
        # Since forecast_horizon=1, squeeze that dimension
        if len(predictions.shape) == 3:
            predictions = predictions.squeeze(
                axis=1
            )  # Shape: (num_windows, num_features)

    # Calculate the correct number of predictions
    num_predictions = len(dataset) - params["window_size"]

    # Ensure predictions match expected length
    if len(predictions) != num_predictions:
        raise ValueError(
            f"Length {len(predictions)} doesn't match expected {num_predictions}"
        )

    # Create DataFrame for predictions - align with correct indices
    predictions_df = pd.DataFrame(
        predictions,
        index=dataset.index[params["window_size"] :],  # Use original dataset indices
        columns=dataset.columns
        if hasattr(dataset, "columns")
        else [f"feature_{i}" for i in range(predictions.shape[1])],
    )

    # Create DataFrame for actual values - these are the target values we're predicting
    actuals_df = pd.DataFrame(
        dataset.iloc[params["window_size"] :].values,
        index=dataset.index[params["window_size"] :],
        columns=dataset.columns
        if hasattr(dataset, "columns")
        else [f"feature_{i}" for i in range(dataset.shape[1])],
    )

    # Calculate metrics
    mae = round(get_mae(predictions_df.values, actuals_df.values), 6)
    rmse = round(get_rmse(predictions_df.values, actuals_df.values), 6)

    # Price-path curve: re-anchored to the true previous value at every step
    predictions_cum = actuals_df.cumsum() - (actuals_df - predictions_df)

    return {
        "mae": mae,
        "rmse": rmse,
        "predictions": predictions_df,
        "predictions_cum": predictions_cum,
    }


class Forecasts:
    """
    Evaluates and visualizes model performances on train/test datasets.

    This class computes metrics like MAE and RMSE, generates forecasts using models,
    and visualizes performance through plots and tables.

    Args:
        models (list): Trained models and baselines (each with ``name`` and ``predict``).
        params (dict): Configuration (window size and windowing settings).
        data (list): [train_data, test_data] as DataFrames of log returns.

    Methods:
        generate_model_performances():
            Computes metrics like MAE/RMSE for each model on train/test datasets and stores results
            in a dictionary.

        get_forecasts(dataset, model):
            Generates forecasts using a specified model on a given dataset.

        plot_model_performances():
            Plots graphs comparing actual vs predicted values for each model on train/test datasets.

        print_model_metrics():
            Displays metrics like MAE/RMSE in tabular format for all models evaluated.
    """

    def __init__(self, models, params, data):
        """
        Initializes the `Forecasts` class for evaluating and visualizing model performances.

        Args:
            models (list): Trained models and baselines (each with ``name`` and ``predict``).
            params (dict): Configuration (window size and windowing settings).
            data (list): [train_data, test_data] as DataFrames of log returns.

        Returns:
            None
        """

        self.models = models
        self.params = params
        self.train_data = data[0]
        self.test_data = data[1]
        self.model_performances = self.generate_model_performances()

    def generate_model_performances(self):
        """
        Computes metrics like MAE/RMSE for each model on train/test datasets
        and stores results in a dictionary.

        Returns:
            list[dict]: List of dictionaries containing metrics and predictions for each
                        model evaluated.
        """

        window_size = self.params["window_size"]
        model_performances = []
        for model in self.models:
            if isinstance(model, ARIMAResultsWrapper):
                model.name = "ARIMA"
            train_performance = get_model_performance(
                model, self.params, self.train_data
            )
            test_performance = get_model_performance(model, self.params, self.test_data)

            model_performances.append(
                {
                    "name": model.name,
                    "train_prediction": np.exp(train_performance["predictions_cum"]),
                    "train_actuals": np.exp(self.train_data[window_size:].cumsum()),
                    "train_log_prediction": train_performance["predictions"],
                    "train_log_actuals": self.train_data[window_size:],
                    "test_prediction": np.exp(test_performance["predictions_cum"]),
                    "test_actuals": np.exp(self.test_data[window_size:].cumsum()),
                    "test_log_prediction": test_performance["predictions"],
                    "test_log_actuals": self.test_data[window_size:],
                    "color": "red",
                    "mae_train": train_performance["mae"],
                    "mae_test": test_performance["mae"],
                    "rmse_train": train_performance["rmse"],
                    "rmse_test": test_performance["rmse"],
                }
            )
        return model_performances

    def plot_model_performances(self):
        """
        Plots the performance of each model on train and test datasets.

        For each model, four subplots compare actual values (black) with the
        model's forecasts: price path and log returns, each in-sample (train set)
        and out-of-sample (test set).

        Args:
            None

        Returns:
            None: Displays the plots for each model's performance.
        """

        for performance in self.model_performances:
            # Plot the graph
            __, axs = plt.subplots(2, 2, figsize=(20, 6))
            plt.subplots_adjust(hspace=0.5, wspace=0.125)

            # Subplot 1: in-sample cumulative return predictions
            axs[0][0].grid()
            axs[0][0].plot(
                performance["train_actuals"], color="black", label="Train Set"
            )
            axs[0][0].plot(
                performance["train_prediction"],
                color=performance["color"],
                label=performance["name"],
            )
            axs[0][0].set_title(performance["name"] + " (in-sample)")
            axs[0][0].tick_params(axis="x", rotation=30)
            axs[0][0].legend(loc="lower right")

            # Subplot 2: out-of-sample cumulative return predictions
            axs[0][1].grid()
            axs[0][1].plot(performance["test_actuals"], color="black", label="Test Set")
            axs[0][1].plot(
                performance["test_prediction"],
                color=performance["color"],
                label=performance["name"],
            )
            axs[0][1].set_title(performance["name"] + " (out-of-sample)")
            axs[0][1].tick_params(axis="x", rotation=30)
            axs[0][1].legend(loc="lower right")

            # Subplot 3: in-sample log return predictions
            axs[1][0].grid()
            axs[1][0].plot(
                performance["train_log_actuals"], color="black", label="Train Set"
            )
            axs[1][0].plot(
                performance["train_log_prediction"],
                color=performance["color"],
                label=performance["name"],
            )
            axs[1][0].set_title(performance["name"] + " (in-sample)")
            axs[1][0].tick_params(axis="x", rotation=30)
            axs[1][0].legend(loc="lower right")

            # Subplot 4: out-of-sample log return predictions
            axs[1][1].grid()
            axs[1][1].plot(
                performance["test_log_actuals"], color="black", label="Test Set"
            )
            axs[1][1].plot(
                performance["test_log_prediction"],
                color=performance["color"],
                label=performance["name"],
            )
            axs[1][1].set_title(performance["name"] + " (out-of-sample)")
            axs[1][1].tick_params(axis="x", rotation=30)
            axs[1][1].legend(loc="lower right")
        plt.show()

    def print_model_metrics(self):
        """
        Prints evaluation metrics (MAE and RMSE) for each model in a tabular format.

        This method summarizes the performance of all models on both training and test
        datasets. It uses the `PrettyTable` library to display the following metrics
        for each model:
        - Mean Absolute Error (MAE) on the training set.
        - Mean Absolute Error (MAE) on the test set.
        - Root Mean Squared Error (RMSE) on the training set.
        - Root Mean Squared Error (RMSE) on the test set.

        Args:
            None

        Returns:
            None: Outputs a formatted table of metrics to the console.

        Example:
            >>> forecasts.print_model_metrics()
            +-------+-----------+----------+------------+-----------+
            | Model | MAE Train | MAE Test | RMSE Train | RMSE Test |
            +-------+-----------+----------+------------+-----------+
            |  LSTM |   0.0123  |  0.0156  |   0.0201   |   0.0254  |
            +-------+-----------+----------+------------+-----------+
        """

        # Initialize table
        table = PrettyTable()

        table.field_names = [
            "Model",
            "MAE Train",
            "MAE Test",
            "RMSE Train",
            "RMSE Test",
        ]
        for performance in self.model_performances:
            # fill the table
            table.add_row(
                [
                    performance["name"],
                    performance["mae_train"],
                    performance["mae_test"],
                    performance["rmse_train"],
                    performance["rmse_test"],
                ]
            )

        print(table.get_string(sortby="MAE Test"))
