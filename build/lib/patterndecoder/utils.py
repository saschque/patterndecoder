# pylint: disable=E1101, R0913, R0903, R0917, R0902, R0801, R0914
"""
Utilities for dataset handling and preprocessing in machine learning projects.

This module provides functions and classes to facilitate data loading, transformation, 
and preparation for model training and evaluation. It supports various formats and 
frameworks, enabling efficient data management for time series forecasting and other tasks.

Key Features:
- Data loading from sources like CSV, JSON, and Yahoo Finance.
- Preprocessing techniques such as normalization and feature engineering.
- Dataset splitting into train/test/validation sets.
- Sliding window creation for time series data.
- Model evaluation metrics (RMSE, MAE).
- Visualization of model performance.

Classes:
- `WindowedDataset`: Generates sliding windows of time series data.
- `Naive`: Implements a simple naive forecasting model.
- `Forecasts`: Evaluates and visualizes model performance.

Functions:
- `get_stock_data`: Downloads historical stock price data from Yahoo Finance.
- `load_config_json`: Loads configuration parameters from a JSON file.
- `train_model`: Compiles and trains a Keras model.
- `rmse`: Computes Root Mean Squared Error (RMSE).
- `mae`: Computes Mean Absolute Error (MAE).
"""

import json
import importlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from prettytable import PrettyTable
from statsmodels.tsa.arima.model import ARIMAResultsWrapper

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
    if '.' not in path:
        getattr(path)
    module_path, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def get_stock_data(params, download=False):
    """
    Downloads historical stock price data from Yahoo Finance.

    Args:
        params (dict): Dictionary containing parameters:
            - ticker (str): Stock ticker symbol.
            - path (str): Directory to save the downloaded data. Default is 'data/'.
            - period (str): Historical period to download (e.g., '3y' for 3 years). Default is '3y'.
            - interval (str): Frequency of the data (e.g., '1d' for daily). Default is '1d'.
            - type (str): File type to save the data (e.g., 'csv').
        download (bool): Whether to download the data. Default is False.

    Returns:
        pd.DataFrame: DataFrame containing historical stock price data.
    """

    path = params["path"]
    ticker = params["ticker"]
    data_type = params["type"]
    # Download the DataFrame from Yahoo Finance
    if download:
        df = yf.download(tickers=[params["ticker"]],
                         period=params["period"],
                         interval=params["interval"],
                         auto_adjust=False,
                         prepost=True,
                         threads=True,
                         proxy=None,
                         progress=False
                         )

        df.columns = df.columns.get_level_values('Price')
        df.to_csv(f"{path}{ticker}.{data_type}")

    # Read the DataFrame from path
    series_dataframe = pd.read_csv(f"{path}{ticker}.{data_type}",
                                   index_col=0,
                                   sep=",")
    series_dataframe.index = pd.to_datetime(series_dataframe.index,
                                           format='%Y-%m-%d',
                                           errors='coerce')

    return series_dataframe

def load_config_json(path="config/config.json"):
    """
    Loads configuration parameters from a JSON file.

    Args:
        path (str): Path to the configuration file. Default is 'config/config.json'.

    Returns:
        tuple: A tuple containing dictionaries for temporal window parameters, training parameters,
            transformer parameters, and data parameters.
    """

    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    return (config["temporal_window_params"],
            config["training_params"],
            config["transformer_params"],
            config["data_params"])

def split_dataset(dataset, test_ratio=0.30):
    """Splits a panda dataframe in two."""
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]

def train_model(model, data, name, config_path="config/config.json"):
    """
    Compiles and trains a Keras model using specified training parameters.

    Args:
        model (tf.keras.Model): The Keras model to compile and train.
        train_set (tf.data.Dataset): Training dataset.
        training_params (dict): Dictionary containing training parameters such as:
            - loss (str or list): Loss function(s) for training.
            - learning_rate (float): Learning rate for the optimizer.
            - metrics (list): List of metrics to evaluate during training.
            - epochs (int): Number of epochs for training.
            - verbose (bool): Whether to display training progress. Default is False.

    Returns:
        tuple: A tuple containing the training history and the trained model.
    """
    # Load configuration parameters
    (window_params, training_params, __, data_params) = load_config_json(config_path)

    # Generating the training and validation sets
    #train_size = int(data.shape[0] * (1-validation_split))
    #train_data = data.iloc[:train_size]
    #val_data = data.iloc[train_size:]
    #train_set, val_set = split_dataset(data, validation_split)
    #val_ds = WindowedDataset(window_params, data_params)(data, shuffle=True)
    train_ds = WindowedDataset(window_params, data_params)(data, shuffle=True)

    # Model checkpoint to save the best model
    file_path = training_params["tmp_model_file"] + name +".keras"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        file_path,
        monitor=training_params["metrics"][0],
        mode="min",
        save_best_only=True,
        verbose=training_params["verbose"]
    )

    # Early stopping
    if training_params["early_stopping"]:
        earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor=training_params["metrics"][0],
            restore_best_weights=False,
            patience=training_params["patience"]
        )
        callbacks = [earlystopping,checkpoint]
    else:
        callbacks = [checkpoint]


    # Compile the model
    optimizer = string_to_object(training_params["optimizer"])(
        learning_rate=training_params["learning_rate"]
    )
    #optimizer = optimizer_class(learning_rate=training_params["learning_rate"])
    #optimizer = tf.keras.optimizers.Lion(learning_rate=training_params["learning_rate"])
    # For Lion optimizer, see:
    # https://arxiv.org/abs/2302.01107
    # https://arxiv.org/abs/2302.06675

    model.compile(
        loss=training_params["loss"],
        optimizer=optimizer,
        metrics=training_params["metrics"]
    )

    if training_params["training"] is False :
        model.fit(train_ds, epochs=1, verbose=0)
        model.load_weights(file_path)
        return None, model

    # Train the model
    history = model.fit(
        train_ds,
        epochs=training_params["epochs"],
        callbacks=callbacks,
        #validation_data=val_ds,
        verbose=training_params["verbose"]
    )

    # Load best weights
    model.load_weights(file_path)

    return history, model

class WindowedDataset():
    """
    Generates sliding windows of time series data for model training.

    This class creates TensorFlow datasets from pandas DataFrames, enabling models 
    to train on sequences of historical data to predict the next day's target.

    Args:
        window_params (dict): Parameters for generating windows, including:
            - observation_window_size: Number of time steps in each input window.
            - output_size: Number of predictions per window.
            - batch_size: Batch size for the dataset.
            - stride: Stride between consecutive windows.
            - shuffle_buffer: Buffer size for shuffling windows. Default is 1000.
        data_params (dict): Parameters specifying columns in the dataset.

    Methods:
        __call__(dataframe, shuffle=False):
            Generates a TensorFlow dataset of sliding windows from a pandas DataFrame.
    """

    def __init__(self, window_params, data_params):
        """
        Initializes the `WindowedDataset` class for generating sliding windows of time series data.

        Args:
            window_params (dict): Dictionary containing parameters for window generation:
                - observation_window_size (int): Number of time steps in each input window.
                - output_size (int): Number of predictions per window.
                - batch_size (int): Batch size for the dataset.
                - stride (int): Stride between consecutive windows.
                - shuffle_buffer (int): Buffer size for shuffling windows.
            data_params (dict): Dictionary containing data-specific parameters:
                - columns (list[str]): List of column names to use as features.

        Returns:
            None
        """

        self.window_size = window_params["observation_window_size"]
        self.output_size = window_params["output_size"]
        self.batch_size = window_params["batch_size"]
        self.stride = window_params["stride"]
        self.shuffle_buffer = window_params["shuffle_buffer"]
        self.columns = data_params["columns"]

    def __call__(self, dataframe, shuffle=False, training_mode=True):
        """
        Generates a TensorFlow Dataset of sliding windows from a pandas DataFrame.

        Args:
            dataframe (pd.DataFrame): Time series data containing features like
                Open, High, Low, Close, Volume.
            shuffle (bool): Whether to shuffle the dataset. Default is False.
            training_mode (bool): Whether to apply training-specific transformations. 
                Default is True.

        Returns:
            tf.data.Dataset: Dataset containing windows with input features (`x`) 
                and next-day targets (`y`).
        """

        # Ensure the dataframe is sorted by date
        dataframe = dataframe.sort_index()

        # Create a new column for the target (next day's value)
        dataframe['target'] = dataframe.shift(-1)

        # Create a dataset of sliding windows
        dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=dataframe[self.columns][:-1],  # Exclude last row (no target available)
            targets=dataframe[self.window_size:-1]['target'],  # Targets aligned with windows
            sequence_length=self.window_size,
            sequence_stride=self.stride,
            shuffle=False,
            batch_size=self.batch_size
        )

        if training_mode:
            if shuffle:
                dataset = dataset.shuffle(self.shuffle_buffer)

        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset


def rmse(test_data, predicted_data):
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

def mae(test_data, predicted_data):
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

class Naive():
    """
    Implements a simple naive forecasting model that returns zeros as predictions.

    Args:
        output_size (int): Number of predictions per sample.
        window_size (int): Size of input windows used in forecasting.

    Methods:
        predict(data, verbose=False):
            Generates naive predictions by returning zeros for all samples in the dataset.
    """

    def __init__(self, output_size, window_size):
        """
        Initializes the `Naive` class, a simple forecasting model that returns zeros as predictions.

        Args:
            output_size (int): Number of predictions per sample.
            window_size (int): Size of input windows used in forecasting.

        Returns:
            None
        """

        self.output_size = output_size
        self.window_size = window_size

    def predict(self, data, verbose=False):
        """
        Generates naive predictions by returning zeros for all samples in the dataset.

        Args:
            data (tf.data.Dataset): Input dataset containing samples for prediction.
            verbose (bool): Whether to display progress during prediction. Default is False.

        Returns:
            np.array: Array filled with zeros matching the shape `(num_samples, output_size)`.
        """

        if verbose:
            print("Naive Model: Predicting...")
        unbatched_dataset = data.unbatch()
        count = 0
        for _ in unbatched_dataset:
            count += 1

        return np.zeros(shape=(count, self.output_size))

class Forecasts():
    """
    Evaluates and visualizes model performances on train/test datasets.

    This class computes metrics like MAE and RMSE, generates forecasts using models,
    and visualizes performance through plots and tables.

    Args:
        models (list): List of tuples containing model name, instance, and color for plotting.
        train_data (pd.DataFrame): Training dataset with ground truth values.
        test_data (pd.DataFrame): Test dataset with ground truth values.
        window_params (dict): Parameters specifying sliding window dimensions and stride.
        data_params (dict): Parameters specifying columns in the dataset.

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

    def __init__(self, models, data, config_path="config/config.json"):
        """
        Initializes the `Forecasts` class for evaluating and visualizing model performances.

        Args:
            models (list[tuple]): List of tuples, where each tuple contains:
                - model_name (str): Name of the model.
                - model: Trained model instance.
                - color (str): Color used for plotting this model's results.
            train_data (pd.DataFrame): Training dataset containing ground truth values.
            test_data (pd.DataFrame): Test dataset containing ground truth values.
            window_params (dict): Dictionary containing parameters for sliding window generation:
                - observation_window_size (int): Number of time steps in each input window.
                - n_steps_ahead (int): Number of steps ahead to predict targets.
            data_params (dict): Dictionary containing data-specific parameters:
                - columns (list[str]): List of column names to use as features.

        Returns:
            None
        """

        self.models = models
        self.train_data = data[0]
        self.test_data = data[1]
        (self.window_params, __, __, self.data_params ) = load_config_json(config_path)
        self.window_size = self.window_params["observation_window_size"]
        self.model_performances = self.generate_model_performances()

    def generate_model_performances(self):
        """
        Computes metrics like MAE/RMSE for each model on train/test datasets 
        and stores results in a dictionary.

        Returns:
            list[dict]: List of dictionaries containing metrics and predictions for each 
                        model evaluated.
        """

        model_performances = []
        for model_name, attn_type, model, color, enabled in self.models:
            if enabled:
                train_predictions,train_actuals = self.get_forecasts(
                    self.train_data, model)
                test_predictions, test_actuals = self.get_forecasts(
                    self.test_data, model)

                mae_train = mae(
                    train_predictions.values,
                    train_actuals.values
                )
                mae_test = mae(
                    test_predictions.values,
                    test_actuals.values
                )
                rmse_train = rmse(
                    train_predictions.values,
                    train_actuals.values
                )
                rmse_test = rmse(
                    test_predictions.values,
                    test_actuals.values
                )

                train_returns_cum = train_actuals.cumsum()
                train_predictions_cum = train_actuals.cumsum() - (train_actuals - train_predictions)
                test_returns_cum = test_actuals.cumsum()
                test_predictions_cum = test_actuals.cumsum() - (test_actuals - test_predictions)

                model_performances.append({
                    "name": model_name,
                    "attn_type": attn_type,
                    "train_prediction": np.exp(train_predictions_cum),
                    "train_returns" : np.exp(train_returns_cum),
                    "train_log_prediction": train_predictions,
                    "train_log_returns" : train_actuals,
                    "test_prediction": np.exp(test_predictions_cum),
                    "test_returns": np.exp(test_returns_cum),
                    "test_log_prediction": test_predictions,
                    "test_log_returns": test_actuals,
                    "color" : color,
                    "mae_train": round(mae_train, 6),
                    "mae_test": round(mae_test, 6),
                    "rmse_train": round(rmse_train, 6),
                    "rmse_test": round(rmse_test, 6),
                })
        return model_performances

    def get_forecasts(self, dataset, model):
        """
        Generates forecasts using a specified model on a given dataset.

        Args:
            dataset (pd.DataFrame): Input time series data with ground truth values.
            model: Trained forecasting model capable of predicting future values.

        Returns:
            tuple: forecasts and ground truth values generated by the model.
        """
        window_size = self.window_params["observation_window_size"]

        # Datasets must not be shuffled to ensure temporal integrity when plotting
        window = WindowedDataset(self.window_params, self.data_params)
        windowed_dataset = window(dataset, shuffle=False, training_mode=False)

        # Make predictions
        # we handle ARIMA predictions seperately
        if isinstance(model, ARIMAResultsWrapper):
            predictions = model.predict(start=window_size+1, end=len(dataset)-1, verbose=0)
        else:
            predictions = model.predict(windowed_dataset, verbose=0)

        # Convert it to DataFrame object
        predictions_df = pd.DataFrame(
            predictions,
            index=dataset[window_size+1:].index,
            columns=self.data_params["columns"]
        )

        return predictions_df, dataset[window_size+1:]

    def plot_model_performances(self):
        """
        Plots the performance of each model on train and test datasets.

        This method generates two subplots for each model:
        1. In-sample predictions (train set): Compares the actual values with the predictions
            made by the model.
        2. Out-of-sample predictions (test set): Compares the actual values with the predictions
            made by the model.

        Each plot includes:
        - Actual values (ground truth) in red.
        - Model predictions in a specified color.

        Args:
            None

        Returns:
            None: Displays the plots for each model's performance.
        """

        for performance in self.model_performances:
            # Plot the graph
            __, axs = plt.subplots(2,2, figsize=(20, 6))
            plt.subplots_adjust(hspace=0.5, wspace=0.125)

            # Subplot 1: in-sample cumulative return predictions
            axs[0][0].grid()
            axs[0][0].plot(
                performance["train_returns"],
                color='red',
                label='Train Set'
            )
            axs[0][0].plot(
                performance["train_prediction"],
                color=performance["color"],
                label=performance["name"]+" "+ performance["attn_type"]
            )
            axs[0][0].set_title(
                performance["name"]+' '+ performance["attn_type"] + ' (in-sample)'
            )
            axs[0][0].tick_params(axis='x', rotation=30)
            axs[0][0].legend(loc='lower right')

            # Subplot 2: out-of-sample cumulative return predictions
            axs[0][1].grid()
            axs[0][1].plot(
                performance["test_returns"],
                color='red',
                label='Test Set'
            )
            axs[0][1].plot(
                performance["test_prediction"],
                color=performance["color"],
                label=performance["name"]+" "+ performance["attn_type"]
            )
            axs[0][1].set_title(
                performance["name"]+' '+ performance["attn_type"] + ' (out-of-sample)'
            )
            axs[0][1].tick_params(axis='x', rotation=30)
            axs[0][1].legend(loc='lower right')

            # Subplot 3: in-sample log return predictions
            axs[1][0].grid()
            axs[1][0].plot(
                performance["train_log_returns"],
                color='red',
                label='Train Set'
            )
            axs[1][0].plot(
                performance["train_log_prediction"],
                color=performance["color"],
                label=performance["name"]+" "+ performance["attn_type"]
            )
            axs[1][0].set_title(
                performance["name"]+' '+ performance["attn_type"] + ' (in-sample)'
            )
            axs[1][0].tick_params(axis='x', rotation=30)
            axs[1][0].legend(loc='lower right')

            # Subplot 2: out-of-sample log return predictions
            axs[1][1].grid()
            axs[1][1].plot(
                performance["test_log_returns"],
                color='red',
                label='Test Set'
            )
            axs[1][1].plot(
                performance["test_log_prediction"],
                color=performance["color"],
                label=performance["name"]+" "+ performance["attn_type"]
            )
            axs[1][1].set_title(
                performance["name"]+' '+ performance["attn_type"] + ' (out-of-sample)'
            )
            axs[1][1].tick_params(axis='x', rotation=30)
            axs[1][1].legend(loc='lower right')
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
            "Attention",
            "MAE Train",
            "MAE Test",
            "RMSE Train",
            "RMSE Test"
        ]
        for performance in self.model_performances:
            # fill the table
            table.add_row([
                performance["name"],
                performance["attn_type"],
                performance["mae_train"],
                performance["mae_test"],
                performance["rmse_train"],
                performance["rmse_test"],
            ])

        print(table.get_string(sortby="MAE Test"))
