"""
Volatility forecasting as a second use case.

Daily returns are almost unpredictable, so the models cannot be told apart on
them. Volatility is predictable, because calm and turbulent periods persist. A
model with a recurrent memory, such as the LSTM in the PatternDecoder, could
have an advantage there. The design of this use case was fixed before any
model was trained on it:

- Target: the next day's Parkinson volatility ln(High / Low) / sqrt(4 ln 2), a
  range-based estimate of the day's volatility, of the DAX and the S&P 500 over
  the period of config/config.yaml (``series: parkinson``).
- Inputs: the past ``window_size`` values of the target and the calendar
  features. The models are trained with the MAE loss and the settings of
  config/config.yaml, once with input_scale 1 and once with input_scale 100;
  the choice between the two is made on the validation MAE.
- Benchmarks: persistence (the previous day's value), an exponentially weighted
  moving average with decay 0.94, the HAR model of Corsi (2009) on the previous
  day's value and its 5- and 22-day means, fitted by least absolute deviations
  on the training data, and the training median.
- Evaluation: the walk-forward evaluation of patterndecoder/robustness.py with
  five test windows and ten seeds; MAE relative to HAR and persistence, with
  Diebold-Mariano tests of the seed-averaged forecasts.
"""

import numpy as np
import pandas as pd

from patterndecoder.training import split_train_validation

EWMA_DECAY = 0.94
HAR_WINDOWS = (1, 5, 22)


def parkinson_volatility(prices):
    """Daily Parkinson volatility from the high and low prices."""
    log_range = np.log(prices["High"] / prices["Low"]).clip(lower=0)
    return log_range / np.sqrt(4 * np.log(2))


def har_regressors(series):
    """The HAR regressors at the end of each day: the value and its 5- and 22-day means."""
    return pd.concat(
        [series.rolling(days).mean().rename(f"mean_{days}") for days in HAR_WINDOWS], axis=1
    )


def har_design(series, dates, extra=None):
    """
    The regressors for forecasting the values on ``dates``: the HAR regressors
    of the previous day and, if given, ``extra(dates)`` (a DataFrame by date).
    """
    design = har_regressors(series).shift(1).loc[dates]
    if extra is not None:
        design = pd.concat([design, extra(dates)], axis=1)
    return design


def fit_har(series, first_target, extra=None):
    """
    Coefficients (constant first) from a least-absolute-deviations fit of each
    day's value on its regressors (see har_design), using all days before the
    date ``first_target``.
    """
    # statsmodels is imported here because only these use cases need it
    # pylint: disable-next=import-outside-toplevel
    from statsmodels.regression.quantile_regression import QuantReg

    design = har_design(series, series.index[series.index < first_target], extra).dropna()
    regressors = np.column_stack([np.ones(len(design)), design.to_numpy()])
    return QuantReg(series[design.index].to_numpy(), regressors).fit(q=0.5).params


def har_forecasts(series, coefficients, dates, extra=None):
    """Forecasts for ``dates`` from the regressors of har_design."""
    design = har_design(series, dates, extra).to_numpy()
    return pd.Series(coefficients[0] + design @ coefficients[1:], index=dates)


def level_benchmarks(train_data, test_data, params, extra=None):
    """
    Test and validation forecasts of persistence, EWMA, HAR and the training
    median, and of HAR-X (HAR plus the regressors ``extra(dates)``) if ``extra``
    is given, with the validation and test targets, in the form of
    robustness.benchmark_forecasts. The HAR models and the median are fitted on
    the data before the validation set (for the validation forecasts) or before
    the test window (for the test forecasts).
    """
    target, window = params["target_column"], params["window_size"]
    series = pd.concat([train_data[target], test_data[target].iloc[window:]])
    fit_data, val_data = split_train_validation(train_data, params)
    val_targets = val_data[target].iloc[window:]
    actuals = test_data[target].iloc[window:]
    previous = series.shift(1)
    ewma = series.ewm(alpha=1 - EWMA_DECAY, adjust=False).mean().shift(1)

    def har(regressors):
        return tuple(
            har_forecasts(series, fit_har(series, dates[0], regressors), dates, regressors)
            for dates in (actuals.index, val_targets.index)
        )

    forecasts = {
        "Persistence": (previous[actuals.index], previous[val_targets.index]),
        "EWMA": (ewma[actuals.index], ewma[val_targets.index]),
        "HAR": har(None),
        "Median": (
            pd.Series(float(train_data[target].median()), index=actuals.index),
            pd.Series(float(fit_data[target].median()), index=val_targets.index),
        ),
    }
    if extra is not None:
        forecasts["HARX"] = har(extra)
    return forecasts, val_targets, actuals


def volatility_benchmarks(train_data, test_data, params):
    """The volatility benchmarks: persistence, EWMA, HAR and the training median."""
    return level_benchmarks(train_data, test_data, params)
