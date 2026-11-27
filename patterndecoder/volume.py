"""
Trading volume as a third use case.

Unlike returns, trading volume has recurring calendar patterns: it differs by
weekday, it rises when options and futures expire on the third Friday of a
month, most of all in March, June, September and December, and it is
persistent. This use case asks whether the models, and the PatternDecoder in
particular, learn such patterns better than linear benchmarks that are given
them. The design was fixed before any model was trained on it:

- Target: the next day's log trading volume of the DAX and the S&P 500 as
  reported by Yahoo Finance, over the period of config/config.yaml
  (``series: log_volume``). Days without reported volume (4 for the DAX, 1 for
  the S&P 500) take the previous day's value. The log volume is centred by its
  median over the first half of the sample, which is training data in every
  test window, so that its values are of order one.
- Inputs: the past ``window_size`` values of the target and the calendar
  features of those days. The models are trained with the MAE loss and the
  current settings of config/config.yaml, except input_scale 1, because the
  centred log volume is already of order one.
- Benchmarks: persistence (the previous day's value), an EWMA with decay 0.94,
  the HAR model (the previous day's value and its 5- and 22-day means), HAR-X,
  which adds dummies for the target day's weekday, for the third Friday of a
  month and for the third Friday of March, June, September and December, and
  the training median. Both HAR models are fitted by least absolute deviations
  on the training data. HAR-X gets the calendar of the target day directly, as
  a forecaster would; the neural models see the calendar features of the past
  days only and have to infer those of the target day.
- Evaluation: the walk-forward evaluation of patterndecoder/robustness.py with
  five test windows and ten seeds; MAE in log volume relative to HAR-X and HAR,
  with Diebold-Mariano tests of the seed-averaged forecasts.
"""

import numpy as np
import pandas as pd

from patterndecoder.volatility import level_benchmarks

EXPIRY_MONTHS = (3, 6, 9, 12)


def log_volume(prices):
    """
    Daily log volume, centred by its median over the first half of the sample;
    days without reported volume take the previous day's value.
    """
    volume = prices["Volume"].where(prices["Volume"] > 0).ffill().bfill()
    values = np.log(volume.astype(float))
    return values - values.iloc[: len(values) // 2].median()


def calendar_dummies(dates):
    """
    Dummies for the weekday (Monday to Thursday; Friday is the base), for the
    third Friday of a month and for the third Friday of March, June, September
    and December, indexed by ``dates``.
    """
    # pylint cannot see the date attributes of a DatetimeIndex
    # pylint: disable=no-member
    dates = pd.DatetimeIndex(dates)
    third_friday = (dates.dayofweek == 4) & (dates.day >= 15) & (dates.day <= 21)
    columns = {
        name: dates.dayofweek == day for day, name in enumerate(("mon", "tue", "wed", "thu"))
    }
    columns["third_friday"] = third_friday
    columns["quarterly_expiry"] = third_friday & dates.month.isin(EXPIRY_MONTHS)
    return pd.DataFrame(columns, index=dates).astype(float)


def volume_benchmarks(train_data, test_data, params):
    """The volume benchmarks: persistence, EWMA, HAR, HAR-X and the training median."""
    return level_benchmarks(train_data, test_data, params, extra=calendar_dummies)
