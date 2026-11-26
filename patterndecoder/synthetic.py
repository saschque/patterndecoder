"""
Synthetic return series with known predictable structure (positive controls).

The DAX returns contain almost no predictable signal, so the models cannot be
told apart on them. These series have a signal of known form and strength,
which shows whether the models and the training setup can learn it at all and
which architecture learns it best. Their design was fixed before any model was
trained on them.

Every series has one row per business day between ``params["start"]`` and
``params["end"]`` (end exclusive), a daily drift of 0.0004 and Student-t noise
(4 degrees of freedom) with a standard deviation of 0.01, close to the DAX
returns. On top of the noise, each series has one kind of structure:

- ``SYNTH-AR``: first-order autoregression, r_t - mu = 0.3 (r_{t-1} - mu) + e_t.
- ``SYNTH-MOTIF``: a fixed 10-day pattern (a jump, momentum, then a reversal)
  that starts with probability 1/20 on any day on which no pattern is running.
- ``SYNTH-CYCLE``: a sine wave with a period of 23.7 trading days, which does not
  match the calendar features, and an amplitude of half the noise level.

The ``Oracle`` column holds a reference forecast of each day's log return from
the true structure: the conditional mean and median for SYNTH-AR and
SYNTH-CYCLE; for SYNTH-MOTIF the pattern value of a running pattern from its
second day on, which assumes knowing where patterns run. ``Close`` holds the
prices 100 * exp(cumulative returns), so the usual pipeline applies unchanged.
"""

import numpy as np
import pandas as pd

PREFIX = "SYNTH-"
SEED = 2026
SIGMA = 0.01
DRIFT = 0.0004
NOISE_DF = 4
AR_COEFFICIENT = 0.3
MOTIF = np.array([2.5, 1.2, 0.8, 0.4, 0.0, -0.4, -0.8, -1.0, -0.8, -0.4])
MOTIF_START_PROBABILITY = 1 / 20
CYCLE_PERIOD = 23.7
CYCLE_AMPLITUDE = 0.5
CYCLE_PHASE = 1.0


def noise(rng, size):
    """Student-t noise scaled to a standard deviation of SIGMA."""
    return rng.standard_t(NOISE_DF, size) * SIGMA / np.sqrt(NOISE_DF / (NOISE_DF - 2))


def autoregressive(rng, size):
    """Returns and oracle forecasts of the AR(1) series."""
    shocks = noise(rng, size)
    returns = np.empty(size)
    oracle = np.full(size, DRIFT)
    returns[0] = DRIFT + shocks[0]
    for day in range(1, size):
        oracle[day] = DRIFT + AR_COEFFICIENT * (returns[day - 1] - DRIFT)
        returns[day] = oracle[day] + shocks[day]
    return returns, oracle


def motif(rng, size):
    """Returns and oracle forecasts of the series with a recurring 10-day pattern."""
    pattern = np.zeros(size)
    oracle = np.full(size, DRIFT)
    day = 0
    while day < size:
        if rng.random() < MOTIF_START_PROBABILITY:
            length = min(len(MOTIF), size - day)
            pattern[day : day + length] = MOTIF[:length] * SIGMA
            # the first day of a pattern cannot be foreseen
            oracle[day + 1 : day + length] += MOTIF[1:length] * SIGMA
            day += length
        else:
            day += 1
    return DRIFT + pattern + noise(rng, size), oracle


def cycle(rng, size):
    """Returns and oracle forecasts of the series with a sine wave."""
    phase = 2 * np.pi * np.arange(size) / CYCLE_PERIOD + CYCLE_PHASE
    wave = CYCLE_AMPLITUDE * SIGMA * np.sin(phase)
    return DRIFT + wave + noise(rng, size), DRIFT + wave


PROCESSES = {"SYNTH-AR": autoregressive, "SYNTH-MOTIF": motif, "SYNTH-CYCLE": cycle}


def is_synthetic(ticker):
    """Whether a ticker names one of the synthetic series."""
    return str(ticker).startswith(PREFIX)


def synthetic_data(params):
    """
    Generates the synthetic series named by ``params["ticker"]``.

    Returns:
        pd.DataFrame: ``Close`` prices and ``Oracle`` forecasts of the log
        returns, indexed by business day.
    """
    ticker = params["ticker"]
    if ticker not in PROCESSES:
        raise ValueError(f"Unknown synthetic series {ticker}; choose from {sorted(PROCESSES)}")
    dates = pd.bdate_range(params["start"], params["end"], inclusive="left", name="Date")
    returns, oracle = PROCESSES[ticker](np.random.default_rng(SEED), len(dates))
    return pd.DataFrame({"Close": 100 * np.exp(np.cumsum(returns)), "Oracle": oracle}, index=dates)


def write_synthetic_data(params, path):
    """Writes the synthetic series named by ``params["ticker"]`` to ``path`` as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_data(params).to_csv(path, date_format="%Y-%m-%d")
