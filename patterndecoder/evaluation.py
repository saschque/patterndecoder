"""
Evaluation beyond MAE and RMSE.

- MedianForecast: constant forecast equal to the median of the training
  returns, the best constant forecast under the MAE loss.
- newey_west_test(): t-test of a zero mean for a serially correlated series.
- diebold_mariano(): Diebold-Mariano test of equal predictive accuracy of
  two forecasts of the same targets.
- forecast_diagnostics(): spread of the forecasts, their correlation with the
  actual returns and the share of correctly predicted signs.
- comparison_table(): these statistics for all models of a Forecasts object.
"""

import numpy as np
import pandas as pd
from scipy import stats


class MedianForecast:  # pylint: disable=too-few-public-methods
    """
    Constant forecast equal to the median of the training target.

    Under the MAE loss the median is the best constant forecast, so this is the
    natural benchmark for models trained with MAE.

    Args:
        train_target (array-like): Target values of the training period.
        forecast_horizon (int): Number of predictions per sample.
    """

    def __init__(self, train_target, forecast_horizon=1):
        self.name = "Median"
        self.value = float(np.median(np.asarray(train_target, dtype=float)))
        self.forecast_horizon = forecast_horizon

    def predict(self, data, verbose=False):
        """
        Returns the constant forecast for every window of a batched dataset.

        Args:
            data (tf.data.Dataset): Batched dataset of (inputs, targets).
            verbose (bool): Whether to print a message.

        Returns:
            np.array: Shape (num_samples, forecast_horizon).
        """
        if verbose:
            print("Median Model: Predicting...")
        count = sum(int(inputs.shape[0]) for inputs, _ in data)
        return np.full((count, self.forecast_horizon), self.value)


def newey_west_test(values, max_lag=None):
    """
    t-test of a zero mean for a serially correlated series.

    The long-run variance of the series is estimated with a Newey-West
    (Bartlett) kernel with ``max_lag`` lags, default floor(n^(1/3)), and the
    statistic is compared with a t-distribution with n - 1 degrees of freedom.

    Args:
        values (array-like): The series, for example daily loss differentials.
        max_lag (int or None): Number of lags in the long-run variance.

    Returns:
        dict: ``mean``, ``statistic``, two-sided ``p_value`` and ``n``.
    """
    values = np.ravel(np.asarray(values, dtype=float))
    count = values.size
    if max_lag is None:
        max_lag = int(count ** (1 / 3))

    centred = values - values.mean()
    variance = centred @ centred / count
    for lag in range(1, max_lag + 1):
        weight = 1 - lag / (max_lag + 1)
        variance += 2 * weight * (centred[lag:] @ centred[:-lag]) / count

    result = {"mean": float(values.mean()), "n": count}
    if variance <= 0:
        result.update(statistic=float("nan"), p_value=float("nan"))
        return result
    statistic = values.mean() / np.sqrt(variance / count)
    result.update(
        statistic=float(statistic),
        p_value=float(2 * stats.t.sf(abs(statistic), df=count - 1)),
    )
    return result


def diebold_mariano(errors_a, errors_b, power=1, max_lag=None):
    """
    Diebold-Mariano test of equal predictive accuracy of two forecasts.

    The loss differential is d_t = |e_a,t|^p - |e_b,t|^p (p = 1: absolute,
    p = 2: squared errors). Its long-run variance is estimated with a
    Newey-West (Bartlett) kernel with ``max_lag`` lags, default floor(n^(1/3)),
    to allow for autocorrelation of the losses, and the statistic is compared
    with a t-distribution with n - 1 degrees of freedom.

    Args:
        errors_a (array-like): Forecast errors of forecast a.
        errors_b (array-like): Forecast errors of forecast b (same targets).
        power (int): 1 for absolute, 2 for squared errors.
        max_lag (int or None): Number of lags in the long-run variance.

    Returns:
        dict: ``mean_diff`` (mean loss differential, negative if forecast a is
        more accurate), ``statistic``, two-sided ``p_value`` and ``n``.
    """
    errors_a = np.ravel(np.asarray(errors_a, dtype=float))
    errors_b = np.ravel(np.asarray(errors_b, dtype=float))
    if errors_a.shape != errors_b.shape:
        raise ValueError("Both forecasts must have the same number of errors")

    result = newey_west_test(np.abs(errors_a) ** power - np.abs(errors_b) ** power, max_lag)
    result["mean_diff"] = result.pop("mean")
    return result


def forecast_diagnostics(predictions, actuals):
    """
    Describes how much a forecast varies and how it relates to the actual values.

    Args:
        predictions (array-like): Forecasts.
        actuals (array-like): Actual values.

    Returns:
        dict: mean and standard deviation of the forecasts, their correlation
        with the actual values (NaN for a constant forecast) and the hit rate,
        the share of days on which forecast and actual value have the same
        sign, counted over days where neither is zero (NaN if there are none,
        e.g. for the zero forecast).
    """
    predictions = np.ravel(np.asarray(predictions, dtype=float))
    actuals = np.ravel(np.asarray(actuals, dtype=float))
    spread = float(np.std(predictions))
    correlation = float(np.corrcoef(predictions, actuals)[0, 1]) if spread > 0 else float("nan")
    counted = (predictions != 0) & (actuals != 0)
    hit_rate = (
        float(np.mean(np.sign(predictions[counted]) == np.sign(actuals[counted])))
        if counted.any()
        else float("nan")
    )
    return {
        "pred_mean": float(np.mean(predictions)),
        "pred_std": spread,
        "correlation": correlation,
        "hit_rate": hit_rate,
    }


def comparison_table(forecasts, benchmarks=("Naive", "Median"), power=1):
    """
    Test-set comparison of every model in a Forecasts object with benchmarks.

    For each model: test MAE, the relative difference in mean loss to each
    benchmark with the Diebold-Mariano p-value, and forecast_diagnostics().

    Args:
        forecasts (Forecasts): Evaluated models (see utils.Forecasts).
        benchmarks (tuple[str]): Names of the benchmark models in ``forecasts``.
        power (int): 1 for absolute, 2 for squared errors in the test.

    Returns:
        pd.DataFrame: One row per model, sorted by test MAE.
    """
    performances = {perf["name"]: perf for perf in forecasts.model_performances}
    first = next(iter(performances.values()))
    actuals = first["test_log_actuals"].to_numpy().ravel()
    bench_errors = {
        bench: performances[bench]["test_log_prediction"].to_numpy().ravel() - actuals
        for bench in benchmarks
    }

    rows = []
    for name, perf in performances.items():
        predictions = perf["test_log_prediction"].to_numpy().ravel()
        errors = predictions - actuals
        row = {"model": name, "test_mae": float(np.mean(np.abs(errors)))}
        row.update(_benchmark_columns(errors, bench_errors, power))
        row.update(forecast_diagnostics(predictions, actuals))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("test_mae").reset_index(drop=True)


def _benchmark_columns(errors, bench_errors, power):
    """Relative loss difference and Diebold-Mariano p-value against each benchmark."""
    columns = {}
    for bench, other in bench_errors.items():
        result = diebold_mariano(errors, other, power=power)
        columns[f"vs_{bench}_%"] = 100 * result["mean_diff"] / np.mean(np.abs(other) ** power)
        columns[f"p_vs_{bench}"] = result["p_value"]
    return columns
