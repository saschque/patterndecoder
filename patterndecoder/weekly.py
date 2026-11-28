# pylint: disable=E1101
"""
Weekly horizon: forecasts of the DAX log return over the next five trading days.

Pre-registered design (committed before the runs):

- Target: the log return of the next five trading days, i.e. the sum of their
  daily log returns, forecast after the close of every trading day. Inputs are
  the daily log returns of the last 30, 60 or 120 days.
- Data: the DAX (^GDAXI, Yahoo Finance) from 1990-01-01 to 2026-02-05, set by the
  overrides {"start": "1990-01-01", "target_days": 5}. Comparison: the same
  design on the period of the paper, 2021-02-05 to 2026-02-05 ({"target_days": 5}).
- Walk-forward: five test windows (from 0.5, 0.6, 0.7, 0.8 and 0.9 of the days,
  width 0.1, about 2008 to 2026); the training data are all earlier days, their
  last 20 % the validation data for early stopping. Targets are summed within
  the training, validation and test data, so no target reaches into a later
  part; the last four days of every part have no complete target, so the last
  four days of each test window get no forecast.
- Window length: every model is trained with 30, 60 and 120 input days; per test
  window and model, the length with the lowest validation MAE (mean over the
  seeds) is evaluated. The test data never enter that choice.
- Models: the 13 models in models/*.yaml with the settings of config/config.yaml,
  trained on the five-day target of every day (overlapping weeks); 10 seeds,
  forecasts averaged over the seeds.
- Benchmarks: zero ("Naive"); the median five-day return of the training days
  ("Median"); a median (LAD) regression on the return of the previous five days
  ("LastWeek"), which captures momentum or reversal, whichever the training days
  show. Like the median, it is fit on all training days for the test forecasts
  and on the fit part for the validation forecasts.
- Tests: Diebold-Mariano tests of the absolute errors of the pooled ensemble
  forecasts with Newey-West variance, (a) on all days, i.e. overlapping weeks,
  with at least four lags because neighbouring targets share four days, and (b)
  on every fifth day of each test window, i.e. non-overlapping weeks, with the
  default lags. A model counts as better than a benchmark only if its MAE is
  lower and p < 0.05 in both tests; p < 0.05 / 13 (Bonferroni over the 13
  models) is reported as well.
- Expected beforehand: little or nothing. With about 900 non-overlapping test
  weeks, a forecast has to explain roughly 3 % of the variance of the five-day
  return to be found with 80 % power, on 2021 to 2026 (about 120 weeks) roughly
  20 %; index returns are not known to be that predictable.

Usage:
    python -m patterndecoder.weekly run --seed 1 --test-start 0.5 --out run.csv
    python -m patterndecoder.weekly summarize runs/*.csv --out summary.csv
"""

import statistics
from collections import defaultdict

import numpy as np
import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg

from patterndecoder import robustness
from patterndecoder.evaluation import diebold_mariano
from patterndecoder.experiments import (
    publish_summary,
    read_csv_rows,
    results_writer,
    walk_forward_main,
)
from patterndecoder.study import load_returns, model_files
from patterndecoder.training import forward_sums, split_train_validation
from patterndecoder.utils import load_config

WINDOW_SIZES = (30, 60, 120)
FIELDS = robustness.FIELDS + ["window_size", "target_days"]
BENCHMARKS = ("Naive", "Median", "LastWeek")
ALPHA = 0.05


def previous_sums(series, days):
    """The return of the ``days`` days up to the previous close of each day."""
    return series.rolling(days).sum().shift(1)


def last_week_forecasts(fit_series, series, days):
    """
    Forecasts of the next ``days`` days' return for every day of ``series``: a
    median (LAD) regression on the previous ``days`` days' return, fit on the
    daily returns ``fit_series``. Days without ``days`` earlier returns get NaN.
    """
    pairs = pd.concat(
        [previous_sums(fit_series, days), forward_sums(fit_series, days)], axis=1
    ).dropna()
    design = np.column_stack([np.ones(len(pairs)), pairs.iloc[:, 0].to_numpy()])
    intercept, slope = QuantReg(pairs.iloc[:, 1].to_numpy(), design).fit(q=0.5).params
    return intercept + slope * previous_sums(series, days)


def benchmark_forecasts(train_data, test_data, params):
    """
    Test and validation forecasts of zero, the median and the last-week
    regression, with the validation and test targets (see
    robustness.benchmark_forecasts).
    """
    forecasts, val_targets, actuals = robustness.benchmark_forecasts(train_data, test_data, params)
    target, days = params["target_column"], params["target_days"]
    fit_data, val_data = split_train_validation(train_data, params)
    forecasts["LastWeek"] = (
        last_week_forecasts(train_data[target], test_data[target], days)[actuals.index],
        last_week_forecasts(fit_data[target], val_data[target], days)[val_targets.index],
    )
    return forecasts, val_targets, actuals


def run(seed, test_start, out_path, width=0.1, config_path="config/config.yaml"):
    """
    Trains every model with each window length for one seed and one test window
    and writes the test forecasts of the models and of the benchmarks.
    """
    params = load_config(config_path)
    days = int(params.get("target_days", 1))
    if days < 2 or not params.get("validation_split"):
        raise ValueError("The weekly study needs target_days above 1 and a validation_split")
    params.update(training=True, verbose=False, seed=seed, model_overrides=False)
    returns = load_returns(params)
    with results_writer(out_path, FIELDS) as (write_row, workdir):

        def write(rows, window):
            for row in rows:
                csv_row = robustness.csv_row(row, seed, test_start)
                write_row(dict(csv_row, window_size=window, target_days=days))

        for window in WINDOW_SIZES:
            sized = dict(params, window_size=window)
            data = robustness.split_window(returns, sized, test_start, width)
            if window == WINDOW_SIZES[0]:
                write(robustness.forecast_rows(*benchmark_forecasts(*data, sized)), 0)
            for model_yaml in model_files():
                write(robustness.model_rows(sized, model_yaml, data, workdir), window)


def best_window(val_by_size):
    """The window length whose validation MAE, averaged over the seeds, is lowest."""
    return min(val_by_size, key=lambda size: statistics.mean(val_by_size[size].values()))


def chosen_windows(rows):
    """(test_start, model) -> the window length chosen on the validation MAE."""
    val = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        by_size = val[(row["test_start"], row["model"])]
        by_size[row["window_size"]][row["seed"]] = float(row["val_mae"])
    return {key: best_window(by_size) for key, by_size in val.items()}


def benchmark_columns(errors, bench_errors, suffix, max_lag):
    """MAE difference (%) and Diebold-Mariano p-value against each benchmark."""
    columns = {}
    for bench, other in bench_errors.items():
        result = diebold_mariano(errors, other, max_lag=max_lag)
        columns[f"vs_{bench}{suffix}_%"] = round(
            100 * result["mean_diff"] / float(np.mean(np.abs(other))), 3
        )
        columns[f"p_vs_{bench}{suffix}"] = round(result["p_value"], 4)
    return columns


def pooled_table(forecasts, days, choices):
    """
    Pooled over the test windows: ensemble and single-run MAE, the chosen window
    lengths, and Diebold-Mariano tests against each benchmark on all days
    (overlapping targets, at least ``days - 1`` lags) and on every ``days``-th
    day of each window (non-overlapping targets, suffix ``_weekly``).
    """
    windows = robustness.windows_of(forecasts)
    models = sorted({m for _, m in forecasts if all((w, m) in forecasts for w in windows)})

    def pooled(model, field, step=1):
        return np.concatenate([forecasts[(w, model)][field][::step] for w in windows])

    table = []
    for model in models:
        errors = pooled(model, "prediction") - pooled("Naive", "actual")
        row = {
            "model": model,
            "days": errors.size,
            "weeks": pooled(model, "prediction", days).size,
            "ensemble_mae": round(float(np.mean(np.abs(errors))), 7),
            "single_run_mae": round(robustness.single_run_mae(forecasts, model, windows), 7),
            "window_sizes": " ".join(choices[(w, model)] for w in windows),
        }
        for step, suffix in ((1, ""), (days, "_weekly")):
            actuals = pooled("Naive", "actual", step)
            bench_errors = {b: pooled(b, "prediction", step) - actuals for b in BENCHMARKS}
            max_lag = max(int(errors.size ** (1 / 3)), days - 1) if step == 1 else None
            row.update(
                benchmark_columns(
                    pooled(model, "prediction", step) - actuals, bench_errors, suffix, max_lag
                )
            )
        row["val_mae"] = round(
            statistics.mean(v for w in windows for v in forecasts[(w, model)]["val_mae"]), 7
        )
        table.append(row)
    return sorted(table, key=lambda r: r["ensemble_mae"])


def beaten(row, alpha):
    """The benchmarks a pooled row beats: lower MAE and p < alpha in both tests."""
    return [
        bench
        for bench in BENCHMARKS
        if all(
            row[f"vs_{bench}{suffix}_%"] < 0 and row[f"p_vs_{bench}{suffix}"] < alpha
            for suffix in ("", "_weekly")
        )
    ]


def summary_text(pooled, per_window, seeds, days):
    """Markdown summary: the design in one paragraph, then one table row per model."""
    windows = {row["test_start"]: (row["first_day"], row["last_day"]) for row in per_window}
    models = [row["model"] for row in pooled if row["model"] not in BENCHMARKS]
    strict = ALPHA / max(len(models), 1)
    reference = next(row["ensemble_mae"] for row in pooled if row["model"] == "Naive")
    labels = [robustness.LABELS[bench] for bench in BENCHMARKS]
    lines = [
        f"Target: log return of the next {days} trading days. Test windows: "
        f"{'; '.join(f'{a} to {b}' for a, b in windows.values())}. {seeds} seeds, "
        f"{pooled[0]['days']} forecasts ({pooled[0]['weeks']} non-overlapping weeks), "
        "averaged over the seeds. Input days per test window chosen on the validation MAE. "
        f"p: Diebold-Mariano test on all days (at least {days - 1} lags) / on every "
        f"{days}th day. Better: lower MAE and p < {ALPHA} in both tests; * also "
        f"p < {strict:.4f} (Bonferroni over {len(models)} models).",
        "",
        "| Model | input days | MAE | "
        + " | ".join(f"vs {label} | p" for label in labels)
        + " | single run vs zero | better than | val MAE |",
        "|---|---|---|" + "---|---|" * len(labels) + "---|---|---|",
    ]
    for row in pooled:
        strong = beaten(row, strict)
        better = [
            robustness.LABELS[bench] + ("*" if bench in strong else "")
            for bench in beaten(row, ALPHA)
        ]
        cells = [
            row["model"],
            "-" if row["model"] in BENCHMARKS else row["window_sizes"],
            f"{row['ensemble_mae']:.6f}",
        ]
        for bench in BENCHMARKS:
            cells += [
                f"{row[f'vs_{bench}_%']:+.2f} %",
                f"{robustness.format_p(row[f'p_vs_{bench}'])} / "
                f"{robustness.format_p(row[f'p_vs_{bench}_weekly'])}",
            ]
        cells += [
            f"{100 * (row['single_run_mae'] / reference - 1):+.2f} %",
            ", ".join(better) or "-",
            f"{row['val_mae']:.6f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summarize(paths, out_path=None):
    """
    Chooses the window length per test window and model on the validation MAE,
    prints the pooled results and optionally writes them and the per-window
    results as CSV.
    """
    rows = read_csv_rows(paths)
    days = int(rows[0]["target_days"])
    choices = chosen_windows(rows)
    forecasts = robustness.ensemble_forecasts(
        row for row in rows if row["window_size"] == choices[(row["test_start"], row["model"])]
    )
    per_window = robustness.window_table(forecasts)
    for row in per_window:
        row["window_size"] = choices[(row["test_start"], row["model"])]
    pooled = pooled_table(forecasts, days, choices)
    seeds = len({row["seed"] for row in rows})
    publish_summary(
        f"Walk-forward evaluation, {days}-day returns",
        summary_text(pooled, per_window, seeds, days),
    )
    if out_path:
        robustness.write_csv(out_path, pooled)
        robustness.write_csv(out_path.replace(".csv", "_windows.csv"), per_window)


def main():
    """Command line entry point."""
    walk_forward_main(__doc__.split("\n\n", maxsplit=1)[0], run, summarize)


if __name__ == "__main__":
    main()
