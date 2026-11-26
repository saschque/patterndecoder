# pylint: disable=E1101
"""
Walk-forward evaluation over several test windows, for any ticker.

``run`` trains every model in models/*.yaml once for one seed and one test
window, given as fractions of the data (default width 0.1). The training data
are everything before the window; their last ``validation_split`` share is the
validation set. The test data start ``window_size`` days earlier, so every day
of the window gets a forecast. ``run`` writes the daily test forecasts of every
model and of two benchmarks: the zero forecast (random walk for the price) and
the median of the training returns (the best constant forecast under MAE).

``summarize`` averages each model's forecasts over the seeds (an ensemble that
removes most of the seed noise) and reports, per window and pooled over all
windows, the test MAE relative to the benchmarks, the mean MAE of the single
runs, the windows in which a model beats each benchmark, and Diebold-Mariano
tests of the pooled ensemble forecasts. It also reports the mean validation
MAE, on which any choice between settings must be based; the test results are
for reporting only.

Usage:
    python -m patterndecoder.robustness run --seed 1 --test-start 0.5 --out run.csv
    python -m patterndecoder.robustness summarize runs/*.csv --out summary.csv
"""

import csv
import statistics
from collections import defaultdict

import numpy as np
import pandas as pd

from patterndecoder.data import get_stock_data
from patterndecoder.evaluation import diebold_mariano
from patterndecoder.experiments import (
    build_model,
    parse_command,
    publish_summary,
    read_csv_rows,
    results_writer,
)
from patterndecoder.study import load_returns, model_files
from patterndecoder.training import split_train_validation
from patterndecoder.utils import compile_and_train, get_model_performance, load_config

FIELDS = ["model", "seed", "test_start", "date", "actual", "prediction", "val_mae"]
TESTED_BENCHMARKS = ("Naive", "Median")
# Rows that are not models; Oracle exists only for synthetic series
BENCHMARKS = TESTED_BENCHMARKS + ("Oracle",)


def row_position(count, share):
    """First row after a share of the data, robust to floating-point error."""
    return int(count * share + 1e-9)


def split_window(returns, params, test_start, width=0.1):
    """
    Splits the returns at a test window given as fractions of the data.

    Windows that follow each other (0.5, 0.6, ...) neither overlap nor leave gaps.

    Returns:
        tuple: (train_data, test_data); test_data starts ``window_size`` days
        before the window so that every day in the window can be forecast.
    """
    count = len(returns)
    start = row_position(count, test_start)
    end = min(count, row_position(count, test_start + width))
    window = params["window_size"]
    if not window < start < end:
        raise ValueError(f"Test window {test_start}+{width} is outside the data")
    return returns.iloc[:start], returns.iloc[start - window : end]


def benchmark_forecasts(train_data, test_data, params, oracle=None):
    """
    Test and validation forecasts of the benchmarks, with the validation and
    test targets. ``oracle`` (pd.Series by date) exists for synthetic series only.
    """
    target, window = params["target_column"], params["window_size"]
    fit_data, val_data = split_train_validation(train_data, params)
    val_targets = val_data[target].iloc[window:]
    actuals = test_data[target].iloc[window:]
    constants = {
        "Naive": (0.0, 0.0),
        "Median": (train_data[target].median(), fit_data[target].median()),
    }
    forecasts = {
        name: (
            pd.Series(float(test), index=actuals.index),
            pd.Series(float(val), index=val_targets.index),
        )
        for name, (test, val) in constants.items()
    }
    if oracle is not None:
        forecasts["Oracle"] = (oracle[actuals.index], oracle[val_targets.index])
    return forecasts, val_targets, actuals


def benchmark_rows(train_data, test_data, params, oracle=None):
    """Daily forecasts and validation MAE of the zero, median and oracle forecasts."""
    forecasts, val_targets, actuals = benchmark_forecasts(train_data, test_data, params, oracle)
    rows = []
    for name, (test_forecast, val_forecast) in forecasts.items():
        val_mae = float(np.mean(np.abs(val_targets - val_forecast)))
        rows += [
            (name, date, actual, test_forecast[date], val_mae) for date, actual in actuals.items()
        ]
    return rows


def model_rows(params, model_yaml, data, workdir):
    """Trains one model; returns its daily test forecasts with its validation MAE."""
    train_data, test_data = data
    model, config_path = build_model(params, model_yaml, workdir)
    history, model = compile_and_train(model, train_data, config_path)
    predictions = get_model_performance(model, params, test_data)["predictions"]
    actuals = test_data[params["target_column"]].iloc[params["window_size"] :]
    val_mae = float(min(history.history["val_mae"]))
    name = load_config(model_yaml)["name"]
    return [
        (name, date, actual, prediction, val_mae)
        for date, actual, prediction in zip(
            actuals.index, actuals.to_numpy(), predictions.iloc[:, 0].to_numpy()
        )
    ]


def csv_row(row, seed, test_start):
    """Formats one (model, date, actual, prediction, val_mae) tuple for the CSV file."""
    name, date, actual, prediction, val_mae = row
    return {
        "model": name,
        "seed": seed,
        "test_start": test_start,
        "date": f"{date:%Y-%m-%d}",
        "actual": round(float(actual), 8),
        "prediction": round(float(prediction), 8),
        "val_mae": round(val_mae, 7),
    }


def run(seed, test_start, out_path, width=0.1, config_path="config/config.yaml"):
    """Trains all models for one seed and one test window and writes the forecasts."""
    params = load_config(config_path)
    if not params.get("validation_split"):
        raise ValueError("The walk-forward evaluation needs a validation_split above 0")
    params.update(training=True, verbose=False, seed=seed, model_overrides=False)
    data = split_window(load_returns(params), params, test_start, width)
    prices = get_stock_data(params)
    oracle = prices["Oracle"] if "Oracle" in prices else None
    with results_writer(out_path, FIELDS) as (write_row, workdir):
        for row in benchmark_rows(*data, params, oracle):
            write_row(csv_row(row, seed, test_start))
        for model_yaml in model_files():
            for row in model_rows(params, model_yaml, data, workdir):
                write_row(csv_row(row, seed, test_start))


def ensemble_forecasts(rows):
    """
    Collects the forecasts per window and model and averages them over the seeds.

    Returns:
        dict: (test_start, model) -> {"dates", "actual", "prediction" (mean over
        the seeds), "seed_errors" (seed -> absolute errors), "val_mae" (list)}.
    """
    collected = defaultdict(lambda: defaultdict(dict))
    val = defaultdict(dict)
    actual = {}
    for row in rows:
        key = (row["test_start"], row["model"])
        collected[key][row["seed"]][row["date"]] = float(row["prediction"])
        val[key][row["seed"]] = float(row["val_mae"])
        actual[(row["test_start"], row["date"])] = float(row["actual"])
    forecasts = {}
    for key, by_seed in collected.items():
        dates = sorted(next(iter(by_seed.values())))
        actuals = np.array([actual[(key[0], date)] for date in dates])
        predictions = {
            seed: np.array([days[date] for date in dates]) for seed, days in by_seed.items()
        }
        forecasts[key] = {
            "dates": dates,
            "actual": actuals,
            "prediction": np.mean(list(predictions.values()), axis=0),
            "seed_errors": {seed: np.abs(p - actuals) for seed, p in predictions.items()},
            "val_mae": list(val[key].values()),
        }
    return forecasts


def windows_of(forecasts):
    """The test windows in the forecasts, in order."""
    return sorted({key[0] for key in forecasts}, key=float)


def window_table(forecasts):
    """Per window and model: ensemble MAE, mean MAE of the single runs and rank."""
    table = []
    for start in windows_of(forecasts):
        models = {model: f for (win, model), f in forecasts.items() if win == start}
        naive = float(np.mean(np.abs(models["Naive"]["actual"])))
        maes = {m: float(np.mean(np.abs(f["prediction"] - f["actual"]))) for m, f in models.items()}
        neural = sorted((m for m in maes if m not in BENCHMARKS), key=maes.get)
        for model in sorted(maes, key=maes.get):
            entry = models[model]
            single = statistics.mean(float(np.mean(e)) for e in entry["seed_errors"].values())
            table.append(
                {
                    "test_start": start,
                    "first_day": entry["dates"][0],
                    "last_day": entry["dates"][-1],
                    "model": model,
                    "ensemble_mae": round(maes[model], 7),
                    "vs_naive_%": round(100 * (maes[model] / naive - 1), 3),
                    "single_run_mae": round(single, 7),
                    "rank": neural.index(model) + 1 if model in neural else "",
                }
            )
    return table


def single_run_mae(forecasts, model, windows):
    """Mean over the seeds of a model's pooled MAE (seeds present in every window)."""
    seeds = set.intersection(*(set(forecasts[(w, model)]["seed_errors"]) for w in windows))
    return statistics.mean(
        float(np.mean(np.concatenate([forecasts[(w, model)]["seed_errors"][s] for w in windows])))
        for s in seeds
    )


def benchmark_tests(errors, bench_errors):
    """MAE difference (%) and Diebold-Mariano p-value against each benchmark."""
    columns = {}
    for bench, other in bench_errors.items():
        result = diebold_mariano(errors, other)
        bench_mae = float(np.mean(np.abs(other)))
        columns[f"vs_{bench}_%"] = round(100 * result["mean_diff"] / bench_mae, 3)
        columns[f"p_vs_{bench}"] = round(result["p_value"], 4)
    return columns


def pooled_table(forecasts):
    """Pooled over all windows: ensemble and single-run MAE, Diebold-Mariano tests."""
    windows = windows_of(forecasts)
    models = sorted({key[1] for key in forecasts})
    models = [m for m in models if all((w, m) in forecasts for w in windows)]

    def pooled(model, field):
        return np.concatenate([forecasts[(w, model)][field] for w in windows])

    actuals = pooled("Naive", "actual")
    bench_errors = {bench: pooled(bench, "prediction") - actuals for bench in TESTED_BENCHMARKS}
    table = []
    for model in models:
        errors = pooled(model, "prediction") - actuals
        row = {
            "model": model,
            "days": errors.size,
            "ensemble_mae": round(float(np.mean(np.abs(errors))), 7),
            "single_run_mae": round(single_run_mae(forecasts, model, windows), 7),
        }
        row.update(benchmark_tests(errors, bench_errors))
        row["val_mae"] = round(
            statistics.mean(v for w in windows for v in forecasts[(w, model)]["val_mae"]), 7
        )
        table.append(row)
    add_oracle_shares(table)
    return sorted(table, key=lambda r: r["ensemble_mae"])


def add_oracle_shares(table):
    """
    Adds the share of the oracle's advantage over the zero forecast that each
    row achieves, for its ensemble and its single runs (synthetic series only).
    """
    maes = {row["model"]: row["ensemble_mae"] for row in table}
    if "Oracle" not in maes:
        return
    gain = maes["Naive"] - maes["Oracle"]
    for row in table:
        for field, share in (
            ("ensemble_mae", "oracle_share_%"),
            ("single_run_mae", "single_oracle_share_%"),
        ):
            row[share] = round(100 * (maes["Naive"] - row[field]) / gain, 1)


def window_wins(window_rows):
    """Per model: windows won against the zero and median forecast, and its rank in each."""
    by_window = defaultdict(dict)
    for row in window_rows:
        by_window[row["test_start"]][row["model"]] = row
    wins = defaultdict(lambda: {"zero": 0, "median": 0, "ranks": []})
    for start in sorted(by_window, key=float):
        rows = by_window[start]
        for model, row in rows.items():
            wins[model]["zero"] += row["ensemble_mae"] < rows["Naive"]["ensemble_mae"]
            wins[model]["median"] += row["ensemble_mae"] < rows["Median"]["ensemble_mae"]
            wins[model]["ranks"].append(row["rank"])
    return wins


def format_p(value):
    """A p-value with three decimals, or "-" where no test applies."""
    return "-" if np.isnan(value) else f"{value:.3f}"


def summary_text(pooled, per_window, seeds):
    """Markdown summary: the windows, then one table row per model."""
    windows = {row["test_start"]: (row["first_day"], row["last_day"]) for row in per_window}
    wins = window_wins(per_window)
    naive = next(row["ensemble_mae"] for row in pooled if row["model"] == "Naive")
    oracle = "oracle_share_%" in pooled[0]
    lines = [
        f"Test windows: {'; '.join(f'{a} to {b}' for a, b in windows.values())}. "
        f"{seeds} seeds, {pooled[0]['days']} test days. Forecasts averaged over the seeds; "
        "single run: mean MAE of the seeds' own forecasts. p: Diebold-Mariano test of the "
        "pooled ensemble forecasts."
        + (" Oracle share: part of the oracle's advantage over the zero forecast reached "
           "by the ensemble / by single runs." if oracle else ""),
        "",
        "| Model | MAE | vs zero | p | vs median | p | single run vs zero "
        "| windows better than zero | windows better than median | rank per window | val MAE |"
        + (" oracle share |" if oracle else ""),
        "|---|---|---|---|---|---|---|---|---|---|---|" + ("---|" if oracle else ""),
    ]
    for row in pooled:
        won = wins[row["model"]]
        ranks = ", ".join(str(r) for r in won["ranks"] if r != "") or "-"
        lines.append(
            f"| {row['model']} | {row['ensemble_mae']:.6f} | {row['vs_Naive_%']:+.2f} % "
            f"| {format_p(row['p_vs_Naive'])} | {row['vs_Median_%']:+.2f} % "
            f"| {format_p(row['p_vs_Median'])} "
            f"| {100 * (row['single_run_mae'] / naive - 1):+.2f} % "
            f"| {won['zero']}/{len(windows)} | {won['median']}/{len(windows)} | {ranks} "
            f"| {row['val_mae']:.6f} |"
            + (f" {round(row['oracle_share_%'])} % / {round(row['single_oracle_share_%'])} % |"
               if oracle else "")
        )
    return "\n".join(lines)


def write_csv(path, table):
    """Writes a list of dicts as CSV."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def summarize(paths, out_path=None):
    """Prints the pooled results; optionally writes them and the per-window results as CSV."""
    rows = read_csv_rows(paths)
    forecasts = ensemble_forecasts(rows)
    per_window = window_table(forecasts)
    pooled = pooled_table(forecasts)
    seeds = len({row["seed"] for row in rows})
    publish_summary("Walk-forward evaluation", summary_text(pooled, per_window, seeds))
    if out_path:
        write_csv(out_path, pooled)
        write_csv(out_path.replace(".csv", "_windows.csv"), per_window)


def main():
    """Command line entry point."""

    def add_run_arguments(run_parser):
        run_parser.add_argument("--seed", type=int, required=True)
        run_parser.add_argument("--test-start", type=float, required=True)
        run_parser.add_argument("--width", type=float, default=0.1)
        run_parser.add_argument("--out", required=True)

    def add_summary_arguments(summary_parser):
        summary_parser.add_argument("--out", help="CSV file for the pooled results")

    args = parse_command(
        __doc__.split("\n\n", maxsplit=1)[0], add_run_arguments, add_summary_arguments
    )
    if args.command == "run":
        run(args.seed, args.test_start, args.out, args.width)
    else:
        summarize(args.paths, args.out)


if __name__ == "__main__":
    main()
