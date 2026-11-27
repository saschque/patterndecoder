# pylint: disable=E1101
"""
Cross-section study: which of the DAX stocks does best the next day?

Instead of forecasting one index, the models rank the stocks of a panel by
their next-day return relative to the others, where studies of single stocks
have found short-term patterns such as reversal. Each trading day gives one
observation per stock, so the evaluation can detect much weaker signals than
on a single series. The design was fixed before any model was trained on the
panel:

- Universe: the 40 members of the DAX on 20 March 2023 (patterndecoder/panel.py),
  a date before the first test window, so that the test has no survivorship
  bias. Stocks enter the panel when their prices start (Daimler Truck in
  December 2021, Porsche AG in September 2022) and stay in it after leaving
  the DAX. Yahoo Finance has no data for Covestro, whose free float fell
  below 10 % in December 2024, so the panel has 39 stocks.
- Data: daily Xetra closing prices adjusted for dividends and splits, daily
  high and low and volume from Yahoo Finance for the date range of
  config/config.yaml (5 February 2021 to 5 February 2026).
- Target: each stock's excess log return of the next day, its log return
  minus the mean log return of the stocks of the panel that day.
- Inputs: for each stock the 30 days before the target day with four
  features: its excess return and the panel's mean return in percent, its log
  volume relative to the mean of the 20 days before, and its daily range
  log(high / low) in percent. The stock's identity is not an input.
- Models: the 13 models of models/*.yaml with the settings of config.yaml
  (MAE loss, last-step readout, no RevIN), one model for all stocks, trained
  on the windows of all stocks. An epoch over the panel holds about 40 times
  as many windows as one over a single series, so an epoch is a quarter pass,
  with at most 10 passes and early stopping after 2 passes without a lower
  validation MAE.
- Benchmarks: the zero forecast; a reversal rule, the LAD regression of the
  excess return on the previous day's excess return; Linear, the LAD
  regression on all inputs of the window; and, for the synthetic panel, the
  oracle.
- Evaluation: the five walk-forward test windows of robustness.py (August 2023
  to February 2026) with ten seeds, forecasts averaged over the seeds. The
  main measure is the information coefficient (IC): the Spearman rank
  correlation between forecasts and excess returns across the stocks of a day,
  averaged over the test days and tested against zero, and against Linear and
  the reversal rule through the daily differences, with Newey-West t-tests as
  for the Diebold-Mariano tests. Also reported: the share of days with a
  positive IC, the MAE relative to the zero forecast, and a portfolio that
  each day buys the fifth of the stocks with the highest forecasts and sells
  the fifth with the lowest short, with equal weights, before and after costs
  of 5 basis points per unit of turnover.
- PatternDecoder: the PatternDecoder and PatternDecoderNoConvBlock, which led
  for trading volume with RevIN (results/revin.md), are compared with the one
  of the seven other neural models with the highest mean validation IC,
  through the daily IC differences as above.
- Positive control: before the DAX run, the same evaluation with two seeds on
  the synthetic panel (``ticker: SYNTH-PANEL``), whose excess returns reverse
  by 10 % of the previous day's; it shows whether the pipeline and the models
  find a cross-sectional signal of known size.
- Small caps (``ticker: SDAX-PANEL``), designed after the DAX run and before
  any run on these stocks: the same study for the 71 companies of the SDAX
  table of the English Wikipedia, stated as of the index review of 10
  September 2020 (patterndecoder/panel.py), a universe fixed long before the
  test windows. Smaller stocks trade less and react more slowly, so
  short-term patterns such as reversal should be stronger there; trading them
  also costs more, so the summary also reports the break-even cost per unit
  of turnover. Yahoo Finance has no prices for eight of the companies
  (Encavis, Hornbach Baumarkt, Klöckner & Co, Leoni, New Work, Steinhoff, Tele
  Columbus, zooplus), most of which have been taken over or delisted since,
  so the panel has 63 stocks. This survivorship can favour a reversal rule,
  because stocks that fell and recovered stay in the panel while some that
  kept falling are missing. Everything else, including the ten seeds and the
  PatternDecoder comparison, is as for the DAX.
- Changes after the positive control, before the DAX run: Linear was first a
  linear model trained like the neural models. In the positive control its
  training failed (37 % of the oracle's IC and a higher MAE than the zero
  forecast, although a linear model can reach the oracle), so it is now the
  LAD regression on all inputs. The DAX run uses ten instead of four seeds,
  because a training job took about 10 instead of the expected 45 minutes.

Usage:
    python -m patterndecoder.cross_section run --seed 1 --test-start 0.5 --out run.csv
    python -m patterndecoder.cross_section summarize runs/*.csv --out summary.csv
"""

import math
import tempfile
from itertools import chain

import numpy as np
import pandas as pd
import tensorflow as tf
from statsmodels.regression.quantile_regression import QuantReg

from patterndecoder.evaluation import newey_west_test
from patterndecoder.experiments import build_model, publish_summary, walk_forward_main
from patterndecoder.panel import (
    daily_ic,
    load_panel,
    long_short,
    panel_data,
    panel_windows,
    split_rows,
)
from patterndecoder.study import model_files
from patterndecoder.utils import fit_model, load_config

FIELDS = [
    "model", "seed", "test_start", "date", "ticker", "actual", "prediction", "val_mae", "val_ic",
]
# Rows that are not neural models; Oracle exists only for the synthetic panel
BENCHMARKS = ("Naive", "Reversal", "Linear", "Oracle")
COMPARED = ("PatternDecoder", "PatternDecoderNoConvBlock")
EPOCHS_PER_PASS = 4
PASSES = 10
PATIENCE_PASSES = 2
SHARE = 0.2
COST = 0.0005
TRADING_DAYS = 252


def dataset(inputs, targets, batch_size, seed=None):
    """Batched dataset of (inputs, targets); with a seed shuffled and repeated, for training."""
    data = tf.data.Dataset.from_tensor_slices((inputs, targets[:, None]))
    if seed is not None:
        data = data.shuffle(len(targets), seed=seed, reshuffle_each_iteration=True).repeat()
    return data.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train(model, sets, params):
    """
    Trains a model on the fit windows of all stocks, with early stopping on the
    validation windows; returns its validation and test forecasts.
    """
    batch = params["batch_size"]
    settings = dict(
        params, epochs=EPOCHS_PER_PASS * PASSES, patience=EPOCHS_PER_PASS * PATIENCE_PASSES
    )
    _, model = fit_model(
        model,
        dataset(*sets["fit"][:2], batch, seed=int(params["seed"])),
        dataset(*sets["val"][:2], batch),
        settings,
        steps_per_epoch=math.ceil(len(sets["fit"][1]) / batch / EPOCHS_PER_PASS),
    )
    return {
        name: model.predict(dataset(*sets[name][:2], 8 * batch), verbose=0)[:, 0]
        for name in ("val", "test")
    }


def reversal_design(inputs):
    """Regressors of the reversal rule: a constant and the previous day's excess return."""
    return np.column_stack([np.ones(len(inputs)), inputs[:, -1, 0].astype(float)])


def window_design(inputs):
    """Regressors of Linear: a constant and all inputs of the window."""
    return np.column_stack([np.ones(len(inputs)), inputs.reshape(len(inputs), -1).astype(float)])


def lad_forecasts(fit, target, design=reversal_design):
    """Forecasts for the ``target`` windows of the LAD regression fitted on the ``fit`` windows."""
    coefficients = QuantReg(fit[1].astype(float), design(fit[0])).fit(q=0.5).params
    return design(target[0]) @ coefficients


def benchmark_forecasts(sets, data):
    """Validation and test forecasts of the zero forecast, the LAD benchmarks and the oracle."""
    training = tuple(np.concatenate([sets["fit"][i], sets["val"][i]]) for i in range(3))
    forecasts = {
        "Naive": {name: np.zeros(len(sets[name][1])) for name in ("val", "test")},
        **{
            name: {
                "val": lad_forecasts(sets["fit"], sets["val"], design),
                "test": lad_forecasts(training, sets["test"], design),
            }
            for name, design in (("Reversal", reversal_design), ("Linear", window_design))
        },
    }
    if "oracle" in data:
        forecasts["Oracle"] = {
            name: data["oracle"][tuple(sets[name][2].T)] for name in ("val", "test")
        }
    return forecasts


def mean_ic(predictions, windows):
    """Mean daily IC of forecasts of the samples (inputs, targets, keys)."""
    frame = pd.DataFrame(
        {"date": windows[2][:, 0], "prediction": predictions, "actual": windows[1]}
    )
    return float(daily_ic(frame).mean())


def forecast_rows(name, forecasts, sets, data, run_id):
    """One model's test forecasts per day and stock, with its validation MAE and IC."""
    _, actuals, keys = sets["test"]
    columns = {
        "model": name,
        "seed": run_id[0],
        "test_start": run_id[1],
        "date": data["dates"][keys[:, 0]].strftime("%Y-%m-%d"),
        "ticker": np.asarray(data["tickers"])[keys[:, 1]],
        "actual": np.round(actuals.astype(float), 8),
        "prediction": np.round(np.asarray(forecasts["test"], dtype=float), 8),
        "val_mae": round(float(np.mean(np.abs(sets["val"][1] - forecasts["val"]))), 7),
        "val_ic": round(mean_ic(forecasts["val"], sets["val"]), 5),
    }
    return pd.DataFrame(columns, columns=FIELDS)


def trained_forecasts(params, sets, workdir):
    """Yields the name and the forecasts of each model of models/*.yaml."""
    for model_yaml in model_files():
        model, config_path = build_model(params, model_yaml, workdir)
        yield load_config(model_yaml)["name"], train(model, sets, load_config(config_path))


def run(seed, test_start, out_path, width=0.1, config_path="config/config.yaml"):
    """Trains all models for one seed and one test window and writes the test forecasts."""
    params = load_config(config_path)
    params.update(training=True, verbose=False, seed=seed, model_overrides=False)
    data = panel_data(load_panel(params), params.get("input_scale", 1))
    rows = split_rows(len(data["dates"]), params, test_start, width)
    sets = {name: panel_windows(data, days, params["window_size"]) for name, days in rows.items()}
    print("Samples:", ", ".join(f"{name} {len(sets[name][1])}" for name in sets), flush=True)
    with open(out_path, "w", newline="", encoding="utf-8") as file, \
            tempfile.TemporaryDirectory() as workdir:
        file.write(",".join(FIELDS) + "\n")
        results = benchmark_forecasts(sets, data).items()
        for name, forecasts in chain(results, trained_forecasts(params, sets, workdir)):
            table = forecast_rows(name, forecasts, sets, data, (seed, test_start))
            table.to_csv(file, header=False, index=False)
            file.flush()
            print(f"{name}: validation MAE {table['val_mae'].iat[0]}, IC {table['val_ic'].iat[0]}")


def load_forecasts(paths):
    """The forecasts of all runs and their seed averages (per model, window, day and stock)."""
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    averaged = (
        frame.groupby(["model", "test_start", "date", "ticker"], sort=False)
        .agg(actual=("actual", "first"), prediction=("prediction", "mean"))
        .reset_index()
    )
    return frame, averaged


def ic_test(ics, reference=None):
    """Newey-West test of the mean daily IC, or of the mean daily IC difference to a reference."""
    values = ics if reference is None else ics - reference
    values = values.dropna()
    if len(values) < 3:
        return {"mean": float("nan"), "p_value": float("nan")}
    return newey_west_test(values.to_numpy())


def portfolio(frame):
    """
    Annualized mean returns before and after costs, Sharpe ratio after costs,
    daily turnover, and the cost per unit of turnover that uses up the gross
    return (break-even, in basis points).
    """
    returns = long_short(frame, SHARE, COST)
    net = returns["net"]
    return {
        "long_short_gross_%": round(100 * TRADING_DAYS * returns["gross"].mean(), 2),
        "long_short_net_%": round(100 * TRADING_DAYS * net.mean(), 2),
        "sharpe_net": round(float(net.mean() / net.std() * np.sqrt(TRADING_DAYS)), 2),
        "turnover": round(float(returns["turnover"].mean()), 3),
        "break_even_bp": round(
            float(1e4 * returns["gross"].mean() / returns["turnover"].mean()), 2
        ),
    }


def daily_mae(frame):
    """Mean absolute error of each day across the stocks."""
    return (frame["prediction"] - frame["actual"]).abs().groupby(frame["date"]).mean()


def model_row(name, frames, ics, validation):
    """Pooled results of one model: IC, tests, MAE, portfolio and validation results."""
    frame, averaged = frames
    seeds = frame[frame["model"] == name].groupby("seed")
    row = {"model": name, "ic": ics[name].mean(), "p_ic": ic_test(ics[name])["p_value"]}
    row["positive_days_%"] = 100 * (ics[name].dropna() > 0).mean()
    row["single_run_ic"] = np.mean([daily_ic(runs).mean() for _, runs in seeds])
    for reference in ("Linear", "Reversal"):
        if reference in ics:
            test = ic_test(ics[name], ics[reference])
            row[f"ic_vs_{reference}"], row[f"p_vs_{reference}"] = test["mean"], test["p_value"]
    errors = daily_mae(averaged[averaged["model"] == name])
    zero = daily_mae(averaged[averaged["model"] == "Naive"])
    row["mae_vs_zero_%"] = 100 * (errors.mean() / zero.mean() - 1)
    row["p_mae"] = newey_west_test((errors - zero).to_numpy())["p_value"]
    if np.isfinite(row["ic"]):
        row.update(portfolio(averaged[averaged["model"] == name]))
    row["val_ic"], row["val_mae"] = validation.loc[name, "val_ic"], validation.loc[name, "val_mae"]
    if "Oracle" in ics:
        row["oracle_share"] = row["ic"] / ics["Oracle"].mean()
    return {
        key: round(value, 5) if isinstance(value, float) else value for key, value in row.items()
    }


def window_rows(averaged, models):
    """Mean IC of each model in each test window, with the number of test days."""
    rows = []
    for (test_start, name), frame in averaged.groupby(["test_start", "model"], sort=True):
        if name in models:
            ics = daily_ic(frame)
            rows.append(
                {
                    "test_start": test_start,
                    "first_day": frame["date"].min(),
                    "last_day": frame["date"].max(),
                    "model": name,
                    "days": int(ics.size),
                    "ic": round(float(ics.mean()), 5),
                }
            )
    return rows


def comparison_text(validation, ics):
    """Tests the PatternDecoder variants against the other model with the best validation IC."""
    others = [
        name for name in validation.index
        if not name.startswith("PatternDecoder") and name not in BENCHMARKS
    ]
    compared = [name for name in COMPARED if name in ics]
    if not others or not compared:
        return ""
    reference = validation.loc[others, "val_ic"].idxmax()
    lines = [f"Comparison model (highest mean validation IC of the other models): {reference}"]
    for name in compared:
        test = ic_test(ics[name], ics[reference])
        lines.append(
            f"- {name} minus {reference}: IC difference {test['mean']:+.4f}, "
            f"p = {test['p_value']:.3f}"
        )
    return "\n".join(lines)


def summary_text(pooled, windows, frames):
    """Markdown summary: test period, table of the pooled results and the PatternDecoder test."""
    frame, averaged = frames
    days = averaged.groupby("date")["ticker"].nunique()
    first = {}
    for row in windows:
        first.setdefault(row["test_start"], f"{row['first_day']} to {row['last_day']}")
    periods = "; ".join(first[start] for start in sorted(first))
    columns = list(pooled[0])
    lines = [
        f"Test windows: {periods}. {frame['seed'].nunique()} seeds, {days.size} test days, "
        f"{days.mean():.1f} stocks per day. IC: Spearman rank correlation between the "
        "seed-averaged forecasts and the excess returns of a day; p: Newey-West test.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for row in sorted(pooled, key=lambda r: -np.nan_to_num(r["ic"], nan=-9)):
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def summarize(paths, out_path=None):
    """Prints the pooled results; optionally writes them and the per-window results as CSV."""
    frames = load_forecasts(paths)
    frame, averaged = frames
    models = list(dict.fromkeys(averaged["model"]))
    ics = {name: daily_ic(averaged[averaged["model"] == name]) for name in models}
    validation = (
        frame.groupby(["model", "seed", "test_start"])[["val_mae", "val_ic"]].first()
        .groupby("model").mean()
    )
    pooled = [model_row(name, frames, ics, validation) for name in models]
    windows = window_rows(averaged, models)
    text = summary_text(pooled, windows, frames) + "\n\n" + comparison_text(validation, ics)
    publish_summary("Cross-section evaluation", text)
    if out_path:
        pd.DataFrame(pooled).to_csv(out_path, index=False)
        pd.DataFrame(windows).to_csv(out_path.replace(".csv", "_windows.csv"), index=False)


def main():
    """Command line entry point."""
    walk_forward_main(__doc__.split("\n\n", maxsplit=1)[0], run, summarize)


if __name__ == "__main__":
    main()
