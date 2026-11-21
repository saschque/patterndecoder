# pylint: disable=E1101
"""
Hyperparameter search on the chronological validation split.

Each trial trains one model with one combination of the GRID below and one
seed, using only the training part of the data (``train_split``). Inside it,
compile_and_train holds out the last ``validation_split`` share for
validation; the score of a trial is the best validation MAE over its epochs.
The test set is never used here.

Usage:
    python -m patterndecoder.tuning run --model models/transformer.yaml --out t.csv
    python -m patterndecoder.tuning summarize results/*.csv
"""

import argparse
import csv
import itertools
import os
import statistics
import tempfile
from collections import defaultdict

import numpy as np
import tensorflow as tf
import yaml

from patterndecoder.training import split_train_validation
from patterndecoder.utils import (
    compile_and_train,
    get_stock_data,
    load_config,
    load_model,
)

GRID = {
    "loss": ["Huber", "mae"],
    "learning_rate": [1.0e-4, 1.0e-3],
    "dropout": [0.1, 0.3],
    "size": ["32x128", "16x64"],
}
SEEDS = [1, 2]
FIELDS = [
    "model",
    "loss",
    "learning_rate",
    "dropout",
    "size",
    "seed",
    "val_mae",
    "rel_to_zero",
    "best_epoch",
    "epochs_run",
    "zero_val_mae",
    "median_val_mae",
]


def load_training_data(params):
    """Returns the log returns of the training period (first train_split share)."""
    prices = get_stock_data(params, False)[params["columns"]]
    returns = np.log(prices / prices.shift(1)).fillna(0)
    return returns.iloc[: int(returns.shape[0] * params["train_split"])]


def validation_baselines(train_data, params):
    """MAE of the zero forecast and of the fit-period median on the validation targets."""
    fit_data, val_data = split_train_validation(train_data, params)
    target = params["target_column"]
    val_targets = val_data[target].to_numpy()[params["window_size"] :]
    median = float(np.median(fit_data[target].to_numpy()))
    return (
        float(np.mean(np.abs(val_targets))),
        float(np.mean(np.abs(val_targets - median))),
    )


def trial_overrides():
    """Yields one dict of config overrides per grid combination."""
    keys = list(GRID)
    for values in itertools.product(*(GRID[key] for key in keys)):
        combo = dict(zip(keys, values))
        d_model, units = (int(part) for part in combo["size"].split("x"))
        yield combo, {
            "loss": [combo["loss"]],
            "learning_rate": combo["learning_rate"],
            "dropout": combo["dropout"],
            "d_model": d_model,
            "units": units,
        }


def run_trial(params, model_yaml, train_data, workdir):
    """Trains one model with the given params; returns its validation history."""
    params = dict(params)
    params["tmp_weights_file"] = os.path.join(workdir, "")
    params["tmp_history_file"] = os.path.join(workdir, "")
    config_path = os.path.join(workdir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(params, file)
    tf.keras.backend.clear_session()
    model = load_model(params, model_yaml)
    history, _ = compile_and_train(model, train_data, config_path)
    return history.history["val_mae"]


def make_row(model_name, combo, seed, val_mae, baselines):
    """Builds one CSV row from a trial's validation history."""
    zero_mae, median_mae = baselines
    best = int(np.argmin(val_mae))
    return {
        **combo,
        "model": model_name,
        "seed": seed,
        "val_mae": round(float(val_mae[best]), 7),
        "rel_to_zero": round(float(val_mae[best]) / zero_mae - 1, 5),
        "best_epoch": best + 1,
        "epochs_run": len(val_mae),
        "zero_val_mae": round(zero_mae, 7),
        "median_val_mae": round(median_mae, 7),
    }


def run(model_yaml, out_path, config_path="config/config.yaml"):
    """Runs every grid combination and seed for one model and writes a CSV."""
    base = load_config(config_path)
    base.update(training=True, early_stopping=True, restore_best=True, verbose=False)
    train_data = load_training_data(base)
    baselines = validation_baselines(train_data, base)
    model_name = os.path.splitext(os.path.basename(model_yaml))[0]
    with open(out_path, "w", newline="", encoding="utf-8") as out_file, \
            tempfile.TemporaryDirectory() as workdir:
        writer = csv.DictWriter(out_file, fieldnames=FIELDS)
        writer.writeheader()
        for combo, overrides in trial_overrides():
            for seed in SEEDS:
                val_mae = run_trial(
                    {**base, **overrides, "seed": seed}, model_yaml, train_data, workdir
                )
                row = make_row(model_name, combo, seed, val_mae, baselines)
                writer.writerow(row)
                out_file.flush()
                print(row, flush=True)


def summarize(paths):
    """Prints a markdown table: mean relative validation MAE per grid combination."""
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))
    by_combo = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in GRID)
        by_combo[key].append(float(row["rel_to_zero"]))
    ranked = sorted(by_combo.items(), key=lambda item: statistics.mean(item[1]))
    lines = [
        f"Trials: {len(rows)}; zero forecast val MAE {rows[0]['zero_val_mae']}, "
        f"fit-median forecast val MAE {rows[0]['median_val_mae']}",
        "",
        "| " + " | ".join(GRID) + " | mean vs zero | best | worst | n |",
        "|" + "---|" * (len(GRID) + 4),
    ]
    for key, values in ranked:
        lines.append(
            "| " + " | ".join(key) + f" | {100 * statistics.mean(values):+.2f} % "
            f"| {100 * min(values):+.2f} % | {100 * max(values):+.2f} % | {len(values)} |"
        )
    text = "\n".join(lines)
    print(text)
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as file:
            file.write("### Hyperparameter search (validation MAE vs zero forecast)\n\n")
            file.write(text + "\n")


def main():
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--out", required=True)
    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    if args.command == "run":
        run(args.model, args.out)
    else:
        summarize(args.paths)


if __name__ == "__main__":
    main()
