# pylint: disable=E1101
"""
Seed and layer study for all models defined in models/*.yaml.

``run`` trains every model once with one seed and one combination of encoder
and decoder layers, and records the best validation MAE (the epoch selected by
compile_and_train) and the test MAE/RMSE of the selected weights.

``summarize`` chooses the layer combination of each model by its mean
validation MAE over the seeds and only then reports the test results of that
combination, averaged over the seeds. Test results never influence the choice.

Usage:
    python -m patterndecoder.study run --seed 1 --enc 3 --dec 2 --out r.csv
    python -m patterndecoder.study summarize results/*.csv
"""

import glob
import statistics
from collections import defaultdict

import numpy as np

from patterndecoder.experiments import (
    build_model,
    parse_command,
    publish_summary,
    read_csv_rows,
    results_writer,
)
from patterndecoder.utils import (
    compile_and_train,
    get_model_performance,
    get_stock_data,
    load_config,
)

FIELDS = [
    "model",
    "seed",
    "enc",
    "dec",
    "val_mae",
    "test_mae",
    "test_rmse",
    "naive_test_mae",
    "naive_test_rmse",
    "epochs_run",
]


def model_files():
    """Returns the model definitions used in the study (all but the example)."""
    return sorted(f for f in glob.glob("models/*.yaml") if "example" not in f)


def load_split(params):
    """Returns the training and test log returns, split as in the notebook."""
    prices = get_stock_data(params, False)[params["columns"]]
    returns = np.log(prices / prices.shift(1)).fillna(0)
    train_size = int(returns.shape[0] * params["train_split"])
    return returns.iloc[:train_size], returns.iloc[train_size:]


def naive_errors(test_data, params):
    """MAE and RMSE of the zero forecast on the test targets the models predict."""
    targets = test_data.to_numpy()[params["window_size"] :]
    return float(np.mean(np.abs(targets))), float(np.sqrt(np.mean(targets**2)))


def train_and_score(params, model_yaml, data, workdir):
    """Trains one model; returns (validation history, test MAE, test RMSE)."""
    train_data, test_data = data
    model, config_path = build_model(params, model_yaml, workdir)
    history, model = compile_and_train(model, train_data, config_path)
    predictions = get_model_performance(model, params, test_data)["predictions"]
    errors = predictions.to_numpy() - test_data.to_numpy()[params["window_size"] :]
    return (
        history.history["val_mae"],
        float(np.mean(np.abs(errors))),
        float(np.sqrt(np.mean(errors**2))),
    )


def make_row(model_yaml, params, scores, naive):
    """Builds one CSV row from a model's scores and the zero-forecast errors."""
    val_mae, test_mae, test_rmse = scores
    return {
        "model": load_config(model_yaml)["name"],
        "seed": params["seed"],
        "enc": params["n_layers_enc"],
        "dec": params["n_layers_dec"],
        "val_mae": round(float(min(val_mae)), 7),
        "test_mae": round(test_mae, 7),
        "test_rmse": round(test_rmse, 7),
        "naive_test_mae": round(naive[0], 7),
        "naive_test_rmse": round(naive[1], 7),
        "epochs_run": len(val_mae),
    }


def run(seed, layers, out_path, config_path="config/config.yaml"):
    """Trains all models for one seed and one (encoder, decoder) layer pair."""
    params = load_config(config_path)
    params.update(
        training=True,
        verbose=False,
        seed=seed,
        n_layers_enc=layers[0],
        n_layers_dec=layers[1],
    )
    data = load_split(params)
    naive = naive_errors(data[1], params)
    with results_writer(out_path, FIELDS) as (write_row, workdir):
        for model_yaml in model_files():
            scores = train_and_score(params, model_yaml, data, workdir)
            write_row(make_row(model_yaml, params, scores, naive))


def read_rows(paths):
    """Reads the CSV files written by ``run``."""
    rows = read_csv_rows(paths)
    for row in rows:
        for key in ("val_mae", "test_mae", "test_rmse", "naive_test_mae", "naive_test_rmse"):
            row[key] = float(row[key])
    return rows


def select_layers(rows):
    """Per model, the (enc, dec) pair with the lowest mean validation MAE."""
    by_config = defaultdict(list)
    for row in rows:
        by_config[(row["model"], row["enc"], row["dec"])].append(row["val_mae"])
    best = {}
    for (model, enc, dec), values in by_config.items():
        score = statistics.mean(values)
        if model not in best or score < best[model][1]:
            best[model] = ((enc, dec), score)
    return {model: layers for model, (layers, _) in best.items()}


def chosen_results(rows, chosen):
    """Test MAE per seed and model for the chosen layers, and the ranks per seed."""
    per_seed = defaultdict(dict)
    for row in rows:
        if (row["enc"], row["dec"]) == chosen[row["model"]]:
            per_seed[row["seed"]][row["model"]] = row["test_mae"]
    ranks = defaultdict(list)
    for results in per_seed.values():
        for position, model in enumerate(sorted(results, key=results.get), start=1):
            ranks[model].append(position)
    return per_seed, ranks


def test_table(rows, chosen):
    """Markdown lines with the seed-averaged test results of the chosen layers."""
    naive = rows[0]["naive_test_mae"]
    per_seed, ranks = chosen_results(rows, chosen)
    seeds = sorted(per_seed, key=int)
    stats = []
    for model, layers in chosen.items():
        values = [per_seed[seed][model] for seed in seeds if model in per_seed[seed]]
        stats.append((statistics.mean(values), model, layers, values))
    lines = [
        f"Seeds: {', '.join(seeds)}; zero forecast test MAE {naive:.6f}",
        "",
        "| Model | enc/dec | mean test MAE | vs zero | sd | below zero | mean rank |",
        "|---|---|---|---|---|---|---|",
    ]
    for mean, model, layers, values in sorted(stats):
        spread = statistics.stdev(values) if len(values) > 1 else 0.0
        lines.append(
            f"| {model} | {layers[0]}/{layers[1]} | {mean:.6f} | "
            f"{100 * (mean / naive - 1):+.2f} % | {100 * spread / naive:.2f} % | "
            f"{sum(v < naive for v in values)}/{len(values)} | "
            f"{statistics.mean(ranks[model]):.1f} |"
        )
    return lines


def summarize(paths):
    """Prints the layer choice (validation) and the test results (seed means)."""
    rows = read_rows(paths)
    lines = ["Layers chosen per model by mean validation MAE over the seeds."]
    lines += test_table(rows, select_layers(rows))
    publish_summary("Seed and layer study", "\n".join(lines))


def main():
    """Command line entry point."""

    def add_run_arguments(run_parser):
        run_parser.add_argument("--seed", type=int, required=True)
        run_parser.add_argument("--enc", type=int, required=True)
        run_parser.add_argument("--dec", type=int, required=True)
        run_parser.add_argument("--out", required=True)

    args = parse_command(__doc__.split("\n\n", maxsplit=1)[0], add_run_arguments)
    if args.command == "run":
        run(args.seed, (args.enc, args.dec), args.out)
    else:
        summarize(args.paths)


if __name__ == "__main__":
    main()
