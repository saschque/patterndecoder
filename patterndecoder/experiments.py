# pylint: disable=E1101
"""
Shared helpers for the experiment scripts (tuning.py and study.py).
"""

import argparse
import contextlib
import csv
import os
import tempfile

import tensorflow as tf
import yaml

from patterndecoder.utils import load_model


def build_model(params, model_yaml, workdir):
    """
    Writes ``params`` to a config file in ``workdir`` and builds the model.

    compile_and_train reads its settings from a config file, so each trial
    gets its own. Returns (model, config_path).
    """
    params = dict(params)
    params["tmp_weights_file"] = os.path.join(workdir, "")
    params["tmp_history_file"] = os.path.join(workdir, "")
    config_path = os.path.join(workdir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(params, file)
    tf.keras.backend.clear_session()
    return load_model(params, model_yaml), config_path


@contextlib.contextmanager
def results_writer(out_path, fields):
    """
    Opens a CSV results file and a temporary working directory.

    Yields (write_row, workdir); write_row writes, flushes and prints a row so
    that partial results survive an aborted job.
    """
    with open(out_path, "w", newline="", encoding="utf-8") as out_file, \
            tempfile.TemporaryDirectory() as workdir:
        writer = csv.DictWriter(out_file, fieldnames=fields)
        writer.writeheader()

        def write_row(row):
            writer.writerow(row)
            out_file.flush()
            print(row, flush=True)

        yield write_row, workdir


def read_csv_rows(paths):
    """Reads and concatenates the rows of several CSV result files."""
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))
    return rows


def publish_summary(title, text):
    """Prints a summary and appends it to the GitHub job summary if available."""
    print(text)
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as file:
            file.write(f"### {title}\n\n{text}\n")


def parse_command(description, add_run_arguments):
    """
    Parses ``run`` / ``summarize`` sub-commands.

    ``add_run_arguments`` receives the ``run`` sub-parser to add its options;
    ``summarize`` takes CSV paths. Returns the parsed arguments.
    """
    parser = argparse.ArgumentParser(description=description)
    commands = parser.add_subparsers(dest="command", required=True)
    add_run_arguments(commands.add_parser("run"))
    commands.add_parser("summarize").add_argument("paths", nargs="+")
    return parser.parse_args()
