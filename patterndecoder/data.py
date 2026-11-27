"""
Market data: download from Yahoo Finance and local caching.

The price data are not part of the repository. On first use, get_stock_data()
downloads the configured ticker for a fixed date range and caches it as a CSV
file under ``data_path``. data_checksum() fingerprints the closing prices, so a
run can tell whether it uses the same observations as the published results.
"""

import hashlib
import warnings
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

from patterndecoder.synthetic import is_synthetic, write_synthetic_data


def data_checksum(dataframe, column="Close"):
    """
    Returns a SHA-256 fingerprint of the dates and prices of one column.

    Prices are rounded to four decimals, so the fingerprint does not depend on
    how the numbers are formatted in the CSV file.

    Args:
        dataframe (pd.DataFrame): Price data with a DatetimeIndex.
        column (str): Column to fingerprint. Default is 'Close'.

    Returns:
        str: Hexadecimal SHA-256 digest.
    """
    rows = (f"{date:%Y-%m-%d},{value:.4f}" for date, value in dataframe[column].items())
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def download_stock_data(params, path):
    """
    Downloads daily price data from Yahoo Finance and writes it to ``path``.

    Args:
        params (dict): Configuration with ticker, start, end (exclusive) and interval.
        path (Path): Target CSV file.
    """
    dataframe = yf.download(
        tickers=[params["ticker"]],
        start=params["start"],
        end=params["end"],
        interval=params["interval"],
        auto_adjust=False,
        prepost=True,
        threads=True,
        progress=False,
    )
    if dataframe.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {params['ticker']}")

    dataframe.columns = dataframe.columns.get_level_values("Price")
    index = pd.to_datetime(dataframe.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    dataframe.index = index.normalize().rename("Date")

    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, date_format="%Y-%m-%d")


def get_stock_data(params, download=False):
    """
    Returns daily price data for ``params["ticker"]``.

    The data are read from ``<data_path><ticker>.<type>``. If that file does not
    exist, or ``download`` is True, they are first downloaded from Yahoo Finance
    for the fixed range ``params["start"]`` to ``params["end"]`` (end exclusive);
    tickers starting with ``SYNTH-`` name synthetic series, which are generated
    instead (see patterndecoder/synthetic.py).
    If the config contains ``data_checksum``, a warning is issued when the
    closing prices differ from the data used for the published results.

    Args:
        params (dict): Configuration with data_path, ticker, type, start, end,
            interval, columns and optionally data_checksum.
        download (bool): Download even if the file already exists. Default is False.

    Returns:
        pd.DataFrame: Price data with a DatetimeIndex.
    """
    path = Path(f"{params['data_path']}{params['ticker']}.{params['type']}")
    if download or not path.exists():
        if is_synthetic(params["ticker"]):
            write_synthetic_data(params, path)
        else:
            download_stock_data(params, path)

    series_dataframe = pd.read_csv(path, index_col=0, sep=",")
    series_dataframe.index = pd.to_datetime(
        series_dataframe.index, format="%Y-%m-%d", errors="coerce"
    )

    expected = params.get("data_checksum")
    if expected and data_checksum(series_dataframe, params["columns"][0]) != expected:
        warnings.warn(
            f"The price data in {path} differ from the data used for the published "
            "results (data_checksum in the config), so results will not match exactly.",
            stacklevel=2,
        )

    return series_dataframe


def main(config_path="config/config.yaml"):
    """Downloads the configured data if needed and reports whether they match."""
    with open(config_path, encoding="utf-8") as file:
        params = yaml.safe_load(file)
    dataframe = get_stock_data(params)
    summary = (
        f"{params['ticker']}: {len(dataframe)} rows, "
        f"{dataframe.index[0].date()} to {dataframe.index[-1].date()}"
    )
    if "Volume" in dataframe:
        missing = int((dataframe["Volume"].fillna(0) <= 0).sum())
        summary += f", volume zero or missing on {missing} days"
    expected = params.get("data_checksum")
    if not expected:
        print(f"{summary}, no checksum configured")
        return
    matches = data_checksum(dataframe, params["columns"][0]) == expected
    print(f"{summary}, checksum {'matches' if matches else 'differs from'} the published data")
    if not matches:
        print("::warning::The price data differ from the data used for the published results")


if __name__ == "__main__":
    main()
