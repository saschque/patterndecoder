"""
Panel of stocks for the cross-section study (patterndecoder/cross_section.py).

``UNIVERSE`` lists the 40 members of the DAX on 20 March 2023 with their Xetra
tickers on Yahoo Finance. ``write_panel`` downloads their daily prices, adjusted
for dividends and splits, with high, low and volume for the date range of the
config (``ticker: DAX-PANEL``), or generates the synthetic panel (``ticker:
SYNTH-PANEL``), and writes them to ``<data_path>panel_<ticker>.csv``.
``panel_data`` turns a panel into input features and targets, ``panel_windows``
into samples, and ``daily_ic`` and ``long_short`` measure cross-sectional
forecasts.

The synthetic panel is the positive control of the study: 40 stocks with a
common market factor and fat-tailed idiosyncratic returns that reverse by
``REVERSAL`` (10 %) of the previous day's idiosyncratic return. The excess
return over the panel mean is then predictable from the previous day's excess
return; the oracle forecast is -REVERSAL times that excess return.

Usage:
    python -m patterndecoder.panel   (writes and describes the panel of the configured ticker)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DAX_PANEL = "DAX-PANEL"
SYNTHETIC_PANEL = "SYNTH-PANEL"
UNIVERSE_DATE = "2023-03-20"
UNIVERSE = (
    "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "BNR.DE",
    "CBK.DE", "CON.DE", "1COV.DE", "DTG.DE", "DBK.DE", "DB1.DE", "DHL.DE", "DTE.DE",
    "EOAN.DE", "FRE.DE", "HNR1.DE", "HEI.DE", "HEN3.DE", "IFX.DE", "MBG.DE", "MRK.DE",
    "MTX.DE", "MUV2.DE", "P911.DE", "PAH3.DE", "QIA.DE", "RHM.DE", "RWE.DE", "SAP.DE",
    "SRT3.DE", "SIE.DE", "ENR.DE", "SHL.DE", "SY1.DE", "VOW3.DE", "VNA.DE", "ZAL.DE",
)
COLUMNS = ["Close", "High", "Low", "Volume"]
FEATURES = ["excess_return", "panel_return", "relative_volume", "range"]
VOLUME_DAYS = 20
REVERSAL = 0.1
SYNTHETIC_STOCKS = 40
SYNTHETIC_SEED = 2026


def row_position(count, share):
    """First row after a share of the data, robust to floating-point error."""
    return int(count * share + 1e-9)


def panel_path(params):
    """CSV file of the panel of the configured ticker."""
    return Path(params["data_path"]) / f"panel_{params['ticker']}.csv"


def long_format(frames):
    """Turns a dict of (days x stocks) DataFrames into rows of Date, Ticker and one column each."""
    long = pd.concat(frames, axis=1).stack(level=1, future_stack=True)
    long.index.names = ["Date", "Ticker"]
    return long.dropna(subset=["Close"]).reset_index()


def download_panel(params, tickers=UNIVERSE):
    """Daily prices (adjusted for dividends and splits), high, low and volume from Yahoo Finance."""
    import yfinance as yf  # pylint: disable=import-outside-toplevel

    frame = yf.download(
        tickers=list(tickers),
        start=params["start"],
        end=params["end"],
        interval="1d",
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=True,
    )
    if frame.empty:
        raise RuntimeError("Yahoo Finance returned no data for the panel")
    index = pd.to_datetime(frame.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    frame.index = index.normalize().rename("Date")
    return long_format({column: frame[column] for column in COLUMNS})


def synthetic_panel(params, stocks=SYNTHETIC_STOCKS, reversal=REVERSAL, seed=SYNTHETIC_SEED):
    """
    Synthetic panel with a known cross-sectional reversal, on business days of
    the configured date range. ``Oracle`` is the expected excess return of each
    day given the day before.
    """
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(params["end"]) - pd.Timedelta(days=1)
    dates = pd.bdate_range(params["start"], end, name="Date")
    scale = rng.uniform(0.01, 0.02, stocks)
    market = 0.01 * rng.standard_t(4, len(dates)) / np.sqrt(2)
    shocks = rng.standard_t(4, (len(dates), stocks)) / np.sqrt(2) * scale
    idiosyncratic = shocks.copy()
    for row in range(1, len(dates)):
        idiosyncratic[row] -= reversal * idiosyncratic[row - 1]
    returns = market[:, None] + idiosyncratic
    excess = idiosyncratic - idiosyncratic.mean(axis=1, keepdims=True)
    oracle = np.full_like(returns, np.nan)
    oracle[1:] = -reversal * excess[:-1]
    close = 100 * np.exp(np.cumsum(returns, axis=0))
    spread = np.abs(returns) + np.abs(rng.normal(0, 0.5, returns.shape)) * scale
    values = {
        "Close": close,
        "High": close * np.exp(spread / 2),
        "Low": close * np.exp(-spread / 2),
        "Volume": np.exp(14 + 20 * np.abs(idiosyncratic) + rng.normal(0, 0.3, returns.shape)),
        "Oracle": oracle,
    }
    tickers = [f"SYN{number:02d}" for number in range(1, stocks + 1)]
    return long_format(
        {name: pd.DataFrame(array, index=dates, columns=tickers) for name, array in values.items()}
    )


def write_panel(params):
    """Downloads or generates the panel of the configured ticker and writes it to its CSV file."""
    if params["ticker"] == SYNTHETIC_PANEL:
        panel = synthetic_panel(params)
    elif params["ticker"] == DAX_PANEL:
        panel = download_panel(params)
    else:
        raise ValueError(f"No panel for ticker {params['ticker']!r}")
    path = panel_path(params)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(path, index=False, date_format="%Y-%m-%d")
    return panel


def load_panel(params):
    """Reads the panel of the configured ticker; writes it first if its file does not exist."""
    path = panel_path(params)
    if not path.exists():
        return write_panel(params)
    return pd.read_csv(path, parse_dates=["Date"])


def describe_panel(panel, params):
    """Prints first and last day, days and days without volume per stock, and missing tickers."""
    stocks = panel.groupby("Ticker").agg(
        first=("Date", "min"),
        last=("Date", "max"),
        days=("Date", "size"),
        no_volume=("Volume", lambda volume: int((volume <= 0).sum())),
    )
    print(stocks.to_string())
    returns = np.log(panel.pivot(index="Date", columns="Ticker", values="Close")).diff()
    large = returns.stack(future_stack=True)[lambda moves: moves.abs() > 0.2]
    print(f"{len(stocks)} stocks, {panel['Date'].nunique()} days; daily moves above 20 %:")
    print(large.round(4).to_string() if len(large) else "none")
    if params["ticker"] == DAX_PANEL:
        missing = sorted(set(UNIVERSE) - set(stocks.index))
        print(f"Tickers without data: {', '.join(missing) if missing else 'none'}")


def panel_data(panel, input_scale=1):
    """
    Features and targets of a panel.

    Returns:
        dict: ``dates`` and ``tickers``; ``features`` (days x stocks x 4): the
        excess log return (log return minus the mean over the stocks of that
        day), the panel's mean log return, the log volume relative to its mean
        over the 20 days before, and the range log(high / low), returns and
        range multiplied by ``input_scale``; ``target`` (days x stocks): the
        excess log return; ``oracle`` for the synthetic panel.
    """
    wide = {
        column: panel.pivot(index="Date", columns="Ticker", values=column).sort_index()
        for column in COLUMNS
    }
    returns = np.log(wide["Close"]).diff()
    market = returns.mean(axis=1)
    excess = returns.sub(market, axis=0)
    log_volume = np.log(wide["Volume"].where(wide["Volume"] > 0))
    mean_volume = log_volume.rolling(VOLUME_DAYS, min_periods=VOLUME_DAYS // 2).mean()
    relative_volume = log_volume - mean_volume.shift(1)
    day_range = np.log(wide["High"] / wide["Low"])
    features = np.stack(
        [
            excess.to_numpy() * input_scale,
            np.broadcast_to(market.to_numpy()[:, None], excess.shape) * input_scale,
            relative_volume.to_numpy(),
            day_range.to_numpy() * input_scale,
        ],
        axis=-1,
    )
    data = {
        "dates": returns.index,
        "tickers": list(returns.columns),
        "features": features.astype("float32"),
        "target": excess.to_numpy(dtype=float, copy=True),
    }
    if "Oracle" in panel:
        oracle = panel.pivot(index="Date", columns="Ticker", values="Oracle")
        oracle = oracle.reindex(index=returns.index, columns=returns.columns)
        data["oracle"] = oracle.to_numpy(dtype=float, copy=True)
    return data


def panel_windows(data, rows, window):
    """
    Samples for the target days ``rows``: the features of each stock on the
    ``window`` days before the target day and its excess return on the target
    day. Stocks with a missing value in either are left out.

    Returns:
        tuple: inputs (samples x window x features), targets (samples,) and
        keys (samples x 2: row of the target day, column of the stock).
    """
    features, target = data["features"], data["target"]
    inputs, targets, keys = [], [], []
    for row in rows:
        block = features[row - window : row]
        stocks = np.flatnonzero(np.isfinite(block).all(axis=(0, 2)) & np.isfinite(target[row]))
        if stocks.size:
            inputs.append(np.transpose(block[:, stocks], (1, 0, 2)))
            targets.append(target[row, stocks])
            keys.append(np.column_stack([np.full(stocks.size, row), stocks]))
    if not inputs:
        raise ValueError("No complete windows for these days")
    return (
        np.concatenate(inputs).astype("float32"),
        np.concatenate(targets).astype("float32"),
        np.concatenate(keys),
    )


def split_rows(count, params, test_start, width=0.1):
    """
    Target days (rows) of one walk-forward window, as in robustness.split_window:
    the test window covers the share test_start to test_start + width of the
    days, the training days are all days before it, and their last
    ``validation_split`` share is the validation set.

    Returns:
        dict: ``fit``, ``val`` and ``test`` arrays of rows.
    """
    window = params["window_size"]
    start = row_position(count, test_start)
    end = min(count, row_position(count, test_start + width))
    if not window < start < end:
        raise ValueError(f"Test window {test_start}+{width} is outside the data")
    n_val = int(start * float(params["validation_split"]))
    return {
        "fit": np.arange(window, start - n_val),
        "val": np.arange(start - n_val, start),
        "test": np.arange(start, end),
    }


def daily_ic(frame):
    """
    Information coefficient of each day: the Spearman rank correlation between
    the forecasts and the outcomes across the stocks of that day (NaN when the
    forecasts are constant).

    Args:
        frame (pd.DataFrame): Columns date, prediction and actual.

    Returns:
        pd.Series: IC by date.
    """
    ranks = frame.groupby("date")[["prediction", "actual"]].rank()
    ranks["date"] = frame["date"]
    correlations = ranks.groupby("date")[["prediction", "actual"]].corr()
    return correlations.xs("prediction", level=1)["actual"]


def long_short(frame, share=0.2, cost=0.0005):
    """
    Daily returns of a portfolio that buys the stocks in the top ``share`` of
    each day's forecasts and sells those in the bottom share short, with equal
    weights per side, before and after costs of ``cost`` per unit of turnover.

    Args:
        frame (pd.DataFrame): Columns date, ticker, prediction and actual
            (the outcome as log return).

    Returns:
        pd.DataFrame: gross, turnover and net by date.
    """
    by_date = frame.groupby("date")["prediction"]
    ranks = by_date.rank(method="first")
    counts = by_date.transform("size")
    picks = np.maximum(1, np.round(share * counts))
    weight = np.where(ranks > counts - picks, 1 / picks, np.where(ranks <= picks, -1 / picks, 0.0))
    weights = frame.assign(weight=weight).pivot(index="date", columns="ticker", values="weight")
    weights = weights.sort_index().fillna(0.0)
    actual = frame.pivot(index="date", columns="ticker", values="actual").reindex_like(weights)
    gross = (weights * actual.fillna(0.0)).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    return pd.DataFrame({"gross": gross, "turnover": turnover, "net": gross - cost * turnover})


def main():
    """Writes and describes the panel of the configured ticker."""
    with open("config/config.yaml", encoding="utf-8") as file:
        params = yaml.safe_load(file)
    describe_panel(write_panel(params), params)


if __name__ == "__main__":
    main()
