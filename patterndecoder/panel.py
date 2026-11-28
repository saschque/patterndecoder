"""
Panel of stocks for the cross-section study (patterndecoder/cross_section.py).

``UNIVERSE`` lists the 40 members of the DAX on 20 March 2023,
``SDAX_UNIVERSE`` the 71 companies of the SDAX table of the English Wikipedia,
stated as of the index review of 10 September 2020, with their Xetra tickers on
Yahoo Finance (for renamed companies the current ticker), and
``FTSE250_UNIVERSE`` the 250 constituents of the FTSE 250 table of the English
Wikipedia after the index changes of 25 April 2023, as archived in May 2023,
with their London tickers. ``write_panel`` downloads their daily prices,
adjusted for dividends and splits, with high, low and volume for the date range
of the config (``ticker: DAX-PANEL``, ``SDAX-PANEL`` or ``FTSE250-PANEL``), or
generates the synthetic panel (``ticker: SYNTH-PANEL``), and writes them to
``<data_path>panel_<ticker>.csv``.
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
SDAX_PANEL = "SDAX-PANEL"
# Renamed companies under their current ticker: DIC Asset (BRNK), S&T (KTN),
# Schaeffler (SHA0) and Shop Apotheke Europe (RDC)
SDAX_UNIVERSE = (
    "1U1.DE", "ADJ.DE", "ADV.DE", "AAD.DE", "AOF.DE", "BYW6.DE", "BFSA.DE", "BDT.DE",
    "GBF.DE", "BVB.DE", "CEC.DE", "CWC.DE", "CCAP.DE", "DMP.DE", "DBAN.DE", "DEQ.DE",
    "PBB.DE", "DEZ.DE", "BRNK.DE", "GIL.DE", "HNL.DE", "DRW3.DE", "DWS.DE", "EUZ.DE",
    "ECV.DE", "FIE.DE", "HABA.DE", "HHFA.DE", "HBM.DE", "HBH.DE", "HYQ.DE", "INH.DE",
    "INS.DE", "JEN.DE", "JST.DE", "JUN3.DE", "KCO.DE", "SKB.DE", "KRN.DE", "KWS.DE",
    "LEO.DE", "LPK.DE", "NA9.DE", "NWO.DE", "NDX1.DE", "NOEJ.DE", "PAT.DE", "PFV.DE",
    "KTN.DE", "SFQ.DE", "SZG.DE", "SHA0.DE", "RDC.DE", "SIX2.DE", "S92.DE", "SHF.DE",
    "STM.DE", "SNH.DE", "SBS.DE", "SZU.DE", "TTK.DE", "TLX.DE", "TC1.DE", "8TRA.DE",
    "VOS.DE", "WCH.DE", "WAC.DE", "WSU.DE", "WUW.DE", "TIMA.DE", "ZO1.DE",
)
FTSE250_PANEL = "FTSE250-PANEL"
# QinetiQ, whose ticker the table leaves out, as QQ.L; candidate current tickers of
# companies renamed since: Shaftesbury Capital (SHC), Redde Northgate (ZIG) and
# National Express (MCG)
FTSE250_UNIVERSE = (
    "3IN.L", "FOUR.L", "ASL.L", "APEO.L", "ATST.L", "ATT.L", "APAX.L", "ASCL.L", "ASHM.L",
    "ASC.L", "AGR.L", "AML.L", "ATG.L", "AGT.L", "BAB.L", "BGFD.L", "BAKK.L", "BCPT.L", "BBY.L",
    "BCG.L", "BNKR.L", "BGEO.L", "BAG.L", "BBGI.L", "AJB.L", "BBH.L", "BWY.L", "BHMG.L", "BYG.L",
    "BRSC.L", "THRG.L", "BRWM.L", "BSIF.L", "BOY.L", "BPT.L", "BVIC.L", "BYIT.L", "CCR.L",
    "CLDN.L", "CGT.L", "CNE.L", "CCL.L", "CEY.L", "CHG.L", "CTY.L", "CKN.L", "CBG.L", "CLI.L",
    "CMCX.L", "COA.L", "CCC.L", "CWK.L", "CRST.L", "CURY.L", "DARK.L", "DPH.L", "DLN.L", "DGI9.L",
    "DPLM.L", "DLG.L", "DSCV.L", "DEC.L", "DOM.L", "DWL.L", "DRX.L", "DOCS.L", "DNLM.L", "EZJ.L",
    "EDIN.L", "EWI.L", "ELM.L", "ENOG.L", "ESNT.L", "EOT.L", "ESCT.L", "FDM.L", "FXPO.L",
    "FCSS.L", "FEML.L", "FEV.L", "FSV.L", "FGT.L", "FGP.L", "FSFL.L", "FUTR.L", "GAW.L", "GCP.L",
    "GEN.L", "GNS.L", "GSCT.L", "GFTU.L", "GRI.L", "GPE.L", "UKW.L", "GRG.L", "HMSO.L", "HVPE.L",
    "HAS.L", "HBR.L", "HTWS.L", "HSL.L", "HRI.L", "HGT.L", "HICL.L", "HIK.L", "HILS.L", "HFG.L",
    "SONG.L", "HWDN.L", "HTG.L", "IBST.L", "ICGT.L", "IGG.L", "IMI.L", "IEM.L", "INCH.L",
    "INDV.L", "IHP.L", "ICP.L", "IDS.L", "INPP.L", "INVP.L", "IPO.L", "ITH.L", "ITV.L", "IWG.L",
    "JLEN.L", "JAM.L", "JMG.L", "JEDT.L", "JGGI.L", "JII.L", "JFJ.L", "JTC.L", "JUP.L", "JUST.L",
    "KNOS.L", "KLR.L", "LRE.L", "LWDB.L", "LIO.L", "LMP.L", "LXI.L", "EMG.L", "MKS.L", "MSLH.L",
    "MDC.L", "MRC.L", "MRCH.L", "MAB.L", "MTO.L", "GROW.L", "MONY.L", "MNKS.L", "MGAM.L",
    "MGNS.L", "MUT.L", "MYI.L", "NEX.L", "NBPE.L", "NETW.L", "NESF.L", "N91.L", "OSB.L", "ORIT.L",
    "OXIG.L", "PHI.L", "PAGE.L", "PIN.L", "PAG.L", "PNN.L", "PNL.L", "PHLL.L", "PETS.L", "PTEC.L",
    "PLUS.L", "PCT.L", "PFD.L", "PHP.L", "PRTC.L", "PZC.L", "QQ.L", "QLT.L", "RAT.L", "REDD.L",
    "RDW.L", "RSW.L", "RHIM.L", "RCP.L", "ROR.L", "RICA.L", "SAFE.L", "SVS.L", "SDP.L", "SOI.L",
    "SAIN.L", "SEIT.L", "SNR.L", "SEQI.L", "SRP.L", "SHB.L", "SRE.L", "SSON.L", "SCT.L", "SXS.L",
    "SPI.L", "SPT.L", "SSPG.L", "STEM.L", "SUPR.L", "SYNC.L", "SYNT.L", "THRL.L", "TATE.L",
    "TBCG.L", "TEP.L", "TMPL.L", "TEM.L", "TRIG.L", "TIFS.L", "TCAP.L", "TRN.L", "TPK.L",
    "BBOX.L", "EBOX.L", "TRY.L", "TUI.L", "TLW.L", "TFIF.L", "UKCM.L", "SHED.L", "VANQ.L",
    "VSVS.L", "VCT.L", "VID.L", "VEIL.L", "VOF.L", "VMUK.L", "VTY.L", "FAN.L", "WPS.L", "WHR.L",
    "WOSG.L", "JDW.L", "SMWH.L", "WTAN.L", "WIZZ.L", "WG.L", "WKP.L", "WWH.L", "SHC.L", "ZIG.L",
    "MCG.L",
)
PANELS = {DAX_PANEL: UNIVERSE, SDAX_PANEL: SDAX_UNIVERSE, FTSE250_PANEL: FTSE250_UNIVERSE}
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


def download_prices(params, tickers):
    """One Yahoo Finance download of daily prices, adjusted for dividends and splits."""
    import yfinance as yf  # pylint: disable=import-outside-toplevel

    frame = yf.download(
        tickers=list(tickers),
        start=params["start"],
        end=params["end"],
        interval="1d",
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
    )
    if frame.empty:
        return pd.DataFrame(columns=["Date", "Ticker", *COLUMNS])
    index = pd.to_datetime(frame.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    frame.index = index.normalize().rename("Date")
    return long_format({column: frame[column] for column in COLUMNS})


def download_panel(params, tickers=UNIVERSE, attempts=3):
    """
    Daily prices, high, low and volume of the tickers from Yahoo Finance; tickers
    without data are downloaded again, up to ``attempts`` times in all.
    """
    parts, missing = [], list(tickers)
    for _ in range(attempts):
        if missing:
            parts.append(download_prices(params, missing))
            missing = sorted(set(missing) - set(parts[-1]["Ticker"]))
    panel = pd.concat(parts, ignore_index=True)
    if panel.empty:
        raise RuntimeError("Yahoo Finance returned no data for the panel")
    return panel


def synthetic_returns(rng, shape, reversal):
    """
    Log returns of a market factor plus idiosyncratic returns that reverse by
    ``reversal`` of the previous day's, both fat-tailed (Student t with 4
    degrees of freedom). Returns (returns, idiosyncratic returns, volatilities).
    """
    scale = rng.uniform(0.01, 0.02, shape[1])
    market = 0.01 * rng.standard_t(4, shape[0]) / np.sqrt(2)
    idiosyncratic = rng.standard_t(4, shape) / np.sqrt(2) * scale
    for row in range(1, shape[0]):
        idiosyncratic[row] -= reversal * idiosyncratic[row - 1]
    return market[:, None] + idiosyncratic, idiosyncratic, scale


def synthetic_panel(params, stocks=SYNTHETIC_STOCKS, reversal=REVERSAL, seed=SYNTHETIC_SEED):
    """
    Synthetic panel with a known cross-sectional reversal, on business days of
    the configured date range. ``Oracle`` is the expected excess return of each
    day given the day before.
    """
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(params["end"]) - pd.Timedelta(days=1)
    dates = pd.bdate_range(params["start"], end, name="Date")
    returns, idiosyncratic, scale = synthetic_returns(rng, (len(dates), stocks), reversal)
    excess = idiosyncratic - idiosyncratic.mean(axis=1, keepdims=True)
    close = 100 * np.exp(np.cumsum(returns, axis=0))
    spread = np.abs(returns) + np.abs(rng.normal(0, 0.5, returns.shape)) * scale
    values = {
        "Close": close,
        "High": close * np.exp(spread / 2),
        "Low": close * np.exp(-spread / 2),
        "Volume": np.exp(14 + 20 * np.abs(idiosyncratic) + rng.normal(0, 0.3, returns.shape)),
        "Oracle": np.vstack([np.full((1, stocks), np.nan), -reversal * excess[:-1]]),
    }
    tickers = [f"SYN{number:02d}" for number in range(1, stocks + 1)]
    return long_format(
        {name: pd.DataFrame(array, index=dates, columns=tickers) for name, array in values.items()}
    )


def write_panel(params):
    """Downloads or generates the panel of the configured ticker and writes it to its CSV file."""
    if params["ticker"] == SYNTHETIC_PANEL:
        panel = synthetic_panel(params)
    elif params["ticker"] in PANELS:
        panel = download_panel(params, PANELS[params["ticker"]])
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
    if params["ticker"] in PANELS:
        missing = sorted(set(PANELS[params["ticker"]]) - set(stocks.index))
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
