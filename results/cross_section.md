# Cross section of German stocks

Results of the cross-section study (`patterndecoder/cross_section.py`), computed with the workflow "Walk-forward evaluation" (tickers `DAX-PANEL`, `SDAX-PANEL` and `SYNTH-PANEL`) on 27 and 28 November 2026; the small caps of the SDAX follow in their own section. Instead of forecasting one index, the models rank the stocks of a panel by their next-day return relative to the others, where studies of single stocks have found short-term patterns such as reversal. Each trading day gives one observation per stock, so the evaluation can detect much weaker signals than on a single series. The design was fixed in `patterndecoder/cross_section.py` before any model was trained on the panel:

- **Universe:** the 40 members of the DAX on 20 March 2023, a date before the first test window, so that the test has no survivorship bias. Stocks enter the panel when their prices start (Daimler Truck in December 2021, Porsche AG in September 2022) and stay in it after leaving the DAX. Yahoo Finance has no data for Covestro, so the panel has 39 stocks, 38.4 per test day on average.
- **Data:** daily Xetra closing prices adjusted for dividends and splits, daily high and low and volume from Yahoo Finance, 5 February 2021 to 5 February 2026. Deutsche Post (DHL) has no reported volume on 64 days; windows that contain such a day are left out.
- **Target and inputs:** each stock's excess log return of the next day (its log return minus the mean of the panel that day), forecast from the 30 days before with four features per day: the stock's excess return, the panel's mean return, the stock's log volume relative to its mean of the 20 days before, and its daily range log(high / low). The stock's identity is not an input.
- **Models:** the 13 models with the settings of `config/config.yaml`, one model for all stocks. An epoch is a quarter pass over the panel, with at most 10 passes and early stopping after 2 passes without a lower validation MAE.
- **Benchmarks:** the zero forecast; a reversal rule, the LAD regression of the excess return on the previous day's excess return; Linear, the LAD regression on all inputs of the window.
- **Evaluation:** the five walk-forward test windows (August 2023 to February 2026) with ten seeds, forecasts averaged over the seeds. Main measure: the information coefficient (IC), the Spearman rank correlation between forecasts and excess returns across the stocks of a day, averaged over the test days and tested against zero, and against Linear and the reversal rule through the daily differences, with Newey-West t-tests. Also reported: the MAE relative to the zero forecast and a portfolio that each day buys the fifth of the stocks with the highest forecasts and sells the fifth with the lowest short, before and after costs of 5 basis points per unit of turnover.
- **PatternDecoder:** the PatternDecoder and PatternDecoderNoConvBlock, which led for trading volume with RevIN (`revin.md`), are compared with the other neural model with the highest mean validation IC.
- **Changes after the positive control, before the DAX run:** Linear was first a linear model trained like the neural models; in the positive control its training failed (37 % of the oracle's IC and a higher MAE than the zero forecast), so it became the LAD regression on all inputs. The DAX run uses ten seeds instead of four, because a training job took about 10 minutes instead of the expected 45. The first DAX download lost SAP to a download error, so that run was cancelled before any results and repeated with a corrected download.

## Positive control

A synthetic panel of 40 stocks with a common market factor and fat-tailed idiosyncratic returns that reverse by 10 % of the previous day's (`patterndecoder/panel.py`); the oracle forecasts -0.1 times the previous day's excess return. Two seeds, same test windows (653 test days, 40 stocks per day); the reversal rule finds the oracle's ranking exactly.

| Model | IC | share of the oracle's IC | vs Linear | p | val IC |
|---|---|---|---|---|---|
| Reversal | 0.0955 | 100 % | +0.0132 | < 0.001 | 0.1007 |
| Oracle | 0.0955 | 100 % | +0.0132 | < 0.001 | 0.1007 |
| PatternDecoderNoConvBlock (1) | 0.0928 | 97 % | +0.0105 | 0.003 | 0.0944 |
| Transformer (2) | 0.0867 | 91 % | +0.0044 | 0.294 | 0.0812 |
| PatternDecoderNoLstmBlock (3) | 0.0863 | 90 % | +0.0040 | 0.323 | 0.0848 |
| Informer (4) | 0.0839 | 88 % | +0.0016 | 0.718 | 0.0823 |
| Linear | 0.0823 | 86 % | - | - | 0.0853 |
| PatternDecoder_Conv (5) | 0.0820 | 86 % | -0.0003 | 0.938 | 0.0814 |
| Transformer_Conv (6) | 0.0799 | 84 % | -0.0024 | 0.604 | 0.0829 |
| PatternDecoder (7) | 0.0793 | 83 % | -0.0030 | 0.491 | 0.0831 |
| Transformer_Logsparse (8) | 0.0767 | 80 % | -0.0056 | 0.216 | 0.0791 |
| DecoderOnlyTransformer_Conv (9) | 0.0760 | 80 % | -0.0063 | 0.192 | 0.0748 |
| DecoderOnlyTransformer (10) | 0.0751 | 79 % | -0.0072 | 0.199 | 0.0645 |
| PatternDecoderBlockNoCrossAttention (11) | 0.0743 | 78 % | -0.0080 | 0.085 | 0.0815 |
| PatternDecoder_Autocorr (12) | 0.0632 | 66 % | -0.0192 | 0.003 | 0.0679 |
| Autoformer (13) | 0.0631 | 66 % | -0.0192 | 0.003 | 0.0627 |

All models find the signal: their ICs reach 66 % to 97 % of the oracle's. PatternDecoderNoConvBlock comes closest (97 %) and is significantly better than Linear (p = 0.003) and than the comparison model, Transformer_Conv (IC difference +0.013, p = 0.002); the PatternDecoder reaches 83 %. The pipeline and the models can therefore find a cross-sectional signal of this kind and size.

## DAX panel

Test windows: 2023-08-02 to 2024-01-31; 2024-02-01 to 2024-07-31; 2024-08-01 to 2025-02-03; 2025-02-04 to 2025-08-04; 2025-08-05 to 2026-02-05. 10 seeds, 638 test days, 38.4 stocks per day. IC: Spearman rank correlation between the seed-averaged forecasts and the excess returns of a day; p: Newey-West test. Rank among the 13 models in brackets; vs Linear and vs reversal: mean daily IC difference with its p-value; MAE vs zero: MAE relative to the zero forecast; long-short: annualized mean return of the portfolio long the top and short the bottom fifth of the stocks, rebalanced daily.

| Model | IC | p | days with IC > 0 | single run IC | vs Linear | p | vs reversal | p | MAE vs zero | long-short p.a. before costs | after costs | Sharpe after costs | val IC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Transformer_Logsparse (1) | 0.0031 | 0.697 | 49.1 % | 0.0015 | +0.0010 | 0.926 | +0.0179 | 0.102 | -0.01 % | +9.9 % | -29.2 % | -1.84 | 0.0080 |
| Linear | 0.0021 | 0.800 | 49.7 % | 0.0021 | +0.0000 | - | +0.0170 | 0.125 | +0.31 % | -5.2 % | -43.7 % | -2.67 | 0.0024 |
| PatternDecoder (2) | -0.0010 | 0.916 | 50.9 % | -0.0029 | -0.0031 | 0.770 | +0.0139 | 0.284 | +0.03 % | -7.5 % | -40.4 % | -2.57 | 0.0113 |
| Transformer (3) | -0.0036 | 0.702 | 50.3 % | -0.0029 | -0.0057 | 0.618 | +0.0113 | 0.386 | +0.02 % | -7.4 % | -40.1 % | -2.51 | 0.0100 |
| Informer (4) | -0.0060 | 0.467 | 45.9 % | -0.0029 | -0.0081 | 0.458 | +0.0089 | 0.501 | +0.04 % | -3.0 % | -35.6 % | -2.36 | 0.0084 |
| PatternDecoderNoConvBlock (5) | -0.0067 | 0.439 | 48.9 % | -0.0031 | -0.0088 | 0.438 | +0.0082 | 0.410 | +0.01 % | -7.9 % | -36.8 % | -2.29 | 0.0103 |
| Autoformer (6) | -0.0078 | 0.324 | 48.9 % | -0.0030 | -0.0099 | 0.368 | +0.0071 | 0.540 | +0.05 % | -12.0 % | -45.0 % | -2.87 | 0.0020 |
| PatternDecoder_Conv (7) | -0.0088 | 0.331 | 48.4 % | -0.0019 | -0.0109 | 0.293 | +0.0060 | 0.587 | +0.02 % | -2.9 % | -41.6 % | -2.54 | 0.0102 |
| DecoderOnlyTransformer_Conv (8) | -0.0091 | 0.200 | 47.3 % | -0.0033 | -0.0112 | 0.269 | +0.0058 | 0.634 | +0.01 % | -10.6 % | -39.8 % | -2.55 | 0.0136 |
| PatternDecoderBlockNoCrossAttention (9) | -0.0096 | 0.277 | 50.8 % | -0.0015 | -0.0117 | 0.266 | +0.0053 | 0.621 | +0.02 % | -19.6 % | -54.6 % | -3.23 | 0.0132 |
| PatternDecoder_Autocorr (10) | -0.0116 | 0.150 | 47.0 % | -0.0040 | -0.0137 | 0.216 | +0.0032 | 0.793 | +0.04 % | -16.1 % | -43.2 % | -2.73 | 0.0046 |
| PatternDecoderNoLstmBlock (11) | -0.0122 | 0.181 | 46.7 % | -0.0067 | -0.0143 | 0.172 | +0.0027 | 0.799 | +0.03 % | -6.9 % | -47.2 % | -2.94 | 0.0105 |
| DecoderOnlyTransformer (12) | -0.0135 | 0.123 | 47.5 % | -0.0027 | -0.0156 | 0.162 | +0.0014 | 0.916 | +0.01 % | -21.0 % | -51.2 % | -3.10 | 0.0111 |
| Reversal | -0.0149 | 0.125 | 45.8 % | -0.0149 | -0.0170 | 0.125 | +0.0000 | - | +0.01 % | -8.2 % | -47.3 % | -2.50 | -0.0031 |
| Transformer_Conv (13) | -0.0195 | 0.022 | 46.6 % | -0.0066 | -0.0216 | 0.051 | -0.0046 | 0.657 | +0.04 % | -18.7 % | -54.2 % | -3.40 | 0.0104 |
| Naive | - | - | - | - | - | - | - | - | +0.00 % | - | - | - | - |

IC per test window:

| Model | 2023-08-02 to 2024-01-31 | 2024-02-01 to 2024-07-31 | 2024-08-01 to 2025-02-03 | 2025-02-04 to 2025-08-04 | 2025-08-05 to 2026-02-05 |
|---|---|---|---|---|---|
| Transformer_Logsparse | -0.0156 | -0.0028 | 0.0156 | -0.0050 | 0.0230 |
| Linear | 0.0091 | -0.0304 | 0.0229 | 0.0133 | -0.0046 |
| PatternDecoder | 0.0087 | -0.0273 | 0.0115 | 0.0123 | -0.0101 |
| Transformer | 0.0049 | -0.0179 | 0.0009 | 0.0017 | -0.0077 |
| Informer | -0.0206 | 0.0098 | 0.0132 | -0.0314 | -0.0012 |
| PatternDecoderNoConvBlock | -0.0000 | -0.0151 | -0.0106 | 0.0139 | -0.0214 |
| Autoformer | 0.0043 | -0.0204 | 0.0208 | -0.0290 | -0.0148 |
| PatternDecoder_Conv | -0.0045 | -0.0269 | 0.0239 | -0.0044 | -0.0324 |
| DecoderOnlyTransformer_Conv | 0.0065 | -0.0181 | -0.0134 | -0.0109 | -0.0097 |
| PatternDecoderBlockNoCrossAttention | 0.0072 | -0.0199 | 0.0180 | -0.0084 | -0.0449 |
| PatternDecoder_Autocorr | 0.0012 | -0.0202 | -0.0103 | -0.0230 | -0.0060 |
| PatternDecoderNoLstmBlock | -0.0284 | -0.0243 | 0.0085 | 0.0152 | -0.0320 |
| DecoderOnlyTransformer | 0.0309 | -0.0289 | -0.0228 | -0.0267 | -0.0202 |
| Reversal | 0.0011 | -0.0170 | -0.0215 | 0.0141 | -0.0509 |
| Transformer_Conv | -0.0311 | -0.0376 | -0.0038 | -0.0145 | -0.0104 |
| Naive | - | - | - | - | - |

Comparison model (highest mean validation IC of the other models): DecoderOnlyTransformer_Conv
- PatternDecoder minus DecoderOnlyTransformer_Conv: IC difference +0.0081, p = 0.381
- PatternDecoderNoConvBlock minus DecoderOnlyTransformer_Conv: IC difference +0.0024, p = 0.792

## Findings for the DAX

1. **No cross-sectional signal is detectable.** The mean ICs of the 13 models lie between -0.019 and +0.003, and none is significantly positive; Transformer_Conv is significantly negative (p = 0.022), about what chance produces in 15 tests. Linear (+0.002) and the reversal rule (-0.015) find nothing either, the share of days with a positive IC is 46 % to 51 %, and no model is positive in all five windows.
2. **The test would have found a modest signal.** The standard error of a mean IC is about 0.009, so a true IC of about 0.017 would be significant and one of 0.024 would be found with a probability of about 80 %. The positive control shows that the models learn an IC of 0.1 almost completely. For the large DAX stocks in 2023 to 2026, the four inputs therefore hold no signal of that size about the next day's relative return.
3. **The forecasts do not beat the zero forecast.** Their MAE is between 0.005 % lower and 0.05 % higher than that of the zero forecast, significantly higher for six of the 15 forecasts.
4. **The portfolios lose money.** Before costs, the daily long-short portfolios earn between -21 % and +10 % a year; they turn over 2.1 to 3.2 times their capital a day, so costs of 5 basis points take 27 to 40 percentage points a year, and all lose after costs (-55 % to -29 % a year, Sharpe ratios -3.4 to -1.8).
5. **The PatternDecoder does not stand out.** It has the second highest IC of the 13 models (-0.001), but neither it nor PatternDecoderNoConvBlock differs significantly from the comparison model, DecoderOnlyTransformer_Conv (IC differences +0.008, p = 0.38, and +0.002, p = 0.79). PatternDecoderNoConvBlock, the best learner in the positive control, ranks 5th.

The result matches a study of the S&P 500 stocks, where daily cross-sectional forecasts of LSTM networks were clearly profitable from 1992 to 2009, while from 2010 their profits after transaction costs fluctuated around zero (Fischer and Krauss, 2018, European Journal of Operational Research 270(2), 654-669).

## SDAX panel (small caps)

Designed after the DAX run and before any run on these stocks (recorded in `patterndecoder/cross_section.py`): the same study for the 71 companies of the SDAX table of the English Wikipedia, stated as of the index review of 10 September 2020, a universe fixed long before the test windows. A clean MDAX list from before the test windows was not available: the Wikipedia table, stated as of January 2023, contains later changes. Yahoo Finance has no prices for eight of the SDAX companies (Encavis, Hornbach Baumarkt, Klöckner & Co, Leoni, New Work, Steinhoff, Tele Columbus, zooplus), most of which have since been taken over or delisted, so the panel has 63 stocks, 60.4 per test day. This survivorship can favour reversal, because stocks that fell and recovered stay in the panel while some that kept falling are missing. The data are used as they are, including the extreme moves of Corestate Capital during its restructuring (-88 % and +72 % in November 2022).

Test windows: 2023-08-02 to 2024-01-31; 2024-02-01 to 2024-07-31; 2024-08-01 to 2025-02-03; 2025-02-04 to 2025-08-04; 2025-08-05 to 2026-02-05. 10 seeds, 638 test days, 60.4 stocks per day. Columns as for the DAX; break-even cost: the cost per unit of turnover at which the portfolio's gross return would be used up.

| Model | IC | p | days with IC > 0 | single run IC | vs Linear | p | vs reversal | p | MAE vs zero | long-short p.a. before costs | after costs | Sharpe after costs | break-even cost | val IC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PatternDecoderNoConvBlock (1) | 0.0398 | < 0.001 | 60.3 % | 0.0280 | +0.0122 | 0.026 | +0.0060 | 0.018 | -0.09 % | +34.3 % | -4.6 % | -0.28 | 4.4 bp | 0.0270 |
| PatternDecoder (2) | 0.0346 | < 0.001 | 58.2 % | 0.0156 | +0.0070 | 0.232 | +0.0008 | 0.868 | -0.04 % | +36.2 % | +0.2 % | +0.01 | 5.0 bp | 0.0150 |
| Reversal | 0.0338 | < 0.001 | 56.9 % | 0.0338 | +0.0062 | 0.324 | +0.0000 | - | -0.04 % | +29.6 % | -9.4 % | -0.52 | 3.8 bp | 0.0291 |
| PatternDecoder_Conv (3) | 0.0328 | < 0.001 | 59.1 % | 0.0174 | +0.0052 | 0.328 | -0.0009 | 0.837 | -0.05 % | +29.6 % | -9.4 % | -0.55 | 3.8 bp | 0.0200 |
| PatternDecoderNoLstmBlock (4) | 0.0324 | < 0.001 | 57.2 % | 0.0213 | +0.0048 | 0.351 | -0.0013 | 0.683 | -0.05 % | +29.7 % | -11.7 % | -0.70 | 3.6 bp | 0.0187 |
| PatternDecoder_Autocorr (5) | 0.0324 | < 0.001 | 60.7 % | 0.0214 | +0.0048 | 0.382 | -0.0014 | 0.812 | -0.04 % | +27.0 % | -7.1 % | -0.44 | 4.0 bp | 0.0160 |
| Informer (6) | 0.0307 | < 0.001 | 56.4 % | 0.0140 | +0.0031 | 0.610 | -0.0031 | 0.580 | -0.04 % | +19.4 % | -16.5 % | -1.01 | 2.7 bp | 0.0152 |
| Transformer_Conv (7) | 0.0298 | < 0.001 | 56.1 % | 0.0138 | +0.0022 | 0.712 | -0.0039 | 0.471 | -0.04 % | +24.7 % | -15.2 % | -0.99 | 3.1 bp | 0.0182 |
| Transformer_Logsparse (8) | 0.0296 | < 0.001 | 57.8 % | 0.0147 | +0.0020 | 0.714 | -0.0042 | 0.401 | -0.04 % | +25.9 % | -15.8 % | -1.03 | 3.1 bp | 0.0171 |
| Transformer (9) | 0.0282 | < 0.001 | 56.3 % | 0.0133 | +0.0006 | 0.920 | -0.0056 | 0.207 | -0.03 % | +26.5 % | -12.1 % | -0.73 | 3.4 bp | 0.0178 |
| Linear | 0.0276 | < 0.001 | 56.4 % | 0.0276 | +0.0000 | - | -0.0062 | 0.324 | +0.14 % | +18.6 % | -17.7 % | -1.05 | 2.6 bp | 0.0083 |
| Autoformer (10) | 0.0246 | < 0.001 | 58.8 % | 0.0148 | -0.0030 | 0.583 | -0.0092 | 0.136 | -0.03 % | +19.9 % | -16.9 % | -1.06 | 2.7 bp | 0.0162 |
| PatternDecoderBlockNoCrossAttention (11) | 0.0225 | < 0.001 | 53.8 % | 0.0117 | -0.0051 | 0.367 | -0.0113 | 0.027 | -0.02 % | +11.2 % | -24.9 % | -1.47 | 1.6 bp | 0.0151 |
| DecoderOnlyTransformer (12) | 0.0222 | < 0.001 | 55.3 % | 0.0076 | -0.0054 | 0.446 | -0.0116 | 0.088 | -0.02 % | +19.2 % | -12.6 % | -0.80 | 3.0 bp | 0.0146 |
| DecoderOnlyTransformer_Conv (13) | 0.0221 | < 0.001 | 56.4 % | 0.0096 | -0.0054 | 0.411 | -0.0116 | 0.093 | -0.02 % | +15.6 % | -13.2 % | -0.81 | 2.7 bp | 0.0139 |
| Naive | - | - | - | - | - | - | - | - | +0.00 % | - | - | - | - | - |

IC per test window:

| Model | 2023-08-02 to 2024-01-31 | 2024-02-01 to 2024-07-31 | 2024-08-01 to 2025-02-03 | 2025-02-04 to 2025-08-04 | 2025-08-05 to 2026-02-05 |
|---|---|---|---|---|---|
| PatternDecoderNoConvBlock | 0.0382 | 0.0281 | 0.0353 | 0.0363 | 0.0608 |
| PatternDecoder | 0.0334 | 0.0251 | 0.0219 | 0.0319 | 0.0606 |
| Reversal | 0.0244 | 0.0234 | 0.0338 | 0.0384 | 0.0487 |
| PatternDecoder_Conv | 0.0325 | 0.0213 | 0.0344 | 0.0243 | 0.0516 |
| PatternDecoderNoLstmBlock | 0.0255 | 0.0244 | 0.0276 | 0.0324 | 0.0523 |
| PatternDecoder_Autocorr | 0.0290 | 0.0178 | 0.0322 | 0.0429 | 0.0399 |
| Informer | 0.0423 | 0.0216 | 0.0164 | 0.0347 | 0.0382 |
| Transformer_Conv | 0.0124 | 0.0306 | 0.0232 | 0.0292 | 0.0537 |
| Transformer_Logsparse | 0.0208 | 0.0218 | 0.0253 | 0.0215 | 0.0585 |
| Transformer | 0.0221 | 0.0198 | 0.0337 | 0.0124 | 0.0528 |
| Linear | 0.0304 | 0.0068 | 0.0283 | 0.0162 | 0.0560 |
| Autoformer | 0.0265 | 0.0205 | 0.0327 | 0.0169 | 0.0262 |
| PatternDecoderBlockNoCrossAttention | 0.0206 | 0.0189 | 0.0107 | 0.0306 | 0.0315 |
| DecoderOnlyTransformer | 0.0291 | 0.0180 | 0.0263 | 0.0182 | 0.0193 |
| DecoderOnlyTransformer_Conv | 0.0344 | 0.0185 | 0.0221 | 0.0298 | 0.0060 |
| Naive | - | - | - | - | - |

Comparison model (highest mean validation IC of the other models): Transformer_Conv
- PatternDecoder minus Transformer_Conv: IC difference +0.0048, p = 0.377
- PatternDecoderNoConvBlock minus Transformer_Conv: IC difference +0.0099, p = 0.043

## Findings for the small caps

1. **Here the models find a signal.** Every model, Linear and the reversal rule have a significantly positive IC (0.022 to 0.040 for the models, p < 0.001 throughout), positive in each of the five test windows; the IC is positive on 54 % to 61 % of the days, and the models' MAE is below that of the zero forecast, significantly for 9 of the 13. The reversal rule alone reaches 0.034: stocks that did worse than the others on one day tend to do better on the next.
2. **The PatternDecoder variants lead.** The five best models are PatternDecoder variants. PatternDecoderNoConvBlock has the highest IC (0.040) and ranks first or second in every window; it is significantly better than Linear (p = 0.026), the reversal rule (p = 0.018) and the comparison model chosen on the validation set, Transformer_Conv (+0.010, p = 0.043, better in four of the five windows). The PatternDecoder itself is second (0.035), but does not differ significantly from the comparison model (p = 0.38). The design specified two comparisons, so with a Bonferroni correction the p-value of PatternDecoderNoConvBlock is 0.086: the hypothesis from the volume study is confirmed in direction, but not strongly. The variant without cross-attention ranks only 11th, which suggests, as a post-hoc reading, that the cross-attention to the LSTM states carries the advantage.
3. **The seed average matters.** Single runs reach much lower ICs (0.008 to 0.028) than the forecasts averaged over ten seeds; for the PatternDecoder 0.016 instead of 0.035.
4. **The signal does not pay for daily trading.** Before costs, the long-short portfolios earn 11 % to 36 % a year, but they turn over 2.3 to 3.3 times their capital a day, so they break even at costs of 1.6 to 5.0 basis points per unit of turnover, below the bid-ask spreads typical of small caps. After 5 basis points only the PatternDecoder's portfolio is just positive (+0.2 % a year).
5. **Small and large caps differ.** In the DAX panel no forecast had a significantly positive IC; in the SDAX panel all have. That fits studies that find short-term reversal mainly in less liquid stocks (Avramov, Chordia and Goyal, 2006, Journal of Finance 61(5)); the survivorship of this panel may add to it, but it affects the models and the benchmarks alike.
