# Weekly horizon: five-day returns of the DAX

Results of `patterndecoder/weekly.py`, computed with the manually started workflow "Walk-forward evaluation" (overrides `{"start": "1990-01-01", "target_days": 5}` and `{"target_days": 5}`) on 28 November 2026. The question is whether the return of the next week is more predictable than the return of the next day. The design was committed in the docstring of `patterndecoder/weekly.py` before the first run:

- **Target:** the log return of the next five trading days (the sum of their daily log returns), forecast after the close of every trading day. Inputs are the daily log returns of the last 30, 60 or 120 days.
- **Data:** the DAX from 2 January 1990 to 5 February 2026 (main run) and, for comparison, the period of the paper (5 February 2021 to 5 February 2026).
- **Walk-forward:** five test windows covering the second half of the data; training data are all earlier days, their last 20 % the validation data for early stopping. Targets are summed within the training, validation and test data, so no target reaches into a later part; the last four days of each test window get no forecast.
- **Window length:** every model is trained with 30, 60 and 120 input days; per test window and model, the length with the lowest validation MAE (mean over the seeds) is evaluated. The test data never enter that choice.
- **Models and seeds:** the 13 models with the settings of `config/config.yaml`, 10 seeds each; the forecasts are averaged over the seeds.
- **Benchmarks:** zero; the median five-day return of the training days (median); a median (LAD) regression on the return of the previous five days (last week), which captures momentum or reversal, whichever the training days show.
- **Criterion:** a model counts as better than a benchmark only if its MAE is lower and the Diebold-Mariano test gives p < 0.05 both on all days (overlapping weeks, at least four Newey-West lags) and on every fifth day (non-overlapping weeks). p < 0.05 / 13 (Bonferroni) is reported with a star.
- **Expected beforehand:** little or nothing. With about 900 non-overlapping test weeks, a forecast has to explain roughly 3 % of the variance of the five-day return to be found with 80 % power, on 2021 to 2026 (about 125 weeks) roughly 20 %.

Columns: **input days** are the window lengths chosen for the five test windows; **MAE** is the test MAE of the seed ensemble, pooled over the windows; **vs zero, vs median, vs last week** give the difference in MAE from each benchmark; **p** is the Diebold-Mariano p-value on all days / on every fifth day; **single run vs zero** is the mean MAE of the seeds' own forecasts against the zero forecast; **val MAE** is the mean validation MAE of the chosen windows.

## Data

Daily closes of the DAX (`^GDAXI`) from Yahoo Finance: 9,132 trading days from 2 January 1990 to 5 February 2026, with no missing prices, no zero returns and 248 to 257 trading days in every full year. The largest daily moves are those of known crashes and rebounds (12 March 2020 -13.1 %, 13 and 28 October 2008 +10.8 % and +10.7 %, 24 March 2020 +10.4 %). The comparison run uses the 1,275 days of the paper (5 February 2021 to 5 February 2026, the data checksum of `config/config.yaml`).

## DAX 1990 to 2026 (main run)


Test windows: 2008-02-07 to 2011-08-31; 2011-09-07 to 2015-04-14; 2015-04-21 to 2018-11-16; 2018-11-23 to 2022-06-29; 2022-07-06 to 2026-01-30. 10 seeds, 4546 forecasts (910 non-overlapping weeks), averaged over the seeds. Input days per test window chosen on the validation MAE. p: Diebold-Mariano test on all days (at least 4 lags) / on every 5th day. Better: lower MAE and p < 0.05 in both tests; * also p < 0.0038 (Bonferroni over 13 models).

| Model | input days | MAE | vs zero | p | vs median | p | vs last week | p | single run vs zero | better than | val MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LastWeek | - | 0.020619 | -1.16 % | 0.019 / 0.266 | -0.06 % | 0.721 / 0.513 | +0.00 % | - / - | -1.16 % | - | 0.020171 |
| Transformer_Logsparse | 60 60 30 120 30 | 0.020625 | -1.13 % | 0.017 / 0.122 | -0.02 % | 0.910 / 0.872 | +0.03 % | 0.904 / 0.605 | -0.85 % | - | 0.020181 |
| PatternDecoderBlockNoCrossAttention | 60 60 120 120 60 | 0.020627 | -1.12 % | 0.020 / 0.171 | -0.02 % | 0.944 / 0.856 | +0.04 % | 0.878 / 0.785 | -0.71 % | - | 0.020185 |
| Median | - | 0.020630 | -1.10 % | 0.016 / 0.131 | +0.00 % | - / - | +0.06 % | 0.721 / 0.513 | -1.10 % | - | 0.020214 |
| PatternDecoder | 60 60 30 120 120 | 0.020635 | -1.08 % | 0.023 / 0.121 | +0.02 % | 0.923 / 0.914 | +0.08 % | 0.782 / 0.659 | -0.82 % | - | 0.020170 |
| Autoformer | 120 60 60 120 120 | 0.020636 | -1.07 % | 0.037 / 0.225 | +0.03 % | 0.918 / 0.800 | +0.09 % | 0.760 / 0.886 | -0.82 % | - | 0.020137 |
| Transformer_Conv | 60 60 60 120 60 | 0.020636 | -1.07 % | 0.027 / 0.185 | +0.03 % | 0.895 / 0.852 | +0.09 % | 0.750 / 0.815 | -0.76 % | - | 0.020189 |
| DecoderOnlyTransformer | 60 60 60 120 120 | 0.020637 | -1.07 % | 0.024 / 0.089 | +0.03 % | 0.877 / 0.765 | +0.09 % | 0.739 / 0.551 | -0.78 % | - | 0.020164 |
| Transformer | 60 60 30 120 60 | 0.020637 | -1.07 % | 0.022 / 0.074 | +0.03 % | 0.884 / 0.583 | +0.09 % | 0.749 / 0.436 | -0.80 % | - | 0.020182 |
| PatternDecoderNoLstmBlock | 60 60 30 120 120 | 0.020637 | -1.07 % | 0.022 / 0.160 | +0.03 % | 0.878 / 0.851 | +0.09 % | 0.739 / 0.812 | -0.84 % | - | 0.020171 |
| PatternDecoder_Conv | 60 30 120 120 30 | 0.020637 | -1.07 % | 0.022 / 0.185 | +0.03 % | 0.886 / 0.739 | +0.09 % | 0.743 / 0.902 | -0.69 % | - | 0.020168 |
| DecoderOnlyTransformer_Conv | 60 60 120 120 30 | 0.020647 | -1.02 % | 0.036 / 0.194 | +0.09 % | 0.703 / 0.768 | +0.14 % | 0.600 / 0.879 | -0.67 % | - | 0.020192 |
| PatternDecoderNoConvBlock | 60 30 60 60 30 | 0.020651 | -1.00 % | 0.039 / 0.201 | +0.10 % | 0.645 / 0.746 | +0.15 % | 0.573 / 0.886 | -0.73 % | - | 0.020177 |
| PatternDecoder_Autocorr | 120 60 60 120 60 | 0.020652 | -1.00 % | 0.063 / 0.344 | +0.11 % | 0.729 / 0.564 | +0.16 % | 0.586 / 0.835 | -0.66 % | - | 0.020113 |
| Informer | 120 60 120 120 60 | 0.020653 | -0.99 % | 0.041 / 0.226 | +0.11 % | 0.591 / 0.679 | +0.17 % | 0.516 / 0.892 | -0.57 % | - | 0.020192 |
| Naive | - | 0.020860 | +0.00 % | - / - | +1.11 % | 0.016 / 0.131 | +1.17 % | 0.019 / 0.266 | +0.00 % | - | 0.020486 |

Per test window, the models follow the constant median: where the median beats the zero forecast clearly (2011 to 2015: -2.4 %; 2022 to 2026: -2.4 %), so do the models; where it does not (2015 to 2018: +0.3 %), no model does either. The number of models below the median is 12, 0, 0, 11 and 13 in the five windows. The PatternDecoder ranks 3, 6, 10, 8 and 5 among the 13 models.

## DAX 2021 to 2026 (period of the paper)

Test windows: 2023-08-02 to 2024-01-25; 2024-02-01 to 2024-07-25; 2024-08-01 to 2025-01-28; 2025-02-04 to 2025-07-29; 2025-08-05 to 2026-01-30. 10 seeds, 618 forecasts (125 non-overlapping weeks), averaged over the seeds. Input days per test window chosen on the validation MAE. p: Diebold-Mariano test on all days (at least 4 lags) / on every 5th day. Better: lower MAE and p < 0.05 in both tests; * also p < 0.0038 (Bonferroni over 13 models).

| Model | input days | MAE | vs zero | p | vs median | p | vs last week | p | single run vs zero | better than | val MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Transformer_Conv | 30 30 30 30 30 | 0.014908 | -2.58 % | 0.161 / 0.634 | -0.64 % | 0.602 / 0.321 | -0.64 % | 0.609 / 0.484 | -1.60 % | - | 0.013783 |
| Informer | 30 30 30 30 30 | 0.014909 | -2.56 % | 0.129 / 0.785 | -0.62 % | 0.543 / 0.472 | -0.63 % | 0.563 / 0.710 | -1.36 % | - | 0.013818 |
| PatternDecoderBlockNoCrossAttention | 30 30 60 60 60 | 0.014973 | -2.15 % | 0.226 / 0.799 | -0.20 % | 0.844 / 0.446 | -0.21 % | 0.843 / 0.705 | -1.15 % | - | 0.013942 |
| Transformer | 30 30 30 30 30 | 0.014988 | -2.06 % | 0.248 / 0.903 | -0.10 % | 0.920 / 0.656 | -0.11 % | 0.918 / 0.905 | -1.15 % | - | 0.013859 |
| PatternDecoder | 30 30 30 30 30 | 0.014998 | -1.99 % | 0.259 / 0.676 | -0.04 % | 0.973 / 0.382 | -0.04 % | 0.970 / 0.560 | -1.39 % | - | 0.013869 |
| DecoderOnlyTransformer | 30 30 30 30 30 | 0.015001 | -1.97 % | 0.259 / 0.705 | -0.02 % | 0.989 / 0.400 | -0.02 % | 0.985 / 0.588 | -1.29 % | - | 0.013848 |
| Median | - | 0.015003 | -1.95 % | 0.232 / 0.877 | +0.00 % | - / - | -0.01 % | 0.986 / 0.200 | -1.95 % | - | 0.013955 |
| LastWeek | - | 0.015004 | -1.95 % | 0.252 / 0.943 | +0.01 % | 0.986 / 0.200 | +0.00 % | - / - | -1.95 % | - | 0.013997 |
| PatternDecoder_Conv | 30 60 30 30 30 | 0.015008 | -1.92 % | 0.279 / 0.703 | +0.03 % | 0.978 / 0.403 | +0.03 % | 0.982 / 0.595 | -1.05 % | - | 0.013895 |
| PatternDecoderNoLstmBlock | 30 30 30 30 30 | 0.015017 | -1.86 % | 0.286 / 0.711 | +0.09 % | 0.938 / 0.402 | +0.08 % | 0.943 / 0.590 | -1.32 % | - | 0.013896 |
| PatternDecoderNoConvBlock | 30 30 30 30 30 | 0.015020 | -1.84 % | 0.302 / 0.952 | +0.11 % | 0.920 / 0.751 | +0.11 % | 0.927 / 0.997 | -1.52 % | - | 0.013873 |
| DecoderOnlyTransformer_Conv | 30 60 30 30 30 | 0.015021 | -1.83 % | 0.293 / 0.763 | +0.12 % | 0.919 / 0.481 | +0.11 % | 0.925 / 0.691 | -0.97 % | - | 0.013919 |
| Transformer_Logsparse | 30 120 30 30 30 | 0.015057 | -1.60 % | 0.370 / 0.909 | +0.36 % | 0.745 / 0.676 | +0.35 % | 0.754 / 0.918 | -0.94 % | - | 0.013879 |
| PatternDecoder_Autocorr | 30 30 60 120 30 | 0.015179 | -0.80 % | 0.664 / 0.877 | +1.17 % | 0.393 / 0.952 | +1.17 % | 0.405 / 0.755 | -0.18 % | - | 0.013916 |
| Autoformer | 30 30 30 60 30 | 0.015215 | -0.57 % | 0.743 / 0.688 | +1.41 % | 0.263 / 0.672 | +1.40 % | 0.281 / 0.501 | -0.06 % | - | 0.013928 |
| Naive | - | 0.015302 | +0.00 % | - / - | +1.99 % | 0.232 / 0.877 | +1.98 % | 0.252 / 0.943 | +0.00 % | - | 0.014357 |

The number of models below the median is 13, 0, 0, 11 and 13 in the five windows; the PatternDecoder ranks 4, 7, 7, 5 and 5.

## Findings

- **No model is better than any benchmark by the pre-registered criterion, in either period.** The five-day horizon does not reveal a pattern that the daily horizon missed.
- **1990 to 2026:** all 13 models are about 1 % better than the zero forecast in MAE (-0.99 % to -1.13 %). On all days this is significant for 12 of them (p = 0.017 to 0.063), on non-overlapping weeks for none (p = 0.074 to 0.344), so the criterion is not met. The constant median does the same (-1.10 %, p = 0.016 / 0.131): the advantage over zero is the upward drift of the DAX, not a pattern. Against the median, the models lie between -0.02 % and +0.11 % (all p >= 0.56), against the last-week regression between +0.03 % and +0.17 % (all p >= 0.43). The last-week regression has the lowest MAE of all, but only 0.06 % below the median (p = 0.72 / 0.51), so the previous week's return carries no usable information about the next one.
- **2021 to 2026:** the models are 0.57 % to 2.58 % better than the zero forecast and the median 1.95 %, none significantly (all p > 0.12); against the median they lie between -0.64 % and +1.41 %. With about 125 non-overlapping weeks, this run could only have found a very large effect.
- **PatternDecoder:** 3rd of the 13 models from 1990 (-1.08 % against zero, +0.02 % against the median, p = 0.92 / 0.91) and 5th from 2021 (-1.99 % and -0.04 %); the PatternDecoder variants are spread over the whole ranking in both runs.
- **Window length:** on the validation data, the models choose mostly 60 or 120 days when trained on the long history (55 of 65 choices) and mostly 30 days on the short one (56 of 65). Longer inputs do not help on the test data either.
- As expected beforehand, five-day index returns are not predictable enough for these models to show it: with 910 non-overlapping test weeks, a forecast explaining roughly 3 % of their variance would have been found with 80 % power.

## Reproduce

```bash
gh workflow run walk-forward.yml -f ticker='^GDAXI' -f overrides='{"start": "1990-01-01", "target_days": 5}'
gh workflow run walk-forward.yml -f ticker='^GDAXI' -f overrides='{"target_days": 5}'
# or locally, per seed and test window:
python -m patterndecoder.weekly run --seed 1 --test-start 0.5 --out wk/seed1_0.5.csv
python -m patterndecoder.weekly summarize wk/*.csv --out weekly_summary.csv
```

`weekly.csv` holds the pooled results of both runs (column `period`), `weekly_windows.csv` the results per test window with the chosen window length.
