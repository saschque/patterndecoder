# Trading volume

Results of the third use case (`patterndecoder/volume.py`), computed with the workflow "Walk-forward evaluation" on 27 November 2026. Unlike returns, trading volume has recurring calendar patterns: it differs by weekday, it rises when options and futures expire on the third Friday of a month, most of all in March, June, September and December, and it is persistent. The question is whether the models, and the PatternDecoder in particular, learn such patterns better than linear benchmarks that are given them. The design was fixed in `patterndecoder/volume.py` before any model was trained on it:

- **Target:** the next day's log trading volume of the DAX and the S&P 500 as reported by Yahoo Finance, from 5 February 2021 to 5 February 2026. Days without reported volume (4 for the DAX, 1 for the S&P 500) take the previous day's value. The log volume is centred by its median over the first half of the sample, which is training data in every test window.
- **Inputs:** the past 30 values of the target and the calendar features of those days. The models are trained with the MAE loss and the current settings of `config/config.yaml` (last-step readout), except input_scale 1, because the centred log volume is already of order one.
- **Benchmarks:** persistence (the previous day's value), an EWMA with decay 0.94, the HAR model (the previous day's value and its 5- and 22-day means), HAR-X, which adds dummies for the target day's weekday, for the third Friday of a month and for the third Friday of March, June, September and December, and the training median. Both HAR models are fitted by least absolute deviations on the training data. HAR-X gets the calendar of the target day directly, as a forecaster would; the neural models see the calendar features of the past days only and have to infer those of the target day.
- **Evaluation:** the walk-forward evaluation of `patterndecoder/robustness.py` with five test windows (August 2023 to February 2026) and ten seeds; MAE in log volume relative to HAR-X and HAR, with Diebold-Mariano tests of the seed-averaged forecasts.

## Results

### DAX (`^GDAXI`)

Test windows: 2023-08-02 to 2024-01-31; 2024-02-01 to 2024-07-31; 2024-08-01 to 2025-02-03; 2025-02-04 to 2025-08-04; 2025-08-05 to 2026-02-05. 10 seeds, 638 test days. Forecasts averaged over the seeds; single run: mean MAE of the seeds' own forecasts. p: Diebold-Mariano test of the pooled ensemble forecasts.

| Model | MAE | vs HAR-X | p | vs HAR | p | single run vs HAR-X | windows better than HAR-X | windows better than HAR | rank per window | val MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| HARX | 0.172565 | +0.00 % | - | -12.70 % | 0.000 | +0.00 % | 0/5 | 5/5 | - | 0.174228 |
| HAR | 0.197659 | +14.54 % | 0.000 | +0.00 % | - | +14.54 % | 0/5 | 0/5 | - | 0.199741 |
| DecoderOnlyTransformer_Conv | 0.203608 | +17.99 % | 0.000 | +3.01 % | 0.349 | +19.74 % | 0/5 | 1/5 | 2, 5, 2, 7, 2 | 0.197866 |
| PatternDecoder_Conv | 0.203765 | +18.08 % | 0.000 | +3.09 % | 0.337 | +20.10 % | 0/5 | 1/5 | 1, 4, 1, 4, 3 | 0.197196 |
| PatternDecoderNoConvBlock | 0.204896 | +18.74 % | 0.000 | +3.66 % | 0.259 | +20.28 % | 0/5 | 1/5 | 12, 2, 3, 1, 6 | 0.198301 |
| PatternDecoderNoLstmBlock | 0.204910 | +18.74 % | 0.000 | +3.67 % | 0.267 | +20.23 % | 0/5 | 1/5 | 6, 9, 5, 6, 1 | 0.198550 |
| PatternDecoder | 0.206479 | +19.65 % | 0.000 | +4.46 % | 0.180 | +21.23 % | 0/5 | 1/5 | 5, 10, 8, 9, 4 | 0.198751 |
| DecoderOnlyTransformer | 0.207062 | +19.99 % | 0.000 | +4.76 % | 0.156 | +21.55 % | 0/5 | 1/5 | 3, 6, 10, 10, 5 | 0.199377 |
| Transformer_Logsparse | 0.208120 | +20.60 % | 0.000 | +5.29 % | 0.127 | +22.28 % | 0/5 | 1/5 | 11, 8, 7, 5, 7 | 0.200359 |
| Transformer | 0.208687 | +20.93 % | 0.000 | +5.58 % | 0.108 | +22.75 % | 0/5 | 1/5 | 4, 11, 12, 8, 8 | 0.199885 |
| Transformer_Conv | 0.209275 | +21.27 % | 0.000 | +5.88 % | 0.086 | +23.00 % | 0/5 | 1/5 | 7, 3, 11, 3, 9 | 0.198683 |
| PatternDecoderBlockNoCrossAttention | 0.209421 | +21.36 % | 0.000 | +5.95 % | 0.090 | +23.15 % | 0/5 | 1/5 | 13, 13, 4, 2, 10 | 0.200132 |
| EWMA | 0.210069 | +21.73 % | 0.000 | +6.28 % | 0.025 | +21.73 % | 0/5 | 1/5 | - | 0.210312 |
| Autoformer | 0.211222 | +22.40 % | 0.000 | +6.86 % | 0.042 | +23.16 % | 0/5 | 1/5 | 9, 1, 6, 13, 12 | 0.199159 |
| PatternDecoder_Autocorr | 0.211542 | +22.59 % | 0.000 | +7.02 % | 0.047 | +23.69 % | 0/5 | 1/5 | 8, 7, 9, 12, 11 | 0.198195 |
| Informer | 0.216910 | +25.70 % | 0.000 | +9.74 % | 0.010 | +27.24 % | 0/5 | 1/5 | 10, 12, 13, 11, 13 | 0.202497 |
| Persistence | 0.228465 | +32.39 % | 0.000 | +15.59 % | 0.000 | +32.39 % | 0/5 | 0/5 | - | 0.238008 |
| Median | 0.244786 | +41.85 % | 0.000 | +23.84 % | 0.000 | +41.85 % | 0/5 | 0/5 | - | 0.220451 |

### S&P 500 (`^GSPC`)

Test windows: 2023-08-07 to 2024-02-02; 2024-02-05 to 2024-08-05; 2024-08-06 to 2025-02-04; 2025-02-05 to 2025-08-06; 2025-08-07 to 2026-02-05. 10 seeds, 628 test days. Forecasts averaged over the seeds; single run: mean MAE of the seeds' own forecasts. p: Diebold-Mariano test of the pooled ensemble forecasts.

| Model | MAE | vs HAR-X | p | vs HAR | p | single run vs HAR-X | windows better than HAR-X | windows better than HAR | rank per window | val MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| HARX | 0.094841 | +0.00 % | - | -8.22 % | 0.003 | +0.00 % | 0/5 | 5/5 | - | 0.091647 |
| HAR | 0.103333 | +8.95 % | 0.003 | +0.00 % | - | +8.95 % | 0/5 | 0/5 | - | 0.101418 |
| Persistence | 0.112393 | +18.51 % | 0.000 | +8.77 % | 0.002 | +18.51 % | 0/5 | 1/5 | - | 0.114973 |
| EWMA | 0.115163 | +21.43 % | 0.000 | +11.45 % | 0.002 | +21.43 % | 0/5 | 0/5 | - | 0.113243 |
| PatternDecoder_Conv | 0.143836 | +51.66 % | 0.000 | +39.20 % | 0.000 | +54.44 % | 0/5 | 0/5 | 13, 1, 4, 1, 1 | 0.123984 |
| Transformer_Conv | 0.147529 | +55.55 % | 0.000 | +42.77 % | 0.000 | +57.95 % | 0/5 | 0/5 | 2, 2, 6, 2, 3 | 0.126244 |
| DecoderOnlyTransformer_Conv | 0.148278 | +56.34 % | 0.000 | +43.49 % | 0.000 | +59.42 % | 0/5 | 0/5 | 12, 3, 5, 3, 2 | 0.125935 |
| Transformer | 0.160614 | +69.35 % | 0.000 | +55.43 % | 0.000 | +70.92 % | 0/5 | 0/5 | 5, 4, 13, 4, 4 | 0.131629 |
| PatternDecoderBlockNoCrossAttention | 0.162323 | +71.15 % | 0.000 | +57.09 % | 0.000 | +73.20 % | 0/5 | 0/5 | 11, 5, 3, 6, 6 | 0.132774 |
| Transformer_Logsparse | 0.163473 | +72.36 % | 0.000 | +58.20 % | 0.000 | +73.40 % | 0/5 | 0/5 | 4, 6, 12, 5, 5 | 0.132674 |
| DecoderOnlyTransformer | 0.167503 | +76.61 % | 0.000 | +62.10 % | 0.000 | +77.92 % | 0/5 | 0/5 | 7, 7, 9, 8, 7 | 0.133631 |
| PatternDecoder | 0.168068 | +77.21 % | 0.000 | +62.65 % | 0.000 | +78.41 % | 0/5 | 0/5 | 9, 8, 8, 7, 13 | 0.134567 |
| PatternDecoderNoLstmBlock | 0.168127 | +77.27 % | 0.000 | +62.70 % | 0.000 | +78.79 % | 0/5 | 0/5 | 10, 10, 7, 9, 11 | 0.133863 |
| Autoformer | 0.168599 | +77.77 % | 0.000 | +63.16 % | 0.000 | +78.37 % | 0/5 | 0/5 | 8, 9, 1, 11, 12 | 0.134676 |
| PatternDecoderNoConvBlock | 0.170120 | +79.37 % | 0.000 | +64.63 % | 0.000 | +80.34 % | 0/5 | 0/5 | 6, 12, 10, 10, 8 | 0.135081 |
| PatternDecoder_Autocorr | 0.170193 | +79.45 % | 0.000 | +64.70 % | 0.000 | +80.34 % | 0/5 | 0/5 | 3, 11, 2, 12, 9 | 0.134389 |
| Informer | 0.173820 | +83.27 % | 0.000 | +68.21 % | 0.000 | +84.10 % | 0/5 | 0/5 | 1, 13, 11, 13, 10 | 0.134967 |
| Median | 0.185013 | +95.08 % | 0.000 | +79.04 % | 0.000 | +95.08 % | 0/5 | 0/5 | - | 0.151923 |

## Findings

1. **The calendar patterns are real and large.** HAR-X is better than HAR by 12.7 % on the DAX and by 8.2 % on the S&P 500 (Diebold-Mariano p < 0.01), and no other forecast beats it in any window.
2. **No neural model learns them as well.** Every model is significantly worse than HAR-X: by 18 % to 26 % on the DAX and by 52 % to 83 % on the S&P 500. On the DAX, the models are about as good as plain HAR (3 % to 10 % worse, significantly so for 3 of the 13) and better than persistence and, except for three models, than the EWMA. On the S&P 500, all 13 are significantly worse than HAR and worse than persistence and the EWMA; they beat only the constant median.
3. **The PatternDecoder does not stand out.** It ranks 5th of the 13 models on the DAX (19.7 % behind HAR-X) and 8th on the S&P 500 (77.2 % behind). The models with convolutional attention lead: DecoderOnlyTransformer_Conv and PatternDecoder_Conv are first and second on the DAX; PatternDecoder_Conv, Transformer_Conv and DecoderOnlyTransformer_Conv are the first three on the S&P 500.
4. **What the comparison leaves open.** The models see only the calendar features of past days and have to infer the target day's position, for example that the next day is the third Friday of a month; HAR-X gets it directly. Giving the models the target day's calendar features would need known-future inputs, which the pipeline does not have. Their weak results on the S&P 500, behind even persistence, also show that they do not track a slowly moving level well from 30 days of input.

## Extension: the calendar of the target day

Design, fixed in `patterndecoder/volume.py` after the results above and before these runs: the models also get the calendar of the target day. With `calendar: next`, every input row carries the calendar features of the following trading day, so the last row carries those of the target day; once with the cyclical calendar features only and once with the expiry dummies of HAR-X added (`expiry_features: true`). All other settings are unchanged; ten seeds and five test windows as above. The benchmarks are the same, so their results do not change.

### DAX (`^GDAXI`)

MAE relative to HAR-X, rank among the 13 models in brackets:

| Model | Calendar of past days | Calendar of the target day | With expiry dummies |
|---|---|---|---|
| DecoderOnlyTransformer_Conv | +18.0 % (1) | +16.3 % (1) | +15.7 % (1) |
| PatternDecoder_Conv | +18.1 % (2) | +16.6 % (2) | +16.1 % (2) |
| PatternDecoderNoConvBlock | +18.7 % (3) | +18.7 % (7) | +16.3 % (3) |
| DecoderOnlyTransformer | +20.0 % (6) | +17.9 % (5) | +16.8 % (4) |
| PatternDecoderNoLstmBlock | +18.7 % (4) | +17.2 % (3) | +16.8 % (5) |
| PatternDecoder | +19.7 % (5) | +17.6 % (4) | +17.3 % (6) |
| Transformer | +20.9 % (8) | +21.1 % (10) | +18.2 % (7) |
| Transformer_Logsparse | +20.6 % (7) | +20.3 % (8) | +18.4 % (8) |
| Transformer_Conv | +21.3 % (9) | +18.4 % (6) | +18.9 % (9) |
| PatternDecoderBlockNoCrossAttention | +21.4 % (10) | +20.5 % (9) | +20.7 % (10) |
| PatternDecoder_Autocorr | +22.6 % (12) | +22.6 % (11) | +23.3 % (11) |
| Autoformer | +22.4 % (11) | +23.3 % (12) | +23.5 % (12) |
| Informer | +25.7 % (13) | +24.0 % (13) | +23.8 % (13) |
| Mean over the models | +20.6 % | +19.6 % | +18.9 % |
| Mean validation MAE | 0.19915 | 0.19761 | 0.19665 |

### S&P 500 (`^GSPC`)

MAE relative to HAR-X, rank among the 13 models in brackets:

| Model | Calendar of past days | Calendar of the target day | With expiry dummies |
|---|---|---|---|
| Transformer_Conv | +55.6 % (2) | +56.7 % (2) | +53.2 % (1) |
| PatternDecoder_Conv | +51.7 % (1) | +53.4 % (1) | +58.2 % (2) |
| DecoderOnlyTransformer_Conv | +56.3 % (3) | +59.3 % (3) | +61.5 % (3) |
| DecoderOnlyTransformer | +76.6 % (7) | +72.9 % (7) | +67.8 % (4) |
| Transformer | +69.3 % (4) | +69.8 % (4) | +70.0 % (5) |
| Transformer_Logsparse | +72.4 % (6) | +70.3 % (5) | +70.4 % (6) |
| PatternDecoderNoLstmBlock | +77.3 % (9) | +76.3 % (9) | +70.9 % (7) |
| PatternDecoder | +77.2 % (8) | +73.8 % (8) | +72.0 % (8) |
| PatternDecoderNoConvBlock | +79.4 % (11) | +77.0 % (10) | +77.0 % (9) |
| Autoformer | +77.8 % (10) | +78.0 % (11) | +77.4 % (10) |
| PatternDecoderBlockNoCrossAttention | +71.2 % (5) | +71.3 % (6) | +79.0 % (11) |
| PatternDecoder_Autocorr | +79.5 % (12) | +80.3 % (12) | +79.8 % (12) |
| Informer | +83.3 % (13) | +83.1 % (13) | +83.0 % (13) |
| Mean over the models | +71.3 % | +70.9 % | +70.8 % |
| Mean validation MAE | 0.13188 | 0.13181 | 0.13138 |

**Findings.**

1. **The calendar is not what the models lack.** The target day's calendar narrows the gap to HAR-X a little on the DAX (on average from 20.6 % to 19.6 %, with the expiry dummies to 18.9 %) and hardly at all on the S&P 500 (71.3 %, 70.9 %, 70.8 %). Every model stays significantly worse than HAR-X, and none beats plain HAR. The validation MAE also falls slightly, so by that criterion the version with the expiry dummies would be chosen.
2. **The PatternDecoder improves like the others.** On the DAX it moves from 19.7 % to 17.3 % behind HAR-X (ranks 5, 4 and 6), on the S&P 500 from 77.2 % to 72.0 % (rank 8 in all three).
3. **The models do not follow the level.** On the S&P 500, the volume of the test windows in 2025 lies about 0.20 above the training median in centred log volume, but the PatternDecoder's forecasts for seed 1 average +0.05 and 0.00 there; in the earlier windows the volume lies below the training median, and the forecasts again stay close to it. The models pull their forecasts towards the level of the training data instead of following the current level, which persistence and HAR do by construction. Normalizing each input window by its own level and adding that level back to the forecast (reversible instance normalization, as used by recent Transformer forecasters) addresses this and would be the next step.
