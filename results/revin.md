# Reversible instance normalization

Results of an extension of the volume and volatility use cases (`patterndecoder/revin.py`), computed with the workflow "Walk-forward evaluation" on 27 November 2026. In the volume use case, the models pull their forecasts towards the level of their training data: on the S&P 500 in 2025, the log volume was 0.20 above its centring value, while the PatternDecoder forecast 0.00 to 0.05 (`volume.md`). Reversible instance normalization (RevIN; Kim et al., 2022), which PatchTST and other recent forecasting models use, addresses this: it normalizes the input of each window by the window's own mean and standard deviation, applies a learnable affine transform, and maps the forecasts back with the same statistics, so that a model learns deviations from the level of the window and its forecasts follow that level. The design was fixed in `patterndecoder/revin.py` before any model was trained with it:

- **Implementation:** RevIN wraps each model (`revin: true`), so that all models, including the LSTM of the PatternDecoder, which reads the raw inputs, see the normalized series. Only the series input is normalized; the calendar features are left as they are.
- **Use cases:** trading volume as in `volume.md` (DAX and S&P 500, input_scale 1, last-step readout, calendar features of the past days) and volatility as in `volatility.md` with the current setup (input_scale 100, last-step readout).
- **Evaluation:** the walk-forward evaluation with five test windows and ten seeds, compared with the same runs without RevIN. The benchmarks are HAR-X and HAR for volume and HAR and persistence for volatility; the validation MAE decides whether RevIN is used.

## Results

Pooled over the five test windows; seed-averaged forecasts. Without RevIN: the runs reported in `volume.md` (calendar of past days) and `volatility.md` (input_scale 100, last-step readout); the benchmarks are unchanged.

### Trading volume

**DAX (`^GDAXI`)**, MAE relative to HAR-X, rank among the 13 models in brackets:

| Model | Without RevIN | With RevIN | Change of MAE |
|---|---|---|---|
| PatternDecoderNoConvBlock | +18.7 % (3) | +9.8 % (1) | -7.5 % |
| Transformer | +20.9 % (8) | +9.8 % (2) | -9.2 % |
| PatternDecoder_Conv | +18.1 % (2) | +10.1 % (3) | -6.8 % |
| DecoderOnlyTransformer_Conv | +18.0 % (1) | +10.2 % (4) | -6.6 % |
| Transformer_Conv | +21.3 % (9) | +10.4 % (5) | -9.0 % |
| PatternDecoderNoLstmBlock | +18.7 % (4) | +10.6 % (6) | -6.8 % |
| PatternDecoder | +19.7 % (5) | +10.9 % (7) | -7.3 % |
| PatternDecoderBlockNoCrossAttention | +21.4 % (10) | +11.1 % (8) | -8.5 % |
| Transformer_Logsparse | +20.6 % (7) | +11.1 % (9) | -7.8 % |
| DecoderOnlyTransformer | +20.0 % (6) | +11.4 % (10) | -7.2 % |
| Informer | +25.7 % (13) | +13.1 % (11) | -10.0 % |
| PatternDecoder_Autocorr | +22.6 % (12) | +14.8 % (12) | -6.3 % |
| Autoformer | +22.4 % (11) | +15.1 % (13) | -6.0 % |
| Mean over the models | +20.6 % | +11.4 % | -7.6 % |
| Mean validation MAE | 0.199150 | 0.196184 | |
| Models significantly worse than HAR-X (p < 0.05) | 13 | 13 | |
| Models better than HAR | 0 | 11 | |

**S&P 500 (`^GSPC`)**, MAE relative to HAR-X, rank among the 13 models in brackets:

| Model | Without RevIN | With RevIN | Change of MAE |
|---|---|---|---|
| PatternDecoderNoConvBlock | +79.4 % (11) | +1.0 % (1) | -43.7 % |
| PatternDecoder_Conv | +51.7 % (1) | +3.4 % (2) | -31.9 % |
| PatternDecoder | +77.2 % (8) | +3.4 % (3) | -41.6 % |
| PatternDecoderNoLstmBlock | +77.3 % (9) | +3.4 % (4) | -41.7 % |
| Transformer_Conv | +55.6 % (2) | +3.5 % (5) | -33.5 % |
| DecoderOnlyTransformer_Conv | +56.3 % (3) | +3.8 % (6) | -33.6 % |
| DecoderOnlyTransformer | +76.6 % (7) | +4.1 % (7) | -41.1 % |
| Transformer | +69.3 % (4) | +4.2 % (8) | -38.5 % |
| Transformer_Logsparse | +72.4 % (6) | +5.2 % (9) | -39.0 % |
| PatternDecoderBlockNoCrossAttention | +71.2 % (5) | +6.8 % (10) | -37.6 % |
| Informer | +83.3 % (13) | +7.9 % (11) | -41.1 % |
| PatternDecoder_Autocorr | +79.5 % (12) | +11.1 % (12) | -38.1 % |
| Autoformer | +77.8 % (10) | +11.3 % (13) | -37.4 % |
| Mean over the models | +71.3 % | +5.3 % | -38.4 % |
| Mean validation MAE | 0.131878 | 0.101863 | |
| Models significantly worse than HAR-X (p < 0.05) | 13 | 3 | |
| Models better than HAR | 0 | 11 | |

### Volatility

**DAX (`^GDAXI`)**, MAE relative to HAR, rank among the 13 models in brackets:

| Model | Without RevIN | With RevIN | Change of MAE |
|---|---|---|---|
| PatternDecoder_Autocorr | +8.0 % (2) | +0.5 % (1) | -6.9 % |
| Transformer_Conv | +13.5 % (8) | +0.7 % (2) | -11.3 % |
| Autoformer | +5.3 % (1) | +0.9 % (3) | -4.2 % |
| PatternDecoder_Conv | +10.5 % (3) | +0.9 % (4) | -8.7 % |
| DecoderOnlyTransformer_Conv | +11.9 % (5) | +1.0 % (5) | -9.7 % |
| Transformer_Logsparse | +16.5 % (13) | +1.6 % (6) | -12.8 % |
| PatternDecoder | +13.4 % (7) | +1.7 % (7) | -10.3 % |
| PatternDecoderNoLstmBlock | +14.6 % (10) | +1.7 % (8) | -11.2 % |
| Transformer | +15.3 % (12) | +1.9 % (9) | -11.6 % |
| DecoderOnlyTransformer | +14.8 % (11) | +1.9 % (10) | -11.3 % |
| PatternDecoderBlockNoCrossAttention | +12.0 % (6) | +2.1 % (11) | -8.9 % |
| Informer | +14.1 % (9) | +2.7 % (12) | -10.0 % |
| PatternDecoderNoConvBlock | +11.0 % (4) | +3.0 % (13) | -7.3 % |
| Mean over the models | +12.4 % | +1.6 % | -9.6 % |
| Mean validation MAE | 0.002184 | 0.001995 | |
| Models significantly worse than HAR (p < 0.05) | 13 | 0 | |
| Models better than persistence | 13 | 13 | |

**S&P 500 (`^GSPC`)**, MAE relative to HAR, rank among the 13 models in brackets:

| Model | Without RevIN | With RevIN | Change of MAE |
|---|---|---|---|
| Transformer | +26.9 % (12) | +3.8 % (1) | -18.2 % |
| PatternDecoderNoLstmBlock | +24.4 % (10) | +3.9 % (2) | -16.5 % |
| Transformer_Logsparse | +28.7 % (13) | +3.9 % (3) | -19.3 % |
| PatternDecoder_Conv | +20.7 % (5) | +4.0 % (4) | -13.8 % |
| PatternDecoder | +23.8 % (9) | +4.0 % (5) | -16.0 % |
| DecoderOnlyTransformer | +22.8 % (7) | +4.2 % (6) | -15.2 % |
| Transformer_Conv | +19.4 % (3) | +4.2 % (7) | -12.7 % |
| DecoderOnlyTransformer_Conv | +19.5 % (4) | +4.3 % (8) | -12.8 % |
| PatternDecoder_Autocorr | +12.8 % (2) | +5.1 % (9) | -6.8 % |
| Autoformer | +10.3 % (1) | +5.5 % (10) | -4.4 % |
| PatternDecoderBlockNoCrossAttention | +24.9 % (11) | +5.8 % (11) | -15.3 % |
| PatternDecoderNoConvBlock | +23.7 % (8) | +5.8 % (12) | -14.4 % |
| Informer | +22.6 % (6) | +6.6 % (13) | -13.1 % |
| Mean over the models | +21.6 % | +4.7 % | -13.7 % |
| Mean validation MAE | 0.002353 | 0.002007 | |
| Models significantly worse than HAR (p < 0.05) | 13 | 9 | |
| Models better than persistence | 2 | 13 | |

## Findings

1. **RevIN improves every model in both use cases.** The test MAE falls for all 13 models in all four runs: by 6 % to 10 % for volume on the DAX and by 32 % to 44 % on the S&P 500, by 4 % to 13 % for volatility on the DAX and by 4 % to 19 % on the S&P 500. The validation MAE falls in 50 of the 52 cases (not for the Autoformer and PatternDecoder_Autocorr on DAX volume), so by the rule of the design RevIN is used in both use cases.
2. **The gap to the linear benchmarks narrows but does not close.** For volume on the S&P 500, the models are now 1 % to 11 % behind HAR-X instead of 52 % to 83 %, and 10 of the 13 are no longer significantly worse; on the DAX they stay 10 % to 15 % behind (all significantly). In both markets, 11 of the 13 models are now better than plain HAR, significantly so for 7 on the S&P 500 and 2 on the DAX. For volatility on the DAX, all 13 models are within 3 % of HAR and none is significantly worse; on the S&P 500 they are 4 % to 7 % behind (9 of the 13 significantly), and all beat persistence. No model beats HAR-X or HAR on the pooled test days.
3. **The decomposition models lose their lead.** Without RevIN, the Autoformer and PatternDecoder_Autocorr were the two best models for volatility in both markets, which fits the idea that series decomposition helps by separating the level. With RevIN, which takes over that job for every model, they rank 1st and 3rd on the DAX but 9th and 10th on the S&P 500, and they are last (12th and 13th) for volume in both markets.
4. **The PatternDecoder variants lead on volume.** With RevIN, PatternDecoderNoConvBlock (the PatternDecoder with a dense instead of a convolutional token embedding) is the best of the 13 models in both markets: on the S&P 500 it is 1.0 % behind HAR-X (p = 0.76), better than HAR-X in 2 of the 5 test windows and the best model in 4 of them; on the DAX it is 9.8 % behind, level with the Transformer. On the S&P 500, the four PatternDecoder variants take the first four places. The PatternDecoder itself ranks 3rd on the S&P 500 (3.4 % behind HAR-X) and 7th on the DAX (10.9 %). For volatility, no architecture stands out: the 13 models lie within 2.5 (DAX) and 2.8 (S&P 500) percentage points of each other, and the PatternDecoder ranks 7th and 5th.
5. **What this does not show.** The ranking of the models was a secondary question of the design, the differences between the leading models are small, and the evaluation tests the models against the benchmarks, not against each other. The lead of the PatternDecoder variants on volume is therefore a hypothesis for a confirmatory run with new seeds and other series, not a result. Single runs are also weaker than the seed averages: the best single-run MAE for volume on the S&P 500 is 2.8 % behind HAR-X.

## Full tables: trading volume

### DAX (`^GDAXI`)

Test windows: 2023-08-02 to 2024-01-31; 2024-02-01 to 2024-07-31; 2024-08-01 to 2025-02-03; 2025-02-04 to 2025-08-04; 2025-08-05 to 2026-02-05. 10 seeds, 638 test days. Forecasts averaged over the seeds; single run: mean MAE of the seeds' own forecasts. p: Diebold-Mariano test of the pooled ensemble forecasts.

| Model | MAE | vs HAR-X | p | vs HAR | p | single run vs HAR-X | windows better than HAR-X | windows better than HAR | rank per window | val MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| HARX | 0.172565 | +0.00 % | - | -12.70 % | 0.000 | +0.00 % | 0/5 | 5/5 | - | 0.174228 |
| PatternDecoderNoConvBlock | 0.189464 | +9.79 % | 0.001 | -4.15 % | 0.072 | +11.02 % | 0/5 | 3/5 | 4, 1, 6, 4, 1 | 0.195443 |
| Transformer | 0.189489 | +9.81 % | 0.000 | -4.13 % | 0.046 | +11.45 % | 0/5 | 3/5 | 9, 3, 1, 3, 4 | 0.194076 |
| PatternDecoder_Conv | 0.189968 | +10.09 % | 0.000 | -3.89 % | 0.046 | +11.58 % | 0/5 | 4/5 | 1, 7, 2, 5, 9 | 0.193502 |
| DecoderOnlyTransformer_Conv | 0.190103 | +10.16 % | 0.000 | -3.82 % | 0.052 | +11.94 % | 0/5 | 4/5 | 2, 8, 5, 2, 6 | 0.193543 |
| Transformer_Conv | 0.190526 | +10.41 % | 0.000 | -3.61 % | 0.075 | +12.28 % | 0/5 | 3/5 | 3, 9, 7, 1, 10 | 0.193641 |
| PatternDecoderNoLstmBlock | 0.190929 | +10.64 % | 0.000 | -3.40 % | 0.099 | +11.78 % | 0/5 | 2/5 | 8, 2, 4, 8, 11 | 0.195162 |
| PatternDecoder | 0.191346 | +10.88 % | 0.000 | -3.19 % | 0.120 | +12.23 % | 0/5 | 3/5 | 7, 4, 8, 6, 13 | 0.195037 |
| PatternDecoderBlockNoCrossAttention | 0.191647 | +11.06 % | 0.000 | -3.04 % | 0.217 | +13.28 % | 0/5 | 2/5 | 5, 10, 3, 11, 2 | 0.197082 |
| Transformer_Logsparse | 0.191785 | +11.14 % | 0.000 | -2.97 % | 0.172 | +12.99 % | 0/5 | 2/5 | 6, 6, 9, 9, 3 | 0.195902 |
| DecoderOnlyTransformer | 0.192227 | +11.39 % | 0.000 | -2.75 % | 0.189 | +12.98 % | 0/5 | 3/5 | 10, 5, 11, 7, 12 | 0.196187 |
| Informer | 0.195225 | +13.13 % | 0.000 | -1.23 % | 0.583 | +15.39 % | 0/5 | 2/5 | 11, 13, 10, 10, 5 | 0.198520 |
| HAR | 0.197659 | +14.54 % | 0.000 | +0.00 % | - | +14.54 % | 0/5 | 0/5 | - | 0.199741 |
| PatternDecoder_Autocorr | 0.198186 | +14.85 % | 0.000 | +0.27 % | 0.914 | +15.32 % | 0/5 | 2/5 | 13, 12, 12, 12, 7 | 0.200558 |
| Autoformer | 0.198556 | +15.06 % | 0.000 | +0.45 % | 0.852 | +15.45 % | 0/5 | 2/5 | 12, 11, 13, 13, 8 | 0.201745 |
| EWMA | 0.210069 | +21.73 % | 0.000 | +6.28 % | 0.025 | +21.73 % | 0/5 | 1/5 | - | 0.210312 |
| Persistence | 0.228465 | +32.39 % | 0.000 | +15.59 % | 0.000 | +32.39 % | 0/5 | 0/5 | - | 0.238008 |
| Median | 0.244786 | +41.85 % | 0.000 | +23.84 % | 0.000 | +41.85 % | 0/5 | 0/5 | - | 0.220451 |

### S&P 500 (`^GSPC`)

Test windows: 2023-08-07 to 2024-02-02; 2024-02-05 to 2024-08-05; 2024-08-06 to 2025-02-04; 2025-02-05 to 2025-08-06; 2025-08-07 to 2026-02-05. 10 seeds, 628 test days. Forecasts averaged over the seeds; single run: mean MAE of the seeds' own forecasts. p: Diebold-Mariano test of the pooled ensemble forecasts.

| Model | MAE | vs HAR-X | p | vs HAR | p | single run vs HAR-X | windows better than HAR-X | windows better than HAR | rank per window | val MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| HARX | 0.094841 | +0.00 % | - | -8.22 % | 0.003 | +0.00 % | 0/5 | 5/5 | - | 0.091647 |
| PatternDecoderNoConvBlock | 0.095798 | +1.01 % | 0.762 | -7.29 % | 0.001 | +2.81 % | 2/5 | 4/5 | 3, 1, 1, 1, 1 | 0.098041 |
| PatternDecoder_Conv | 0.098022 | +3.35 % | 0.321 | -5.14 % | 0.020 | +5.12 % | 1/5 | 4/5 | 8, 2, 5, 7, 2 | 0.099290 |
| PatternDecoder | 0.098096 | +3.43 % | 0.321 | -5.07 % | 0.024 | +5.58 % | 2/5 | 4/5 | 2, 5, 7, 3, 3 | 0.100299 |
| PatternDecoderNoLstmBlock | 0.098101 | +3.44 % | 0.313 | -5.06 % | 0.020 | +5.47 % | 2/5 | 4/5 | 4, 8, 3, 4, 5 | 0.100304 |
| Transformer_Conv | 0.098160 | +3.50 % | 0.312 | -5.01 % | 0.032 | +5.96 % | 2/5 | 4/5 | 5, 6, 2, 6, 4 | 0.100131 |
| DecoderOnlyTransformer_Conv | 0.098476 | +3.83 % | 0.265 | -4.70 % | 0.037 | +5.82 % | 1/5 | 4/5 | 6, 3, 4, 8, 6 | 0.099871 |
| DecoderOnlyTransformer | 0.098695 | +4.06 % | 0.241 | -4.49 % | 0.047 | +6.30 % | 2/5 | 4/5 | 1, 4, 6, 5, 10 | 0.100605 |
| Transformer | 0.098836 | +4.21 % | 0.231 | -4.35 % | 0.063 | +6.39 % | 2/5 | 4/5 | 7, 9, 8, 2, 8 | 0.101721 |
| Transformer_Logsparse | 0.099786 | +5.21 % | 0.133 | -3.43 % | 0.141 | +7.73 % | 1/5 | 4/5 | 10, 10, 9, 9, 7 | 0.103237 |
| PatternDecoderBlockNoCrossAttention | 0.101300 | +6.81 % | 0.070 | -1.97 % | 0.471 | +9.96 % | 1/5 | 4/5 | 9, 11, 10, 12, 9 | 0.105077 |
| Informer | 0.102294 | +7.86 % | 0.025 | -1.01 % | 0.680 | +10.68 % | 0/5 | 4/5 | 11, 7, 11, 10, 11 | 0.103831 |
| HAR | 0.103333 | +8.95 % | 0.003 | +0.00 % | - | +8.95 % | 0/5 | 0/5 | - | 0.101418 |
| PatternDecoder_Autocorr | 0.105356 | +11.09 % | 0.003 | +1.96 % | 0.452 | +11.68 % | 0/5 | 2/5 | 12, 13, 12, 11, 13 | 0.105723 |
| Autoformer | 0.105581 | +11.32 % | 0.002 | +2.17 % | 0.396 | +11.93 % | 0/5 | 3/5 | 13, 12, 13, 13, 12 | 0.106096 |
| Persistence | 0.112393 | +18.51 % | 0.000 | +8.77 % | 0.002 | +18.51 % | 0/5 | 1/5 | - | 0.114973 |
| EWMA | 0.115163 | +21.43 % | 0.000 | +11.45 % | 0.002 | +21.43 % | 0/5 | 0/5 | - | 0.113243 |
| Median | 0.185013 | +95.08 % | 0.000 | +79.04 % | 0.000 | +95.08 % | 0/5 | 0/5 | - | 0.151923 |

## Full tables: volatility

### DAX (`^GDAXI`)

Test windows: 2023-08-02 to 2024-01-31; 2024-02-01 to 2024-07-31; 2024-08-01 to 2025-02-03; 2025-02-04 to 2025-08-04; 2025-08-05 to 2026-02-05. 10 seeds, 638 test days. Forecasts averaged over the seeds; single run: mean MAE of the seeds' own forecasts. p: Diebold-Mariano test of the pooled ensemble forecasts.

| Model | MAE | vs HAR | p | vs persistence | p | single run vs HAR | windows better than HAR | windows better than persistence | rank per window | val MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| HAR | 0.001924 | +0.00 % | - | -22.21 % | 0.000 | +0.00 % | 0/5 | 5/5 | - | 0.001935 |
| PatternDecoder_Autocorr | 0.001933 | +0.46 % | 0.815 | -21.85 % | 0.000 | +1.35 % | 1/5 | 5/5 | 8, 4, 1, 2, 4 | 0.001976 |
| Transformer_Conv | 0.001937 | +0.66 % | 0.720 | -21.70 % | 0.000 | +2.20 % | 2/5 | 5/5 | 4, 7, 2, 1, 8 | 0.001977 |
| Autoformer | 0.001941 | +0.89 % | 0.658 | -21.52 % | 0.000 | +1.66 % | 1/5 | 5/5 | 6, 2, 3, 5, 5 | 0.001985 |
| PatternDecoder_Conv | 0.001942 | +0.90 % | 0.616 | -21.52 % | 0.000 | +1.90 % | 1/5 | 5/5 | 11, 6, 5, 3, 3 | 0.001977 |
| DecoderOnlyTransformer_Conv | 0.001943 | +0.99 % | 0.577 | -21.44 % | 0.000 | +2.05 % | 2/5 | 5/5 | 5, 5, 8, 4, 2 | 0.001979 |
| Transformer_Logsparse | 0.001955 | +1.60 % | 0.337 | -20.97 % | 0.000 | +3.10 % | 1/5 | 5/5 | 7, 8, 10, 7, 10 | 0.001998 |
| PatternDecoder | 0.001957 | +1.70 % | 0.388 | -20.89 % | 0.000 | +2.52 % | 1/5 | 5/5 | 12, 10, 6, 6, 7 | 0.001992 |
| PatternDecoderNoLstmBlock | 0.001957 | +1.73 % | 0.353 | -20.87 % | 0.000 | +2.44 % | 1/5 | 5/5 | 9, 11, 4, 9, 6 | 0.001996 |
| Transformer | 0.001960 | +1.87 % | 0.309 | -20.76 % | 0.000 | +2.95 % | 2/5 | 5/5 | 3, 13, 9, 10, 12 | 0.002006 |
| DecoderOnlyTransformer | 0.001960 | +1.88 % | 0.334 | -20.75 % | 0.000 | +2.69 % | 1/5 | 5/5 | 10, 12, 7, 8, 9 | 0.001998 |
| PatternDecoderBlockNoCrossAttention | 0.001965 | +2.11 % | 0.192 | -20.57 % | 0.000 | +3.92 % | 1/5 | 5/5 | 2, 3, 12, 11, 11 | 0.002017 |
| Informer | 0.001976 | +2.67 % | 0.165 | -20.14 % | 0.000 | +4.47 % | 1/5 | 5/5 | 1, 1, 11, 13, 13 | 0.002035 |
| PatternDecoderNoConvBlock | 0.001981 | +2.97 % | 0.122 | -19.91 % | 0.000 | +3.60 % | 0/5 | 5/5 | 13, 9, 13, 12, 1 | 0.002004 |
| EWMA | 0.002133 | +10.86 % | 0.001 | -13.76 % | 0.000 | +10.86 % | 0/5 | 5/5 | - | 0.002137 |
| Median | 0.002288 | +18.92 % | 0.000 | -7.50 % | 0.103 | +18.92 % | 0/5 | 5/5 | - | 0.002329 |
| Persistence | 0.002474 | +28.56 % | 0.000 | +0.00 % | - | +28.56 % | 0/5 | 0/5 | - | 0.002461 |

### S&P 500 (`^GSPC`)

Test windows: 2023-08-07 to 2024-02-02; 2024-02-05 to 2024-08-05; 2024-08-06 to 2025-02-04; 2025-02-05 to 2025-08-06; 2025-08-07 to 2026-02-05. 10 seeds, 628 test days. Forecasts averaged over the seeds; single run: mean MAE of the seeds' own forecasts. p: Diebold-Mariano test of the pooled ensemble forecasts.

| Model | MAE | vs HAR | p | vs persistence | p | single run vs HAR | windows better than HAR | windows better than persistence | rank per window | val MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| HAR | 0.002040 | +0.00 % | - | -15.91 % | 0.000 | +0.00 % | 0/5 | 5/5 | - | 0.001958 |
| Transformer | 0.002118 | +3.82 % | 0.082 | -12.70 % | 0.001 | +5.54 % | 1/5 | 4/5 | 9, 6, 10, 1, 2 | 0.001998 |
| PatternDecoderNoLstmBlock | 0.002119 | +3.87 % | 0.082 | -12.66 % | 0.001 | +4.76 % | 1/5 | 4/5 | 5, 5, 11, 3, 4 | 0.001990 |
| Transformer_Logsparse | 0.002120 | +3.93 % | 0.044 | -12.60 % | 0.002 | +5.85 % | 1/5 | 4/5 | 4, 4, 3, 5, 6 | 0.001996 |
| PatternDecoder_Conv | 0.002122 | +4.02 % | 0.019 | -12.53 % | 0.001 | +4.92 % | 1/5 | 4/5 | 2, 3, 7, 6, 9 | 0.001992 |
| PatternDecoder | 0.002122 | +4.03 % | 0.075 | -12.52 % | 0.001 | +5.05 % | 1/5 | 4/5 | 6, 8, 13, 2, 5 | 0.001994 |
| DecoderOnlyTransformer | 0.002125 | +4.16 % | 0.072 | -12.42 % | 0.001 | +5.39 % | 1/5 | 4/5 | 7, 7, 12, 4, 3 | 0.001997 |
| Transformer_Conv | 0.002126 | +4.19 % | 0.029 | -12.39 % | 0.002 | +5.32 % | 1/5 | 4/5 | 1, 1, 6, 8, 11 | 0.001986 |
| DecoderOnlyTransformer_Conv | 0.002127 | +4.26 % | 0.017 | -12.32 % | 0.001 | +5.14 % | 1/5 | 4/5 | 3, 2, 9, 7, 10 | 0.001994 |
| PatternDecoder_Autocorr | 0.002145 | +5.14 % | 0.006 | -11.59 % | 0.003 | +5.87 % | 0/5 | 4/5 | 10, 11, 2, 10, 7 | 0.002015 |
| Autoformer | 0.002152 | +5.47 % | 0.006 | -11.31 % | 0.004 | +5.95 % | 0/5 | 4/5 | 11, 10, 1, 12, 8 | 0.002022 |
| PatternDecoderBlockNoCrossAttention | 0.002158 | +5.78 % | 0.004 | -11.05 % | 0.008 | +7.64 % | 0/5 | 4/5 | 8, 9, 5, 9, 13 | 0.002026 |
| PatternDecoderNoConvBlock | 0.002159 | +5.81 % | 0.030 | -11.03 % | 0.005 | +6.62 % | 1/5 | 4/5 | 12, 13, 8, 13, 1 | 0.002026 |
| Informer | 0.002175 | +6.62 % | 0.009 | -10.35 % | 0.009 | +8.63 % | 0/5 | 4/5 | 13, 12, 4, 11, 12 | 0.002052 |
| EWMA | 0.002388 | +17.02 % | 0.000 | -1.60 % | 0.759 | +17.02 % | 0/5 | 4/5 | - | 0.002203 |
| Persistence | 0.002426 | +18.92 % | 0.000 | +0.00 % | - | +18.92 % | 0/5 | 0/5 | - | 0.002415 |
| Median | 0.002732 | +33.89 % | 0.000 | +12.59 % | 0.081 | +33.89 % | 0/5 | 1/5 | - | 0.002716 |
