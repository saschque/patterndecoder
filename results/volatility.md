# Volatility forecasting

Results of the second use case (`patterndecoder/volatility.py`), computed with the workflow "Walk-forward evaluation" on 26 November 2026. Daily returns contain almost no predictable signal, so the architectures cannot be told apart on them. Volatility is predictable, because calm and turbulent periods persist, and a model with a recurrent memory, such as the LSTM in the PatternDecoder, could have an advantage there. The design was fixed in `patterndecoder/volatility.py` before any model was trained on it:

- **Target:** the next day's Parkinson volatility ln(High / Low) / sqrt(4 ln 2), a range-based estimate of the day's volatility, of the DAX and the S&P 500 from 5 February 2021 to 5 February 2026.
- **Inputs:** the past 30 values of the target and the calendar features. The models are trained with the MAE loss and the settings of `config/config.yaml`, once with input_scale 1 and once with input_scale 100; the choice between the two is made on the validation MAE.
- **Benchmarks:** persistence (the previous day's value), an exponentially weighted moving average with decay 0.94 (EWMA), the HAR model of Corsi (2009) on the previous day's value and its 5- and 22-day means, fitted by least absolute deviations on the training data, and the training median.
- **Evaluation:** the walk-forward evaluation of `patterndecoder/robustness.py` with five consecutive test windows (August 2023 to February 2026) and ten seeds; MAE relative to HAR and persistence, with Diebold-Mariano tests of the seed-averaged forecasts.

The design names GARCH-type models neither as a benchmark nor as a target: they describe close-to-close returns, whereas the target here is the intraday range, for which HAR is the standard benchmark.

After the design was fixed, the last-step readout was chosen as the new default on the DAX returns (see `positive_control.md`). The volatility runs with this setup are reported separately below; they are not part of the pre-specified design.

## Choice between the pre-specified settings

By the rule of the design, input_scale 100 is chosen: its validation MAE is lower than with input_scale 1 for 12 of the 13 models on the DAX and for all 13 on the S&P 500 (mean 0.0022487 against 0.0023582 on the DAX, 0.0023956 against 0.0026702 on the S&P 500). The last-step readout lowers the validation MAE further for 13 of 13 models on the DAX and 11 of 13 on the S&P 500.

## Results

### DAX (`^GDAXI`), input_scale 1, mean readout (pre-specified)

Test windows: 2023-08-02 to 2024-01-31; 2024-02-01 to 2024-07-31; 2024-08-01 to 2025-02-03; 2025-02-04 to 2025-08-04; 2025-08-05 to 2026-02-05. 10 seeds, 638 test days. Forecasts averaged over the seeds; single run: mean MAE of the seeds' own forecasts. p: Diebold-Mariano test of the pooled ensemble forecasts.

| Model | MAE | vs HAR | p | vs persistence | p | single run vs HAR | windows better than HAR | windows better than persistence | rank per window | val MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| HAR | 0.001924 | +0.00 % | - | -22.21 % | 0.000 | +0.00 % | 0/5 | 5/5 | - | 0.001935 |
| EWMA | 0.002133 | +10.86 % | 0.001 | -13.76 % | 0.000 | +10.86 % | 0/5 | 5/5 | - | 0.002137 |
| Autoformer | 0.002243 | +16.59 % | 0.001 | -9.31 % | 0.036 | +25.68 % | 0/5 | 5/5 | 6, 1, 2, 12, 12 | 0.002280 |
| Informer | 0.002258 | +17.34 % | 0.001 | -8.73 % | 0.059 | +24.21 % | 0/5 | 5/5 | 7, 3, 1, 13, 1 | 0.002246 |
| DecoderOnlyTransformer | 0.002282 | +18.61 % | 0.000 | -7.74 % | 0.090 | +25.32 % | 0/5 | 4/5 | 2, 4, 5, 8, 9 | 0.002312 |
| Median | 0.002288 | +18.92 % | 0.000 | -7.50 % | 0.103 | +18.92 % | 0/5 | 5/5 | - | 0.002329 |
| PatternDecoder_Conv | 0.002288 | +18.92 % | 0.000 | -7.50 % | 0.107 | +29.54 % | 0/5 | 4/5 | 4, 7, 4, 5, 11 | 0.002403 |
| DecoderOnlyTransformer_Conv | 0.002290 | +19.00 % | 0.000 | -7.43 % | 0.111 | +31.20 % | 0/5 | 4/5 | 3, 6, 9, 6, 4 | 0.002356 |
| PatternDecoderNoLstmBlock | 0.002303 | +19.70 % | 0.000 | -6.89 % | 0.136 | +27.26 % | 0/5 | 4/5 | 1, 9, 7, 11, 8 | 0.002339 |
| PatternDecoder | 0.002304 | +19.74 % | 0.000 | -6.86 % | 0.136 | +29.04 % | 0/5 | 4/5 | 5, 8, 6, 10, 10 | 0.002367 |
| PatternDecoder_Autocorr | 0.002318 | +20.47 % | 0.000 | -6.29 % | 0.154 | +40.88 % | 0/5 | 5/5 | 8, 2, 13, 1, 13 | 0.002385 |
| PatternDecoderBlockNoCrossAttention | 0.002322 | +20.65 % | 0.000 | -6.15 % | 0.192 | +29.75 % | 0/5 | 3/5 | 9, 5, 3, 9, 6 | 0.002352 |
| Transformer | 0.002358 | +22.53 % | 0.000 | -4.69 % | 0.311 | +33.69 % | 0/5 | 3/5 | 10, 11, 10, 3, 5 | 0.002435 |
| Transformer_Conv | 0.002385 | +23.95 % | 0.000 | -3.59 % | 0.465 | +30.03 % | 0/5 | 3/5 | 11, 13, 8, 4, 3 | 0.002407 |
| PatternDecoderNoConvBlock | 0.002412 | +25.35 % | 0.000 | -2.50 % | 0.616 | +32.89 % | 0/5 | 3/5 | 12, 12, 12, 2, 2 | 0.002318 |
| Transformer_Logsparse | 0.002419 | +25.70 % | 0.000 | -2.23 % | 0.638 | +30.90 % | 0/5 | 3/5 | 13, 10, 11, 7, 7 | 0.002456 |
| Persistence | 0.002474 | +28.56 % | 0.000 | +0.00 % | - | +28.56 % | 0/5 | 0/5 | - | 0.002461 |


### DAX (`^GDAXI`), input_scale 100, mean readout (pre-specified)

Test windows: 2023-08-02 to 2024-01-31; 2024-02-01 to 2024-07-31; 2024-08-01 to 2025-02-03; 2025-02-04 to 2025-08-04; 2025-08-05 to 2026-02-05. 10 seeds, 638 test days. Forecasts averaged over the seeds; single run: mean MAE of the seeds' own forecasts. p: Diebold-Mariano test of the pooled ensemble forecasts.

| Model | MAE | vs HAR | p | vs persistence | p | single run vs HAR | windows better than HAR | windows better than persistence | rank per window | val MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| HAR | 0.001924 | +0.00 % | - | -22.21 % | 0.000 | +0.00 % | 0/5 | 5/5 | - | 0.001935 |
| Autoformer | 0.002126 | +10.50 % | 0.002 | -14.05 % | 0.000 | +14.61 % | 0/5 | 5/5 | 1, 1, 4, 13, 8 | 0.002099 |
| EWMA | 0.002133 | +10.86 % | 0.001 | -13.76 % | 0.000 | +10.86 % | 0/5 | 5/5 | - | 0.002137 |
| Informer | 0.002167 | +12.62 % | 0.003 | -12.40 % | 0.002 | +26.02 % | 0/5 | 5/5 | 6, 5, 1, 12, 2 | 0.002299 |
| PatternDecoder_Conv | 0.002179 | +13.25 % | 0.000 | -11.91 % | 0.001 | +19.59 % | 0/5 | 5/5 | 5, 9, 10, 2, 4 | 0.002243 |
| DecoderOnlyTransformer_Conv | 0.002179 | +13.25 % | 0.000 | -11.91 % | 0.001 | +17.92 % | 0/5 | 5/5 | 4, 2, 9, 11, 6 | 0.002233 |
| Transformer_Conv | 0.002180 | +13.28 % | 0.000 | -11.89 % | 0.002 | +20.20 % | 0/5 | 5/5 | 8, 8, 3, 1, 3 | 0.002286 |
| DecoderOnlyTransformer | 0.002194 | +14.02 % | 0.000 | -11.31 % | 0.003 | +19.11 % | 0/5 | 5/5 | 9, 4, 7, 9, 1 | 0.002205 |
| PatternDecoderNoConvBlock | 0.002198 | +14.22 % | 0.000 | -11.16 % | 0.003 | +20.79 % | 0/5 | 5/5 | 10, 3, 8, 7, 7 | 0.002240 |
| PatternDecoderBlockNoCrossAttention | 0.002202 | +14.42 % | 0.000 | -11.00 % | 0.005 | +20.61 % | 0/5 | 5/5 | 2, 10, 2, 10, 12 | 0.002290 |
| PatternDecoder_Autocorr | 0.002204 | +14.52 % | 0.000 | -10.92 % | 0.003 | +20.12 % | 0/5 | 5/5 | 3, 6, 13, 5, 13 | 0.002196 |
| PatternDecoder | 0.002220 | +15.38 % | 0.000 | -10.25 % | 0.010 | +21.27 % | 0/5 | 5/5 | 7, 11, 6, 4, 5 | 0.002233 |
| PatternDecoderNoLstmBlock | 0.002237 | +16.27 % | 0.000 | -9.56 % | 0.016 | +21.86 % | 0/5 | 4/5 | 11, 7, 5, 8, 11 | 0.002254 |
| Transformer_Logsparse | 0.002282 | +18.59 % | 0.000 | -7.75 % | 0.064 | +25.25 % | 0/5 | 3/5 | 12, 13, 11, 3, 9 | 0.002320 |
| Median | 0.002288 | +18.92 % | 0.000 | -7.50 % | 0.103 | +18.92 % | 0/5 | 5/5 | - | 0.002329 |
| Transformer | 0.002295 | +19.24 % | 0.000 | -7.25 % | 0.085 | +24.86 % | 0/5 | 3/5 | 13, 12, 12, 6, 10 | 0.002336 |
| Persistence | 0.002474 | +28.56 % | 0.000 | +0.00 % | - | +28.56 % | 0/5 | 0/5 | - | 0.002461 |


### DAX (`^GDAXI`), input_scale 100, last-step readout (setup chosen on DAX returns)

Test windows: 2023-08-02 to 2024-01-31; 2024-02-01 to 2024-07-31; 2024-08-01 to 2025-02-03; 2025-02-04 to 2025-08-04; 2025-08-05 to 2026-02-05. 10 seeds, 638 test days. Forecasts averaged over the seeds; single run: mean MAE of the seeds' own forecasts. p: Diebold-Mariano test of the pooled ensemble forecasts.

| Model | MAE | vs HAR | p | vs persistence | p | single run vs HAR | windows better than HAR | windows better than persistence | rank per window | val MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| HAR | 0.001924 | +0.00 % | - | -22.21 % | 0.000 | +0.00 % | 0/5 | 5/5 | - | 0.001935 |
| Autoformer | 0.002025 | +5.26 % | 0.036 | -18.12 % | 0.000 | +11.95 % | 1/5 | 5/5 | 2, 1, 1, 2, 2 | 0.002054 |
| PatternDecoder_Autocorr | 0.002077 | +7.96 % | 0.007 | -16.02 % | 0.000 | +16.48 % | 0/5 | 5/5 | 1, 3, 9, 1, 6 | 0.002127 |
| PatternDecoder_Conv | 0.002127 | +10.52 % | 0.005 | -14.03 % | 0.000 | +17.33 % | 0/5 | 5/5 | 3, 8, 8, 6, 3 | 0.002188 |
| EWMA | 0.002133 | +10.86 % | 0.001 | -13.76 % | 0.000 | +10.86 % | 0/5 | 5/5 | - | 0.002137 |
| PatternDecoderNoConvBlock | 0.002136 | +11.02 % | 0.004 | -13.64 % | 0.000 | +16.63 % | 0/5 | 5/5 | 8, 6, 3, 3, 1 | 0.002161 |
| DecoderOnlyTransformer_Conv | 0.002152 | +11.85 % | 0.001 | -13.00 % | 0.001 | +17.91 % | 0/5 | 5/5 | 6, 5, 10, 4, 5 | 0.002191 |
| PatternDecoderBlockNoCrossAttention | 0.002156 | +12.04 % | 0.002 | -12.85 % | 0.001 | +19.87 % | 0/5 | 5/5 | 7, 2, 7, 5, 13 | 0.002235 |
| PatternDecoder | 0.002182 | +13.41 % | 0.001 | -11.78 % | 0.003 | +18.28 % | 0/5 | 5/5 | 4, 10, 12, 8, 11 | 0.002195 |
| Transformer_Conv | 0.002184 | +13.50 % | 0.001 | -11.72 % | 0.003 | +19.79 % | 0/5 | 4/5 | 11, 4, 6, 7, 9 | 0.002199 |
| Informer | 0.002196 | +14.11 % | 0.002 | -11.24 % | 0.007 | +20.47 % | 0/5 | 5/5 | 9, 11, 2, 13, 4 | 0.002211 |
| PatternDecoderNoLstmBlock | 0.002204 | +14.56 % | 0.001 | -10.89 % | 0.008 | +19.64 % | 0/5 | 4/5 | 10, 7, 11, 10, 8 | 0.002188 |
| DecoderOnlyTransformer | 0.002209 | +14.82 % | 0.000 | -10.69 % | 0.008 | +20.46 % | 0/5 | 5/5 | 5, 9, 13, 11, 12 | 0.002202 |
| Transformer | 0.002218 | +15.26 % | 0.001 | -10.35 % | 0.012 | +21.67 % | 0/5 | 4/5 | 12, 12, 5, 9, 10 | 0.002205 |
| Transformer_Logsparse | 0.002242 | +16.53 % | 0.001 | -9.36 % | 0.032 | +22.58 % | 0/5 | 4/5 | 13, 13, 4, 12, 7 | 0.002243 |
| Median | 0.002288 | +18.92 % | 0.000 | -7.50 % | 0.103 | +18.92 % | 0/5 | 5/5 | - | 0.002329 |
| Persistence | 0.002474 | +28.56 % | 0.000 | +0.00 % | - | +28.56 % | 0/5 | 0/5 | - | 0.002461 |


### S&P 500 (`^GSPC`), input_scale 1, mean readout (pre-specified)

Test windows: 2023-08-07 to 2024-02-02; 2024-02-05 to 2024-08-05; 2024-08-06 to 2025-02-04; 2025-02-05 to 2025-08-06; 2025-08-07 to 2026-02-05. 10 seeds, 628 test days. Forecasts averaged over the seeds; single run: mean MAE of the seeds' own forecasts. p: Diebold-Mariano test of the pooled ensemble forecasts.

| Model | MAE | vs HAR | p | vs persistence | p | single run vs HAR | windows better than HAR | windows better than persistence | rank per window | val MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| HAR | 0.002040 | +0.00 % | - | -15.91 % | 0.000 | +0.00 % | 0/5 | 5/5 | - | 0.001958 |
| EWMA | 0.002388 | +17.02 % | 0.000 | -1.60 % | 0.759 | +17.02 % | 0/5 | 4/5 | - | 0.002203 |
| Persistence | 0.002426 | +18.92 % | 0.000 | +0.00 % | - | +18.92 % | 0/5 | 0/5 | - | 0.002415 |
| PatternDecoder_Autocorr | 0.002699 | +32.28 % | 0.000 | +11.24 % | 0.105 | +58.39 % | 0/5 | 1/5 | 2, 1, 6, 4, 12 | 0.002652 |
| Autoformer | 0.002709 | +32.78 % | 0.000 | +11.66 % | 0.109 | +43.29 % | 0/5 | 1/5 | 1, 2, 2, 13, 2 | 0.002456 |
| Median | 0.002732 | +33.89 % | 0.000 | +12.59 % | 0.081 | +33.89 % | 0/5 | 1/5 | - | 0.002716 |
| Informer | 0.002748 | +34.67 % | 0.000 | +13.24 % | 0.084 | +41.94 % | 0/5 | 1/5 | 6, 8, 1, 11, 1 | 0.002485 |
| DecoderOnlyTransformer | 0.002788 | +36.64 % | 0.000 | +14.90 % | 0.033 | +48.00 % | 0/5 | 0/5 | 3, 3, 7, 2, 5 | 0.002637 |
| DecoderOnlyTransformer_Conv | 0.002844 | +39.40 % | 0.000 | +17.22 % | 0.018 | +53.35 % | 0/5 | 0/5 | 4, 4, 8, 12, 11 | 0.002652 |
| PatternDecoderNoLstmBlock | 0.002848 | +39.58 % | 0.000 | +17.37 % | 0.015 | +50.46 % | 0/5 | 0/5 | 5, 6, 10, 8, 6 | 0.002694 |
| PatternDecoder | 0.002879 | +41.11 % | 0.000 | +18.66 % | 0.012 | +52.43 % | 0/5 | 0/5 | 8, 7, 9, 9, 4 | 0.002660 |
| PatternDecoderBlockNoCrossAttention | 0.002882 | +41.25 % | 0.000 | +18.77 % | 0.015 | +52.57 % | 0/5 | 1/5 | 7, 9, 4, 5, 13 | 0.002682 |
| PatternDecoder_Conv | 0.002909 | +42.59 % | 0.000 | +19.90 % | 0.008 | +55.26 % | 0/5 | 0/5 | 9, 5, 11, 10, 8 | 0.002682 |
| PatternDecoderNoConvBlock | 0.002979 | +46.01 % | 0.000 | +22.78 % | 0.005 | +56.80 % | 0/5 | 1/5 | 12, 10, 5, 7, 7 | 0.002648 |
| Transformer_Conv | 0.002999 | +46.99 % | 0.000 | +23.61 % | 0.003 | +56.82 % | 0/5 | 1/5 | 11, 12, 3, 6, 3 | 0.002753 |
| Transformer | 0.003073 | +50.59 % | 0.000 | +26.63 % | 0.001 | +62.02 % | 0/5 | 0/5 | 10, 13, 12, 1, 10 | 0.002847 |
| Transformer_Logsparse | 0.003147 | +54.26 % | 0.000 | +29.71 % | 0.000 | +62.93 % | 0/5 | 0/5 | 13, 11, 13, 3, 9 | 0.002865 |


### S&P 500 (`^GSPC`), input_scale 100, mean readout (pre-specified)

Test windows: 2023-08-07 to 2024-02-02; 2024-02-05 to 2024-08-05; 2024-08-06 to 2025-02-04; 2025-02-05 to 2025-08-06; 2025-08-07 to 2026-02-05. 10 seeds, 628 test days. Forecasts averaged over the seeds; single run: mean MAE of the seeds' own forecasts. p: Diebold-Mariano test of the pooled ensemble forecasts.

| Model | MAE | vs HAR | p | vs persistence | p | single run vs HAR | windows better than HAR | windows better than persistence | rank per window | val MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| HAR | 0.002040 | +0.00 % | - | -15.91 % | 0.000 | +0.00 % | 0/5 | 5/5 | - | 0.001958 |
| EWMA | 0.002388 | +17.02 % | 0.000 | -1.60 % | 0.759 | +17.02 % | 0/5 | 4/5 | - | 0.002203 |
| Autoformer | 0.002394 | +17.33 % | 0.000 | -1.34 % | 0.816 | +22.41 % | 0/5 | 4/5 | 1, 1, 2, 11, 1 | 0.002198 |
| Persistence | 0.002426 | +18.92 % | 0.000 | +0.00 % | - | +18.92 % | 0/5 | 0/5 | - | 0.002415 |
| Informer | 0.002524 | +23.70 % | 0.000 | +4.02 % | 0.531 | +36.48 % | 0/5 | 3/5 | 8, 13, 1, 2, 6 | 0.002405 |
| PatternDecoderNoLstmBlock | 0.002553 | +25.12 % | 0.000 | +5.22 % | 0.388 | +30.81 % | 0/5 | 3/5 | 5, 7, 4, 5, 8 | 0.002395 |
| PatternDecoder_Autocorr | 0.002560 | +25.45 % | 0.000 | +5.49 % | 0.329 | +31.16 % | 0/5 | 2/5 | 4, 2, 12, 3, 7 | 0.002344 |
| PatternDecoder_Conv | 0.002567 | +25.84 % | 0.000 | +5.82 % | 0.312 | +33.06 % | 0/5 | 3/5 | 6, 5, 10, 9, 4 | 0.002356 |
| Transformer_Conv | 0.002569 | +25.91 % | 0.000 | +5.87 % | 0.325 | +33.73 % | 0/5 | 3/5 | 9, 9, 7, 1, 3 | 0.002436 |
| DecoderOnlyTransformer | 0.002576 | +26.27 % | 0.000 | +6.18 % | 0.315 | +32.29 % | 0/5 | 3/5 | 10, 6, 5, 13, 2 | 0.002343 |
| PatternDecoder | 0.002584 | +26.66 % | 0.000 | +6.51 % | 0.274 | +32.85 % | 0/5 | 2/5 | 2, 8, 11, 7, 13 | 0.002372 |
| DecoderOnlyTransformer_Conv | 0.002586 | +26.77 % | 0.000 | +6.60 % | 0.269 | +31.76 % | 0/5 | 2/5 | 7, 4, 9, 12, 10 | 0.002360 |
| PatternDecoderNoConvBlock | 0.002596 | +27.22 % | 0.000 | +6.98 % | 0.250 | +37.03 % | 0/5 | 1/5 | 3, 3, 13, 10, 12 | 0.002414 |
| PatternDecoderBlockNoCrossAttention | 0.002599 | +27.39 % | 0.000 | +7.12 % | 0.264 | +35.94 % | 0/5 | 2/5 | 11, 12, 3, 4, 9 | 0.002501 |
| Transformer_Logsparse | 0.002639 | +29.35 % | 0.000 | +8.77 % | 0.166 | +36.08 % | 0/5 | 2/5 | 12, 11, 6, 6, 11 | 0.002497 |
| Transformer | 0.002661 | +30.40 % | 0.000 | +9.65 % | 0.140 | +38.92 % | 0/5 | 3/5 | 13, 10, 8, 8, 5 | 0.002521 |
| Median | 0.002732 | +33.89 % | 0.000 | +12.59 % | 0.081 | +33.89 % | 0/5 | 1/5 | - | 0.002716 |


### S&P 500 (`^GSPC`), input_scale 100, last-step readout (setup chosen on DAX returns)

Test windows: 2023-08-07 to 2024-02-02; 2024-02-05 to 2024-08-05; 2024-08-06 to 2025-02-04; 2025-02-05 to 2025-08-06; 2025-08-07 to 2026-02-05. 10 seeds, 628 test days. Forecasts averaged over the seeds; single run: mean MAE of the seeds' own forecasts. p: Diebold-Mariano test of the pooled ensemble forecasts.

| Model | MAE | vs HAR | p | vs persistence | p | single run vs HAR | windows better than HAR | windows better than persistence | rank per window | val MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| HAR | 0.002040 | +0.00 % | - | -15.91 % | 0.000 | +0.00 % | 0/5 | 5/5 | - | 0.001958 |
| Autoformer | 0.002251 | +10.34 % | 0.002 | -7.21 % | 0.169 | +18.45 % | 0/5 | 4/5 | 1, 1, 1, 2, 1 | 0.002155 |
| PatternDecoder_Autocorr | 0.002301 | +12.77 % | 0.000 | -5.17 % | 0.322 | +22.12 % | 0/5 | 3/5 | 2, 2, 3, 1, 6 | 0.002251 |
| EWMA | 0.002388 | +17.02 % | 0.000 | -1.60 % | 0.759 | +17.02 % | 0/5 | 4/5 | - | 0.002203 |
| Persistence | 0.002426 | +18.92 % | 0.000 | +0.00 % | - | +18.92 % | 0/5 | 0/5 | - | 0.002415 |
| Transformer_Conv | 0.002435 | +19.35 % | 0.000 | +0.36 % | 0.952 | +27.47 % | 0/5 | 3/5 | 5, 7, 2, 3, 2 | 0.002322 |
| DecoderOnlyTransformer_Conv | 0.002439 | +19.51 % | 0.000 | +0.50 % | 0.933 | +27.09 % | 0/5 | 3/5 | 4, 5, 6, 5, 3 | 0.002327 |
| PatternDecoder_Conv | 0.002462 | +20.68 % | 0.000 | +1.48 % | 0.805 | +29.87 % | 0/5 | 3/5 | 3, 11, 5, 6, 5 | 0.002357 |
| Informer | 0.002502 | +22.63 % | 0.000 | +3.12 % | 0.636 | +31.92 % | 0/5 | 3/5 | 8, 3, 4, 12, 8 | 0.002390 |
| DecoderOnlyTransformer | 0.002506 | +22.84 % | 0.000 | +3.29 % | 0.617 | +29.04 % | 0/5 | 3/5 | 6, 10, 7, 11, 7 | 0.002367 |
| PatternDecoderNoConvBlock | 0.002523 | +23.67 % | 0.000 | +3.99 % | 0.523 | +32.00 % | 0/5 | 3/5 | 9, 4, 13, 4, 12 | 0.002337 |
| PatternDecoder | 0.002527 | +23.83 % | 0.000 | +4.13 % | 0.529 | +30.24 % | 0/5 | 3/5 | 10, 8, 11, 7, 4 | 0.002361 |
| PatternDecoderNoLstmBlock | 0.002538 | +24.40 % | 0.000 | +4.61 % | 0.482 | +29.78 % | 0/5 | 3/5 | 11, 6, 8, 8, 10 | 0.002371 |
| PatternDecoderBlockNoCrossAttention | 0.002548 | +24.87 % | 0.000 | +5.00 % | 0.444 | +34.12 % | 0/5 | 3/5 | 7, 9, 10, 10, 11 | 0.002450 |
| Transformer | 0.002589 | +26.91 % | 0.000 | +6.72 % | 0.323 | +32.76 % | 0/5 | 2/5 | 13, 12, 9, 9, 9 | 0.002435 |
| Transformer_Logsparse | 0.002627 | +28.74 % | 0.000 | +8.26 % | 0.235 | +35.10 % | 0/5 | 1/5 | 12, 13, 12, 13, 13 | 0.002462 |
| Median | 0.002732 | +33.89 % | 0.000 | +12.59 % | 0.081 | +33.89 % | 0/5 | 1/5 | - | 0.002716 |

## Findings

1. **HAR is the best forecast on both markets.** Every neural model is significantly worse in every setting (Diebold-Mariano p ≤ 0.04), and no model beats HAR in more than one of the five windows.
2. **With the pre-specified settings, the models add little.** With input_scale 1 their seed-averaged forecasts are about as good as the constant median on the DAX (+16.6 % to +25.7 % against HAR, median +18.9 %) and worse than persistence on the S&P 500. With the chosen input_scale 100, the best model, the Autoformer, is about as good as the EWMA (DAX +10.5 % against +10.9 %; S&P 500 +17.3 % against +17.0 %).
3. **With the last-step readout, all models improve.** The Autoformer is the best neural model on both markets (DAX +5.3 % against HAR, p = 0.036; S&P 500 +10.3 %), followed by the PatternDecoder variant with auto-correlation (+8.0 % and +12.8 %); both beat the EWMA and persistence. Both use series decomposition, which separates a smooth level from fluctuations. Across all six runs, the Autoformer is the best neural model five times; on the S&P 500 with input_scale 1 it is second, behind the PatternDecoder variant with auto-correlation.
4. **The PatternDecoder does not stand out.** With the last-step readout it ranks 7th of the 13 models on the DAX (+13.4 % against HAR) and 9th on the S&P 500 (+23.8 %); with the pre-specified settings it ranks 7th to 10th.

## Extension: reversible instance normalization

With reversible instance normalization of each input window (`revin: true`, with input_scale 100 and the last-step readout; design and results in `revin.md`), the test MAE of every model falls, by 4 % to 13 % on the DAX and by 4 % to 19 % on the S&P 500, and by the rule of that design it is used. On the DAX, all models are then within 3 % of HAR and none is significantly worse; on the S&P 500 they are 4 % to 7 % behind, 9 of the 13 significantly. No model beats HAR. The lead of the Autoformer and the PatternDecoder variant with auto-correlation disappears on the S&P 500 (ranks 10 and 9); the PatternDecoder ranks 7th on the DAX and 5th on the S&P 500.
