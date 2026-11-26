# Walk-forward evaluation

Results of `patterndecoder/robustness.py`, computed with the manually started workflow "Walk-forward evaluation" on 26 November 2026. For each of ten seeds (1 to 10), every model was trained once per test window. The five test windows are consecutive, each covers 10 % of the data, and together they cover its second half (August 2023 to February 2026). The models of a window are trained on all earlier data, whose last 20 % serve for early stopping; all other settings are those of `config/config.yaml`, including 2 encoder layers and 1 decoder layer for every model.

- **MAE:** test MAE of the forecasts averaged over the ten seeds (seed ensemble), pooled over all windows.
- **vs zero, vs median:** difference in MAE from the zero forecast (random walk for the price) and from the constant median of the training returns, the best constant forecast under the MAE loss.
- **p:** Diebold-Mariano test of the pooled ensemble forecasts against the benchmark (absolute errors, Newey-West variance).
- **single run vs zero:** mean MAE of the seeds' own forecasts against the zero forecast, that is, what a single training run achieves on average.
- **val MAE:** mean validation MAE over seeds and windows, the criterion for choosing between settings.

The hyperparameters and the layer pair were chosen on the DAX with data up to the end of the original training period (70 % of the data). The first two DAX windows therefore lie inside the period on which these choices were made; the last three cover the original test period. The S&P 500 runs use the settings chosen for the DAX. The daily forecasts of every training run are available as artifacts of the workflow runs; this folder keeps the summaries (`walk_forward.csv`, pooled, and `walk_forward_windows.csv`, per window).

## DAX (`^GDAXI`), input_scale 1 (setting of the published results)

Test windows: 2023-08-02 to 2024-01-31; 2024-02-01 to 2024-07-31; 2024-08-01 to 2025-02-03; 2025-02-04 to 2025-08-04; 2025-08-05 to 2026-02-05. 10 seeds, 638 test days. Forecasts averaged over the seeds; single run: mean MAE of the seeds' own forecasts. p: Diebold-Mariano test of the pooled ensemble forecasts.

| Model | MAE | vs zero | p | vs median | p | single run vs zero | windows better than zero | windows better than median | rank per window | val MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| Informer | 0.006681 | -0.64 % | 0.285 | -0.11 % | 0.737 | -0.13 % | 4/5 | 2/5 | 1, 1, 7, 1, 9 | 0.006446 |
| Median | 0.006688 | -0.53 % | 0.279 | +0.00 % | - | -0.53 % | 5/5 | 0/5 | - | 0.006444 |
| Transformer_Conv | 0.006692 | -0.47 % | 0.443 | +0.06 % | 0.843 | -0.01 % | 4/5 | 3/5 | 2, 4, 2, 2, 7 | 0.006441 |
| PatternDecoderNoLstmBlock | 0.006699 | -0.36 % | 0.560 | +0.17 % | 0.630 | -0.08 % | 4/5 | 2/5 | 7, 2, 13, 4, 3 | 0.006443 |
| PatternDecoder_Autocorr | 0.006701 | -0.34 % | 0.605 | +0.19 % | 0.623 | +0.01 % | 3/5 | 2/5 | 9, 3, 10, 8, 6 | 0.006450 |
| PatternDecoder | 0.006702 | -0.31 % | 0.605 | +0.22 % | 0.539 | -0.01 % | 4/5 | 1/5 | 5, 6, 9, 3, 12 | 0.006447 |
| DecoderOnlyTransformer_Conv | 0.006704 | -0.30 % | 0.587 | +0.23 % | 0.457 | +0.12 % | 3/5 | 1/5 | 11, 10, 4, 10, 1 | 0.006454 |
| DecoderOnlyTransformer | 0.006704 | -0.29 % | 0.603 | +0.24 % | 0.480 | -0.00 % | 3/5 | 2/5 | 4, 5, 11, 12, 10 | 0.006448 |
| Autoformer | 0.006705 | -0.27 % | 0.647 | +0.26 % | 0.474 | -0.10 % | 3/5 | 2/5 | 13, 7, 1, 7, 4 | 0.006433 |
| PatternDecoderBlockNoCrossAttention | 0.006707 | -0.24 % | 0.676 | +0.29 % | 0.371 | +0.42 % | 3/5 | 1/5 | 10, 8, 5, 13, 11 | 0.006462 |
| Transformer | 0.006708 | -0.23 % | 0.696 | +0.30 % | 0.386 | +0.23 % | 4/5 | 2/5 | 3, 12, 12, 5, 5 | 0.006451 |
| Transformer_Logsparse | 0.006710 | -0.21 % | 0.750 | +0.32 % | 0.368 | +0.33 % | 3/5 | 1/5 | 12, 9, 3, 11, 8 | 0.006452 |
| PatternDecoderNoConvBlock | 0.006712 | -0.18 % | 0.745 | +0.35 % | 0.274 | +0.24 % | 3/5 | 0/5 | 6, 11, 6, 9, 13 | 0.006448 |
| PatternDecoder_Conv | 0.006714 | -0.14 % | 0.794 | +0.39 % | 0.274 | +0.25 % | 2/5 | 1/5 | 8, 13, 8, 6, 2 | 0.006450 |
| Naive | 0.006724 | +0.00 % | - | +0.53 % | 0.279 | +0.00 % | 0/5 | 0/5 | - | 0.006467 |

Per test window (seed-averaged forecasts; rank among the 13 models):

| Window | Zero MAE | Median vs zero | PatternDecoder vs zero (rank) | Models better than zero | Models better than median | Best model |
|---|---|---|---|---|---|---|
| 2023-08-02 to 2024-01-31 | 0.005367 | -0.39 % | -0.38 % (5) | 12/13 | 4/13 | Informer (-1.19 %) |
| 2024-02-01 to 2024-07-31 | 0.005577 | -0.88 % | -1.10 % (6) | 12/13 | 8/13 | Informer (-1.80 %) |
| 2024-08-01 to 2025-02-03 | 0.006577 | -0.68 % | +0.13 % (9) | 1/13 | 0/13 | Autoformer (-0.14 %) |
| 2025-02-04 to 2025-08-04 | 0.010280 | -0.31 % | -0.08 % (3) | 5/13 | 0/13 | Informer (-0.25 %) |
| 2025-08-05 to 2026-02-05 | 0.005835 | -0.54 % | -0.41 % (12) | 13/13 | 8/13 | DecoderOnlyTransformer_Conv (-0.93 %) |

## S&P 500 (`^GSPC`), input_scale 1

Test windows: 2023-08-07 to 2024-02-02; 2024-02-05 to 2024-08-05; 2024-08-06 to 2025-02-04; 2025-02-05 to 2025-08-06; 2025-08-07 to 2026-02-05. 10 seeds, 628 test days. Forecasts averaged over the seeds; single run: mean MAE of the seeds' own forecasts. p: Diebold-Mariano test of the pooled ensemble forecasts.

| Model | MAE | vs zero | p | vs median | p | single run vs zero | windows better than zero | windows better than median | rank per window | val MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| Median | 0.006427 | -0.66 % | 0.025 | +0.00 % | - | -0.66 % | 5/5 | 0/5 | - | 0.006286 |
| PatternDecoderNoLstmBlock | 0.006456 | -0.21 % | 0.472 | +0.45 % | 0.113 | +0.30 % | 3/5 | 1/5 | 1, 8, 12, 9, 1 | 0.006285 |
| PatternDecoderNoConvBlock | 0.006457 | -0.19 % | 0.552 | +0.47 % | 0.046 | +0.23 % | 4/5 | 1/5 | 4, 4, 1, 10, 4 | 0.006274 |
| Autoformer | 0.006458 | -0.19 % | 0.568 | +0.48 % | 0.033 | +0.11 % | 4/5 | 1/5 | 3, 2, 7, 11, 6 | 0.006274 |
| DecoderOnlyTransformer | 0.006464 | -0.09 % | 0.778 | +0.57 % | 0.063 | +0.37 % | 3/5 | 1/5 | 2, 9, 11, 4, 5 | 0.006277 |
| Transformer_Logsparse | 0.006465 | -0.08 % | 0.806 | +0.59 % | 0.012 | +0.56 % | 3/5 | 0/5 | 5, 5, 4, 2, 12 | 0.006280 |
| PatternDecoder | 0.006466 | -0.06 % | 0.856 | +0.60 % | 0.035 | +0.47 % | 3/5 | 1/5 | 10, 6, 5, 3, 2 | 0.006286 |
| Informer | 0.006467 | -0.04 % | 0.924 | +0.62 % | 0.036 | +0.70 % | 3/5 | 0/5 | 6, 7, 10, 1, 13 | 0.006295 |
| PatternDecoder_Autocorr | 0.006469 | -0.02 % | 0.959 | +0.65 % | 0.022 | +0.51 % | 4/5 | 1/5 | 7, 3, 13, 13, 3 | 0.006287 |
| Transformer | 0.006469 | -0.01 % | 0.976 | +0.65 % | 0.042 | +0.61 % | 4/5 | 1/5 | 12, 1, 3, 7, 11 | 0.006279 |
| Naive | 0.006470 | +0.00 % | - | +0.66 % | 0.025 | +0.00 % | 0/5 | 0/5 | - | 0.006307 |
| PatternDecoderBlockNoCrossAttention | 0.006473 | +0.04 % | 0.902 | +0.71 % | 0.015 | +0.72 % | 3/5 | 0/5 | 9, 11, 2, 5, 9 | 0.006297 |
| PatternDecoder_Conv | 0.006479 | +0.14 % | 0.642 | +0.81 % | 0.004 | +0.59 % | 3/5 | 1/5 | 8, 12, 6, 12, 8 | 0.006285 |
| DecoderOnlyTransformer_Conv | 0.006480 | +0.15 % | 0.640 | +0.82 % | 0.004 | +0.55 % | 3/5 | 1/5 | 11, 10, 8, 8, 7 | 0.006290 |
| Transformer_Conv | 0.006488 | +0.28 % | 0.443 | +0.95 % | 0.002 | +0.78 % | 3/5 | 0/5 | 13, 13, 9, 6, 10 | 0.006281 |

Per test window (seed-averaged forecasts; rank among the 13 models):

| Window | Zero MAE | Median vs zero | PatternDecoder vs zero (rank) | Models better than zero | Models better than median | Best model |
|---|---|---|---|---|---|---|
| 2023-08-07 to 2024-02-02 | 0.005927 | -0.02 % | +2.08 % (10) | 0/13 | 0/13 | PatternDecoderNoLstmBlock (+0.56 %) |
| 2024-02-05 to 2024-08-05 | 0.005872 | -0.55 % | +0.09 % (6) | 4/13 | 1/13 | Transformer (-0.60 %) |
| 2024-08-06 to 2025-02-04 | 0.006278 | -0.93 % | -0.64 % (5) | 13/13 | 0/13 | PatternDecoderNoConvBlock (-0.89 %) |
| 2025-02-05 to 2025-08-06 | 0.008978 | -0.73 % | -0.45 % (3) | 13/13 | 0/13 | Informer (-0.68 %) |
| 2025-08-07 to 2026-02-05 | 0.005289 | -1.06 % | -1.28 % (2) | 13/13 | 8/13 | PatternDecoderNoLstmBlock (-1.29 %) |

## DAX (`^GDAXI`), input_scale 100 (inputs in percent)

Test windows: 2023-08-02 to 2024-01-31; 2024-02-01 to 2024-07-31; 2024-08-01 to 2025-02-03; 2025-02-04 to 2025-08-04; 2025-08-05 to 2026-02-05. 10 seeds, 638 test days. Forecasts averaged over the seeds; single run: mean MAE of the seeds' own forecasts. p: Diebold-Mariano test of the pooled ensemble forecasts.

| Model | MAE | vs zero | p | vs median | p | single run vs zero | windows better than zero | windows better than median | rank per window | val MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| Informer | 0.006669 | -0.81 % | 0.225 | -0.28 % | 0.493 | -0.03 % | 5/5 | 4/5 | 3, 4, 11, 1, 10 | 0.006422 |
| Autoformer | 0.006675 | -0.72 % | 0.240 | -0.19 % | 0.605 | -0.46 % | 5/5 | 3/5 | 12, 6, 1, 2, 5 | 0.006406 |
| PatternDecoder_Autocorr | 0.006676 | -0.71 % | 0.275 | -0.18 % | 0.618 | -0.38 % | 5/5 | 3/5 | 8, 1, 3, 10, 13 | 0.006425 |
| PatternDecoder | 0.006677 | -0.69 % | 0.240 | -0.16 % | 0.642 | -0.25 % | 5/5 | 4/5 | 1, 8, 7, 4, 11 | 0.006437 |
| DecoderOnlyTransformer | 0.006678 | -0.69 % | 0.288 | -0.16 % | 0.674 | -0.28 % | 5/5 | 3/5 | 11, 2, 5, 3, 7 | 0.006431 |
| Transformer_Conv | 0.006678 | -0.68 % | 0.229 | -0.15 % | 0.647 | -0.17 % | 5/5 | 3/5 | 5, 10, 4, 7, 4 | 0.006415 |
| DecoderOnlyTransformer_Conv | 0.006682 | -0.62 % | 0.262 | -0.09 % | 0.789 | -0.15 % | 5/5 | 3/5 | 9, 3, 12, 11, 1 | 0.006433 |
| Transformer | 0.006683 | -0.61 % | 0.327 | -0.08 % | 0.836 | -0.18 % | 5/5 | 3/5 | 2, 11, 6, 9, 6 | 0.006433 |
| PatternDecoderNoLstmBlock | 0.006686 | -0.56 % | 0.368 | -0.03 % | 0.936 | -0.20 % | 5/5 | 3/5 | 7, 5, 10, 8, 8 | 0.006431 |
| Transformer_Logsparse | 0.006686 | -0.56 % | 0.364 | -0.03 % | 0.936 | -0.07 % | 5/5 | 1/5 | 10, 12, 2, 6, 3 | 0.006435 |
| Median | 0.006688 | -0.53 % | 0.279 | +0.00 % | - | -0.53 % | 5/5 | 0/5 | - | 0.006444 |
| PatternDecoderNoConvBlock | 0.006693 | -0.46 % | 0.412 | +0.07 % | 0.836 | +0.11 % | 4/5 | 3/5 | 6, 7, 9, 13, 12 | 0.006438 |
| PatternDecoder_Conv | 0.006693 | -0.45 % | 0.467 | +0.08 % | 0.836 | -0.01 % | 4/5 | 2/5 | 13, 9, 8, 12, 2 | 0.006440 |
| PatternDecoderBlockNoCrossAttention | 0.006698 | -0.37 % | 0.518 | +0.16 % | 0.679 | +0.24 % | 4/5 | 3/5 | 4, 13, 13, 5, 9 | 0.006450 |
| Naive | 0.006724 | +0.00 % | - | +0.53 % | 0.279 | +0.00 % | 0/5 | 0/5 | - | 0.006467 |

Per test window (seed-averaged forecasts; rank among the 13 models):

| Window | Zero MAE | Median vs zero | PatternDecoder vs zero (rank) | Models better than zero | Models better than median | Best model |
|---|---|---|---|---|---|---|
| 2023-08-02 to 2024-01-31 | 0.005367 | -0.39 % | -0.95 % (1) | 12/13 | 9/13 | PatternDecoder (-0.95 %) |
| 2024-02-01 to 2024-07-31 | 0.005577 | -0.88 % | -1.22 % (8) | 13/13 | 11/13 | PatternDecoder_Autocorr (-2.04 %) |
| 2024-08-01 to 2025-02-03 | 0.006577 | -0.68 % | -0.21 % (7) | 12/13 | 0/13 | Autoformer (-0.64 %) |
| 2025-02-04 to 2025-08-04 | 0.010280 | -0.31 % | -0.41 % (4) | 12/13 | 5/13 | Informer (-0.88 %) |
| 2025-08-05 to 2026-02-05 | 0.005835 | -0.54 % | -0.94 % (11) | 13/13 | 13/13 | DecoderOnlyTransformer_Conv (-1.60 %) |

## Summary

- **DAX:** Averaged over the seeds, all 13 models have a slightly lower test MAE than the zero forecast ($-0.14$ % to $-0.64$ %), but no difference is significant ($p \ge 0.28$). All except the Informer ($-0.11$ %, $p = 0.74$) are behind the median forecast, which beats the zero forecast in every window. Single runs are on average about as good as the zero forecast ($-0.13$ % to $+0.42$ %). The PatternDecoder is fifth of the 13 models ($-0.31$ % against the zero forecast, $+0.22$ % against the median forecast), with ranks between 3 and 12 in the five windows.
- **S&P 500:** The median forecast beats the zero forecast significantly ($-0.66$ %, $p = 0.025$), reflecting the strong rise of the index. The models lie between $-0.21$ % and $+0.28$ % of the zero forecast, none significantly, and 11 of the 13 are significantly worse than the median forecast ($p < 0.05$).
- **Input scaling:** With the return inputs in percent (`input_scale: 100`), the validation MAE of every model is lower (on average by 0.27 %), and so is its test MAE (on average by 0.30 %). Ten of the 13 models are then slightly ahead of the median forecast on the DAX, none significantly ($p \ge 0.49$). The published results use unscaled inputs; adopting the scaling would require retraining the study and the notebook.
- **Overall:** No model shows forecasting skill beyond the drift that holds across test windows or markets, and the rankings change from window to window.
