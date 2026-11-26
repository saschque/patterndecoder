# Positive control: synthetic series with known patterns

Results of the walk-forward evaluation (`patterndecoder/robustness.py`) on the synthetic series of `patterndecoder/synthetic.py`, computed with the workflow "Walk-forward evaluation" on 26 November 2026. The DAX returns contain almost no predictable signal, so a tie between the models says little about the architectures. These series have a signal of known form and strength, which shows whether the models and the training setup can learn a signal at all and which architecture learns which kind of signal. Their design was fixed before any model was trained on them.

- **Series:** an AR(1) process (coefficient 0.3); a recurring 10-day pattern (a jump, momentum, then a reversal) that starts with probability 1/20 on any day without a running pattern; and a sine wave with a period of 23.7 trading days and an amplitude of half the noise level. All have Student-t noise with a standard deviation of 0.01, a drift of 0.0004 and the 1,305 business days of the DAX sample period.
- **Measure:** the share of the oracle's advantage over the zero forecast that a model reaches. 100 % is as good as the oracle forecast from the true structure, 0 % as good as the zero forecast; negative values are worse than the zero forecast. Each cell shows the seed ensemble / the mean of the single runs, pooled over the test windows starting at 50 %, 70 % and 90 % of the data (each 10 % long), with five seeds.
- **Reference:** in 200 simulated samples of the same length, a linear AR(1) fitted by least squares reaches almost all of the oracle's advantage on the AR series (MAE −5.8 % against −6.0 % for the oracle, relative to the zero forecast) but little on the pattern (−1.5 % against −9.0 %) and the sine wave (−0.8 % against −8.5 %).
- **Settings:** the published settings (input_scale 1, mean of all time steps as readout); percent inputs (input_scale 100); percent inputs with the last time step as readout; percent inputs with target_scale 100; and all three. The first two ran before the change that stops models from building layers they do not use, which changes the random initialization but not the structure of the models.

## SYNTH-AR (AR(1))

| Model | published | percent inputs | percent + last step | percent + target scale | all three |
|---|---|---|---|---|---|
| PatternDecoderNoConvBlock | 1 / -8 | -29 / -44 | 76 / 58 | 0 / -0 | 64 / 54 |
| DecoderOnlyTransformer | 2 / -3 | -49 / -59 | 65 / 47 | -4 / -6 | 43 / 36 |
| PatternDecoderNoLstmBlock | 3 / -3 | -66 / -76 | 64 / 53 | 0 / -0 | 42 / 36 |
| DecoderOnlyTransformer_Conv | -3 / -10 | -109 / -119 | 45 / 26 | -22 / -24 | 45 / 40 |
| Autoformer | 10 / 7 | -52 / -56 | 42 / 26 | -2 / -2 | 23 / 20 |
| PatternDecoder_Conv | -3 / -11 | -147 / -165 | 35 / 21 | -15 / -17 | 31 / 25 |
| PatternDecoder | 7 / 0 | -59 / -68 | 33 / 18 | -7 / -9 | 36 / 29 |
| Transformer_Conv | -2 / -12 | -104 / -121 | 26 / 12 | -47 / -50 | -2 / -9 |
| Transformer_Logsparse | 0 / -4 | -112 / -123 | 23 / 1 | -21 / -23 | 50 / 39 |
| PatternDecoder_Autocorr | -10 / -17 | -69 / -82 | 15 / -5 | -1 / -2 | 21 / 18 |
| PatternDecoderBlockNoCrossAttention | -4 / -14 | -12 / -36 | 5 / -19 | 0 / -3 | 27 / 18 |
| Transformer | 5 / -1 | -45 / -57 | 4 / -13 | -31 / -33 | 22 / 17 |
| Informer | -17 / -29 | -95 / -112 | -9 / -33 | -54 / -56 | -73 / -80 |

## SYNTH-MOTIF (recurring pattern)

| Model | published | percent inputs | percent + last step | percent + target scale | all three |
|---|---|---|---|---|---|
| PatternDecoder | 3 / -1 | 3 / -2 | 29 / 22 | 4 / 3 | 13 / 11 |
| DecoderOnlyTransformer_Conv | 2 / -1 | 5 / 1 | 27 / 18 | 2 / 1 | 17 / 14 |
| Informer | -3 / -7 | -7 / -13 | 25 / 18 | -11 / -15 | 3 / -2 |
| PatternDecoderNoLstmBlock | 5 / 0 | 4 / -1 | 25 / 17 | 3 / 2 | 14 / 13 |
| PatternDecoder_Conv | -2 / -5 | 2 / -6 | 24 / 16 | 2 / 0 | 16 / 14 |
| Transformer_Conv | -5 / -9 | -7 / -12 | 23 / 14 | -3 / -7 | 7 / 4 |
| Transformer | -1 / -8 | 2 / -4 | 22 / 16 | 0 / -0 | 12 / 9 |
| DecoderOnlyTransformer | 2 / -2 | 1 / -5 | 22 / 15 | 3 / 2 | 13 / 12 |
| PatternDecoderNoConvBlock | -4 / -9 | -6 / -11 | 22 / 15 | 0 / -1 | 10 / 9 |
| Transformer_Logsparse | 3 / -2 | 4 / -0 | 21 / 12 | 7 / 5 | 14 / 12 |
| PatternDecoder_Autocorr | -2 / -11 | -6 / -10 | 20 / 15 | -1 / -2 | 14 / 13 |
| Autoformer | 5 / 3 | -3 / -6 | 18 / 11 | -1 / -1 | 13 / 13 |
| PatternDecoderBlockNoCrossAttention | -2 / -7 | 3 / -5 | 10 / 4 | 4 / 2 | 8 / 5 |

## SYNTH-CYCLE (sine wave)

| Model | published | percent inputs | percent + last step | percent + target scale | all three |
|---|---|---|---|---|---|
| PatternDecoderBlockNoCrossAttention | 48 / 22 | 30 / -19 | 45 / 14 | 53 / 31 | 51 / 35 |
| Transformer | 53 / 24 | 19 / -23 | -3 / -17 | 17 / 9 | 7 / 3 |
| DecoderOnlyTransformer | 22 / 9 | 10 / -2 | -6 / -17 | 11 / 10 | 2 / -3 |
| Informer | 7 / -11 | -23 / -37 | -8 / -15 | 4 / 2 | -9 / -10 |
| PatternDecoder_Conv | 48 / 19 | 10 / -28 | -9 / -22 | 21 / 16 | 5 / -1 |
| PatternDecoder_Autocorr | 8 / -3 | -8 / -29 | -10 / -17 | 4 / 3 | -5 / -6 |
| PatternDecoderNoConvBlock | 36 / 19 | 25 / 9 | -12 / -16 | 7 / 6 | 1 / -1 |
| Transformer_Logsparse | 8 / -4 | -2 / -16 | -14 / -27 | 21 / 18 | 7 / 2 |
| PatternDecoderNoLstmBlock | 5 / -8 | 15 / 4 | -14 / -23 | 13 / 12 | 1 / -2 |
| DecoderOnlyTransformer_Conv | 22 / -4 | 9 / -12 | -15 / -24 | 18 / 14 | 6 / 3 |
| PatternDecoder | 13 / 5 | 16 / -2 | -16 / -27 | 10 / 9 | -1 / -6 |
| Transformer_Conv | 6 / -21 | -36 / -80 | -18 / -29 | 2 / -3 | 2 / -5 |
| Autoformer | -6 / -8 | -23 / -25 | -22 / -31 | -0 / -1 | -1 / -2 |

## Choice of the setup on the DAX

Walk-forward evaluation on the DAX returns with five seeds and five windows; mean over the 13 models:

| Setting | Validation MAE | Lower validation MAE than percent inputs | Test MAE vs zero forecast | PatternDecoder test vs zero |
|---|---|---|---|---|
| percent inputs | 0.0064286 | - | −0.56 % | −0.54 % |
| percent inputs + last step | 0.0063926 | 13 of 13 models | −0.48 % | −0.15 % |
| percent inputs + target scale | 0.0064382 | 2 of 13 | −0.44 % | −0.40 % |
| all three | 0.0064199 | 7 of 13 | −0.58 % | −0.52 % |

The constant median forecast has a validation MAE of 0.0064445. By its validation MAE, percent inputs with the last-step readout are the new setup: they are better for every model. On the DAX test windows the mean result of the models is about the same as before, and for the PatternDecoder it is worse.

## Findings

1. **With the published settings, the models hardly learn the AR signal or the pattern.** Their seed ensembles reach at most 10 % of the oracle's advantage on the AR series and 5 % on the pattern, and most single runs are worse than the zero forecast, while their validation MAE looks favourable, because the best epoch is chosen on a small validation set. Only the sine wave is partly learned (up to 53 %, by the Transformer).
2. **Percent inputs alone make the AR results worse:** the models then fit the noise more strongly.
3. **With the last time step as readout, most models learn the AR signal** (up to 76 %, by the PatternDecoder variant without convolution) **and part of the pattern** (10 % to 29 %). The decoders mask their self-attention causally, so only the last step sees the whole window, and averaging over all steps diluted the signal. The sine wave, whose phase has to be estimated from the whole window, is then learned worse by almost all models; only the PatternDecoder variant without cross-attention keeps a high share (45 %).
4. **Scaling the targets does not help.**
5. **On the pattern series, the PatternDecoder reaches the highest share with the new setup** (29 %), narrowly ahead of DecoderOnlyTransformer_Conv (27 %). This exploratory result is checked with new seeds below.

## Confirmatory run on the pattern series

Hypothesis, fixed before the run: with the new setup, the PatternDecoder has the highest oracle share of the 13 models on SYNTH-MOTIF (seed ensemble, pooled over all five test windows), with seeds 6 to 10, which had not been used before.

Test windows: 2023-08-08 to 2024-02-06; 2024-02-07 to 2024-08-06; 2024-08-07 to 2025-02-05; 2025-02-06 to 2025-08-06; 2025-08-07 to 2026-02-05. 5 seeds, 653 test days. Forecasts averaged over the seeds; single run: mean MAE of the seeds' own forecasts. p: Diebold-Mariano test of the pooled ensemble forecasts. Oracle share: part of the oracle's advantage over the zero forecast reached by the ensemble / by single runs.

| Model | MAE | vs zero | p | vs median | p | single run vs zero | windows better than zero | windows better than median | rank per window | val MAE | oracle share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Oracle | 0.007512 | -7.96 % | 0.000 | -7.88 % | 0.000 | -7.96 % | 4/5 | 4/5 | - | 0.008005 | 100 % / 100 % |
| PatternDecoder | 0.008025 | -1.67 % | 0.009 | -1.58 % | 0.014 | -0.91 % | 3/5 | 3/5 | 2, 4, 5, 5, 2 | 0.008460 | 21 % / 12 % |
| DecoderOnlyTransformer | 0.008031 | -1.59 % | 0.011 | -1.51 % | 0.016 | -0.90 % | 3/5 | 3/5 | 5, 3, 1, 10, 6 | 0.008464 | 20 % / 11 % |
| Transformer_Conv | 0.008047 | -1.40 % | 0.052 | -1.32 % | 0.069 | -0.71 % | 3/5 | 3/5 | 10, 6, 3, 4, 5 | 0.008455 | 18 % / 9 % |
| PatternDecoder_Autocorr | 0.008055 | -1.30 % | 0.099 | -1.21 % | 0.124 | -0.78 % | 3/5 | 3/5 | 8, 13, 2, 1, 4 | 0.008445 | 16 % / 10 % |
| PatternDecoder_Conv | 0.008056 | -1.29 % | 0.088 | -1.21 % | 0.110 | -0.57 % | 3/5 | 3/5 | 13, 7, 6, 13, 1 | 0.008469 | 16 % / 7 % |
| PatternDecoderNoLstmBlock | 0.008057 | -1.27 % | 0.033 | -1.19 % | 0.047 | -0.53 % | 3/5 | 3/5 | 3, 5, 11, 7, 7 | 0.008469 | 16 % / 7 % |
| Autoformer | 0.008059 | -1.25 % | 0.121 | -1.17 % | 0.146 | -0.71 % | 3/5 | 3/5 | 9, 11, 4, 3, 8 | 0.008437 | 16 % / 9 % |
| Transformer_Logsparse | 0.008060 | -1.24 % | 0.046 | -1.16 % | 0.062 | -0.57 % | 4/5 | 4/5 | 6, 1, 12, 8, 9 | 0.008460 | 16 % / 7 % |
| DecoderOnlyTransformer_Conv | 0.008065 | -1.18 % | 0.126 | -1.10 % | 0.159 | -0.54 % | 3/5 | 3/5 | 11, 10, 7, 12, 3 | 0.008463 | 15 % / 7 % |
| Transformer | 0.008071 | -1.11 % | 0.077 | -1.02 % | 0.106 | -0.49 % | 3/5 | 3/5 | 7, 2, 8, 6, 12 | 0.008469 | 14 % / 6 % |
| PatternDecoderNoConvBlock | 0.008073 | -1.08 % | 0.122 | -1.00 % | 0.155 | -0.67 % | 4/5 | 4/5 | 1, 12, 9, 11, 10 | 0.008494 | 14 % / 8 % |
| Informer | 0.008080 | -1.00 % | 0.067 | -0.92 % | 0.090 | -0.01 % | 3/5 | 3/5 | 12, 8, 10, 2, 11 | 0.008480 | 13 % / 0 % |
| PatternDecoderBlockNoCrossAttention | 0.008146 | -0.18 % | 0.702 | -0.10 % | 0.842 | +0.58 % | 3/5 | 3/5 | 4, 9, 13, 9, 13 | 0.008524 | 2 % / -7 % |
| Median | 0.008154 | -0.09 % | 0.217 | +0.00 % | - | -0.09 % | 2/5 | 0/5 | - | 0.008575 | 1 % / 1 % |
| Naive | 0.008161 | +0.00 % | - | +0.09 % | 0.217 | +0.00 % | 0/5 | 3/5 | - | 0.008557 | 0 % / 0 % |

The hypothesis holds: the PatternDecoder reaches the highest share (21 % for the ensemble, 12 % for single runs). Its lead over the decoder-only Transformer (20 % and 11 %; MAE 0.008025 against 0.008031) is very small, though. Across the exploratory and the confirmatory run, the PatternDecoder is the model that learns the most of the recurring pattern, but it cannot be called clearly better than the next models.
