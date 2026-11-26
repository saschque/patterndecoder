# Seed and layer study

All 13 models in `models/*.yaml` were trained with the tuned config (MAE loss, learning rate 1e-4, dropout 0.3, d_model 16, units 64, early stopping on the validation MAE) for 10 seeds (1-10) and every pair of 1-3 encoder and 1-3 decoder layers: 1,170 trainings in total. The test metrics use the final closing price of 5 February 2026; an earlier version used a preliminary intraday value for that last test day, which raised every test MAE by 0.0000146 without changing any difference between models. All rows are in `seed_layer_study.csv`; code in `patterndecoder/study.py`, workflow "Seed and layer study".

## Protocol

1. One pair of encoder and decoder layers is used for all models, so that all models have the same depth. It is the pair with the lowest mean validation MAE over all models and seeds: 2 encoder and 1 decoder layer (1/1 and 3/1 are practically equal). The validation set is the last 20 % of the training period; the test set plays no part in the choice. The encoder count only changes the architecture of the models with an encoder stack (Transformer variants, Informer, Autoformer). The PatternDecoder and decoder-only models have none; for them it only changes the random initialisation, because the unused encoder is still constructed.
2. For that pair, the test MAE of every model is averaged over the 10 seeds. The zero forecast (random walk for the price) has test MAE 0.007584.

## Result

| Model | enc/dec | mean test MAE | vs zero | sd | below zero | mean rank |
|---|---|---|---|---|---|---|
| PatternDecoderNoLstmBlock | 2/1 | 0.007559 | -0.33 % | 0.27 % | 9/10 | 5.0 |
| PatternDecoder | 2/1 | 0.007561 | -0.31 % | 0.24 % | 9/10 | 6.1 |
| DecoderOnlyTransformer | 2/1 | 0.007566 | -0.23 % | 0.23 % | 9/10 | 6.5 |
| DecoderOnlyTransformer_Conv | 2/1 | 0.007567 | -0.23 % | 0.18 % | 9/10 | 6.0 |
| Transformer_Conv | 2/1 | 0.007569 | -0.19 % | 0.20 % | 9/10 | 7.5 |
| Autoformer | 2/1 | 0.007571 | -0.18 % | 0.23 % | 7/10 | 7.9 |
| PatternDecoder_Conv | 2/1 | 0.007571 | -0.17 % | 0.28 % | 7/10 | 7.4 |
| PatternDecoderNoConvBlock | 2/1 | 0.007572 | -0.16 % | 0.25 % | 8/10 | 7.6 |
| Transformer | 2/1 | 0.007573 | -0.15 % | 0.35 % | 7/10 | 6.8 |
| PatternDecoder_Autocorr | 2/1 | 0.007574 | -0.13 % | 0.27 % | 8/10 | 7.5 |
| PatternDecoderBlockNoCrossAttention | 2/1 | 0.007576 | -0.11 % | 0.56 % | 6/10 | 7.9 |
| Transformer_Logsparse | 2/1 | 0.007576 | -0.11 % | 0.33 % | 7/10 | 8.2 |
| Informer | 2/1 | 0.007577 | -0.09 % | 0.59 % | 7/10 | 6.6 |

"sd" is the standard deviation over the seeds, "below zero" counts the seeds in which the model beats the zero forecast, and "mean rank" is the average rank among the 13 models per seed.

## Reading the result

- **All 13 models** are below the zero forecast on average, by 0.09 % to 0.33 % in MAE. The PatternDecoder (-0.31 %, 9 of 10 seeds below zero, t = -4.0 over the seeds) is second, practically tied with PatternDecoderNoLstmBlock (-0.33 %).
- **No pair of models can be told apart.** Paired over the seeds, none of the differences between the PatternDecoder and another model is clear (all |t| < 1.7). Depending on the seed the PatternDecoder ranks anywhere from 1st (seeds 5 and 10) to 13th (seed 9).
- **Ablation.** With equal depth, neither removing the LSTM, the convolution nor the cross-attention makes a measurable difference.
- **The seed spread is not the whole uncertainty.** All runs use the same 353 test days. The sampling uncertainty of the test period itself, about ±18 % for the MAE of the zero forecast in a block bootstrap, is not included.
- **Median forecast.** A constant forecast of the median training return reaches test MAE 0.007560, as good as the best models: with an MAE loss they mainly learn to predict the median return. Beating the zero forecast therefore largely reflects the positive median of the period, not forecasting of individual days.
- **Per-model layers.** An earlier version of this summary chose the layers per model. With that choice the PatternDecoder (1/1) had the lowest mean test MAE (-0.36 %). It was replaced by one shared pair so that all models are compared at the same depth.
