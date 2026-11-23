# Seed and layer study

All 13 models in `models/*.yaml` were trained with the tuned config (MAE loss, learning rate 1e-4, dropout 0.3, d_model 16, units 64, early stopping on the validation MAE) for 10 seeds (1-10) and every pair of 1-3 encoder and 1-3 decoder layers: 1,170 trainings in total. All rows are in `seed_layer_study.csv`; code in `patterndecoder/study.py`, workflow "Seed and layer study".

## Protocol

1. One pair of encoder and decoder layers is used for all models, so that all models have the same depth. It is the pair with the lowest mean validation MAE over all models and seeds: 2 encoder and 1 decoder layer (1/1 and 3/1 are practically equal). The validation set is the last 20 % of the training period; the test set plays no part in the choice. Decoder-only models ignore the encoder setting.
2. For that pair, the test MAE of every model is averaged over the 10 seeds. The zero forecast (random walk for the price) has test MAE 0.007599.

## Result

| Model | enc/dec | mean test MAE | vs zero | sd | below zero | mean rank |
|---|---|---|---|---|---|---|
| PatternDecoderNoLstmBlock | 2/1 | 0.007574 | -0.33 % | 0.26 % | 9/10 | 5.0 |
| PatternDecoder | 2/1 | 0.007575 | -0.31 % | 0.24 % | 9/10 | 6.1 |
| DecoderOnlyTransformer | 2/1 | 0.007581 | -0.23 % | 0.23 % | 9/10 | 6.5 |
| DecoderOnlyTransformer_Conv | 2/1 | 0.007581 | -0.23 % | 0.18 % | 9/10 | 6.0 |
| Autoformer | 2/1 | 0.007582 | -0.21 % | 0.21 % | 9/10 | 7.4 |
| Transformer_Conv | 2/1 | 0.007584 | -0.19 % | 0.19 % | 9/10 | 7.6 |
| PatternDecoder_Conv | 2/1 | 0.007585 | -0.17 % | 0.28 % | 7/10 | 7.4 |
| PatternDecoderNoConvBlock | 2/1 | 0.007587 | -0.16 % | 0.25 % | 8/10 | 7.6 |
| PatternDecoder_Autocorr | 2/1 | 0.007587 | -0.16 % | 0.26 % | 8/10 | 7.6 |
| Transformer | 2/1 | 0.007588 | -0.15 % | 0.34 % | 7/10 | 7.0 |
| PatternDecoderBlockNoCrossAttention | 2/1 | 0.007590 | -0.11 % | 0.56 % | 6/10 | 7.8 |
| Transformer_Logsparse | 2/1 | 0.007590 | -0.11 % | 0.33 % | 7/10 | 8.4 |
| Informer | 2/1 | 0.007594 | -0.07 % | 0.62 % | 7/10 | 6.6 |

"sd" is the standard deviation over the seeds, "below zero" counts the seeds in which the model beats the zero forecast, and "mean rank" is the average rank among the 13 models per seed.

## Reading the result

- **All 13 models** are below the zero forecast on average, by 0.07 % to 0.33 % in MAE. The PatternDecoder (-0.31 %, 9 of 10 seeds below zero, t = -4.0 over the seeds) is second, practically tied with PatternDecoderNoLstmBlock (-0.33 %).
- **No pair of models can be told apart.** Paired over the seeds, none of the differences between the PatternDecoder and another model is clear (all |t| < 1.6). Depending on the seed the PatternDecoder ranks anywhere from 1st (seeds 5 and 10) to 13th (seed 9).
- **Ablation.** With equal depth, neither removing the LSTM, the convolution nor the cross-attention makes a measurable difference.
- **The seed spread is not the whole uncertainty.** All runs use the same 353 test days. The sampling uncertainty of the test period itself, about ±18 % for the MAE of the zero forecast in a block bootstrap, is not included.
- **Median forecast.** A constant forecast of the median training return reaches test MAE 0.007574, as good as the best models: with an MAE loss they mainly learn to predict the median return. Beating the zero forecast therefore largely reflects the positive median of the period, not forecasting of individual days.
- **Per-model layers.** An earlier version of this summary chose the layers per model. With that choice the PatternDecoder (1/1) had the lowest mean test MAE (-0.36 %). It was replaced by one shared pair so that all models are compared at the same depth.
