# Seed and layer study

All 13 models in `models/*.yaml` were trained with the tuned config (MAE loss, learning rate 1e-4, dropout 0.3, d_model 16, units 64, early stopping on the validation MAE) for 10 seeds (1-10) and every pair of 1-3 encoder and 1-3 decoder layers: 1,170 trainings in total. All rows are in `seed_layer_study.csv`; code in `patterndecoder/study.py`, workflow "Seed and layer study".

## Protocol

1. For every model, the layer pair with the lowest mean validation MAE over the 10 seeds is chosen. The validation set is the last 20 % of the training period; the test set plays no part in the choice.
2. For the chosen pair, the test MAE is averaged over the 10 seeds. The zero forecast (random walk for the price) has test MAE 0.007599.

## Result

| Model | enc/dec | mean test MAE | vs zero | sd | below zero | mean rank |
|---|---|---|---|---|---|---|
| PatternDecoder | 1/1 | 0.007572 | -0.36 % | 0.22 % | 9/10 | 4.1 |
| PatternDecoderNoLstmBlock | 2/1 | 0.007574 | -0.33 % | 0.26 % | 9/10 | 5.8 |
| DecoderOnlyTransformer | 3/2 | 0.007574 | -0.32 % | 0.38 % | 8/10 | 6.3 |
| Transformer_Logsparse | 1/1 | 0.007577 | -0.28 % | 0.36 % | 8/10 | 6.4 |
| DecoderOnlyTransformer_Conv | 1/1 | 0.007577 | -0.28 % | 0.16 % | 10/10 | 5.5 |
| Autoformer | 2/1 | 0.007582 | -0.21 % | 0.21 % | 9/10 | 7.1 |
| PatternDecoder_Conv | 2/1 | 0.007585 | -0.17 % | 0.28 % | 7/10 | 7.5 |
| Transformer_Conv | 1/1 | 0.007586 | -0.17 % | 0.22 % | 8/10 | 7.4 |
| PatternDecoderNoConvBlock | 3/1 | 0.007587 | -0.16 % | 0.24 % | 8/10 | 7.6 |
| PatternDecoder_Autocorr | 2/1 | 0.007587 | -0.16 % | 0.26 % | 8/10 | 7.5 |
| Informer | 2/1 | 0.007594 | -0.07 % | 0.62 % | 7/10 | 6.6 |
| Transformer | 1/2 | 0.007618 | +0.26 % | 0.58 % | 4/10 | 10.1 |
| PatternDecoderBlockNoCrossAttention | 1/3 | 0.007619 | +0.26 % | 0.66 % | 4/10 | 9.1 |

"sd" is the standard deviation over the seeds, "below zero" counts the seeds in which the model beats the zero forecast, and "mean rank" is the average rank among the 13 models per seed.

## Reading the result

- **PatternDecoder** (1 encoder, 1 decoder layer) has the lowest mean test MAE and the best mean rank. It beats the zero forecast in 9 of 10 seeds, on average by 0.36 % in MAE (t = -5.0 over the seeds) and in RMSE as well.
- **Against the other models** the lead is small. Paired over the seeds it is clear only against the Transformer (0.61 %, t = -3.4) and the variant without cross-attention (0.62 %, t = -3.2). Against PatternDecoderNoLstmBlock, DecoderOnlyTransformer, DecoderOnlyTransformer_Conv and Transformer_Logsparse the difference is 0.03-0.07 % and not distinguishable from seed noise.
- **Seed noise is larger than most gaps.** PatternDecoder alone ranges from rank 1 to rank 10 depending on the seed.
- **The seed spread is not the whole uncertainty.** All runs use the same 353 test days. The sampling uncertainty of the test period itself, about ±18 % for the MAE of the zero forecast in a block bootstrap, is not included.
- **Median forecast.** A constant forecast of the median training return reaches test MAE 0.007574. The best models are at this level: with an MAE loss they mainly learn to predict the median return. Beating the zero forecast therefore largely reflects the positive median of the period, not forecasting of individual days.
