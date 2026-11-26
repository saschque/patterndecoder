# Seed and layer study

All 13 models in `models/*.yaml` were trained with the tuned config (MAE loss, learning rate 1e-4, dropout 0.3, d_model 16, units 64, early stopping on the validation MAE) and the current setup (inputs in percent, last time step as readout) for 10 seeds (1-10) and every pair of 1-3 encoder and 1-3 decoder layers: 1,170 trainings in total. The test metrics use the final closing price of 5 February 2026. All rows are in `seed_layer_study.csv`; code in `patterndecoder/study.py`, workflow "Seed and layer study". The results of the earlier setup (mean of all time steps as readout, unscaled inputs) are summarized at the end; their rows are in `seed_layer_study_mean_readout.csv`.

## Protocol

1. One pair of encoder and decoder layers is used for all models, so that all models have the same depth. It is the pair with the lowest mean validation MAE over all models and seeds: 2 encoder and 1 decoder layer (1/1 and 3/1 are practically equal). The validation set is the last 20 % of the training period; the test set plays no part in the choice. The encoder count only affects the models with an encoder stack (Transformer variants, Informer, Autoformer); the PatternDecoder and decoder-only models have none, so it does not affect them.
2. For that pair, the test MAE of every model is averaged over the 10 seeds. The zero forecast (random walk for the price) has test MAE 0.007584.

## Result

| Model | enc/dec | mean test MAE | vs zero | sd | below zero | mean rank |
|---|---|---|---|---|---|---|
| DecoderOnlyTransformer_Conv | 2/1 | 0.007541 | -0.57 % | 0.36 % | 10/10 | 4.9 |
| Transformer_Conv | 2/1 | 0.007544 | -0.53 % | 0.35 % | 9/10 | 5.6 |
| PatternDecoder_Autocorr | 2/1 | 0.007554 | -0.40 % | 0.32 % | 9/10 | 7.0 |
| Autoformer | 2/1 | 0.007556 | -0.37 % | 0.55 % | 8/10 | 7.8 |
| PatternDecoder_Conv | 2/1 | 0.007556 | -0.37 % | 0.48 % | 8/10 | 6.4 |
| Transformer_Logsparse | 2/1 | 0.007557 | -0.36 % | 0.62 % | 6/10 | 6.7 |
| PatternDecoderNoConvBlock | 2/1 | 0.007560 | -0.32 % | 0.27 % | 8/10 | 6.6 |
| Transformer | 2/1 | 0.007563 | -0.28 % | 0.47 % | 9/10 | 7.0 |
| PatternDecoderNoLstmBlock | 2/1 | 0.007563 | -0.28 % | 0.47 % | 8/10 | 6.8 |
| DecoderOnlyTransformer | 2/1 | 0.007565 | -0.25 % | 0.37 % | 7/10 | 7.4 |
| PatternDecoderBlockNoCrossAttention | 2/1 | 0.007569 | -0.20 % | 0.43 % | 7/10 | 8.1 |
| Informer | 2/1 | 0.007570 | -0.18 % | 0.55 % | 6/10 | 8.0 |
| PatternDecoder | 2/1 | 0.007576 | -0.11 % | 0.40 % | 4/10 | 8.7 |

"sd" is the standard deviation over the seeds, "below zero" counts the seeds in which the model beats the zero forecast, and "mean rank" is the average rank among the 13 models per seed.

## Reading the result

- **All 13 models** are below the zero forecast on average, by 0.11 % to 0.57 % in MAE. DecoderOnlyTransformer_Conv is best (-0.57 %, 10 of 10 seeds below zero, t = -5.0 over the seeds), followed by Transformer_Conv (-0.53 %). The PatternDecoder is last (-0.11 %, 4 of 10 seeds below zero, t = -0.9).
- **Few pairs of models can be told apart.** Paired over the seeds, the PatternDecoder is behind DecoderOnlyTransformer_Conv (t = 2.6) and Transformer_Conv (t = 2.1); with twelve comparisons, even these differences are not conclusive. Depending on the seed the PatternDecoder ranks anywhere from 1st (seed 10) to 13th (seed 1); with seed 5, the seed of the single run in the notebook, it is 2nd.
- **Ablation.** The variants without the LSTM, the convolution or the cross-attention are all slightly ahead of the full PatternDecoder (-0.28 %, -0.32 % and -0.20 %), but not clearly.
- **The seed spread is not the whole uncertainty.** All runs use the same 353 test days. The sampling uncertainty of the test period itself, about ±18 % for the MAE of the zero forecast in a block bootstrap, is not included.
- **Median forecast.** A constant forecast of the median training return reaches test MAE 0.007559 (-0.33 %), as good as the middle of the field; the best model is 0.24 % better. Beating the zero forecast therefore largely reflects the positive median of the period, not forecasting of individual days.

## Earlier setup

With the mean of all time steps as readout and unscaled inputs (the setup published until 26 November 2026), the same protocol also chose 2 encoder and 1 decoder layer. All models were below the zero forecast (-0.09 % to -0.33 %), and the PatternDecoder was second (-0.31 %, 9 of 10 seeds below zero), practically tied with PatternDecoderNoLstmBlock (-0.33 %). In that setup, the unused encoder was still constructed in the PatternDecoder and decoder-only models, so their encoder count changed their random initialisation. The rankings change with the setup; the overall result, no model clearly better than the others or than the median forecast, does not.

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
