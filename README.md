# Transformer-Based Architectures for Temporal Forecasting: A Study on Financial Time Series Data
![Build Status](https://github.com/saschque/patterndecoder/actions/workflows/pylint.yml/badge.svg)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green.svg)](https://shields.io/)

This study examines the effectiveness of Transformer-based models for short-term financial time series forecasting, specifically focusing on one-day-ahead prediction using log returns derived from daily closing prices of the DAX40 index. Various Transformer architectures are evaluated for their predictive performance, including the standard Transformer with multiple attention mechanisms (full, convolutional, LogSparse), specialized variants (Informer, Autoformer), and the newly introduced PatternDecoder architecture, a hybrid decoder-only Transformer. Additionally, baseline models (Naïve, Conv1D-LSTM) are analyzed to provide performance benchmarks. The models are evaluated using MAE and RMSE across 30-day input windows for single-step forecasting.  

Averaged over ten training seeds, with the same depth for all models (chosen on a validation set), all Transformer-type models - the PatternDecoder included - are slightly better than the zero-return (random-walk) baseline, by 0.1 % to 0.3 % in MAE. The PatternDecoder is among the best, but the differences between the models are within seed noise, and a constant forecast of the median return is about as good as the best models. This is consistent with the noise-dominated nature of one-day-ahead returns and with simple baseline models remaining highly relevant in this domain. The input also contains **calendar features** (sine/cosine encodings of day of week, day of month and month); their effect was not tested separately.

## Introduction

Time series forecasting is a critical task in financial analysis, enabling traders, investors, and financial institutions to make informed decisions based on predicted market movements. Traditional statistical methods, including naïve approaches, have limitations in capturing complex non-linear patterns and short-term dependencies in financial data. Deep learning approaches, particularly transformer-based architectures, offer promising alternatives due to their ability to model sequential data effectively.

## Model Architectures

### Transformer

The **Transformer** model, initially introduced in *[Attention is All You Need](https://arxiv.org/abs/1706.03762)*, leverages self-attention mechanisms to capture short-term dependencies and complex patterns in sequential data. Unlike recurrent neural networks, transformers process the entire sequence simultaneously, allowing them to capture relationships between any positions in the sequence regardless of their distance. Key components include:

- **Multi-Head Attention**: Allows the model to jointly attend to information from different representation subspaces
- **Positional Encoding**: Provides position information since transformers lack inherent sequential processing
- **Feed-Forward Networks**: Process the attention output through non-linear transformations

In this study, the Transformer is evaluated with full, convolutional, and LogSparse attention mechanisms.

### Time Series Transformer

The **Time Series Transformer**, as proposed in *[Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting](https://arxiv.org/abs/1907.00235)*, enhances the standard transformer architecture by addressing challenges specific to time series data:

- **Convolutional Self-Attention**: Incorporates local pattern recognition capabilities using convolutional operations
- **LogSparse Attention**: Reduces memory complexity from $O(L^2)$ to $O(L(\log L)^2)$ by using a logarithmic sampling strategy
- **Memory Efficiency**: Enables processing of longer sequences with limited computational resources

### Informer

The **Informer** architecture, introduced in *[Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436)*, improves transformer efficiency for long sequence time-series forecasting through:

- **ProbSparse Attention**: Reduces computational complexity to $O(L(\log L))$ by letting only the most informative queries attend fully
- **Distilling Mechanism**: Progressively halves the sequence length at each layer through convolutional operations
- **Direct Multi-step Forecasting** (original paper): Predicts the entire output sequence in one forward pass; not used here, as this study forecasts one step ahead ($h=1$)

### Autoformer

The **Autoformer**, as presented in *[Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://arxiv.org/abs/2106.13008)*, introduces components for time series modeling:

- **Series Decomposition**: Separates time series into trend and seasonal components using moving average filters
- **Auto-Correlation Mechanism**: Replaces self-attention with an auto-correlation mechanism that captures lag-based periodicities
- **Progressive Decomposition**: Applies the decomposition block repeatedly inside the encoder and decoder layers; the decoder accumulates the trend components extracted at each layer

### Decoder-Only Transformer
**Decoder-only transformers** are neural network architectures composed exclusively of stacked decoder blocks, each containing masked self-attention and feed-forward layers, optimized for autoregressive tasks such as time series prediction. This architecture underpins models like GPT and PatternDecoder, enabling efficient autoregressive forecasting through residual connections and layer normalization. (see *[Q. Chen (2025)](https://arxiv.org/html/2504.16361v1)* )

### PatternDecoder
A decoder-only transformer model specifically designed for immediate-term financial time series forecasting. The PatternDecoder architecture addresses the specific requirements of short-sequence financial forecasting and is evaluated with full, convolutional, and Auto-Correlation attention mechanisms. Additional calendar features (cyclical day-of-week, day-of-month and month encodings) are included as inputs; whether they help was not tested.

### Baseline Models
For the study, **Naïve** (zero return, i.e. a random walk for the price), **Moving Average** and **Conv1D-LSTM** models are used as baselines to evaluate forecast quality. These models provide reference points for assessing whether Transformer-based models deliver meaningful improvements in short-term financial forecasting.

## Installation

To run the code in this repository, clone the repository and install the package and its dependencies:

```bash
git clone https://github.com/saschque/patterndecoder.git
cd patterndecoder
python setup.py install
# or
# make init
```

## Dataset

The study uses historical daily closing prices of the DAX40 index spanning from February 5, 2021, to February 5, 2026, providing 1,275 trading days of information. The data is preprocessed by:
1. Converting prices to daily log returns using $r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$, which makes the series approximately stationary in the mean (the ADF and KPSS tests agree; the variance is not constant)

2. Using a $30$-day input window ($L=w=30$) for 1-day ahead ($h=1$) prediction

3. Splitting chronologically into training ($70\%$) and testing ($30\%$) sets, yielding $892$ training and $383$ test observations

4. Adding cyclical calendar features (sine/cosine of day of week, day of month and month) via utils.py as further inputs, so that each input step has 7 features. The return inputs can be multiplied by `input_scale` in `config/config.yaml` (default 1, as in the published results; the targets stay unscaled)

The price data are not included in the repository. On first use, `get_stock_data` (`patterndecoder/data.py`) downloads the daily DAX index data (`^GDAXI`, 5 February 2021 to 5 February 2026) from Yahoo Finance with yfinance and caches them in `data/`. A checksum in `config/config.yaml` (`data_checksum`) issues a warning if the downloaded closing prices differ from the data used for the published results, for example after a revision by the data provider.
You may also use a custom dataset. Make sure to align your dataset with the preprocessing methods used in this project. 


## Usage

1. Clone the repository and install the package along with its dependencies:
```bash
git clone https://github.com/saschque/patterndecoder.git
cd patterndecoder
python setup.py install
# or
# make init
```

2. Run the Jupyter notebook to train and evaluate the given models:
```bash
jupyter notebook patterndecoder_evaluation_study.ipynb
```

3. To train a specific model (e.g., Transformer with full attention), first you need a model "models/transformer.yaml" like this:
```yaml
name: Transformer
architecture: patterndecoder.transformer.TransformerBlock
attention: patterndecoder.attention.MultiHeadAttention
```
4. Load the configuration, create the model, and train it in Python:

```python
from patterndecoder.utils import load_config, load_model, compile_and_train
import tensorflow as tf

# Clear any previous Keras session
tf.keras.backend.clear_session()

# Load parameters from config.yaml
params = load_config("config/config.yaml")

# Load and instantiate the model from the YAML config
transformer_model = load_model(params, "models/transformer.yaml")

# Compile and train the model
transformer_history, transformer_model = compile_and_train(transformer_model, train_data)

```

## Evaluation

Models are trained with MAE loss and the AdamW optimizer (learning rate $1\times10^{-4}$), with early stopping on a validation set (the last 20 % of the training period) and a fixed seed; loss, learning rate, dropout and model size were chosen by a validation search (`patterndecoder/tuning.py`). Models are evaluated using MAE and RMSE on training and test sets. In a study over ten seeds and 1-3 encoder and decoder layers (`patterndecoder/study.py`, results in `results/seed_layer_study.md`), one layer pair (2 encoder, 1 decoder) was chosen for all models on the validation set. With it, all 13 Transformer-type models have a mean test MAE below the zero-return baseline (0.007584); the PatternDecoder reaches 0.007561, second behind PatternDecoderNoLstmBlock (0.007559), and no two models can be told apart across seeds. A constant median forecast reaches 0.007559, and in the notebook's run Diebold-Mariano tests find no model significantly different from the zero or the median forecast, so the result shows no forecasting skill beyond the drift of the period. See the notebook's "Discussion of Results" for details and limitations.

A walk-forward evaluation (`patterndecoder/robustness.py`, results in `results/walk_forward.md`) retrains all models for ten seeds on five consecutive test windows that cover the second half of the data (August 2023 to February 2026), for the DAX and the S&P 500. It confirms this picture. On the DAX, the seed-averaged forecasts of all models are slightly but not significantly better than the zero forecast, and all except the Informer are behind the median forecast; on the S&P 500, 11 of the 13 models are significantly worse than the median forecast. Scaling the return inputs to percent (`input_scale: 100`) lowers the validation MAE of every model by about 0.3 %; the published results use unscaled inputs. The evaluation runs as the manually started workflow "Walk-forward evaluation" (inputs: a ticker or one of the synthetic series `SYNTH-AR`, `SYNTH-MOTIF` and `SYNTH-CYCLE`, and config values to change as JSON, for example `{"input_scale": 100}`) or locally:

```bash
mkdir -p wf
python -m patterndecoder.robustness run --seed 1 --test-start 0.7 --out wf/seed1_0.7.csv
python -m patterndecoder.robustness summarize wf/*.csv --out wf_summary.csv
```

## Project Structure

- `setup.py`: Setup file to create the Python project package `patterndecoder`
- `config/`: Contains configuration files, e.g., `config.yaml`
- `data/`: Local cache for the downloaded market data (DAX40, not part of the repository)
- `docs/`: Supporting documentation
- `models/`: Model configurations and pre-trained weights
- `patterndecoder/`: Core Python modules  
  - `attention.py`: Implementation of various attention mechanisms  
  - `embedding.py`: Implementation of embedding mechanisms  
  - `transformer.py`: Base Transformer implementation  
  - `informer.py`: Informer model implementation  
  - `autoformer.py`: Autoformer model implementation  
  - `patterndecoder.py`: PatternDecoder model implementation  
  - `data.py`: Download and caching of the market data from Yahoo Finance
  - `utils.py`: Data processing, feature engineering (includes calendaric features), training and evaluation functions
  - `training.py`: Random seeds and the chronological validation split
  - `tuning.py`: Hyperparameter search on the validation set
  - `study.py`: Seed and layer study over all models
  - `robustness.py`: Walk-forward evaluation over several test windows, for any ticker
  - `synthetic.py`: Synthetic return series with known patterns and oracle forecasts, used as positive controls
  - `experiments.py`: Helpers shared by the tuning, study and walk-forward scripts
  - `evaluation.py`: Median baseline, Diebold-Mariano test and forecast diagnostics
- `tests/`: Smoke tests for the models, the training pipeline, the evaluation and the walk-forward evaluation (`python -m pytest tests`)
- `results/`: Results of the seed and layer study (`seed_layer_study.csv`, `seed_layer_study.md`) and of the walk-forward evaluation (`walk_forward.md`, `walk_forward.csv`, `walk_forward_windows.csv`)
- `.github/workflows/`: Pylint and tests on every push, plus manually started workflows that run the whole notebook ("Notebook full run", with or without retraining), the hyperparameter search, the seed and layer study and the walk-forward evaluation on GitHub Actions
- `patterndecoder_evaluation_study.ipynb`: Main Jupyter notebook for model evaluation and analysis

## License

This project is licensed under the Apache-2.0 license - see the `LICENSE` file for details.
