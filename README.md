# Transformer-Based Architectures for Temporal Forecasting: A Study on Financial Time Series Data
![Build Status](https://github.com/saschque/patterndecoder/actions/workflows/pylint.yml/badge.svg)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green.svg)](https://shields.io/)

This study examines the effectiveness of Transformer-based models for short-term financial time series forecasting, specifically focusing on one-day-ahead prediction using log returns derived from daily closing prices of the DAX40 index. Various Transformer architectures are evaluated for their predictive performance, including the standard Transformer with multiple attention mechanisms (full, convolutional, LogSparse), specialized variants (Informer, Autoformer), and the newly introduced PatternDecoder architecture, a hybrid decoder-only Transformer. Additionally, baseline models (Naïve, Conv1D-LSTM) are analyzed to provide performance benchmarks. The models are evaluated using MAE and RMSE across 30-day input windows for single-step forecasting.  

In the current run, no trained model - the PatternDecoder included - demonstrably improves on the zero-return (random-walk) baseline: all test errors lie within about 4 % of each other, most of these differences are too small to resolve with 353 test points, one training run per model and no fixed seeds, and the ranking of the models differed between two executions of the notebook. This is consistent with the noise-dominated nature of one-day-ahead returns and with simple baseline models remaining highly relevant in this domain. The input also contains **calendar features** (sine/cosine encodings of day of week, day of month and month); their effect was not tested separately.

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

4. Adding cyclical calendar features (sine/cosine of day of week, day of month and month) via utils.py as further inputs, so that each input step has 7 features

You can use the provided dataset in the `data/` directory or download updated data via `get_stock_data(params, download=True)`; note that this overwrites the snapshot and changes all results.
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

Models are evaluated using MAE and RMSE on training and test sets, with Huber loss and AdamW optimizer (learning rate $1\times10^{-4}$). In the current run no model, including the PatternDecoder variants, demonstrably beats the zero-return (Naïve) baseline: the test MAEs of all 16 rows lie within 3.9 % of each other, well inside the bootstrap uncertainty of the baseline's own error (test MAE 0.0076). This highlights the difficulty of short-term financial forecasting; the study does not allow conclusions about the benefit of calendar features, which were not tested separately. See the notebook's "Discussion of Results" for details and limitations (one run per model, no fixed seeds, no validation set).

## Project Structure

- `setup.py`: Setup file to create the Python project package `patterndecoder`
- `config/`: Contains configuration files, e.g., `config.yaml`
- `data/`: Directory containing daily market data (DAX40)
- `docs/`: Supporting documentation
- `models/`: Model configurations and pre-trained weights
- `patterndecoder/`: Core Python modules  
  - `attention.py`: Implementation of various attention mechanisms  
  - `embedding.py`: Implementation of embedding mechanisms  
  - `transformer.py`: Base Transformer implementation  
  - `informer.py`: Informer model implementation  
  - `autoformer.py`: Autoformer model implementation  
  - `patterndecoder.py`: PatternDecoder model implementation  
  - `utils.py`: Data processing, feature engineering (includes calendaric features), and evaluation functions
- `patterndecoder_evaluation_study.ipynb`: Main Jupyter notebook for model evaluation and analysis

## License

This project is licensed under the Apache-2.0 license - see the `LICENSE` file for details.
