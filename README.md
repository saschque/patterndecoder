# Transformer-Based Architectures for Temporal Forecasting: A Study on Financial Time Series Data
![Build Status](https://github.com/saschque/patterndecoder/actions/workflows/pylint.yml/badge.svg)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green.svg)](https://shields.io/)

This study examines the effectiveness of transformer-based models for financial time series forecasting, specifically focusing on log returns derived from daily closing prices of the DAX40 index. We propose a decoder-only transformer model specifically designed for immediate-term financial time series forecasting: The PatternDecoder architecture addresses the specific requirements of short-sequence financial forecasting and is evaluated with full, convolutional, and Auto-Correlation attention mechanisms. Various transformer architectures are evaluated alongside for their predictive performance, including the standard Transformer encoder and its specialized variations Informer and Autoformer, that are designed to improve performance on time series data.

## Introduction

Time series forecasting is a critical task in financial analysis, enabling traders, investors, and financial institutions to make informed decisions based on predicted market movements. Traditional statistical methods like ARIMA and exponential smoothing have limitations in capturing complex non-linear patterns and short-term dependencies in financial data. Deep learning approaches, particularly transformer-based architectures, offer promising alternatives due to their ability to model sequential data effectively.

## Model Architectures

### Transformer

The **Transformer** model, initially introduced in *[Attention is All You Need](https://arxiv.org/abs/1706.03762)*, leverages self-attention mechanisms to capture short-term dependencies and complex patterns in sequential data. Unlike recurrent neural networks, transformers process the entire sequence simultaneously, allowing them to capture relationships between any positions in the sequence regardless of their distance. The key components include:

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

The **Informer** architecture, introduced in *[Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436)*, aims to improve transformer efficiency for long sequence time-series forecasting through:

- **ProbSparse Attention**: Reduces computational complexity to $O(L(\log L))$ by focusing on the most important query-key pairs
- **Distilling Mechanism**: Progressively halves the sequence length at each layer through convolutional operations
- **Direct Multi-step Forecasting**: Predicts the entire output sequence in one forward pass

### Autoformer

The **Autoformer**, as presented in *[Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://arxiv.org/abs/2106.13008)*, introduces innovative components for time series modeling:

- **Series Decomposition**: Separates time series into trend and seasonal components using moving average filters
- **Auto-Correlation Mechanism**: Replaces self-attention with an auto-correlation mechanism that captures lag-based periodicities
- **Progressive Decomposition**: Uses depth-decay to progressively reduce the influence of the trend component at deeper layers

### Decoder-Only Transformer
**Decoder-only transformers** are neural network architectures composed exclusively of stacked decoder blocks, each containing masked self-attention and feed-forward layers, optimized for autoregressive text generation and time series tasks. This architecture underpins large language models like GPT-3 and GPT-4, enabling efficient generation of coherent and contextually relevant information through mechanisms such as residual connections and layer normalization. (see *[Q. Chen (2025)](https://arxiv.org/html/2504.16361v1)* )

### PatternDecoder
A decoder-only transformer model specifically designed for immediate-term financial time series forecasting. The PatternDecoder architecture addresses the specific requirements of short-sequence financial forecasting and is evaluated with full, convolutional, and Auto-Correlation attention mechanisms.

### Baseline Models
For the study, Naive and $ARIMA(0,0,0)$ models were used as baseline to evaluate the forecast quality of the models. Additionally, hybrid models incorporating Conv1D and LSTM layers are analyzed as baseline models to assess their potential for capturing both local patterns and temporal dynamics in time series data.

## Installation

To run the code in this repository, you need to clone the repository and install the package and its dependencies:

```
git clone https://github.com/saschque/patterndecoder.git
cd patterndecoder
python setup.py install
# or
# make init
```

## Dataset

The study uses historical daily closing prices of the DAX40 index spanning from March 28, 2022, to March 28, 2025, providing 768 trading days of information. The data is preprocessed by:
1. Converting prices to daily log returns using the formula $r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$ to ensure stationarity, as confirmed by ADF and KPSS tests
2. Using a $30$-day input window ($L=w=30$) for 1-day ahead ($h=1$) prediction
3. Splitting into training ($70\%$) and testing ($30\%$) sets, yielding $537$ training observations and $231$ test observations

You can use the provided dataset in the `data` directory or download updated data using the commented code in the notebook.
You may also use a custom dataset. Make sure to align your dataset with the preprocessing methods used in this project. 

## Usage

1. Clone the repository and install the package and its dependencies:
```
git clone https://github.com/saschque/patterndecoder.git
cd patterndecoder
python setup.py install
# or
# make init
```

2. Run the Jupyter notebook to train and evaluate models:
```
jupyter notebook time_series_transformer_evaluation_paper.ipynb
```

3. To train a specific model (e.g., Transformer with full attention):
```
from patterndecoder.utils import load_config, load_model, compile_and_train
import tensorflow as tf

tf.keras.backend.clear_session()

# Load parameters and hyperparameters from config.json
params = load_config('config/config.yaml')

# Initialize the Transformer model
transformer_model = load_model(params, "models/transformer.yaml")

# Compile and train the model
__, transformer_model = compile_and_train(transformer_model, train_data)
```

## Evaluation

The models are evaluated based on their Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) performance across training and testing datasets. The study uses Huber loss as the loss function and AdamW optimizer with learning rate 1×10⁻⁴. This study provides insights into the ability of transformer-based models and their variations to handle short-term dependencies, model complex temporal dynamics, and improve forecasting accuracy in immediate-term financial time series prediction. It offers a comparative perspective on the performance of these models relative to traditional methods and other deep learning techniques in the context of financial time series forecasting.

## Project Structure

- `setup.py`: Setup file to create the python project package `patterndecoder`
- `patterndecoder/`: Directory containing the modules of the package
- `patterndecoder/attention.py`: Implementation of various attention mechanisms
- `patterndecoder/embedding.py`: Implementation of various embedding mechanisms
- `patterndecoder/transformer.py`: Implementation of the base Transformer model
- `patterndecoder/informer.py`: Implementation of the Informer model
- `patterndecoder/autoformer.py`: Implementation of the Autoformer model
- `patterndecoder/patterndecoder.py`: Implementation of the PatternDecoder model
- `patterndecoder/utils.py`: Utility functions for data processing and evaluation
- `patterndecoder_evaluation_study.ipynb`: Main notebook for the evaluation
- `data/`: Directory containing the daily market data

## License

This project is licensed under the Apache-2.0 license - see the `LICENSE` file for details.
