# pylint: disable=E1101, R0913, R0903, R0917, R0902, R0801

"""
Core PatternDecoder architecture components for TensorFlow 2.x.

This module implements PatternDecoder models that utilize only the decoder
component of transformer architectures for sequence-to-sequence tasks.
These models bypass the encoder and rely on self-attention mechanisms
within the decoder to process input sequences directly.

Classes:
    PatternDecoderBlock: Decoder-only transformer model for sequence processing.
    AutoPatternDecoderBlock: PatternDecoder variant using AutoCorrelation attention
        and decomposition.

The PatternDecoder architecture is particularly suitable for tasks where
the input and output sequences have similar structure and the model can
learn patterns through self-attention without explicit encoder-decoder separation.
"""

import tensorflow as tf
from patterndecoder.transformer import TransformerBlock
from patterndecoder.autoformer import AutoDecoder, SeriesDecomp


class DecoderOnlyTransformer(TransformerBlock):
    """
    Decoder-only transformer model for sequence-to-sequence tasks.
    """

    def call(self, inputs, training=True):
        """
        Applies the DecoderOnlyTransformer for sequence processing.

        Processes inputs through token embedding, then applies the decoder
        with self-attention to both queries and keys/values from the same
        embedded input, followed by a final feed-forward projection.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, window_size, n_features).
            training (bool): A boolean indicating whether the layer is in training mode.
                Defaults to True.

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Input shape: batch_size x window_size x n_features

        # Token Embedding
        # Output shape: batch_size x window_size x d_model
        embedding = self.embed(inputs)

        # Decoder outputs
        # Output shape: batch_size x window_size x d_model
        dec_outputs = self.decoder(embedding, embedding, training=training)

        # Output shape: batch_size x d_model
        return self.ffn(dec_outputs, training=training)


class PatternDecoderBlock(TransformerBlock):
    """
    Decoder-only transformer model for sequence-to-sequence tasks.

    This class implements a transformer architecture that bypasses the encoder
    and utilizes only the decoder block. The PatternDecoder processes input
    sequences through self-attention mechanisms to capture relationships
    between elements and generate appropriate output sequences.

    Inherits all parameters from TransformerBlock:
        params (dict): Dictionary containing configuration parameters:
            - batch_size: Batch size for input tensors
            - window_size: Length of input sequences
            - n_features: Number of input features
            - output_size: Size of the output (prediction horizon)
            - d_model: Dimensionality of the model
            - units: Number of units in feed-forward networks
            - n_heads: Number of attention heads
            - n_layers_enc: Number of encoder layers
            - activation: Activation function for feed-forward networks
            - dropout: Dropout rate
            - l1_reg: L1 regularization coefficient
            - l2_reg: L2 regularization coefficient
        attn : Attention mechanism for encoder, decoder and cross attention

    Note:
        Uses only the decoder component, making it suitable for tasks where
        input-output sequences have similar structure and patterns can be
        learned through self-attention alone.
    """

    def __init__(self, params, attn, name, **kwargs):
        """
        Initializes the PatternDecoderBlock.

        Args:
            params (dict): Dictionary containing configuration parameters:
                - batch_size: Batch size for input tensors
                - window_size: Length of input sequences
                - n_features: Number of input features
                - output_size: Size of the output (prediction horizon)
                - d_model: Dimensionality of the model
                - units: Number of units in feed-forward networks
                - n_heads: Number of attention heads
                - n_layers_enc: Number of encoder layers
                - activation: Activation function for feed-forward networks
                - dropout: Dropout rate
                - l1_reg: L1 regularization coefficient
                - l2_reg: L2 regularization coefficient
            attn: Attention mechanism for decoder and cross attention
            **kwargs: Additional keyword arguments for the Keras Model base class.
        """
        super().__init__(params, attn, name, **kwargs)
        self.lstm_encode = tf.keras.layers.LSTM(
            params["d_model"],
            kernel_initializer=tf.keras.initializers.HeNormal(),
            return_sequences=True,
        )

    def call(self, inputs, training=True):
        """
        Applies the PatternDecoderBlock for sequence processing.

        Processes inputs through token embedding, then applies the decoder
        with self-attention to both queries and keys/values from the same
        embedded input, followed by a final feed-forward projection.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, window_size, n_features).
            training (bool): A boolean indicating whether the layer is in training mode.
                Defaults to True.

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Input shape: batch_size x window_size x n_features

        # Token Embedding
        # Output shape: batch_size x window_size x d_model
        embedding = self.embed(inputs)
        enc = self.lstm_encode(inputs)
        enc = self.layernorm1(enc + embedding)

        # Decoder outputs
        # Output shape: batch_size x window_size x d_model
        dec_outputs = self.decoder(embedding, enc, training=training)

        # Output shape: batch_size x d_model
        return self.ffn(dec_outputs, training=training)


class AutoPatternDecoderBlock(PatternDecoderBlock):
    """
    AutoCorrelation-based PatternDecoder for time series forecasting.

    This class extends PatternDecoderBlock by using AutoCorrelation attention
    and series decomposition mechanisms from the Autoformer architecture.
    It combines decoder-only processing with auto-correlation for discovering
    time-delay dependencies and trend-seasonal component separation.

    Inherits all parameters from PatternDecoderBlock, but uses AutoDecoder
    instead of standard TransformerDecoder for enhanced time series processing
    with decomposition capabilities.

    Inherits all parameters from PatternDecoderBlock:
        params (dict): Dictionary containing configuration parameters:
            - batch_size: Batch size for input tensors
            - window_size: Length of input sequences
            - n_features: Number of input features
            - output_size: Size of the output (prediction horizon)
            - d_model: Dimensionality of the model
            - units: Number of units in feed-forward networks
            - n_heads: Number of attention heads
            - n_layers_enc: Number of encoder layers
            - activation: Activation function for feed-forward networks
            - dropout: Dropout rate
            - l1_reg: L1 regularization coefficient
            - l2_reg: L2 regularization coefficient
        attn : Attention mechanism for encoder, decoder and cross attention
    """

    def __init__(self, params, attn, name, kernel_size=5, **kwargs):
        """
        Initializes the AutoPatternDecoderBlock with AutoCorrelation attention.

        Sets up the model with AutoDecoder instead of standard decoder to enable
        auto-correlation attention and series decomposition for time series data.

        Args:
            params (dict): Dictionary containing configuration parameters:
                - batch_size: Batch size for input tensors
                - window_size: Length of input sequences
                - n_features: Number of input features
                - output_size: Size of the output (prediction horizon)
                - d_model: Dimensionality of the model
                - units: Number of units in feed-forward networks
                - n_heads: Number of attention heads
                - n_layers_enc: Number of encoder layers
                - activation: Activation function for feed-forward networks
                - dropout: Dropout rate
                - l1_reg: L1 regularization coefficient
                - l2_reg: L2 regularization coefficient
            attn: Attention mechanism for decoder and cross attention
            **kwargs: Additional keyword arguments for the Keras Model base class.
        """
        super().__init__(params, attn, name, **kwargs)
        self.decoder = AutoDecoder(
            params["units"],
            params["d_model"],
            params["dropout"],
            params["n_layers_dec"],
            attn,
            params["activation"],
        )
        self.series_decomp = SeriesDecomp(kernel_size)
