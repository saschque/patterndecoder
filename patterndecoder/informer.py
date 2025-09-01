# pylint: disable=E1101, R0913, R0903, R0917, R0902, R0801

"""
Informer model implementation for time series forecasting.

This module implements the Informer model based on the paper "Informer: Beyond
Efficient Transformer for Long Sequence Time-Series Forecasting". The Informer
addresses limitations of standard Transformers for long sequence forecasting
through ProbSparse attention and progressive distillation.

Classes:
    Encoder: Encoder network with attention distilling for sequence length reduction.
    EncoderLayer: Building block with self-attention, feed-forward, and distilling components.
    InformerBlock: Complete Informer model combining encoder and decoder for forecasting.

Key Features:
    - ProbSparse attention mechanism for O(L log L) complexity
    - Progressive distillation to halve sequence length at each layer
    - Efficient processing of long time series sequences
    - Attention distilling to highlight important temporal patterns
"""

import tensorflow as tf
from patterndecoder.transformer import TransformerBlock
from patterndecoder.transformer import EncoderLayer as TransformerEncoderLayer
from patterndecoder.transformer import Encoder as TransformerEncoder


class Encoder(TransformerEncoder):
    """
    Encoder layer for the Informer model.

    This layer consists of a stack of EncoderLayer instances and performs
    attention distilling to reduce sequence length and highlight important
    information. The distilling process progressively halves the sequence length
    at each layer, making the model efficient for processing long sequences.

    Args:
        units (int): The number of units in the feed-forward network.
        d_model (int): The dimensionality of the model.
        dropout (float): The dropout rate.
        n_layers (int): The number of encoder layers.
        attn_type (tf.keras.layers.Layer): The type of attention mechanism to use (ProbSparse).

    Attributes:
        layers (list): A list of EncoderLayer instances.
    """

    def __init__(
        self, units, d_model, dropout, n_layers, attn_type, activation, **kwargs
    ):
        """
        Initialize the Informer Encoder.

        Args:
            units (int): The number of units in the feed-forward network.
            d_model (int): The dimensionality of the model.
            dropout (float): The dropout rate.
            n_layers (int): The number of encoder layers.
            attn_type (tf.keras.layers.Layer): The type of attention mechanism to use.
            activation (str): Activation function for the feed-forward network.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """

        super().__init__(
            units, d_model, dropout, n_layers, attn_type, activation, **kwargs
        )
        self.layers = [
            EncoderLayer(
                units,
                d_model,
                dropout,
                attn_type,
                activation,
                name=f"encoder_layer_{i}",
            )
            for i in range(n_layers)
        ]

    def call(self, inputs, training=True):
        """
        Apply the Informer Encoder with progressive distillation.

        Processes inputs through all encoder layers and concatenates outputs from
        each layer. This concatenation preserves information from different levels
        of abstraction as the sequence length is progressively reduced.

        Args:
            inputs (tf.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
            training (bool, optional): Whether the model is in training mode. Defaults to True.

        Returns:
            tf.Tensor: Concatenated output tensor from all encoder layers with
                shape (batch_size, total_seq_len, d_model) where total_seq_len
                is the sum of sequence lengths from all layers.
        """
        enc_outputs = inputs
        encoder_outputs = []

        for encoder_layer in self.layers:
            enc_outputs = encoder_layer(enc_outputs)
            encoder_outputs.append(enc_outputs)

        enc_outputs = tf.concat(encoder_outputs, axis=1)
        return enc_outputs


class EncoderLayer(TransformerEncoderLayer):
    """
    EncoderLayer for the Informer model.

    This layer includes self-attention (using ProbSparse attention), a feed-forward network,
    layer normalization, dropout, and distilling components to reduce sequence length.
    The distilling process uses convolutional operations with max pooling to downsample
    the sequence, focusing on the most important features.

    Args:
        units: The number of units in the feed-forward network.
        d_model: The dimensionality of the model.
        dropout: The dropout rate.
        attn_type: The type of attention mechanism to use (ProbSparse).
        name: The name of the layer.
    """

    def __init__(
        self,
        units,
        d_model,
        dropout,
        attn_type,
        activation,
        name="informer_encoder_layer",
        **kwargs,
    ):
        """
        Initializes the Informer-specific EncoderLayer.

        Args:
            units (int): The number of units in the feed-forward network.
            d_model (int): The dimensionality of the model.
            dropout (float): The dropout rate.
            attn_type: The type of attention mechanism to use.
            activation (str): Activation function for the feed-forward network.
            name (str): The name of the layer. Default is "informer_encoder_layer".
            **kwargs: Additional keyword arguments for the Keras Layer base class.
        """
        super().__init__(
            units, d_model, dropout, attn_type, activation, name=name, **kwargs
        )

        # Add distilling components specific to Informer
        self.distilling_layer = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    filters=d_model, kernel_size=3, padding="causal"
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ELU(),
                tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding="same"),
            ]
        )

    def call(self, inputs, training=True):
        """
        Applies the Informer-specific EncoderLayer with distillation.

        Processes inputs through the parent transformer layer (self-attention and
        feed-forward) followed by the distilling layer that reduces sequence length
        by half using convolutional operations and max pooling.

        Args:
            inputs (tf.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
            training (bool): A boolean indicating whether the layer is in training mode.
                Defaults to True.

        Returns:
            tf.Tensor: The output tensor with reduced sequence length, approximately
                shape (batch_size, seq_len//2, d_model).
        """
        # Call the parent class's call method
        ffn_output = super().call(inputs, training=training)

        # Apply attention distilling
        ffn_output = self.distilling_layer(ffn_output)

        return ffn_output


class InformerBlock(TransformerBlock):
    """
    Complete Informer model block for time series forecasting.

    Combines an Informer encoder with progressive distillation and a standard
    transformer decoder. The encoder efficiently processes long input sequences
    using ProbSparse attention and distilling mechanisms, while the decoder
    generates predictions based on the encoder's output.

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
            (ProbSparse recommended)

    Note:
        The encoder uses ProbSparse attention and distilling layers that
        progressively reduce sequence length.
    """

    def call(self, inputs, training=True):
        """
        Applies the complete Informer model for time series forecasting.

        Processes inputs through the Informer encoder with progressive distillation
        and attention mechanisms, then through the decoder to generate predictions.
        The distillation process reduces computational complexity while preserving
        important temporal patterns.

        Args:
            inputs (tf.Tensor): Input time series tensor of shape
                (batch_size, window_size, n_features).
            training (bool): A boolean indicating whether the layer is in training mode.
                Defaults to True.

        Returns:
            tf.Tensor: Predicted output tensor of shape (batch_size, output_size).
        """
        # Call the parent class's call method
        outputs = super().call(inputs, training=training)

        return outputs
