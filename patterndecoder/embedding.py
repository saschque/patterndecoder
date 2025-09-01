# pylint: disable=E1101, R0913, R0903, R0917, R0902, R0801

"""
Embedding and positional encoding layers for Transformer models.

This module implements various embedding strategies for sequence modeling,
including LSTM-based embeddings, convolutional token embeddings, and
positional encodings for transformer architectures.

Classes:
    LSTMEmbedding: LSTM-based embedding layer for sequential feature extraction.
    TokenEmbedding: Convolutional token embedding with circular padding.
    SinusoidalPositionalEncoding: Fixed sinusoidal positional encodings for transformers.
    PositionalEncoding: Learnable positional embeddings using embedding layers.
"""

import tensorflow as tf


class LSTMEmbedding(tf.keras.layers.Layer):
    """
    LSTM-based embedding layer for sequential feature extraction.

    This layer applies LSTM processing to input sequences to create
    contextual embeddings that capture temporal dependencies.
    """

    def __init__(self, d_model, **kwargs):
        """
        Initializes the LSTMEmbedding layer.

        Args:
            d_model (int): The dimensionality of the embeddings.
            **kwargs: Additional keyword arguments for the Keras Layer base class.
        """
        super().__init__(**kwargs)
        initializer = tf.keras.initializers.HeNormal()
        self.lstm = tf.keras.layers.LSTM(
            d_model, kernel_initializer=initializer, return_sequences=True
        )

    def call(self, inputs):
        """
        Applies LSTM processing to input sequences.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            tf.Tensor: LSTM-processed embeddings of shape (batch_size, seq_len, d_model).
        """
        return self.lstm(inputs)


class TokenEmbedding(tf.keras.layers.Layer):
    """
    Convolutional token embedding layer with circular padding.

    Applies 1D convolution to input sequences using circular padding to
    preserve sequence length and capture local patterns in the input.
    """

    def __init__(self, d_model, **kwargs):
        """
        Initializes the TokenEmbedding layer.

        Args:
            d_model (int): The dimensionality of the output embeddings.
            **kwargs: Additional keyword arguments for the Keras Layer base class.
        """
        super().__init__(**kwargs)
        self.padding = 1
        self.token_conv = None
        self.d_model = d_model

    def build(self, input_shape):
        """
        Builds the TokenEmbedding layer.
        Args:
            input_shape: The shape of the input tensor.
        """
        super().build(input_shape)

        # Create a Conv1D layer
        # Use He initialization
        initializer = tf.keras.initializers.HeNormal()

        self.token_conv = tf.keras.layers.Conv1D(
            filters=self.d_model,
            kernel_size=3,
            kernel_initializer=initializer,
            padding="valid",  # We'll handle circular padding manually
        )

    def call(self, x):
        """
        Applies convolutional token embeddings with circular padding.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, seq_len, input_features).

        Returns:
            tf.Tensor: Convolved embeddings of shape (batch_size, seq_len, d_model).
        """
        # x shape: [batch, seq_len, c_in]

        # Implement circular padding
        padding = self.padding

        left_pad = x[:, -padding:, :]
        right_pad = x[:, :padding, :]
        x_padded = tf.concat([left_pad, x, right_pad], axis=1)

        # Apply Conv1D
        x_conv = self.token_conv(x_padded)

        return x_conv


class SinusoidalPositionalEncoding(tf.keras.layers.Layer):
    """
    Fixed sinusoidal positional encoding for transformer models.

    Implements the sinusoidal positional encoding from "Attention Is All You Need"
    that adds position information to embeddings using sine and cosine functions.
    """

    def __init__(self, d_model, max_len=5000, **kwargs):
        """
        Initializes the SinusoidalPositionalEncoding layer.

        Args:
            d_model (int): The dimensionality of the embeddings.
            max_len (int): Maximum sequence length to precompute. Default is 5000.
            **kwargs: Additional keyword arguments for the Keras Layer base class.
        """
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        self.pe = None

    def build(self, input_shape):
        """
        Builds the SinusoidalPositionalEncoding layer.
        Args:
            input_shape: The shape of the input tensor.
        """
        super().build(input_shape)
        # Compute PE once during build
        pe = self._compute_pe(self.max_len, self.d_model)
        self.pe = self.add_weight(
            name="positional_encoding",
            shape=pe.shape,
            initializer="zeros",
            trainable=False,
        )
        self.pe.assign(pe)

    def _compute_pe(self, max_len, d_model):
        """
        Computes sinusoidal positional encodings.

        Args:
            max_len (int): Maximum sequence length.
            d_model (int): Model dimensionality.

        Returns:
            tf.Tensor: Positional encodings of shape (1, max_len, d_model).
        """
        position = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(
            tf.range(0, d_model, 2, dtype=tf.float32)
            * -(tf.math.log(10000.0) / d_model)
        )

        pe = tf.zeros((max_len, d_model))
        pe = tf.tensor_scatter_nd_update(
            pe,
            [[i, j] for i in range(max_len) for j in range(0, d_model, 2)],
            tf.reshape(tf.sin(position * div_term), [-1]),
        )
        pe = tf.tensor_scatter_nd_update(
            pe,
            [[i, j] for i in range(max_len) for j in range(1, d_model, 2)],
            tf.reshape(tf.cos(position * div_term[: d_model // 2]), [-1]),
        )

        return pe[tf.newaxis, :, :]  # Add batch dimension

    def call(self, inputs):
        """
        Adds sinusoidal positional encodings to input embeddings.

        Args:
            inputs (tf.Tensor): Input embeddings of shape (batch_size, seq_len, d_model).

        Returns:
            tf.Tensor: Input embeddings with added positional encodings.
        """
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pe[:, :seq_len, :]


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Learnable positional encoding using embedding layers.

    Implements trainable positional embeddings that are learned during training,
    as an alternative to fixed sinusoidal encodings.
    """

    def __init__(self, output_dim, input_dim, **kwargs):
        """
        Initializes the PositionalEncoding layer.

        Args:
            input_dim: The number of positions.
            output_dim: The dimensionality of the embeddings.
        """
        super().__init__(**kwargs)
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=input_dim, output_dim=output_dim
        )
        self.input_dim = input_dim
        self.output_dim = output_dim

    def build(self, input_shape):
        """
        Builds the PositionalEncoding layer.

        Args:
            input_shape: The shape of the input tensor.
        """
        super().build(input_shape)
        # Ensure the position embeddings are initialized correctly
        self.position_embeddings.build((self.input_dim, self.output_dim))

    def call(self, inputs):
        """
        Adds learnable positional embeddings to input embeddings.

        Args:
            inputs (tf.Tensor): Input embeddings of shape (batch_size, seq_len, d_model).

        Returns:
            tf.Tensor: Input embeddings with added positional encodings.
        """
        positions = tf.range(start=0, limit=self.input_dim, delta=1)
        positions = self.position_embeddings(positions)
        positions = tf.expand_dims(positions, axis=0)

        return inputs + positions
