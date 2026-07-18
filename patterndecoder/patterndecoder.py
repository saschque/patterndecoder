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
from patterndecoder.embedding import DenseTokenEmbedding


class DecoderLayerNoCrossAttention(tf.keras.layers.Layer):
    """
    A single layer of the decoder without cross-attention.

    """

    def __init__(
        self,
        units,
        d_model,
        dropout,
        attn_type,
        activation,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.self_attention = attn_type

        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    filters=units,
                    kernel_size=1,
                    activation=activation,
                ),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Conv1D(
                    filters=d_model,
                    kernel_size=1,
                ),
                tf.keras.layers.Dropout(dropout),
            ]
        )

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=True):
        """
        Applies the decoder layer without cross-attention.

        """

        # Causal self-attention
        seq_len = tf.shape(inputs)[1]
        causal_mask = 1.0 - tf.linalg.band_part(
            tf.ones((seq_len, seq_len)),
            -1,
            0,
        )

        self_attention_output = self.self_attention(
            inputs,
            inputs,
            inputs,
            mask=causal_mask,
        )

        self_attention_output = self.dropout1(
            self_attention_output,
            training=training,
        )

        self_attention_output = self.layernorm1(inputs + self_attention_output)

        # No cross-attention here

        ffn_output = self.ffn(self_attention_output)
        ffn_output = self.layernorm2(self_attention_output + ffn_output)

        return ffn_output


class DecoderNoCrossAttention(tf.keras.layers.Layer):
    """
    Decoder block without cross-attention.
    This class implements a decoder-only transformer block that processes
    input sequences through self-attention mechanisms without cross-attention.
    Attributes:
        layers (list): A list of DecoderLayerNoCrossAttention instances.
    """

    def __init__(
        self,
        units,
        d_model,
        dropout,
        n_layers,
        attn_type,
        activation,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.layers = [
            DecoderLayerNoCrossAttention(
                units,
                d_model,
                dropout,
                attn_type,
                activation,
                name=f"decoder_layer_no_cross_{i}",
            )
            for i in range(n_layers)
        ]

    def call(self, inputs, training=True):
        """
        Applies the decoder block without cross-attention.
        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, window_size, d_model).
            training (bool): A boolean indicating whether the layer is in training mode.
                Defaults to True.
        Returns:
            tf.Tensor: Output tensor of shape (batch_size, window_size, d_model).
        """

        outputs = inputs

        for layer in self.layers:
            outputs = layer(
                outputs,
                training=training,
            )

        return outputs


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


class PatternDecoderBlockNoCrossAttention(PatternDecoderBlock):
    """
    Decoder-only transformer model for sequence-to-sequence tasks without cross-attention.
    This class implements a transformer architecture that bypasses the encoder
    and utilizes only the decoder block without cross-attention. The PatternDecoder processes input
    sequences through self-attention mechanisms to capture relationships
    between elements and generate appropriate output sequences.

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

    def __init__(self, params, attn, name, **kwargs):
        """
        Initializes the PatternDecoderBlockNoCrossAttention.
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
            name: Name of the model
            **kwargs: Additional keyword arguments for the Keras Model base class.
        """

        super().__init__(params, attn, name, **kwargs)

        self.decoder = DecoderNoCrossAttention(
            params["units"],
            params["d_model"],
            params["dropout"],
            params["n_layers_dec"],
            attn,
            params["activation"],
        )

    def call(self, inputs, training=True):
        """
        Applies the PatternDecoderBlockNoCrossAttention for sequence processing.

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
        embedding = self.lstm_encode(embedding)
        # enc = self.lstm_encode(inputs)
        # enc = self.layernorm1(embedding)

        # Decoder outputs
        # Output shape: batch_size x window_size x d_model
        dec_outputs = self.decoder(embedding, embedding, training=training)

        # Output shape: batch_size x d_model
        return self.ffn(dec_outputs, training=training)


class PatternDecoderNoLstmBlock(TransformerBlock):
    """
    Decoder-only transformer model for sequence-to-sequence tasks without LSTM encoding.
    This class implements a transformer architecture that bypasses the encoder
    and utilizes only the decoder block without LSTM encoding. The PatternDecoder processes input
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
    """

    def __init__(self, params, attn, name, **kwargs):
        """
        Initializes the PatternDecoderNoLstmBlock.
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
            name: Name of the model
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
        Applies the PatternDecoderNoLstmBlock for sequence processing.

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
        embedding = self.layernorm1(embedding)

        # Decoder outputs
        # Output shape: batch_size x window_size x d_model
        dec_outputs = self.decoder(embedding, embedding, training=training)

        # Output shape: batch_size x d_model
        return self.ffn(dec_outputs, training=training)


class PatternDecoderNoConvBlock(PatternDecoderBlock):
    """
    Decoder-only transformer model for sequence-to-sequence tasks without convolutional embedding.
    This class implements a transformer architecture that bypasses the encoder
    and utilizes only the decoder block without convolutional embedding. The PatternDecoder
    processes input sequences through self-attention mechanisms to capture relationships
    between elements and generate appropriate output sequences.

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

    def __init__(self, params, attn, name="PatternDecoderNoConv", **kwargs):
        super().__init__(params, attn, name, **kwargs)

        self.token_embedding = DenseTokenEmbedding(params["d_model"])

    def call(self, inputs, training=True):
        """
        Applies the PatternDecoderNoConvBlock for sequence processing.

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
        embedding = self.token_embedding(inputs)
        embedding = self.encoding(embedding)
        enc = self.lstm_encode(inputs)
        enc = self.layernorm1(embedding + enc)

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
