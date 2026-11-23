# pylint: disable=E1101, R0913, R0903, R0917, R0902, R0801

"""
Core Transformer architecture components for TensorFlow 2.x.

This module implements a complete Transformer model architecture designed for
sequence-to-sequence tasks, particularly time series forecasting and temporal
data analysis. The implementation includes encoder-decoder structure with
configurable attention mechanisms and embedding layers.

Classes:
    Encoder: Multi-layer transformer encoder stack for input sequence processing.
    EncoderLayer: Single transformer encoder layer with self-attention and feed-forward networks.
    Decoder: Multi-layer transformer decoder stack for output sequence generation.
    DecoderLayer: Single transformer decoder layer with self-attention, cross-attention,
        and feed-forward networks.
    TransformerBlock: Complete transformer model combining encoder and decoder with embeddings.

Note on attention wiring:
    Attention mechanisms are passed around as a (attention_class, kwargs) specification
    (see patterndecoder.attention.build_attention) rather than as a single shared
    instance. Every layer/role (each encoder layer, each decoder layer's self-attention,
    each decoder layer's cross-attention) calls build_attention() to construct its own
    independently-weighted attention instance. This prevents different layers/roles from
    unintentionally sharing Q/K/V weights.

Dependencies:
    - transformer_models.embedding: TokenEmbedding and SinusoidalPositionalEncoding
    - Custom attention mechanisms from transformer_models.attention

Example:
    ```
    # Initialize transformer for time series forecasting
    transformer = TransformerBlock(
        params={"batch_size": 32, "window_size": 60, ...},
        attn=(MultiHeadAttention, {"d_model": 128, "n_heads": 8}),
        name="transformer",
    )

    # Build and compile model
    model = transformer.build_model()
    ```
"""

import tensorflow as tf
from patterndecoder.embedding import TokenEmbedding, SinusoidalPositionalEncoding
from patterndecoder.attention import build_attention


def build_ffn(units, d_model, dropout, activation):
    """
    Builds the position-wise feed-forward block shared by encoder/decoder layers.

    Used by EncoderLayer, DecoderLayer and DecoderLayerNoCrossAttention so the
    Conv1D -> Dropout -> Conv1D -> Dropout pattern is defined once instead of
    being copy-pasted in each layer class.

    Args:
        units (int): Number of filters in the inner Conv1D (expansion layer).
        d_model (int): Number of filters in the outer Conv1D (projects back to d_model).
        dropout (float): Dropout rate applied after each Conv1D.
        activation (str): Activation function for the inner Conv1D.

    Returns:
        tf.keras.Sequential: The feed-forward block.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv1D(filters=units, kernel_size=1, activation=activation),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Conv1D(filters=d_model, kernel_size=1),
            tf.keras.layers.Dropout(dropout),
        ]
    )


class Encoder(tf.keras.layers.Layer):
    """
    Encoder layer for transformer-based models.

    This class implements a stack of encoder layers, each applying self-attention
    and feed-forward networks to process input sequences. It's designed to capture
    complex patterns and dependencies in the input data.

    Args:
        units (int): The number of units in the feed-forward network.
        d_model (int): The dimensionality of the model.
        dropout (float): The dropout rate.
        n_layers (int): The number of encoder layers.
        attn_type (tuple): (attention_class, kwargs) specification for building a
            fresh attention instance per layer.
        **kwargs: Additional keyword arguments to pass to the BaseEncoder.

    Attributes:
        layers (list): A list of EncoderLayer instances.

    Methods:
        call(inputs, training=True): Applies the encoder to the input tensor.
    """

    def __init__(
        self, units, d_model, dropout, n_layers, attn_type, activation, **kwargs
    ):
        """
        Initializes the Encoder.

        Args:
            units (int): The number of units in the feed-forward network.
            d_model (int): The dimensionality of the model.
            dropout (float): The dropout rate.
            attn_type: (attention_class, kwargs) specification for building attention layers.
            activation (str): Activation function for the feed-forward network.
            name (str): The name of the layer. Default is "encoder_layer".
            **kwargs: Additional keyword arguments for the Keras Layer base class.
        """
        super().__init__(**kwargs)
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
        Applies the Encoder.

        Args:
            inputs (tf.Tensor): The input tensor.
            training (bool): A boolean indicating whether the layer is in training mode.
                Defaults to True.

        Returns:
            tf.Tensor: The output tensor after applying self-attention and feed-forward network.
        """
        enc_outputs = inputs
        for encoder_layer in self.layers:
            enc_outputs = encoder_layer(enc_outputs, training=training)
        return enc_outputs


class EncoderLayer(tf.keras.layers.Layer):
    """
    EncoderLayer for the Transformer model.

    This layer includes self-attention, a feed-forward network, layer normalization, and dropout.
    The self-attention mechanism allows the model to weigh the importance of different
    positions in the input sequence when processing each position.
    """

    def __init__(
        self,
        units,
        d_model,
        dropout,
        attn_type,
        activation,
        name="encoder_layer",
        **kwargs,
    ):
        """
        Initializes the EncoderLayer.

        Args:
            units: The number of units in the feed-forward network.
            d_model: The dimensionality of the model.
            dropout: The dropout rate.
            attn_type: (attention_class, kwargs) specification used to build this
                layer's own, independently-weighted self-attention instance.
            name: The name of the layer.
        """
        super().__init__(name=name, **kwargs)
        self.self_attention = build_attention(attn_type)
        self.ffn = build_ffn(units, d_model, dropout, activation)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=True):
        """
        Applies the EncoderLayer.

        Args:
            x: The input tensor.
            training: A boolean indicating whether the layer is in training mode.

        Returns:
            The output tensor.
        """
        # Self attention layer
        self_attention_output = self.self_attention(inputs, inputs, inputs)
        self_attention_output = self.dropout1(self_attention_output, training=training)
        self_attention_output = self.layernorm1(inputs + self_attention_output)

        # Feed Forward layer
        ffn_output = self.ffn(self_attention_output)
        ffn_output = self.layernorm2(self_attention_output + ffn_output)

        return ffn_output


class Decoder(tf.keras.layers.Layer):
    """
    Decoder for the Transformer model.

    This layer consists of a stack of DecoderLayer instances. Each layer processes
    the output of the previous layer, allowing the model to generate increasingly
    refined outputs based on the encoder's representations.
    """

    def __init__(
        self, units, d_model, dropout, n_layers, attn_type, activation, **kwargs
    ):
        """
        Initializes the Decoder.

        Args:
            units (int): The number of units in the feed-forward network.
            d_model (int): The dimensionality of the model.
            n_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
            n_layers (int): The number of decoder layers.
            attn_type: (attention_class, kwargs) specification for building attention layers.
            activation (str): Activation function for the feed-forward network.
            **kwargs: Additional keyword arguments for the Keras Layer base class.
        """
        super().__init__(**kwargs)
        self.layers = [
            DecoderLayer(
                units,
                d_model,
                dropout,
                attn_type,
                activation,
                name=f"decoder_layer_{i}",
            )
            for i in range(n_layers)
        ]

    def call(self, inputs, enc_outputs, training=True):
        """
        Applies the Decoder.

        Args:
            inputs: The input tensor.
            enc_outputs: The output tensor from the encoder.
            training: A boolean indicating whether the layer is in training mode.

        Returns:
            The output tensor.
        """
        dec_outputs = inputs
        for decoder_layer in self.layers:
            dec_outputs = decoder_layer(dec_outputs, enc_outputs, training=training)
        return dec_outputs


class DecoderLayer(tf.keras.layers.Layer):
    """
    DecoderLayer for the Transformer model.

    This layer includes self-attention, encoder-decoder attention, a feed-forward network,
    layer normalization, and dropout. The self-attention mechanism in the decoder is typically
    masked to prevent positions from attending to future positions. The encoder-decoder
    attention allows the decoder to focus on relevant parts of the input sequence.

    Self-attention and cross-attention each get their own, independently-weighted
    attention instance (built via build_attention) rather than sharing one.
    """

    def __init__(
        self,
        units,
        d_model,
        dropout,
        attn_type,
        activation,
        name="decoder_layer",
        **kwargs,
    ):
        """
        Initializes the DecoderLayer.

        Args:
            units (int): The number of units in the feed-forward network.
            d_model (int): The dimensionality of the model.
            n_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
            attn_type: (attention_class, kwargs) specification used to build this layer's
                self-attention and cross-attention instances (each gets its own).
            activation (str): Activation function for the feed-forward network.
            name (str): The name of the layer. Default is "decoder_layer".
            **kwargs: Additional keyword arguments for the Keras Layer base class.
        """
        super().__init__(name=name, **kwargs)
        self.self_attention = build_attention(attn_type)
        self.encoder_decoder_attention = build_attention(attn_type)
        self.ffn = build_ffn(units, d_model, dropout, activation)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, encoder_outputs, training=True):
        """
        Applies the DecoderLayer.

        Args:
            inputs: The input tensor.
            encoder_outputs: The output tensor from the encoder.
            training: A boolean indicating whether the layer is in training mode.

        Returns:
            The output tensor.
        """
        # Self attention layer
        # Create proper lookahead_mask (0=allowed, 1=blocked)
        seq_len = tf.shape(inputs)[1]
        causal_mask = 1.0 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

        self_attention_output = self.self_attention(
            inputs, inputs, inputs, attn_mask=causal_mask
        )
        self_attention_output = self.dropout1(self_attention_output)
        self_attention_output = self.layernorm1(self_attention_output + inputs)

        # Cross attention layer
        cross_attention = self.encoder_decoder_attention(
            self_attention_output, encoder_outputs, encoder_outputs
        )
        cross_attention = self.dropout1(cross_attention, training=training)
        cross_attention = self.layernorm2(cross_attention + self_attention_output)

        # Feed Forward layer
        ffn_output = self.ffn(cross_attention)
        ffn_output = self.layernorm3(ffn_output + cross_attention)

        return ffn_output


class TransformerBlock(tf.keras.Model):
    """
    TransformerBlock.

    This class defines the Transformer block, consisting of an encoder and a decoder.
    The Transformer architecture is designed to handle sequence-to-sequence tasks
    by using self-attention mechanisms to capture relationships between elements in
    the input sequence and generate appropriate output sequences.
    """

    def __init__(self, params, attn, name, **kwargs):
        """
        Initializes the TransformerBlock.

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
            attn : (attention_class, kwargs) specification. Each encoder layer, and each
                decoder layer's self-attention and cross-attention, builds its own
                independent instance from this spec (see build_attention).
            **kwargs: Additional keyword arguments for the Keras Model base class.
        """
        super().__init__(name=name, **kwargs)

        self.params = params
        # inputs
        self.inputs = tf.keras.Input(
            shape=(params["batch_size"], params["window_size"], params["n_features"]),
            name="inputs",
        )

        # Embedding Layers
        self.token_embedding = TokenEmbedding(params["d_model"])

        self.encoding = SinusoidalPositionalEncoding(
            params["d_model"],
            # window_params["window_size"]
        )
        # Layer Normalization for Embedding
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Attention specification (attention_class, kwargs); a fresh instance is
        # built per layer/role by the Encoder/Decoder below.
        self.attn = attn
        # Encoder. Subclasses without an encoder stack (PatternDecoder and
        # decoder-only models) never call it, but it is still constructed here,
        # so n_layers_enc changes their random initialisation, not their structure.
        self.encoder = Encoder(
            params["units"],
            params["d_model"],
            params["dropout"],
            params["n_layers_enc"],
            attn,
            params["activation"],
        )

        # Decoder
        self.decoder = Decoder(
            params["units"],
            params["d_model"],
            params["dropout"],
            params["n_layers_dec"],
            attn,
            params["activation"],
        )

        # Global Average Pooling and Dropout layers
        self.global_avg_pooling = tf.keras.layers.GlobalAveragePooling1D()

        # Final Projection Layer
        # The final layer may use l1 and l2 regularization to prevent overfitting
        kernel_regularizer = tf.keras.regularizers.l1_l2(
            l1=params["l1_reg"],
            l2=params["l2_reg"],
        )

        # The weights of the final layer are initialized with RandomNormal
        # (close to zero) so that training converges faster
        kernel_initializer = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=0.001, seed=42
        )

        # Final output layer with l1 l2 regularizer and RandomNorm initializer
        self.output_dense = tf.keras.layers.Dense(
            units=params["forecast_horizon"],
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
        )

    def get_config(self):
        """
        Returns a JSON-serialisable description of the model.

        Keras needs it to write the ``.keras`` checkpoints in compile_and_train;
        newer Keras versions refuse to save without it because ``attn`` holds a
        class. The project only ever reloads the weights (``load_weights``) and
        rebuilds the architecture from the model YAML via ``load_model``, so this
        config describes the model but is not used to reconstruct it.
        """
        attn_class, attn_kwargs = self.attn
        return {
            "name": self.name,
            "trainable": self.trainable,
            "params": {
                key: list(value) if isinstance(value, (list, tuple)) else value
                for key, value in self.params.items()
            },
            "attention": f"{attn_class.__module__}.{attn_class.__qualname__}",
            "attention_kwargs": dict(attn_kwargs),
        }

    def build(self, input_shape):
        """
        Build method for the TransformerBlock.

        Args:
            input_shape: The shape of the input tensor, typically
                        (batch_size, window_size, n_features)
        """

        # Build the embedding layer if needed
        if hasattr(self, "embedding") and hasattr(self.embedding, "build"):
            self.embedding.build(input_shape)

        # Mark the layer as built
        super().build(input_shape)

    def build_model(self):
        """
        Builds and returns the complete Transformer model.

        Creates a Keras functional model by connecting the input layer to the
        transformer block's call method and wrapping it in a Model instance.

        Returns:
            tf.keras.Model: The complete transformer model with defined inputs and outputs.
        """
        outputs = self.call(self.inputs)
        return tf.keras.Model(inputs=self.inputs, outputs=outputs, name="transformer")

    def ffn(self, inputs, training=True):
        """
        Applies the feed-forward network to the inputs.

        Args:
            inputs: The input tensor.
            training: A boolean indicating whether the layer is in training mode.

        Returns:
            The output tensor.
        """
        # Feed-forward network

        # Global Average Pooling
        # Output shape: batch_size x d_model
        avg_output = self.global_avg_pooling(inputs, training=training)

        # Final dense projection layer
        # Output shape: batch_size x forecast_horizon
        dense_output = self.output_dense(avg_output, training=training)

        return dense_output

    def embed(self, inputs, training=True):
        """
        Applies the embedding and positional encoding to the inputs.

        Args:
            inputs: The input tensor.
            training: A boolean indicating whether the layer is in training mode.

        Returns:
            The embedded tensor with positional encoding.
        """
        # Token Embedding
        # Output shape: batch_size x window_size x d_model
        embedding = self.token_embedding(inputs, training=training)

        # Positional Encoding
        # Output shape: batch_size x window_size x d_model
        embedding = self.encoding(embedding)

        return embedding

    def call(self, inputs, training=True):
        """
        Applies the TransformerBlock.

        Args:
            inputs: The input tensor.
            training: A boolean indicating whether the layer is in training mode.

        Returns:
            The output tensor.
        """
        # Input shape: batch_size x window_size x n_features

        # Token Embedding
        # Output shape: batch_size x window_size x d_model
        embedding = self.embed(inputs)

        # Encoder outputs
        # Output shape: batch_size x window_size x d_model
        enc_outputs = self.encoder(embedding, training=training)

        # Decoder outputs
        # Output shape: batch_size x window_size x d_model
        dec_outputs = self.decoder(embedding, enc_outputs, training=training)

        # Final dense projection layer
        # Output shape: batch_size x forecast_horizon
        transformer_output = self.ffn(dec_outputs, training=training)

        return transformer_output
