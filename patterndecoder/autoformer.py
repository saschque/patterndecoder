# pylint: disable=E1101, R0913, R0903, R0917, R0902, R0801

"""
Autoformer model implementation for time series forecasting.

This module implements the Autoformer model based on the paper "Autoformer:
Decomposition Transformers with Auto-Correlation for Long-Term Time Series Forecasting".

The Autoformer addresses limitations of standard Transformers for long sequence
forecasting through auto-correlation mechanisms and series decomposition that
separates trend and seasonal components.

Classes:
    SeriesDecomp: Decomposition layer that separates time series into trend and seasonal components.
    AutoEncoder: Encoder with auto-correlation attention and series decomposition.
    AutoEncoderLayer: Building block with self-attention, feed-forward, and decomposition 
        components.
    AutoDecoder: Decoder with auto-correlation attention and trend-seasonal processing.
    AutoDecoderLayer: Decoder layer with self-attention, cross-attention, and decomposition.
    AutoformerBlock: Complete Autoformer model combining encoder and decoder.

Key Features:
    - Auto-correlation mechanism for discovering time-delay dependencies
    - Series decomposition to separate trend and seasonal components
    - Fast Fourier Transform for efficient frequency domain computation
    - Progressive decomposition throughout the network layers
"""

import tensorflow as tf
from patterndecoder.transformer import TransformerBlock
from patterndecoder.transformer import EncoderLayer as TransformerEncoderLayer
from patterndecoder.transformer import DecoderLayer as TransformerDecoderLayer

class SeriesDecomp(tf.keras.layers.Layer):
    """
    Series Decomposition layer that separates a time series into trend and seasonal components.
    
    This layer uses a moving average filter to extract the trend component,
    and the residual becomes the seasonal component, enabling the model to
    process each component separately for better forecasting performance.
    
    Args:
        kernel_size (int): The size of the moving average kernel for trend extraction.
        **kwargs: Additional keyword arguments for the Keras Layer base class.
    """
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.moving_avg = tf.keras.layers.AveragePooling1D(
            pool_size=kernel_size,
            strides=1,
            padding='same'
        )

    def call(self, x):
        """
        Applies series decomposition to separate trend and seasonal components.
        
        Args:
            x (tf.Tensor): Input time series tensor of shape (batch_size, seq_len, features).
        
        Returns:
            tuple: A tuple containing:
                - res (tf.Tensor): Seasonal component (residual after trend removal).
                - moving_avg (tf.Tensor): Trend component (moving average).
        """
        moving_avg = self.moving_avg(x)
        res = x - moving_avg
        return res, moving_avg

class AutoEncoder(tf.keras.layers.Layer):
    """
    Encoder for the Autoformer model.

    This layer consists of a stack of EncoderLayer instances, each applying
    series decomposition and auto-correlation mechanisms to the input sequence.

    Args:
        units (int): The number of units in the feed-forward network.
        d_model (int): The dimensionality of the model.
        dropout (float): The dropout rate.
        n_layers (int): The number of encoder layers.
        attn_type (tf.keras.layers.Layer): The type of attention mechanism to use
    Attributes:
        layers (list): A list of AutoEncoderLayer instances.
    """
    def __init__(self, units, d_model, dropout, n_layers, attn_type, activation,
                 kernel_size=5, **kwargs):
        """
        Initialize the Autoformer Encoder.
        
        Args:
            units (int): The number of units in the feed-forward network.
            d_model (int): The dimensionality of the model.
            dropout (float): The dropout rate.
            n_layers (int): The number of encoder layers.
            attn_type (tf.keras.layers.Layer): The type of attention mechanism to use.
            activation (str): Activation function for the feed-forward network.
            kernel_size (int): Kernel size for series decomposition. Default is 5.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        super().__init__(**kwargs)
        self.layers = [
            AutoEncoderLayer(
                units,
                d_model,
                dropout,
                attn_type,
                activation,
                kernel_size,
                name=f"autoencoder_layer_{i}"
            )
            for i in range(n_layers)
        ]
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=True):
        """
        Applies the Autoformer Encoder with auto-correlation and decomposition.
        
        Processes inputs through all encoder layers, each applying auto-correlation
        attention and series decomposition to separate trend and seasonal components.
        
        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            training (bool): Whether the model is in training mode. Defaults to True.
        
        Returns:
            tf.Tensor: Encoded output tensor with layer normalization applied.
        """
        enc_outputs = inputs
        for encoder_layer in self.layers:
            enc_outputs = encoder_layer(enc_outputs, training=training)

        return self.layernorm1(enc_outputs)

class AutoEncoderLayer(TransformerEncoderLayer):
    """
    EncoderLayer for the Autoformer model.
    
    This layer includes self-attention (using auto-correlation), a feed-forward network,
    layer normalization, dropout, and series decomposition. The series decomposition
    separates the signal into trend and seasonal components after the self-attention,
    allowing the model to process each component more effectively.
    
    Args:
        units: The number of units in the feed-forward network.
        d_model: The dimensionality of the model.
        dropout: The dropout rate.
        attn_type: The type of attention mechanism to use (AutoCorrelation).
        kernel_size: The kernel size for series decomposition.
        name: The name of the layer.
    """
    def __init__(self, units, d_model, dropout, attn_type, activation,
                 kernel_size=5, name="autoencoder_layer", **kwargs):
        """
        Initializes the Autoformer EncoderLayer.
        
        Args:
            units (int): The number of units in the feed-forward network.
            d_model (int): The dimensionality of the model.
            dropout (float): The dropout rate.
            attn_type: The type of attention mechanism to use (AutoCorrelation).
            activation (str): Activation function for the feed-forward network.
            kernel_size (int): The kernel size for series decomposition. Default is 5.
            name (str): The name of the layer. Default is "autoencoder_layer".
            **kwargs: Additional keyword arguments for the Keras Layer base class.
        """
        super().__init__(units, d_model, dropout, attn_type, activation,
                         name=name, **kwargs)

        # initialize Autoformer-specific layers
        self.decomp1 = SeriesDecomp(kernel_size)
        self.decomp2 = SeriesDecomp(kernel_size)

    def build(self, input_shape):
        """
        Build method for the Autoformer EncoderLayer.
        
        Args:
            input_shape: The shape of the input tensor, typically
                        (batch_size, window_size, n_features)
        """
        # To suppress the warnings
        super().build(input_shape)
        self.layernorm1 = None
        self.layernorm2 = None


    def call(self, inputs, training=True):
        """
        Applies the Autoformer EncoderLayer with decomposition.
        
        Processes inputs through auto-correlation self-attention, applies series
        decomposition, then processes through convolutional feed-forward layers
        with additional decomposition to extract seasonal components.
        
        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            training (bool): Whether the layer is in training mode. Defaults to True.
        
        Returns:
            tf.Tensor: Output tensor after auto-correlation attention and decomposition.
        """

        # Self attention layer
        self_attention_output = self.self_attention(inputs, inputs, inputs)
        self_attention_output = self.dropout1(self_attention_output, training=training)

        #Series Decomp
        res1, __ = self.decomp1(self_attention_output + inputs)

        # Feed Forward layer
        ffn = self.ffn(res1)
        res2, _ = self.decomp2(res1 + ffn)

        return res2

class AutoDecoder(tf.keras.layers.Layer):
    """
    Decoder for the Autoformer model with auto-correlation and decomposition.
    
    This decoder processes input sequences through multiple AutoDecoderLayer instances,
    each applying auto-correlation attention, cross-attention with encoder outputs,
    and series decomposition. It maintains separate trend and seasonal components
    throughout the decoding process.
    
    Args:
        units (int): The number of units in the feed-forward network.
        d_model (int): The dimensionality of the model.
        dropout (float): The dropout rate.
        n_layers (int): The number of decoder layers.
        attn_type: The type of attention mechanism to use (AutoCorrelation).
        activation (str): Activation function for the feed-forward network.
        kernel_size (int): Kernel size for series decomposition. Default is 5.
        **kwargs: Additional keyword arguments for the Keras Layer base class.
    """
    def __init__(self, units, d_model, dropout, n_layers, attn_type, activation,
                 kernel_size=5, **kwargs):
        """
        Initialize the Autoformer Decoder.
        
        Args:
            units (int): The number of units in the feed-forward network.
            d_model (int): The dimensionality of the model.
            dropout (float): The dropout rate.
            n_layers (int): The number of decoder layers.
            attn_type: The type of attention mechanism to use (AutoCorrelation).
            activation (str): Activation function for the feed-forward network.
            kernel_size (int): Kernel size for series decomposition. Default is 5.
            **kwargs: Additional keyword arguments for the Keras Layer base class.
        """
        super().__init__(**kwargs)
        self.layers = [
            AutoDecoderLayer(
                units,
                d_model,
                dropout,
                attn_type,
                activation,
                kernel_size,
                name=f"autodecoder_layer_{i}"
            )
            for i in range(n_layers)
        ]
        self.series_decomp = SeriesDecomp(kernel_size)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)


    def call(self, inputs, enc_outputs, training=True):
        """
        Applies the Autoformer Decoder with trend-seasonal decomposition.
        
        Initializes with series decomposition to separate trend and seasonal components,
        then processes through decoder layers while accumulating trend information.
        
        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            enc_outputs (tf.Tensor): Encoder output tensor for cross-attention.
            training (bool): Whether the model is in training mode. Defaults to True.
        
        Returns:
            tf.Tensor: Final output combining seasonal and trend components.
        """
        init_res, init_trend = self.series_decomp(inputs)
        dec_outputs = tf.zeros_like(init_res)
        trend = init_trend
        for decoder_layer in self.layers:
            dec_outputs, residual_trend = decoder_layer(dec_outputs, enc_outputs, training=training)
            trend = trend + residual_trend
        return self.layernorm1(dec_outputs + trend)

class AutoDecoderLayer(TransformerDecoderLayer):
    """
    Decoder layer for the Autoformer model with decomposition and auto-correlation.
    
    This layer includes self-attention (using auto-correlation), cross-attention
    with encoder outputs, feed-forward processing, and multiple series decomposition
    blocks. It processes both seasonal and trend components separately and returns
    both the seasonal output and accumulated trend information.
    
    Args:
        units (int): The number of units in the feed-forward network.
        d_model (int): The dimensionality of the model.
        dropout (float): The dropout rate.
        attn_type: The type of attention mechanism to use (AutoCorrelation).
        activation (str): Activation function for the feed-forward network. Default is "relu".
        kernel_size (int): Kernel size for series decomposition. Default is 5.
        name (str): The name of the layer. Default is "autodecoder_layer".
        **kwargs: Additional keyword arguments for the Keras Layer base class.
    """
    def __init__(self, units, d_model, dropout, attn_type, activation="relu",
                 kernel_size=5, name="autodecoder_layer", **kwargs):
        """
        Initializes the Autoformer DecoderLayer.
        
        Args:
            units (int): The number of units in the feed-forward network.
            d_model (int): The dimensionality of the model.
            dropout (float): The dropout rate.
            attn_type: The type of attention mechanism to use (AutoCorrelation).
            activation (str): Activation function for the feed-forward network. Default is "relu".
            kernel_size (int): Kernel size for series decomposition. Default is 5.
            name (str): The name of the layer. Default is "autodecoder_layer".
            **kwargs: Additional keyword arguments for the Keras Layer base class.
        """
        super().__init__(units, d_model, dropout, attn_type, activation,
                         name=name, **kwargs)

        self.kernel_size = kernel_size
        self.d_model = d_model

        # Initialize Autoformer-specific layer
        self.decomp1 = SeriesDecomp(kernel_size)
        self.decomp2 = SeriesDecomp(kernel_size)
        self.decomp3 = SeriesDecomp(kernel_size)

        self.projection = tf.keras.layers.Conv1D(
            filters = d_model,
            kernel_size=3,
            padding="causal",
            activation = activation
        )

    def build(self, input_shape):
        """
        Build method for the Autoformer DecoderLayer.
        
        Args:
            input_shape: The shape of the input tensor, typically
                        (batch_size, window_size, n_features)
        """
        super().build(input_shape)
        # To suppress the warnings
        self.layernorm1 = None
        self.layernorm2 = None
        self.layernorm3 = None

    def call(self, inputs, encoder_outputs, training=True):
        """
        Applies the Autoformer DecoderLayer with trend-seasonal processing.
        
        Processes inputs through masked self-attention, series decomposition,
        cross-attention with encoder outputs, additional decomposition, and
        feed-forward layers, tracking both seasonal and trend components.
        
        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            encoder_outputs (tf.Tensor): Encoder output tensor for cross-attention.
            training (bool): Whether the layer is in training mode. Defaults to True.
        
        Returns:
            tuple: A tuple containing:
                - res (tf.Tensor): Seasonal component output.
                - residual_trend (tf.Tensor): Accumulated trend component.
        """
        # Create a causal mask for self attentions
        seq_len = tf.shape(inputs)[1]
        causal_mask = 1.0 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

        # Self attention layer
        self_attention_output = self.self_attention(inputs, inputs, inputs, mask=causal_mask)

        self_attention_output = self.dropout1(self_attention_output)
        #self_attention_output = self.layernorm1(inputs + self_attention_output)

        # Series Decomposition 1
        res, trend1 = self.decomp1(self_attention_output + inputs)

        # Cross attention layer
        cross_attention = self.encoder_decoder_attention(res, encoder_outputs, encoder_outputs)
        cross_attention = self.dropout1(cross_attention, training=training)
        #cross_attention = self.layernorm2(self_attention_output + cross_attention)

        # Series Decomposition 2
        res, trend2 = self.decomp2(cross_attention + res)

        # Feed Forward layer
        ffn = self.ffn(res)
        #ffn = self.layernorm3(ffn + res)

        # Series Decomposition 3
        res, trend3 = self.decomp3(ffn + res)

        # Trend Projection
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend)

        return res, residual_trend

class AutoformerBlock(TransformerBlock):
    """
    Complete Autoformer model block for time series forecasting.
    
    Combines an Autoformer encoder and decoder with auto-correlation attention
    and series decomposition. The model separates trend and seasonal components
    throughout processing, enabling better long-term time series forecasting.
    
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
            (AutoCorrelation recommended)
    
    Note:
        Uses AutoCorrelation attention and series decomposition throughout
        the architecture for improved time series forecasting performance.
    """
    def __init__(self, params, attn, name, **kwargs):
        """
        Initializes the AutoformerBlock.
        """
        super().__init__(params, attn, name, **kwargs)
        self.encoder = AutoEncoder(
            params["units"],
            params["d_model"],
            params["dropout"],
            params["n_layers_enc"],
            attn,
            params["activation"],
        )

        self.decoder = AutoDecoder(
            params["units"],
            params["d_model"],
            params["dropout"],
            params["n_layers_enc"],
            attn,
            params["activation"],
        )

    def call(self, inputs, training=True):
        """
        Applies the complete Autoformer model for time series forecasting.
        
        Processes inputs through the Autoformer encoder with auto-correlation
        and decomposition, then through the decoder to generate forecasts
        while maintaining trend and seasonal component separation.
        
        Args:
            inputs (tf.Tensor): Input time series tensor of shape
                (batch_size, window_size, n_features).
            training (bool): Whether the model is in training mode. Defaults to True.
        
        Returns:
            tf.Tensor: Predicted output tensor of shape (batch_size, output_size).
        """
        outputs = super().call(inputs, training=training)
        return outputs
