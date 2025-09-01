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
from transformer_models.transformer import TransformerBlock
from transformer_models.autoformer import AutoDecoder, SeriesDecomp

class PatternDecoderBlock(TransformerBlock):
    """
    Decoder-only transformer model for sequence-to-sequence tasks.
    
    This class implements a transformer architecture that bypasses the encoder 
    and utilizes only the decoder block. The PatternDecoder processes input 
    sequences through self-attention mechanisms to capture relationships 
    between elements and generate appropriate output sequences.
    
    Inherits all parameters from TransformerBlock:
        window_params (dict): Window configuration including batch_size,
            observation_window_size, n_features, and output_size.
        transformer_params (dict): Architecture parameters including d_model,
            units, n_layers_dec, and activation.
        training_params (dict): Training configuration including dropout,
            l1_reg, and l2_reg.
        attn (dict): Attention mechanisms with dec_attn for the decoder.
    
    Note:
        Uses only the decoder component, making it suitable for tasks where
        input-output sequences have similar structure and patterns can be
        learned through self-attention alone.
    """
    def __init__(self, window_params, transformer_params, training_params, attn, **kwargs):
        """
        Initializes the AutoPatternDecoderBlock with AutoCorrelation attention.
        
        Sets up the model with AutoDecoder instead of standard decoder to enable
        auto-correlation attention and series decomposition for time series data.
        
        Args:
            window_params (dict): Window configuration including batch_size,
                observation_window_size, n_features, and output_size.
            transformer_params (dict): Architecture parameters including d_model,
                units, n_layers_dec, and activation.
            training_params (dict): Training configuration including dropout.
            attn (dict): Attention mechanisms with enc_attn (used as decoder attention).
            **kwargs: Additional keyword arguments for the Keras Model base class.
        """
        super().__init__(window_params, transformer_params, training_params, attn, **kwargs)
        self.lstm_encode =tf.keras.layers.LSTM(
            transformer_params["d_model"],
            kernel_initializer=tf.keras.initializers.HeNormal(),
            return_sequences=True
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
        enc = self.layernorm1(enc+embedding)

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
    
    Args:
        window_params (dict): Window configuration parameters.
        transformer_params (dict): Architecture parameters.
        training_params (dict): Training configuration parameters.
        attn (dict): Attention mechanisms (should use AutoCorrelation).
        **kwargs: Additional keyword arguments for the Keras Model base class.
    """

    def __init__(self, window_params, transformer_params, training_params, attn, 
                 kernel_size=5, **kwargs):
        """
        Initializes the AutoPatternDecoderBlock with AutoCorrelation attention.
        
        Sets up the model with AutoDecoder instead of standard decoder to enable
        auto-correlation attention and series decomposition for time series data.
        
        Args:
            window_params (dict): Window configuration including batch_size,
                observation_window_size, n_features, and output_size.
            transformer_params (dict): Architecture parameters including d_model,
                units, n_layers_dec, and activation.
            training_params (dict): Training configuration including dropout.
            attn (dict): Attention mechanisms with enc_attn (used as decoder attention).
            **kwargs: Additional keyword arguments for the Keras Model base class.
        """
        super().__init__(window_params, transformer_params, training_params, attn, **kwargs)
        self.decoder = AutoDecoder(
            transformer_params["units"],
            transformer_params["d_model"],
            training_params["dropout"],
            transformer_params["n_layers_dec"],
            attn,
            transformer_params["activation"],
        )
        self.series_decomp = SeriesDecomp(kernel_size)
