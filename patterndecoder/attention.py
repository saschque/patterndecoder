# pylint: disable=E1101, R0913, R0903, R0917, R0902, R0801, R0914

"""
Attention mechanisms for deep learning models.

This module implements various attention mechanisms designed for sequence modeling,
time series forecasting, and natural language processing. The implementations focus
on efficiency improvements and specialized patterns for temporal data.

Classes:
    MultiHeadAttention: Standard multi-head attention mechanism from "Attention is All You Need".
    ConvolutionalMultiHeadAttention: Combines 1D convolution with multi-head attention for 
        local patterns.
    LogSparseAttention: Implements logarithmic sparse attention to reduce complexity.
    ConvLogSparseAttention: Combines convolutional processing with logarithmic sparse attention.
    ProbSparseAttention: Probabilistic sparse attention for efficient processing of long sequences.
    AutocorrelationAttention: Frequency-domain attention using auto-correlation for time series.
"""

import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Implements the multi-head attention mechanism as described in "Attention is All You Need".

    This layer projects queries, keys, and values into multiple subspaces (heads),
    applies scaled dot-product attention separately in each head, and concatenates
    the results. This allows the model to focus on different parts of the input sequence.

    Args:
        d_model (int): Dimensionality of the input and output.
        n_heads (int): Number of attention heads.
        **kwargs: Additional keyword arguments for the Keras Layer base class.

    Methods:
        call(xq, xk, xv, mask=None): Applies multi-head attention.
    """
    def __init__(self, d_model, n_heads, **kwargs):
        """
        Initializes the Multi-Head Attention layer.

        Args:
            d_model (int): Dimensionality of the input and output.
            n_heads (int): Number of attention heads.
            **kwargs: Additional keyword arguments for the Keras Layer base class.
        """
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.d_model = d_model
        # Ensure attention_filters is divisible by n_heads
        assert d_model % n_heads == 0, "attention_filters must be divisible by n_heads"
        self.depth = d_model // n_heads

        # Layer components
        self.wq = None
        self.wk = None
        self.wv = None
        self.attention_projection = None

    def _scaled_dot_product_attention(self, q, k, v, mask):
        """
        Calculates the scaled dot-product attention.

        The scaled dot-product attention computes the similarity between queries and keys,
        scales it by the square root of the key dimension (to stabilize gradients), and applies
        a softmax function to generate attention weights. These weights are then used to compute
        a weighted sum of the values.

        Args:
            q (tf.Tensor): Query tensor of shape (..., seq_len_q, depth).
            k (tf.Tensor): Key tensor of shape (..., seq_len_k, depth).
            v (tf.Tensor): Value tensor of shape (..., seq_len_v, depth_v).
            mask (tf.Tensor or None): Optional mask tensor with shape broadcastable to 
                (..., seq_len_q, seq_len_k).

        Returns:
            tf.Tensor: The output tensor after applying attention.
            tf.Tensor: The attention weights.
        """
        batch_size = tf.shape(q)[0]
        qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(q)[-1], tf.float32)
        logits = qk / tf.sqrt(dk)

        if mask is not None:
            if len(mask.shape) == 2:
                mask = tf.expand_dims(mask, 0)
                mask = tf.expand_dims(mask, 1)
                mask = tf.tile(mask, [batch_size, self.n_heads, 1, 1])
            elif len(mask.shape) == 3:
                mask = tf.expand_dims(mask, 1)
                mask = tf.tile(mask, [1, self.n_heads, 1, 1])

            logits += mask * -1e9

        weights = tf.nn.softmax(logits, axis=-1)

        return tf.matmul(weights, v), weights

    def _split_heads(self, x, batch_size):
        """
        Splits the input tensor into multiple attention heads.
        
        Reshapes the input tensor from (batch_size, seq_len, d_model) to 
        (batch_size, n_heads, seq_len, depth) where depth = d_model // n_heads.
        
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            batch_size (int): Batch size dimension.
            
        Returns:
            tf.Tensor: Reshaped tensor of shape (batch_size, n_heads, seq_len, depth).
        """
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth))

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def build(self, input_shape):
        """
        Initializes the query, key, value, and output projection layers.
        
        Creates dense layers for:
        - wq: Query projection layer (input_dim -> d_model)
        - wk: Key projection layer (input_dim -> d_model) 
        - wv: Value projection layer (input_dim -> d_model)
        - attention_projection: Final output projection layer (d_model -> d_model)
        
        Args:
            input_shape: Shape of the input tensor.
        """
        super().build(input_shape)
        self.wq = tf.keras.layers.Dense(self.d_model)
        self.wk = tf.keras.layers.Dense(self.d_model)
        self.wv = tf.keras.layers.Dense(self.d_model)
        self.attention_projection = tf.keras.layers.Dense(self.d_model)

    def call(self, xq, xk, xv, mask=None):
        """
        Applies multi-head attention to the input tensors.
        
        Projects the inputs through query, key, and value transformations,
        splits them into multiple heads, applies scaled dot-product attention,
        and combines the results through a final projection.
        
        Args:
            xq (tf.Tensor): Query tensor of shape (batch_size, seq_len_q, d_model).
            xk (tf.Tensor): Key tensor of shape (batch_size, seq_len_k, d_model).
            xv (tf.Tensor): Value tensor of shape (batch_size, seq_len_v, d_model).
            mask (tf.Tensor, optional): Attention mask of shape (seq_len, seq_len) or 
                (batch_size, seq_len, seq_len). Defaults to None.
        
        Returns:
            tf.Tensor: Output tensor of shape (batch_size, seq_len_q, d_model).
        """
        batch_size = tf.shape(xq)[0]

        q = self.wq(xq)  # (B, L, D)
        k = self.wk(xk)
        v = self.wv(xv)

        q = self._split_heads(q, batch_size)  # (B, H, L, d)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        scaled_attention, _ = self._scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])  # (B, L, H, d)
        concat = tf.reshape(scaled_attention, (batch_size, -1, self.d_model)) # (B, L, D)

        return self.attention_projection(concat)

class ConvolutionalMultiHeadAttention(MultiHeadAttention):
    """
    Convolutional Multi-Head Attention for time series data.

    Combines 1D convolution for local temporal patterns with multi-head attention
    for long-range temporal dependencies.

    Args:
        d_model (int): Dimensionality of the input and output.
        n_heads (int): Number of attention heads.
        kernel_size (int): Size of the 1D convolution kernel. Default is 3.
        strides (int): Convolution strides. Default is 1.
        padding (str): Padding type for convolution. Default is 'causal'.
        dilation_rate (int): Dilation rate for convolution. Default is 1.
        **kwargs: Additional keyword arguments for the Keras Layer base class.
    """
    def __init__(self, d_model, n_heads, kernel_size=3, strides=1, padding='causal',
                 dilation_rate=1, **kwargs):
        """
        Initializes the Convolutional Multi-Head Attention layer.
        
        Args:
            d_model (int): Dimensionality of the input and output.
            n_heads (int): Number of attention heads.
            kernel_size (int): Size of the 1D convolution kernel. Default is 3.
            strides (int): Convolution strides. Default is 1.
            padding (str): Padding type for convolution. Default is 'causal'.
            dilation_rate (int): Dilation rate for convolution. Default is 1.
            **kwargs: Additional keyword arguments for the Keras Layer base class.
        """
        super().__init__(d_model, n_heads, **kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate

    def build(self, input_shape):
        """
        Initialize 1D convolution and temporal attention components for
        the query, key, and value
        
        Creates Conv1D layers for:
        - wq: Query projection layer (input_dim -> d_model)
        - wk: Key projection layer (input_dim -> d_model) 
        - wv: Value projection layer (input_dim -> d_model)
        
        Args:
            input_shape: Shape of the input tensor.
        """
        super().build(input_shape)

        # 1D Convolutional component for local temporal patterns
        self.wq = tf.keras.layers.Conv1D(
            filters=self.d_model,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
        )

        self.wk = tf.keras.layers.Conv1D(
            filters=self.d_model,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
        )

        self.wv = tf.keras.layers.Conv1D(
            filters=self.d_model,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
        )

class LogSparseAttention(MultiHeadAttention):
    """
    Logarithmic Sparse Attention mechanism.

    This layer reduces computational complexity by combining local window-based
    attention with exponentially spaced connections. It achieves O(L log L)
    complexity while capturing both local details and long-range dependencies.

    Args:
        d_model (int): Dimensionality of the input and output.
        n_heads (int): Number of attention heads.
        kernel_size (int): Size of the local window for sparse masking. Default is 3.
        **kwargs: Additional keyword arguments for the Keras Layer base class.

    Methods:
        create_logsparse_mask(seq_length): Creates a combined local + exponential sparse mask.
        call(xq, xk, xv, mask=None): Applies logarithmic sparse attention.
    """
    def create_logsparse_mask(self, seq_length, kernel_size=3):
        """
        Creates a logarithmic sparse mask for attention.

        This method generates a mask that combines local window-based attention with
        exponentially spaced connections, allowing for efficient capture of both local
        and long-range dependencies.

        Args:
            seq_length (int): Length of the input sequence.
            kernel_size (int): Size of the local attention window. Default is 3.

        Returns:
            tf.Tensor: Logarithmic sparse mask of shape (seq_length, seq_length)
                where 1.0 indicates allowed attention and 0.0 indicates blocked attention.
        """
        i = tf.range(seq_length, dtype=tf.int32)
        j = tf.range(seq_length, dtype=tf.int32)
        i, j = tf.meshgrid(i, j, indexing='ij')
        d = i - j

        # Local window mask
        local_mask = (d >= 0) & (d < kernel_size)

        # Exponential step mask
        is_power_of_two = (d > 0) & (tf.bitwise.bitwise_and(d, d - 1) == 0)
        exp_mask = is_power_of_two | (d == 0)

        # Combine masks and ensure causality
        combined_mask = (local_mask | exp_mask) & (d >= 0)

        return tf.cast(combined_mask, tf.float32)

    def call(self, xq, xk, xv, mask=None):
        """
        Applies logarithmic sparse attention to reduce computational complexity.
        
        Creates a logarithmic sparsity mask and combines it with any provided mask
        to limit attention to local windows and exponentially spaced positions.

        Args:
            xq (tf.Tensor): Query tensor of shape (batch_size, seq_len, d_model).
            xk (tf.Tensor): Key tensor of shape (batch_size, seq_len, d_model).
            xv (tf.Tensor): Value tensor of shape (batch_size, seq_len, d_model).
            mask (tf.Tensor, optional): Additional attention mask. Defaults to None.

        Returns:
            tf.Tensor: Output tensor after applying logarithmic sparse attention.
        """
        seq_len = tf.shape(xq)[1]
        log_mask = self.create_logsparse_mask(seq_len)
        if mask is not None:
            # Both masks: 0=allowed, 1=blocked
            # Use maximum to block positions blocked by either mask
            combined_mask = tf.maximum(mask, log_mask)
        else:
            # Encoder case: only use log sparse mask
            combined_mask = log_mask

        att = super().call(xq, xk, xv, combined_mask)

        return att

class ConvLogSparseAttention(LogSparseAttention):
    """
    Convolutional Logarithmic Sparse Attention mechanism.
    
    Combines convolutional processing for local pattern extraction with 
    logarithmic sparse attention for efficient long-range dependency modeling.
    This reduces computational complexity while maintaining the ability to 
    capture both local and global temporal patterns.
    
    Args:
        d_model (int): Dimensionality of the input and output.
        n_heads (int): Number of attention heads.
        kernel_size (int): Size of the 1D convolution kernel. Default is 3.
        strides (int): Convolution strides. Default is 1.
        padding (str): Padding type for convolution. Default is 'causal'.
        dilation_rate (int): Dilation rate for convolution. Default is 1.
        **kwargs: Additional keyword arguments for the Keras Layer base class.
    """
    def __init__(self, d_model, n_heads, kernel_size=3, strides=1, padding='causal',
                 dilation_rate=1, **kwargs):
        """
        Initializes the Convolutional Logarithmic Sparse Attention layer.
        
        Args:
            d_model (int): Dimensionality of the input and output.
            n_heads (int): Number of attention heads.
            kernel_size (int): Size of the 1D convolution kernel. Default is 3.
            strides (int): Convolution strides. Default is 1.
            padding (str): Padding type for convolution. Default is 'causal'.
            dilation_rate (int): Dilation rate for convolution. Default is 1.
            **kwargs: Additional keyword arguments for the Keras Layer base class.
        """
        super().__init__(d_model, n_heads, **kwargs)
        self.conv_filters = d_model
        self.attention_filters = d_model
        self.kernel_size = kernel_size
        self.n_heads = n_heads
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate

        # Ensure attention_filters is divisible by n_heads
        assert d_model % n_heads == 0, "attention_filters must be divisible by n_heads"
        self.depth = d_model // n_heads

        # Layer components
        self.conv1d = None
        self.attention_projection = None
        self.final_projection = None

    def build(self, input_shape):
        """
        Initialize 1D convolution and temporal attention components.
        """
        super().build(input_shape)

        # 1D Convolutional component for local temporal patterns
        self.conv1d = tf.keras.layers.Conv1D(
            filters=self.conv_filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            use_bias=False
        )

        # Projection after attention
        self.attention_projection = tf.keras.layers.Dense(self.attention_filters, use_bias=False)

        # Final projection to control output dimensions
        self.final_projection = tf.keras.layers.Dense(self.attention_filters)

    def call(self, xq, xk, xv, mask=None):
        """
        Applies convolutional logarithmic sparse attention.
        
        Processes the query input through both convolutional and attention paths,
        combines them, and applies a final projection.
        
        Args:
            xq (tf.Tensor): Query input tensor of shape (batch_size, seq_len, features).
            xk (tf.Tensor): Key input tensor of shape (batch_size, seq_len, features).
            xv (tf.Tensor): Value input tensor of shape (batch_size, seq_len, features).
            mask (tf.Tensor, optional): Attention mask tensor. Defaults to None.
            
        Returns:
            tf.Tensor: Output tensor with combined convolutional and attention features.
        """
        #Conv Path
        conv_output = self.conv1d(xq)

        # Attention path from parent  class
        attention_output = super().call(xq, xk, xv, mask)

        # Concatenate convolutional and attention outputs
        combined_output = tf.concat([conv_output, attention_output], axis=-1)

        # Final projection
        output = self.final_projection(combined_output)

        return output

class ProbSparseAttention(MultiHeadAttention):
    """
    ProbSparse Attention mechanism from the Informer model.
    
    Reduces attention complexity by selecting only "active" queries based on 
    query sparsity measurement, achieving O(L log L) complexity instead of O(L²).
    
    Args:
        d_model (int): Dimensionality of the input and output.
        n_heads (int): Number of attention heads.
        c_factor (int): Sampling factor for selecting active queries. Default is 5.
        **kwargs: Additional keyword arguments for the Keras Layer base class.
    """

    def __init__(self, d_model, n_heads, c_factor=5, **kwargs):
        """
        Initializes the ProbSparse Attention layer.
        
        Args:
            d_model (int): Dimensionality of the input and output.
            n_heads (int): Number of attention heads.
            c_factor (int): Sampling factor for selecting active queries. Default is 5.
            **kwargs: Additional keyword arguments for the Keras Layer base class.
        """
        super().__init__(d_model, n_heads, **kwargs)
        self.c_factor = c_factor

    def _calculate_query_sparsity(self, q, k):
        """
        Calculate query sparsity measurement efficiently using sampled keys.
        
        Args:
            q: Query tensor (batch_size, n_heads, seq_len_q, d_k)
            k: Key tensor (batch_size, n_heads, seq_len_k, d_k)
            
        Returns:
            sparsity_measurement: Tensor (batch_size, n_heads, seq_len_q)
        """
        d_k = tf.shape(k)[3]

        q_k = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(d_k, tf.float32))
        max_vals = tf.reduce_max(q_k, axis=-1)
        mean_vals = tf.reduce_mean(q_k, axis=-1)

        sparsity_measurement = max_vals - mean_vals

        return sparsity_measurement

    def _prob_sparse_attention(self, q, k, v, mask=None):
        """
        Implements ProbSparse attention mechanism.
        
        Args:
            q: Query tensor (batch_size, n_heads, seq_len_q, d_k)
            k: Key tensor (batch_size, n_heads, seq_len_k, d_k)  
            v: Value tensor (batch_size, n_heads, seq_len_v, d_v)
            mask: Optional mask tensor
            
        Returns:
            output: Attention output tensor
            attention_weights: Attention weights for selected queries
        """
        batch_size = tf.shape(q)[0]
        n_heads = tf.shape(q)[1]
        seq_len_q = tf.shape(q)[2]
        d_k = tf.shape(q)[3]

        # Calculate number of active queries to select: c = c_factor * ln(L_Q)
        c = tf.cast(
            tf.math.ceil(self.c_factor * tf.math.log(tf.cast(seq_len_q, tf.float32))),
            tf.int32
        )
        c = tf.minimum(c, seq_len_q)

        # Calculate query sparsity measurement
        sparsity_measurement = self._calculate_query_sparsity(q, k)

        # Select top-c active queries
        _, top_indices = tf.nn.top_k(sparsity_measurement, k=c)

        # Create indices for gathering selected queries
        batch_indices = tf.range(batch_size)[:, None, None]
        head_indices = tf.range(n_heads)[None, :, None]
        batch_indices = tf.tile(batch_indices, [1, n_heads, c])
        head_indices = tf.tile(head_indices, [batch_size, 1, c])
        gather_indices = tf.stack([batch_indices, head_indices, top_indices], axis=-1)

        # Gather selected queries (Q_reduced)
        q_reduced = tf.gather_nd(q, gather_indices)  # (batch_size, n_heads, c, d_k)

        # Compute attention scores for selected queries only
        scores = tf.matmul(q_reduced, k, transpose_b=True) / tf.sqrt(tf.cast(d_k, tf.float32))

        # Apply mask to selected queries if provided

        if mask is not None:
            mask_expanded = tf.expand_dims(tf.expand_dims(mask, 0), 0)
            mask_expanded = tf.tile(mask_expanded, [batch_size, n_heads, 1, 1])
            # [batch, heads, c, seq_len_k]
            mask_reduced = tf.gather_nd(mask_expanded, gather_indices)
            scores += (mask_reduced * -1e9)

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scores, axis=-1)

        # Apply attention to values
        attended_values = tf.matmul(attention_weights, v)  # (batch_size, n_heads, c, d_k)

        # Create output tensor and place attended values at correct positions
        output = tf.zeros([batch_size, n_heads, seq_len_q, tf.shape(v)[-1]], dtype=v.dtype)

        # For lazy queries, Initialize with mean of values
        mean_attended = tf.reduce_mean(attended_values, axis=2, keepdims=True)
        mean_attended = tf.tile(mean_attended, [1, 1, seq_len_q, 1])
        output = output + mean_attended

        # Update selected positions with computed attention values
        output = tf.tensor_scatter_nd_update(output, gather_indices, attended_values)

        return output, attention_weights

    def call(self, xq, xk, xv, mask=None):
        """
        Applies ProbSparse attention to reduce computational complexity.
        
        Selects active queries based on sparsity measurement and applies attention
        only to these queries, achieving O(L log L) complexity instead of O(L²).
        
        Args:
            xq (tf.Tensor): Query input tensor of shape (batch_size, seq_len, d_model).
            xk (tf.Tensor): Key input tensor of shape (batch_size, seq_len, d_model).
            xv (tf.Tensor): Value input tensor of shape (batch_size, seq_len, d_model).
            mask (tf.Tensor, optional): Attention mask tensor. Defaults to None.
            
        Returns:
            tf.Tensor: Output tensor after applying ProbSparse attention.
        """
        batch_size = tf.shape(xq)[0]

        # Project inputs to query, key, value
        q = self.wq(xq)
        k = self.wk(xk)
        v = self.wv(xv)

        # Split into multiple heads
        q = self._split_heads(q, batch_size)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        # Apply ProbSparse attention
        scaled_attention, _ = self._prob_sparse_attention(q, k, v, mask)

        # Reshape back to original dimensions
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])
        concat = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # Final linear projection
        return self.attention_projection(concat)

class AutocorrelationAttention(MultiHeadAttention):
    """
    AutoCorrelation-based Attention mechanism for time series forecasting.

    This layer replaces traditional self-attention with an auto-correlation mechanism,
    which captures time-delay dependencies in the frequency domain using FFT. It is
    effective for periodic or seasonal time series data.

    Args:
        d_model (int): Dimensionality of the input and output.
        n_heads (int): Number of attention heads.
        **kwargs: Additional keyword arguments for the Keras Layer base class.

    Methods:
        call(xq, xk, xv, mask=None): Applies auto-correlation-based attention
            in the frequency domain.
    """

    def call(self, xq, xk, xv, mask=None):
        """
        Applies auto-correlation-based attention in the frequency domain.
        
        Uses Fast Fourier Transform (FFT) to compute auto-correlation between
        queries and keys, selects top-k correlated positions, and applies
        attention weights to values based on correlation scores.

        Args:
            xq (tf.Tensor): Query tensor of shape (batch_size, seq_len, d_model).
            xk (tf.Tensor): Key tensor of shape (batch_size, seq_len, d_model).
            xv (tf.Tensor): Value tensor of shape (batch_size, seq_len, d_model).
            mask (tf.Tensor, optional): Attention mask (currently not used in 
                autocorrelation). Defaults to None.

        Returns:
            tf.Tensor: Output tensor after applying autocorrelation-based attention.
        """
        batch_size = tf.shape(xq)[0]
        seq_len = tf.shape(xq)[1]

        # Project inputs to query, key, value
        q = self.wq(xq)
        k = self.wk(xk)
        v = self.wv(xv)

        # Split into multiple heads
        q = self._split_heads(q, batch_size)  # [batch, heads, seq, depth]
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        # Get dimensions for later use
        depth = tf.shape(q)[-1]

        # Rearrange for FFT: [batch, heads, seq, depth] -> [batch, heads, depth, seq]
        q_t = tf.transpose(q, [0, 1, 3, 2])
        k_t = tf.transpose(k, [0, 1, 3, 2])

        # Compute FFT along the last axis (sequence dimension)
        q_fft = tf.signal.rfft(q_t)  # Complex tensor
        k_fft = tf.signal.rfft(k_t)  # Complex tensor

        # Compute autocorrelation in the frequency domain
        # Shape: [batch, heads, depth, fft_length]
        autocorr_fft = q_fft * tf.math.conj(k_fft)

        # Inverse FFT to get autocorrelation - specify exact output length
        autocorr = tf.signal.irfft(autocorr_fft, fft_length=[seq_len])

        # Transpose back: [batch, heads, depth, seq] -> [batch, heads, seq, depth]
        autocorr = tf.transpose(autocorr, [0, 1, 3, 2])

        # Ensure we only take the first seq_len elements
        autocorr = autocorr[:, :, :seq_len, :]

        # Scale autocorrelation
        scaled_autocorr = autocorr / tf.math.sqrt(tf.cast(depth, tf.float32))

        # Apply mask if provided
        if mask is not None:
            mask_expanded = tf.expand_dims(tf.expand_dims(mask, 0), 0)
            mask_expanded = tf.tile(mask_expanded, [batch_size, self.n_heads, 1, 1])
            #pass  # Skip masking for now to avoid shape issues

        # Select top-k correlations for each position
        # scaled_autocorr shape: [batch, heads, seq, depth]
        autocorr_seq_len = tf.shape(scaled_autocorr)[2]
        top_k = tf.maximum(autocorr_seq_len // 5, 1)
        top_k = tf.minimum(top_k, autocorr_seq_len)

        # Find top correlations across sequence dimension (axis=2)
        top_values, top_indices = tf.math.top_k(
            tf.reduce_mean(scaled_autocorr, axis=-1),  # Average across depth
            k=top_k
        )

        # Gather values based on top indices
        # top_indices shape: [batch, heads, top_k]
        # v shape: [batch, heads, seq, depth]

        # Create proper indices for gathering
        batch_idx = tf.range(batch_size)[:, None, None]
        head_idx = tf.range(self.n_heads)[None, :, None]
        batch_idx = tf.tile(batch_idx, [1, self.n_heads, top_k])
        head_idx = tf.tile(head_idx, [batch_size, 1, top_k])

        gather_indices = tf.stack([batch_idx, head_idx, top_indices], axis=-1)

        # Gather values: [batch, heads, top_k, depth]
        v_selected = tf.gather_nd(v, gather_indices)

        # Compute attention weights and weighted sum
        attention_weights = tf.nn.softmax(top_values, axis=-1)  # [batch, heads, top_k]

        # Weighted sum: [batch, heads, depth]
        attended_values = tf.einsum('bhk,bhkd->bhd', attention_weights, v_selected)

        # Expand to match original sequence length
        # [batch, heads, depth] -> [batch, heads, seq, depth]
        output = tf.tile(
            tf.expand_dims(attended_values, axis=2),
            [1, 1, seq_len, 1]
        )

        # Reshape back to original dimensions
        # [batch, heads, seq, depth] -> [batch, seq, heads, depth]
        output = tf.transpose(output, [0, 2, 1, 3])

        # [batch, seq, heads, depth] -> [batch, seq, d_model]
        output = tf.reshape(output, [batch_size, seq_len, self.d_model])

        # Final projection
        return self.attention_projection(output)


class FullAutocorrelationAttention(MultiHeadAttention):
    """
    Full AutoCorrelation-based Attention mechanism for time series forecasting.

    This layer replaces traditional self-attention with an auto-correlation mechanism,
    which captures time-delay dependencies in the frequency domain using FFT.

    Args:
        d_model (int): Dimensionality of the input and output.
        n_heads (int): Number of attention heads.
        **kwargs: Additional keyword arguments for the Keras Layer base class.

    Methods:
        call(xq, xk, xv, mask=None): Applies the full auto-correlation-based attention
            in the frequency domain.
    """
    def call(self, xq, xk, xv, mask=None):
        """
        Applies full auto-correlation-based attention in the frequency domain.
        
        Uses Fast Fourier Transform (FFT) to compute auto-correlation between
        queries and keys, selects top-k correlated positions, and applies
        attention weights to values based on correlation scores.

        Args:
            xq (tf.Tensor): Query tensor of shape (batch_size, seq_len, d_model).
            xk (tf.Tensor): Key tensor of shape (batch_size, seq_len, d_model).
            xv (tf.Tensor): Value tensor of shape (batch_size, seq_len, d_model).
            mask (tf.Tensor, optional): Attention mask (currently not used in 
                autocorrelation). Defaults to None.

        Returns:
            tf.Tensor: Output tensor after applying autocorrelation-based attention.
        """
        batch_size = tf.shape(xq)[0]
        seq_len = tf.shape(xq)[1]

        # Project inputs
        q = self.wq(xq)
        k = self.wk(xk)
        v = self.wv(xv)

        # Split into multiple heads
        q = self._split_heads(q, batch_size)  # [batch, heads, seq, depth]
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        # Compute autocorrelation in frequency domain
        q_t = tf.transpose(q, [0, 1, 3, 2])  # [batch, heads, depth, seq]
        k_t = tf.transpose(k, [0, 1, 3, 2])

        q_fft = tf.signal.rfft(q_t)
        k_fft = tf.signal.rfft(k_t)
        autocorr_fft = q_fft * tf.math.conj(k_fft)
        autocorr = tf.signal.irfft(autocorr_fft, fft_length=[seq_len])
        autocorr = tf.transpose(autocorr, [0, 1, 3, 2])  # [batch, heads, seq, depth]

        # Attention weights computation (fixed shape handling)
        scaled_autocorr = autocorr / tf.math.sqrt(tf.cast(tf.shape(q)[-1], tf.float32))
        # [batch, heads, seq]
        autocorr_scores = tf.reduce_mean(scaled_autocorr, axis=-1)
        attention_weights = tf.nn.softmax(autocorr_scores, axis=-1)

        # Apply attention weights directly (avoiding problematic FFT aggregation)
        # Shape-safe implementation
        # [batch, heads, seq, 1]
        attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1)
        attended_values = v * attention_weights_expanded  # [batch, heads, seq, depth]

        # Sum over sequence dimension for aggregation
        # [batch, heads, 1, depth]
        output = tf.reduce_sum(attended_values, axis=2, keepdims=True)
        output = tf.tile(output, [1, 1, seq_len, 1])  # [batch, heads, seq, depth]

        # Reshape and project
        output = tf.transpose(output, [0, 2, 1, 3])  # [batch, seq, heads, depth]
        output = tf.reshape(output, [batch_size, seq_len, self.d_model])
        return self.attention_projection(output)
