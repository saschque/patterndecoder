# pylint: disable=E1101
"""
Reversible instance normalization (RevIN; Kim et al., 2022) around a model.

Neural forecasters tend to pull their forecasts towards the level of their
training data, which hurts on series whose level moves, such as trading
volume. RevIN normalizes the series input of each window by the window's own
mean and standard deviation, applies a learnable affine transform, and maps
the forecasts back with the same statistics, so that the wrapped model learns
deviations from the level of the window and its forecasts follow that level.
Only the series feature (the first input column) is normalized; the calendar
features stay as they are. Enabled with ``revin: true`` in the config.

Evaluation, fixed before any run with it: the volume use case of
patterndecoder/volume.py (DAX and S&P 500, input_scale 1, calendar of the past
days) and the volatility use case of patterndecoder/volatility.py with the
current setup (input_scale 100, last-step readout), each with ten seeds and
five test windows, compared with the same runs without RevIN. The questions
are whether RevIN narrows the gap to the benchmarks (HAR-X and HAR for volume,
HAR and persistence for volatility) and how it changes the ranking of the
models; the validation MAE decides whether it is used.
"""

import tensorflow as tf

EPSILON = 1e-5


class RevIN(tf.keras.Model):  # pylint: disable=too-many-ancestors
    """
    Wraps a forecasting model with reversible instance normalization.

    Args:
        model (tf.keras.Model): The wrapped model; it receives the normalized
            inputs and keeps its name, which the checkpoint files use.
        input_scale (float): input_scale of the windowed dataset; the forecasts
            are returned in the units of the targets.
    """

    def __init__(self, model, input_scale=1, **kwargs):
        kwargs.setdefault("name", model.name)
        super().__init__(**kwargs)
        self.inner = model
        self.input_scale = float(input_scale)
        self.gamma = self.add_weight(name="revin_gamma", shape=(), initializer="ones")
        self.beta = self.add_weight(name="revin_beta", shape=(), initializer="zeros")

    def call(self, inputs, training=False):
        """Normalizes the series input, runs the model and maps its forecasts back."""
        series = inputs[..., :1]
        mean = tf.reduce_mean(series, axis=1, keepdims=True)
        std = tf.math.reduce_std(series, axis=1, keepdims=True) + EPSILON
        normalized = (series - mean) / std * self.gamma + self.beta
        outputs = self.inner(tf.concat([normalized, inputs[..., 1:]], axis=-1), training=training)
        outputs = (outputs - self.beta) / (self.gamma + EPSILON)
        return (outputs * std[:, 0, :] + mean[:, 0, :]) / self.input_scale

    def get_config(self):
        """
        Describes the wrapper and the wrapped model. As for TransformerBlock,
        Keras needs it to write the checkpoints; the project only reloads weights.
        """
        return {
            "name": self.name,
            "trainable": self.trainable,
            "input_scale": self.input_scale,
            "model": self.inner.get_config(),
        }
