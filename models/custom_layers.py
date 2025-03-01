import tensorflow as tf
from tensorflow.keras.layers import Layer

class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class CustomConv2D(Layer):
    def __init__(self, filters, kernel_size, activation=None, **kwargs):
        super(CustomConv2D, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, activation=activation, padding='same')

    def call(self, inputs):
        return self.conv(inputs)

    def get_config(self):
        config = super(CustomConv2D, self).get_config()
        config.update({
            'filters': self.conv.filters,
            'kernel_size': self.conv.kernel_size,
            'activation': self.conv.activation,
        })
        return config