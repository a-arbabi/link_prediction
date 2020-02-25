import tensorflow as tf
from models import transformer


class config:
    num_layers = 3
    d_model = 256
    num_heads = 4
    dff = 256


class TrecModel(tf.keras.layers.Layer):
    def __init__(self, config, hpo_size):
        super(TrecModel, self).__init__()
        self.encoder = transformer.Encoder(
            config.num_layers,
            config.d_model,
            config.num_heads,
            config.dff,
            hpo_size+2,
            rate=0.1,
        )
        self.decoder = tf.keras.layers.Dense(hpo_size)

    def call(self, x_inp, training, mask):
#        x_inp = x['hpo_inputs']
        out = self.encoder(x_inp, training, mask)
        out = self.decoder(out[:,0,:])
        return out