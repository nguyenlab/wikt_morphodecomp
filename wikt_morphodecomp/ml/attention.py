#-*- coding: utf8 -*-

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


class Attention(Layer):

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert(len(input_shape) == 3)

        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1],),
                                      initializer='uniform',
                                      trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        a_weights = K.softmax(self.kernel)
        result = K.dot(a_weights, x)

        return K.squeeze(result, 0)


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def get_config(self):
        return super(Attention, self).get_config()

    
