# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 16:58:33 2021

@author: Computer
"""
import tensorflow as tf

# Define a sampling layer
class NormalSamplingLayer(tf.keras.layers.Layer):
    """
        This class inherits frow keras klass Layer. It takes in a vector of
        means and a vector of log-variances. The input should be anle to be
        unpacked. Then for each input, we will sample from a normal
        distribution with the given mean and varaices. This sample is what is 
        fed forward.
    """
    
    def __init__(self, **kwargs):
        super(NormalSamplingLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        pass
        

    def call(self, inputs):
        """
            Defines the arcitechture with the input and sends the signal
            forward.
            
            The input consists of the mean and log-variances, and we then 
            sample a new sample from the corresponding normal distribution
            which is then passed forward.
            
            Parameters
            ----------
            inputs : tuple/list (None,2,None)
                Containimg the array of means and log variances.
        """
        z_mean, z_log_var = tf.unstack(inputs, axis = 1)
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z_samp = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z_samp
    
    def get_config(self):
        config = super(NormalSamplingLayer,self).get_config()
        return config

if __name__ == '__main__':
    import numpy as np
    layer = NormalSamplingLayer()
    x = tf.constant(np.arange(2000, dtype = "float32").reshape(100,2,10))
    inputs = tf.keras.Input((2,10))
    
    y = layer(x)
    
    






