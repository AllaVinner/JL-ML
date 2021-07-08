# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 18:39:40 2021

@author: Computer
"""


import tensorflow as tf
from tensorflow import keras

class Autoencoder(keras.Model):
    
    def __init__(self, encoder, decoder, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        
        self.total_loss_tracker = keras.metrics.Mean(name = 'total_loss')
        
    
    @property
    def metrics(self):
        return [self.total_loss_tracker]
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                        keras.losses.binary_crossentropy(data,reconstruction),
                        axis = (1,2)
                )
            )
            
            total_loss = reconstruction_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients( zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        
        return {"Loss": self.total_loss_tracker.result()}





