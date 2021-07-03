# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 16:56:22 2021

@author: Computer
"""
import tensorflow as tf
from tensorflow import keras

"""
This file contains the class Variational autoencoder. This is a general framework
for how a general VA could look like. 
"""


class VariationalAutoencoder(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        """
            Initiate the variational autoencoder
            
            The class also initiate three types of metrics.
            1) The total loss
            2) The reconstruction loss
                The loss between the input data and the reconstruction. This 
                is set to be binary crossentropy.
            3) The Kullback-Leibler loss
                The Kluback-Leibler divergence between the latent distribution
                given by the mean and variance of the input, and the unit
                normal distribution.
            
            Parameters
            ----------
            encoder : keras.Model
                An encoder which projects the input into some latent space.
            decoder : keras.Model
                An deocder which "un"-projects a point in the latent space
                back to a point in the input space.
            **kwargs : 
                See documentation for keras.Model
                
        """
        # Init the super of 'VaAu' on the instance of 'self'.
        super(VariationalAutoencoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    
    
    # @property is a python decorator. 
    # Declares the function as a property
    # This makes the metrics into a readonly attribute. We can hence call the
    # method simply by writing model.metrics
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]


    def train_step(self, data):
        """
        Performs a single step of training on the data given. 
        
        A self defined train_step function usually consists of the forward
        pass, loss calculation, backpropagation, and metric updates.
            
        Parameters
        ----------
        data : array
            Input data
        
        """
        with tf.GradientTape() as tape:
            # Forward propagation
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Calculate losses
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        # Calculate and apply gradient
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        # Return training message
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    


