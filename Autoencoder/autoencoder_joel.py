# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 20:50:39 2021

@author: joelw
"""
import tensorflow as tf
from tensorflow import keras

class Autoencoder(keras.Model):
    """
        This class 
    
    """
    def __init__(self, encoder, decoder, sampler = None, **kwargs):
        """
        The encoder, decoder, and sampler need to be compatible with eachother.
        
        
        Parameters
        ----------
        encoder : Layer/Model
            The enocder. Takes an images and encodes it into a latent space.
            The encoder should be callable with the argument training.
        decoder : Layer/Model
            Decodes a 
        
        """
        super(Autoencoder, self).__init__(**kwargs)
        # Set up metrics
        self.loss_tracker = keras.metrics.Mean(name="total_loss")
        
        # Initiate model structure 
        self.encoder = encoder
        self.decoder = decoder
        
    def call(self, inputs, training = False, **kwargs):
        x = inputs
        x = self.encoder(x, training = training)
        x = self.decoder(x, training = training)
        outputs = x
        return outputs
    
    @property
    def latent_dim(self):
        return self.decoder.input_shape[-1]
    
    def encode(self, inputs, **kwargs):
        latent = self.encoder(inputs, training = False)
        return latent
    
    def get_config(self):
        config = {"encoder": self.encoder,
                  "decoder": self.decoder}
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
    def train_step(self, inputs, **kwargs):
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
            latent            = self.encoder(inputs, training = True)
            reconstruction    = self.decoder(latent, training= True)

            # Calculate losses
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(inputs, reconstruction), axis=(1, 2)
                )
            )
            
            total_loss = reconstruction_loss
        # Calculate and apply gradient
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.loss_tracker.update_state(total_loss)
  
        # Return training message
        return {
            "loss": self.loss_tracker.result(),
        }
    
    
if __name__ == '__main__':
    import numpy as np
    import numpy.random as rnd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow import keras
    from variational_autoencoder import VariationalAutoencoder
    
    # For dimension reduction
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    # for visualization and evaluation
    from latent_plane_mosaic import LatentPlaneMosaic
    from latent_interpolation_mosaic import LatentInterpolationMosaic
    from sample_scatter_gui import SampleScatterGUI
    
    # Setup data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
    mnist_labels = np.concatenate([y_train, y_test], axis=0)
    input_shape = mnist_digits.shape[1:]
    input_shape
    
    # Set up pyper parameters
    latent_dim = 10
    
    # Setup architecture
    encoder = keras.Sequential([
        keras.Input(input_shape),
        keras.layers.Conv2D(4, (3,3), padding = "same", activation = "relu"),
        keras.layers.Conv2D(4, (3,3), padding = "same", activation = "relu"),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Conv2D(8, (3,3), padding = "same", activation = "relu"),
        keras.layers.Conv2D(8, (3,3), padding = "same", activation = "relu"),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Conv2D(16, (3,3), padding = "same", activation = "relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(latent_dim),
    ], name = "Encoder" )
    
    decoder = keras.Sequential([
                keras.Input((latent_dim,)),
                keras.layers.Dense(16*7*7, activation = "relu"),
                keras.layers.Reshape((7,7,16)),
                keras.layers.Conv2DTranspose(16, (3,3), activation = "relu", padding = "same"),
                keras.layers.Conv2DTranspose(16, (3,3), activation = "relu", padding = "same"),
                keras.layers.UpSampling2D((2,2)),
                keras.layers.Conv2DTranspose(8, (3,3), activation = "relu", padding = "same"),
                keras.layers.Conv2DTranspose(8, (3,3), activation = "relu", padding = "same"),
                keras.layers.UpSampling2D((2,2)),
                keras.layers.Conv2DTranspose(8, (3,3), activation = "relu", padding = "same"),
                keras.layers.Conv2DTranspose(1, (3,3), activation = "sigmoid", padding = "same"),
    ], name = "Decoder")
    va = Autoencoder(encoder, decoder)
    va.compile(optimizer = "adam")
    
    
    
    #va.train_step(mnist_digits[0:10])
    
    va.fit(mnist_digits,
          epochs = 1,
          batch_size = 256)
    
    
    
    
    

    