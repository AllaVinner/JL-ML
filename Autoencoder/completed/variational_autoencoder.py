# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 20:50:39 2021

@author: joelw
"""
import tensorflow as tf
from tensorflow import keras
import variational_autoencoder_models as vae_models
import keras
import numpy as np

class VariationalAutoencoder(keras.Model):
    """
        This class 
    
    """
    def __init__(self, encoder, decoder, latent_loss = None,  **kwargs):
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
        super(VariationalAutoencoder, self).__init__(**kwargs)
        # Initiate model structure 
        self.encoder, self.decoder = encoder, decoder
        self.sampler = NormalSamplingLayer()
        
        if self.latent_loss is None:
            self.latent_loss =  keras.losses.BinaryCrossentropy
        else:
            self.latent_loss = latent_loss
            
        #if encoder is built, check that it is compatible with the given
        #decoder
        if self.encoder.built:
            pass
            #self._check_encoder_decoder_compatibility(self.encoder.input.shape)
        
        
    @tf.function
    def call(self, inputs, training = False, **kwargs):
        encoded_distribution = self.encoder(inputs)
        latent_sample = self.sampler(encoded_distribution, training = training)
        outputs = self.decoder(latent_sample)
        
        #self.add_loss(self.latent_loss(encoded_distribution, encoded_distribution))
        self.add_loss(lambda inputs, outputs: keras.losses.mse(inputs, outputs))
        return outputs
   

    @property
    def latent_dim(self):
        return self.decoder.input_shape[-1]
    
    def encode(self, inputs, **kwargs):
        latent = self.encoder(inputs, training = False)
        z_mean = self.sampler(latent, training = False)
        return z_mean
    
    def get_config(self):
        config = {"encoder": self.encoder,
                  "decoder": self.decoder}
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
    def build(self, input_shape):
        """
            Check if the given encoder and decoder are compatible
            before building.
        """
        #self._check_encoder_decoder_compatibility(input_shape)
        super(VariationalAutoencoder, self).build(input_shape)
        
    
    @tf.function
    def train_step(self, inputs, **kwargs):
        with tf.GradientTape() as tape:
            encoded_distribution = self.encoder(inputs)
            latent_sample = self.sampler(encoded_distribution)
            outputs = self.decoder(latent_sample)
            loss = self.compute_loss(inputs, outputs, encoded_distribution, latent_sample)
    
    def compute_loss(self, inputs, outputs, encoded_distribution, latent_sample):
        reconstruction_loss = self.loss(inputs, outputs)
        
        latent_loss = self.latent_loss(encoded_distribution, latent_sample)
        
        return latent_loss + reconstruction_loss
 
    def _latent_loss(encoded_distribution, latent_sample):
        # TODO
        pass
    
    def _reconstruciton_loss(inputs, outputs):
        # TODO
        pass
 
    def _check_encoder_decoder_compatibility(self, input_shape):
        """
            Checks if the encoder and decoder are compatible with the given
            shape of the input. It raises ValueErrors in three occations.
            1. The encoder is built, and its input shape does not match the 
            given input shape.
            2. The decoder is built, and its input_shape does not match the
            output shape of the encoder.
            3. The input of the encoder do not match the output of the decoder.

        Parameters
        ----------
        input_shape : TensorShape
            The shape of the tensor fed to the model. Note, including the 
            batch dimensios as None.

        Returns
        -------
        None.

        """
        #1. if encoder is built, checks if its input shape is compatible with
        #the passed input shape.
        #else the encoder is built.
        if self.encoder.built:
            input_shape.assert_is_compatible_with(self.encoder.input.shape)
        else:
            inputs = keras.Input(batch_input_shape = input_shape) 
            encoded = self.encoder(inputs)
        
        #2. if decoder is built, checks if its input shape is compatible with
        #the output of the encoder.
        #else the decoder is built.
        if self.decoder.built:
            self.encoder.output.shape.assert_is_compatible_with(self.decoder.input.shape)
        else:
            outputs = self.decoder(encoded)
        
        #3. checks if the input of the encoder is compatible with the output of
        #the decoder.
        self.encoder.input.shape.assert_is_compatible_with(self.decoder.output.shape)
         

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
        

    def call(self, inputs, training = False, **kwargs):
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
        if not training: return z_mean
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z_samp = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z_samp

    def get_config(self):
        config = super(NormalSamplingLayer,self).get_config()
        return config      

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def kl_normal_loss(y_true, y_pred, **kwargs):
    z_mean, z_log_var = tf.unstack(y_pred, axis = 1)
    kl_loss = -0.5 * (1 + z_log_var - tf.exp(z_log_var) - tf.square(z_mean))
    #kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    return kl_loss

if __name__ == '__main__':
    """
    input_shape = (28,28,1)
    latent_dim = 23
    inputs = keras.Input( input_shape)
    model =  vae_models.get_mnist_cnn_shallow(input_shape, latent_dim)
    model(inputs)
    """
    # Process data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    num_samples = 100
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
    mnist_labels = np.concatenate([y_train, y_test], axis=0)
    input_shape = mnist_digits.shape[1:]
    mnist_digits = mnist_digits[0:num_samples]
    mnist_labels = mnist_labels[0:num_samples]
    
    input_shape = (28,28,1)
    latent_dim = 23
    inputs = keras.Input( input_shape)
    model =  vae_models.get_mnist_cnn_shallow(input_shape, latent_dim)
    
    model.compile(optimizer = "adam",loss = 'mse')

    model.fit(mnist_digits,mnist_digits,
          epochs = 1,
          batch_size = 50)
    
    
    
    
    
    