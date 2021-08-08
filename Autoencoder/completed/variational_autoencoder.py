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
        super(VariationalAutoencoder, self).__init__(**kwargs)
        
        # Initiate model structure 
        self.encoder, self.decoder, self.sampler = encoder, decoder, sampler
        if self.sampler is None: self.sampler = NormalSamplingLayer()



        #initiate loss tracker
        self.loss_tracker_total = keras.metrics.Mean(name="total_loss")
        self.loss_tracker_reconstruction = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.loss_tracker_latent = keras.metrics.Mean(name="latent_loss")
        
        #if encoder is built, check that it is compatible with the given
        #decoder
        if self.encoder.built:
            self._check_encoder_decoder_compatibility(self.encoder.input.shape)
        
        
    def call(self, inputs, training = False, **kwargs):
        latent_distribution = self.encoder(inputs)
        latent_sample       = self.sampler(latent_distribution, training = training)
        outputs             = self.decoder(latent_sample)
        return outputs

    def compile(self, reconstruction_loss = None, latent_loss = None, **kwargs):
        super(VariationalAutoencoder, self).compile(**kwargs)
        
        if reconstruction_loss is None:
            self.reconstruction_loss = keras.losses.binary_crossentropy
        else:
            self.reconstruction_loss = reconstruction_loss
        
        if latent_loss is None:
            self.latent_loss = kl_normal_divergence
        else:
            self.latentloss = latent_loss
      
        
    @property
    def latent_dim(self):
        return self.decoder.input_shape[-1]
    
    def encode(self, inputs, **kwargs):
        latent_distribution = self.encoder(inputs, training = False)
        z_mean = self.sampler(latent_distribution, training = False)
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
        self._check_encoder_decoder_compatibility(input_shape)
        super(VariationalAutoencoder, self).build(input_shape)
        
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
        if not self.built: self.build(inputs.shape)
        with tf.GradientTape() as tape:
            # Forward propagation
            latent_distribution  = self.encoder(inputs, training = True)
            latent_sample        = self.sampler(latent_distribution, training= True)
            reconstruction       = self.decoder(latent_sample, training= True)
            # Calculate loss
            loss_reconstruction = self.reconstruction_loss(inputs, reconstruction)
            loss_latent         = self.latent_loss(latent_distribution)
            loss_total          = loss_reconstruction + loss_latent
            
        # Calculate and apply gradient
        grads = tape.gradient(loss_total, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.loss_tracker_reconstruction.update_state(loss_reconstruction)
        self.loss_tracker_latent.update_state(loss_latent)
        self.loss_tracker_total.update_state(loss_total)
        # Return training message
        return {
            "Total loss": self.loss_tracker_total.result(),
            "Reconstruction loss": self.loss_tracker_reconstruction.result(),
            "Latent loss": self.loss_tracker_latent.result(),
               }
  
    def _check_encoder_decoder_compatibility(self, input_shape):
        """
            Checks if the encoder and decoder are compatible with the given
            shape of the input. It raises ValueErrors in three occations.
            1. The encoder is built, and its input shape does not match the 
            given input shape.
            2. The sampler is built, and its input_shape does not match the
            output shape of the encoder.
            3. The decoder is built, and its input_shape does not match the
            output shape of the sampler.
            4. The input of the encoder do not match the output of the decoder.

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
        inputs = keras.Input(batch_input_shape = input_shape) 
        latent_distribution = self.encoder(inputs)
        
        #2. if sampler is built, checks if its input shape is compatible with
        #the output shape of the encoder.
        #else the sampler is built.
        if self.sampler.built:
            self.encoder.output.shape.assert_is_compatible_with(self.sampler.input.shape)
        latent_sample = self.sampler(latent_distribution)
            
        #3. if decoder is built, checks if its input shape is compatible with
        #the output of the encoder.
        #else the decoder is built.
        if self.decoder.built:
            self.sampler.output.shape.assert_is_compatible_with(self.decoder.input.shape)
        outputs = self.decoder(latent_sample)
        
        #4. checks if the input of the encoder is compatible with the output of
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


def kl_normal_divergence(distribution, **kwargs):
    """
    

    Parameters
    ----------
    distribution : Tensor, shape = (None, 2, None)
        Tensor which defines a multi-dimensional normal distribution.
        The first dimension is the batch size and he last is the space
        dimension. Each element in the batch is hence a (2,dim) tensor. The
        first vector represents the mean point of the distribution and the 
        second represents the log-variance in that dimension.
        
    **kwargs : TYPE
        Will be ignored.

    Returns
    -------
    loss : Tensor shape (None, None)
        The .

    """
    z_mean, z_log_var = tf.unstack(distribution, axis = 1)
    loss = -0.5 * (1 + z_log_var - tf.exp(z_log_var) - tf.square(z_mean))
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return loss

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
    
    model.compile(optimizer = "adam")

    model.fit(mnist_digits,
          epochs = 2,
          batch_size = 50)
    
    
    
    
    
    