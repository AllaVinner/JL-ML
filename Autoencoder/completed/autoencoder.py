# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 20:50:39 2021

@author: joelw
"""
import tensorflow as tf
from tensorflow import keras
import autoencoder_models as ae_models
import keras
import numpy as np

class Autoencoder(keras.Model):
    """
        This class 
    
    """
    def __init__(self, encoder, decoder, **kwargs):
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
        # Initiate model structure 
        self.encoder, self.decoder = encoder, decoder
        
        #if encoder is built, check that it is compatible with the given
        #decoder
        if self.encoder.built:
            self._assert_encoder_decoder_compatibility(self.encoder.input.shape)
        
        
        
    def call(self, inputs, **kwargs):
        encoded = self.encoder(inputs)
        outputs = self.decoder(encoded)
        return outputs
    
    @property
    def latent_dim(self):
        return self.decoder.input_shape[-1]
    
    def encode(self, inputs, **kwargs):
        return self.encoder(inputs)
    
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
        self._assert_encoder_decoder_compatibility(input_shape)
        super(Autoencoder, self).build(input_shape)
        
 
  
    def _assert_encoder_decoder_compatibility(self, input_shape):
        """
            Checks if the encoder and decoder are compatible with the given
            shape of the input. Raises ValueErrors in three situations.
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
         
        """
        # TODO:
        # Decision: Is is better to output messages or is it better to use the tensor 
        #functions?
        
        if self.decoder.built:
        assert(self.encoder.output_shape == self.decoder.input.shape),\
        "The output shape of encoder do not match input shape of decoder\n\
        Encoder output shape: {encoder_shape}\n\
        Decoder input shape: {decoder_shape}".format(
         encoder_shape = self.encoder.output_shape,
         decoder_shape = self.decoder.input_shape) 
        else:     
        outputs = self.decoder(encoded)
       
        assert(self.encoder.input_shape == self.decoder.output_shape),\
        "The input shape of encoder do not match output shape of decoder\n\
        Encoder input shape: {encoder_shape}\n\
        Decoder output shape: {decoder_shape}".format(
        encoder_shape = self.encoder.input_shape,
        decoder_shape = self.decoder.output_shape) 
        """



if __name__ == '__main__':
    input_shape = (28,28,1)
    latent_dim = 23
    inputs = keras.Input( input_shape)
    model =  ae_models.get_model_cnn_shallow(input_shape, latent_dim)
    #model(inputs)
    
    
    
    
    
    
    