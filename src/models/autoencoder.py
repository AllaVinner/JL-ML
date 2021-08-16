import numpy as np
from tensorflow import keras

class Autoencoder(keras.Model):

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
    
    def compile(self, loss=None, **kwargs):
        if loss is None:
            loss = keras.losses.binary_crossentropy
        
        super(Autoencoder, self).compile(loss=loss, **kwargs)
    
    @property
    def latent_shape(self):
        return self.decoder.input_shape[1:]
    
    @property
    def latent_dim(self):
        return np.prod(self.decoder.input_shape[1:])
    
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
        # Check if the given encoder and decoder are compatible
        # before building.
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
        if self.encoder.built:
            input_shape.assert_is_compatible_with(self.encoder.input.shape)
            
        inputs = keras.Input(batch_input_shape = input_shape) 
        encoded = self.encoder(inputs)
        
        #2. if decoder is built, checks if its input shape is compatible with
        #the output of the encoder.
        if self.decoder.built:
            self.encoder.output.shape.assert_is_compatible_with(self.decoder.input.shape)
        outputs = self.decoder(encoded)
        
        #3. checks if the input of the encoder is compatible with the output of
        #the decoder.
        self.encoder.input.shape.assert_is_compatible_with(self.decoder.output.shape)
         
    
    
    
    
    
    