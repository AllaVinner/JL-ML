import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

class Autoencoder(Model):
    def __init__(self, input_shape, latent_dim=2, 
                 model_type=None, encoder=None, decoder=None):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        if encoder is not None and decoder is not None:
            self.encoder, self.decoder = encoder, decoder
        else:
            self.encoder, self.decoder = self._create_model(input_shape, latent_dim, model_type)
        
    def _create_model(self, input_shape, latent_dim, model_type):
        
        # Basic dense model
        if model_type is None:
            # input_shape = (None,28,28,1)  =>  input_len = 28*28*1 = 784
            input_len = tf.reduce_prod(input_shape).numpy() # Disregards batch length
            
            # Encoder
            encoder_input = keras.Input(shape=input_shape, name='input 1') # Adds a batch dimension
            x = layers.Flatten()(encoder_input)
            encoder_output = layers.Dense(latent_dim, activation='relu', name='enc_dense1')(x)
            encoder = keras.Model(encoder_input, encoder_output, name='encoder')
            
            # Decoder
            decoder_input = keras.Input(shape=(latent_dim,))
            x = layers.Dense(input_len, activation='sigmoid')(decoder_input)
            decoder_output = layers.Reshape(input_shape)(x)
            
            decoder = keras.Model(decoder_input, decoder_output, name='decoder')
            
            return encoder, decoder
        
        if model_type == "cnn":
            
            encoder = keras.Sequential([
                layers.Conv2D(filters=32,kernel_size=3, strides=(2, 2),
                              activation='relu', input_shape=input_shape),
                layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2),
                              activation='relu'),
                
                layers.Flatten(),
                layers.Dense(latent_dim),
            ])
            
            decoder = keras.Sequential([
                # Each node corresponds to one "pixel"?
                layers.Dense(7*7*32, input_shape=(latent_dim,), activation='relu'),
                layers.Reshape(target_shape=(7, 7, 32)),
                
                layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2),
                                       padding='same', activation='relu'),
                layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2),
                                       padding='same', activation='relu'),
                layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1),
                                       padding='same', activation='sigmoid'),
            ])
            
            return encoder, decoder
        
        # More model types here
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded