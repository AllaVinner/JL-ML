# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:56:49 2021

@author: joelw
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras

def init_dense_autoencoder(latent_shape, input_shape):
    input_dim = np.prod(input_shape)
    latent_dim = np.prod(latent_shape)
    """ Initiate encoder """
    encoder_input = keras.layers.Input(input_shape)
    net = encoder_input
    net = keras.layers.Flatten()(net)
    net = keras.layers.Dense(latent_dim, activation = 'relu')(net)
    net = keras.layers.Reshape(latent_shape)(net)
    encoder_output = net
    encoder = keras.Model(encoder_input, encoder_output)
    encoder.summary()
    
    """ Initiate decoder """
    decoder_input =keras.layers.Input(latent_shape)
    net = decoder_input
    net = keras.layers.Flatten()(net)
    net = keras.layers.Dense(input_dim, activation = 'sigmoid')(net)
    net = keras.layers.Reshape(input_shape)(net)
    decoder_output = net
    decoder = keras.Model(decoder_input, decoder_output)
    decoder.summary()
    
    """ Initiate autoencoder """
    encoded = encoder(encoder_input)
    decoded = decoder(encoded)
    autoencoder = keras.Model(encoder_input, decoded)
    
    
    """ Return our coders """
    return autoencoder, encoder, decoder

def init_linear_autoencoder(latent_shape, input_shape):
    input_dim = np.prod(input_shape)
    latent_dim = np.prod(latent_shape)
    """ Initiate encoder """
    encoder_input = keras.layers.Input(input_shape)
    net = encoder_input
    net = keras.layers.Flatten()(net)
    net = keras.layers.Dense(latent_dim, activation = 'linear')(net)
    net = keras.layers.Reshape(latent_shape)(net)
    encoder_output = net
    encoder = keras.Model(encoder_input, encoder_output)
    encoder.summary()
    
    """ Initiate decoder """
    decoder_input =keras.layers.Input(latent_shape)
    net = decoder_input
    net = keras.layers.Flatten()(net)
    net = keras.layers.Dense(input_dim, activation = 'linear')(net)
    net = keras.layers.Reshape(input_shape)(net)
    decoder_output = net
    decoder = keras.Model(decoder_input, decoder_output)
    decoder.summary()
    
    """ Initiate autoencoder """
    encoded = encoder(encoder_input)
    decoded = decoder(encoded)
    autoencoder = keras.Model(encoder_input, decoded)
    
    
    """ Return our coders """
    return autoencoder, encoder, decoder

def init_cnn_autoencoder(latent_shape, input_shape):
    input_dim = np.prod(input_shape)
    latent_dim = np.prod(latent_shape)
    
    """ Initiate encoder """
    encoder_input = keras.layers.Input(input_shape)
    net = encoder_input
    # 2D cnn want to have a channel dimesnion
    if len(input_shape) == 2: 
        net = keras.layers.Reshape((*input_shape,1))(net)
    net = keras.layers.Conv2D(8, (3, 3), activation='relu')(net)
    net = keras.layers.MaxPooling2D((2, 2), padding='same')(net)
    net = keras.layers.Conv2D(16, (3, 3), activation='relu')(net)
    net = keras.layers.MaxPooling2D((2, 2), padding='same')(net)
    net = keras.layers.Conv2D(32, (3, 3), activation='relu')(net)
    net = keras.layers.MaxPooling2D((2, 2), padding='same')(net)
    net = keras.layers.Flatten()(net)
    net = keras.layers.Dense(latent_dim,activation='relu')
    net = keras.layers.Reshape(latent_shape)(net)
    encoder_output = net
    encoder = keras.Model(encoder_input, encoder_output)
    encoder.summary()
    
    """ Initiate decoder """
    decoder_input =keras.layers.Input(latent_shape)
    net = decoder_input
    net = keras.layers.Flatten()(net)
    net = keras.layers.Dense(input_dim, activation = 'linear')(net)
    net = keras.layers.Reshape(input_shape)(net)
    decoder_output = net
    decoder = keras.Model(decoder_input, decoder_output)
    decoder.summary()
    
    """ Initiate autoencoder """
    encoded = encoder(encoder_input)
    decoded = decoder(encoded)
    autoencoder = keras.Model(encoder_input, decoded)
    
    
    """ Return our coders """
    return autoencoder, encoder, decoder




if __name__ == "__main__":
    auto, enco, deco = init_dense_autoencoder((2,), (3,4))






