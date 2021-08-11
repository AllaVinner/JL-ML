# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 08:45:15 2021

@author: joelw
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from variational_autoencoder import VariationalAutoencoder

# variational_autoencoder_utils


# Architectures


def get_mnist_cnn_deep(input_shape, latent_dim):
    # OUTPUT_SHAPE = (28,28,1)
    # Variational autoencoder
    encoder = keras.Sequential([
        keras.layers.InputLayer(input_shape=input_shape),
        keras.layers.Reshape((28,28,1)),
        keras.layers.Conv2D(8, (3,3), padding = "same", activation = "relu"),
        keras.layers.Conv2D(8, (3,3), padding = "same", activation = "relu"),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Conv2D(16, (3,3), padding = "same", activation = "relu"),
        keras.layers.Conv2D(16, (3,3), padding = "same", activation = "relu"),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Conv2D(32, (3,3), padding = "same", activation = "relu"),
        keras.layers.Conv2D(32, (3,3), padding = "same", activation = "relu"),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Conv2D(64, (3,3), padding = "same", activation = "relu"),
        keras.layers.Conv2D(64, (3,3), padding = "same", activation = "relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(200, activation = 'relu'),
        keras.layers.Dense(100, activation = 'relu'),
        keras.layers.Dense(2*latent_dim),
        keras.layers.Reshape((2,latent_dim)),
    ], name = "Encoder" )
    
    decoder = keras.Sequential([
        keras.layers.InputLayer(input_shape=(latent_dim,)),
        keras.layers.Dense(200, activation = "relu"),
        keras.layers.Dense(3*3*64, activation = "relu"),
        keras.layers.Reshape((3,3,64)),
        keras.layers.Conv2DTranspose(32, (3,3), activation = "relu", padding = "same"),
        keras.layers.Conv2DTranspose(32, (3,3), activation = "relu", strides = 2),
        keras.layers.UpSampling2D((2,2)),
        keras.layers.Conv2DTranspose(16, (3,3), activation = "relu", padding = "same"),
        keras.layers.Conv2DTranspose(16, (3,3), activation = "relu", padding = "same"),
        keras.layers.Conv2DTranspose(16, (3,3), activation = "relu", padding = "same"),
        keras.layers.UpSampling2D((2,2)),
        keras.layers.Conv2DTranspose(16, (3,3), activation = "relu", padding = "same"),
        keras.layers.Conv2DTranspose(1, (3,3), activation = "sigmoid", padding = "same"),
        keras.layers.Reshape(input_shape),
    ], name = "Decoder")
    
    model = VariationalAutoencoder(encoder, decoder)
    
    return model



def get_mnist_cnn_shallow(input_shape, latent_dim):
    # OUTPUT_SHAPE = (28,28,1)
    # Variational autoencoder
    encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Reshape((28,28,1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(2*latent_dim),
            tf.keras.layers.Reshape((2,latent_dim)),
        ]
    )

    decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same',
                activation = "sigmoid"),
            tf.keras.layers.Reshape(input_shape),
        ]
    )
    
    # Initiate model
    model = VariationalAutoencoder(encoder, decoder)
    
    return model


def get_mnist_dense_shallow(input_shape, latent_dim, intermediat_dim = 200):
    encoder = keras.Sequential([
        keras.layers.InputLayer(input_shape=input_shape),
        keras.layers.Flatten(),
        keras.layers.Dense(intermediat_dim, activation="relu"),
        keras.layers.Dense(2*latent_dim),
        keras.layers.Reshape((2,latent_dim)),
        ], name = "Encoder")
    
    decoder = keras.Sequential([
        keras.layers.InputLayer(input_shape=(latent_dim,)),
        keras.layers.Dense(intermediat_dim, activation = "relu"),
        keras.layers.Dense(np.prod(input_shape), activation = "sigmoid"),
        keras.layers.Reshape(input_shape),
        ], name = "Decoder")
    # Initiate model
    model = VariationalAutoencoder(encoder, decoder)
    
    return model


def get_mnist_dense_deep(input_shape, latent_dim, intermediat_dim = (200,100,50)):
    encoder = keras.Sequential([
        keras.layers.InputLayer(input_shape=input_shape),
        keras.layers.Flatten(),
        keras.layers.Dense(intermediat_dim[0], activation="relu"),
        keras.layers.Dense(intermediat_dim[2], activation="relu"),
        keras.layers.Dense(intermediat_dim[2], activation="relu"),
        keras.layers.Dense(2*latent_dim),
        keras.layers.Reshape((2,latent_dim)),
        ], name = "Encoder")
    
    decoder = keras.Sequential([
        keras.layers.InputLayer(input_shape=(latent_dim,)),
        keras.layers.Dense(intermediat_dim[2], activation = "relu"),
        keras.layers.Dense(intermediat_dim[1], activation = "relu"),
        keras.layers.Dense(intermediat_dim[0], activation = "relu"),
        keras.layers.Dense(np.prod(input_shape), activation = "sigmoid"),
        keras.layers.Reshape(input_shape),
        ], name = "Decoder")
    
    # Initiate model
    model = VariationalAutoencoder(encoder, decoder)
    
    return model

