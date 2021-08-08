# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 17:44:59 2021

@author: joelw
"""



import tensorflow as tf
from tensorflow import keras
import numpy as np
#tf.random.set_seed(2)
from variational_autoencoder import VariationalAutoencoder
import variational_autoencoder_models as vae_models
from cont_bern_loss import cont_bern_loss

#Preprocess mnist data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#num_samples = 100
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
mnist_labels = np.concatenate([y_train, y_test], axis=0)
input_shape = mnist_digits.shape[1:]
#mnist_digits = mnist_digits[0:num_samples]
#mnist_labels = mnist_labels[0:num_samples]

#Set variables
latent_dim = 10

#Create model



##############################################################################
model = vae_models.get_mnist_cnn_deep(input_shape, latent_dim)
model.compile(optimizer = "adam",
                   reconstruction_loss = keras.losses.binary_crossentropy,
                   reconstruction_factor = 10,
                   latent_factor = 1)

model.fit(mnist_digits,
      epochs = 20,
      batch_size = 512)
model(keras.Input(input_shape))
model.save("model_bc_rf_10_lf_1")

##############################################################################
model = vae_models.get_mnist_cnn_deep(input_shape, latent_dim)
model.compile(optimizer = "adam",
                   reconstruction_loss = keras.losses.binary_crossentropy,
                   reconstruction_factor = 100,
                   latent_factor = 1)

model.fit(mnist_digits,
      epochs = 20,
      batch_size = 512)
model(keras.Input(input_shape))
model.save("model_bc_rf_100_lf_1")

##############################################################################
model = vae_models.get_mnist_cnn_deep(input_shape, latent_dim)
model.compile(optimizer = "adam",
                   reconstruction_loss = keras.losses.binary_crossentropy,
                   reconstruction_factor = 1000,
                   latent_factor = 1)

model.fit(mnist_digits,
      epochs = 20,
      batch_size = 512)
model(keras.Input(input_shape))
model.save("model_bc_rf_1000_lf_1")










