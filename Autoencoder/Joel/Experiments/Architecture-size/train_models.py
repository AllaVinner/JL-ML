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
import autoencoder_models as ae_models
import variational_autoencoder_models as vae_models
from cont_bern_loss import cont_bern_loss

#Preprocess mnist data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#num_samples = 100
mnist_digits = np.expand_dims(x_train, -1).astype("float32") / 255
mnist_labels = np.concatenate([y_train, y_test], axis=0)
input_shape = mnist_digits.shape[1:]
#mnist_digits = mnist_digits[0:num_samples]
#mnist_labels = mnist_labels[0:num_samples]

#Set variables
latent_dim = 10

#Create model

##############################################################################
model = ae_models.get_mnist_cnn_deep(input_shape, latent_dim)
model.compile(optimizer = "adam", loss = cont_bern_loss)

model.fit(mnist_digits,mnist_digits,
      epochs = 30,
      batch_size = 512)
model(keras.Input(input_shape))
model.save("model_ae_cb")


##############################################################################
model = ae_models.get_mnist_cnn_deep(input_shape, latent_dim)
model.compile(optimizer = "adam", loss = keras.losses.binary_crossentropy)

model.fit(mnist_digits,mnist_digits,
      epochs = 30,
      batch_size = 512)
model(keras.Input(input_shape))
model.save("model_ae")

##############################################################################
##############################################################################
model = vae_models.get_mnist_cnn_deep(input_shape, latent_dim)
model.compile(optimizer = "adam",
              reconstruction_loss = keras.losses.binary_crossentropy,
              reconstruction_factor = 10,
              latent_factor = 1)

model.fit(mnist_digits,
      epochs = 30,
      batch_size = 512)
model(keras.Input(input_shape))
model.save("model_rf10")

##############################################################################
model = vae_models.get_mnist_cnn_deep(input_shape, latent_dim)
model.compile(optimizer = "adam",
                   reconstruction_loss = keras.losses.binary_crossentropy,
                   reconstruction_factor = 100,
                   latent_factor = 1)

model.fit(mnist_digits,
      epochs = 30,
      batch_size = 512)
model(keras.Input(input_shape))
model.save("model_rf100")

##############################################################################
model = vae_models.get_mnist_cnn_deep(input_shape, latent_dim)
model.compile(optimizer = "adam",
                   reconstruction_loss = keras.losses.binary_crossentropy,
                   reconstruction_factor = 1000,
                   latent_factor = 1)

model.fit(mnist_digits,
      epochs = 30,
      batch_size = 512)
model(keras.Input(input_shape))
model.save("model_rf1000")

##############################################################################
##############################################################################
model = vae_models.get_mnist_cnn_deep(input_shape, latent_dim)
model.compile(optimizer = "adam",
              reconstruction_loss = cont_bern_loss,
              reconstruction_factor = 10,
              latent_factor = 1)

model.fit(mnist_digits,
      epochs = 30,
      batch_size = 512)
model(keras.Input(input_shape))
model.save("model_cb_rf10")

##############################################################################
model = vae_models.get_mnist_cnn_deep(input_shape, latent_dim)
model.compile(optimizer = "adam",
                   reconstruction_loss = cont_bern_loss,
                   reconstruction_factor = 100,
                   latent_factor = 1)

model.fit(mnist_digits,
      epochs = 30,
      batch_size = 512)
model(keras.Input(input_shape))
model.save("model_cb_rf100")

##############################################################################
model = vae_models.get_mnist_cnn_deep(input_shape, latent_dim)
model.compile(optimizer = "adam",
                   reconstruction_loss = cont_bern_loss,
                   reconstruction_factor = 1000,
                   latent_factor = 1)

model.fit(mnist_digits,
      epochs = 30,
      batch_size = 512)
model(keras.Input(input_shape))
model.save("model_cb_rf1000")

##############################################################################







