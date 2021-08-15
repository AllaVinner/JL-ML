# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 17:44:59 2021

@author: joelw
"""



import tensorflow as tf
from tensorflow import keras
import numpy as np
#tf.random.set_seed(2)

from models.variational_autoencoder import VariationalAutoencoder
from models.autoencoder import Autoencoder
from models.load_premade import load_premade_model
from models.continuous_bernoulli_loss import continuous_bernoulli_loss

#Preprocess mnist data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#num_samples = 100
train_digits = np.expand_dims(x_train, -1).astype("float32") / 255



#Set variables
latent_dim = 10
input_shape = train_digits.shape[1:]

# Train models

##############################################################################
model_type = 'autoencoder'
model_name = 'mnist_cnn_shallow'
loss       = continuous_bernoulli_loss
optimizer  = 'adam'
epochs     = 1
name       = 'ae_test'
save_path  = 'models\\saved\\'

# Create and train model
model = load_premade_model(model_type = model_type, model_name = model_name,
                           input_shape = input_shape,
                           latent_dim = latent_dim)
model.compile(optimizer = optimizer, loss = loss)

model.fit(train_digits, train_digits,
          epochs = epochs,
          batch_size = 512)
model.save(save_path + name)

##############################################################################

model_type = 'variational_autoencoder'
model_name = 'mnist_cnn_shallow'
loss       = tf.keras.losses.binary_crossentropy
optimizer  = 'adam'
epochs     = 1
name       = 'vae_test'
save_path  = 'models\\saved\\'

# Create and train model
model = load_premade_model(model_type = model_type, model_name = model_name,
                           input_shape = input_shape,
                           latent_dim = latent_dim)
model.compile(optimizer = optimizer, loss = loss)

model.fit(train_digits,
          epochs = epochs,
          batch_size = 512)
model.save(save_path + name)

##############################################################################

