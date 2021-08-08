# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 17:19:18 2021

@author: joelw
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

from variational_autoencoder import VariationalAutoencoder
import variational_autoencoder_models as vae_models


#Preprocess mnist data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
num_samples = 100
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
mnist_labels = np.concatenate([y_train, y_test], axis=0)
input_shape = mnist_digits.shape[1:]
mnist_digits = mnist_digits[0:num_samples]
mnist_labels = mnist_labels[0:num_samples]

#Set variables
latent_dim = 10

#Create model
model =  vae_models.get_mnist_cnn_shallow(input_shape, latent_dim)

#Compile and train model   
model.compile(optimizer = "adam")

model.fit(mnist_digits,
      epochs = 2,
      batch_size = 50)







