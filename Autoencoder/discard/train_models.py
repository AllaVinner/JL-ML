# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 08:57:43 2021

@author: joelw
"""
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow import keras
from variational_autoencoder import VariationalAutoencoder
from variational_autoencoder import NormalSamplingLayer
import variational_autoencoder_utils as va_utils
import autoencoder_utils as ae_utils
import os

if __name__ == "__main__":
    
    # Process data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
    mnist_labels = np.concatenate([y_train, y_test], axis=0)
    input_shape = mnist_digits.shape[1:]
    
    # Train Autoencoder
    
    # Setup model arcitectures to be trained
    models = defaultdict(dict)
    models["cnn_shallow"]["get_model"]   =  ae_utils.get_model_cnn_shallow
    models["cnn_deep"]["get_model"]       = ae_utils.get_model_cnn_deep
    models["dense_deep"]["get_model"]    = ae_utils.get_model_dense_deep
    models["dense_shallow"]["get_model"] = ae_utils.get_model_dense_shallow
    
    # Train the targeted models
    latent_dims = [2,4,8,16,50, 100]
    i = 0
    for i in range(len(latent_dims)):
        name = "AE_cnn_deep" + "_" + "latent_dim" + "_" + str(latent_dims[i])
        model = models["cnn_deep"]["get_model"](input_shape, latent_dims[i])
        i = i +1
        model.compile(optimizer = "adam")
        model(keras.Input(input_shape))
        model.summary()
        model.fit(mnist_digits,
              epochs = 30,
              batch_size = 256)
        
        model.save("Models" + os.path.sep +name)



    # Train Variational Autoencoder
    
    
    # Setup model arcitectures to be trained
    models = defaultdict(dict)
    models["cnn_shallow"]["get_model"]   =  va_utils.get_model_cnn_shallow
    models["cnn_deep"]["get_model"]       = va_utils.get_model_cnn_deep
    models["dense_deep"]["get_model"]    = va_utils.get_model_dense_deep
    models["dense_shallow"]["get_model"] = va_utils.get_model_dense_shallow
    
    # Train the targeted models
    i = 0
    for i in range(len(latent_dims)):
        name = "VAE_cnn_deep" + "_" + "latent_dim" + "_" + str(latent_dims[i])
        model = models["cnn_deep"]["get_model"](input_shape, latent_dims[i])
        i = i +1
        model.compile(optimizer = "adam")
        model(keras.Input(input_shape))
        model.summary()
        model.fit(mnist_digits,
              epochs = 30,
              batch_size = 256)
        
        model.save("Models" + os.path.sep +name)






