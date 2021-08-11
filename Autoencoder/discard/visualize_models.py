# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 09:25:15 2021

@author: joelw
"""




import tensorflow as tf
from tensorflow import keras
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from variational_autoencoder import VariationalAutoencoder

# For dimension reduction
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# for visualization and evaluation
from latent_plane_mosaic import LatentPlaneMosaic
from latent_interpolation_mosaic import LatentInterpolationMosaic
from sample_scatter_gui import SampleScatterGUI

def get_pca(encoded):
    pca_va = PCA(n_components = 2).fit(encoded)
    va_pca = pca_va.transform(encoded)
    va_vectors = (pca_va.components_.T *3*np.sqrt(pca_va.explained_variance_)).T
    va_origin = np.mean(encoded, axis = 0)
    return va_vectors, va_origin

def plot_reconstructions(models, mnist_digits, num_digits = 5):
    """
        Plots the num_digits randomly chosen digits, original and
        reconstructed from all models in models
    """
    num_models = len(models)
    fig, axs = plt.subplots(num_models+1,num_digits)
    indeces = np.random.randint(40000, size = (num_digits,))
    for i, ind in enumerate(indeces):
        axs[0,i].imshow(mnist_digits[ind])
        axs[0,i].axis("off")
        axs[0,i].set_title("MNIST digit")
    
    model_i = 1
    for name, model in models.items():
        reconstructions = model(mnist_digits[indeces])
        for i, rec in enumerate(reconstructions):
            axs[model_i, i].imshow(rec)
            axs[model_i,i].axis("off")
            axs[model_i,i].set_title(name)
        model_i = model_i + 1
        
def plot_latent_interpolation_mosaic(models, mnist_digits):
    indeces = np.random.randint(10000,size = (3,))
    digits = mnist_digits[indeces]
    
    fig, axs = plt.subplots(2,2)
    axis_coord = [(0,0),(0,1),(1,0), (1,1)]
    axis_i = 0
    for name, model in models.items():
        
        mosaic = LatentInterpolationMosaic(model.encode,
                                  model.decoder,
                                  digits,
                                  num_row = 15,
                                  num_col = 15).mosaic 
        
        axs[axis_coord[axis_i]].imshow(mosaic)
        axs[axis_coord[axis_i]].axis("off")
        axs[axis_coord[axis_i]].set_title(name)
        axis_i = axis_i + 1

def plot_scatter(models, data, labels):

    fig, axs = plt.subplots(2,2)
    axis_coord = [(0,0),(0,1),(1,0), (1,1)]
    axis_i = 0
    for name, model in models.items():
        
        encoded = model.encode(data).numpy()
        
        for i in range(10):
            axs[axis_coord[axis_i]].scatter(
                encoded[labels == i,0],
                encoded[labels == i,1],
                label = str(i),
                )
            
        axs[axis_coord[axis_i]].axis("equal")
        axs[axis_coord[axis_i]].set_title(name)
        axs[axis_coord[axis_i]].legend()
        axis_i = axis_i + 1


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
    mnist_labels = np.concatenate([y_train, y_test], axis=0)
    input_shape = mnist_digits.shape[1:]
    
    models = {}
    model_names = ["cnn_deep", "cnn_shallow", "dense_deep", "dense_shallow"]

    for name in model_names:   
        models[name] = keras.models.load_model("Models/"+name,
                              custom_objects={"VariationalAutoencoder": VariationalAutoencoder})
        
        
    #plot_reconstructions(models, mnist_digits)
    plot_latent_interpolation_mosaic(models, mnist_digits)
    #plot_scatter(models, mnist_digits[0:20000], mnist_labels[0:20000])
    plt.show()










