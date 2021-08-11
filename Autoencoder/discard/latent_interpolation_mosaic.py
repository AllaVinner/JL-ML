# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 14:15:42 2021

@author: joelw
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class LatentInterpolationMosaic():
    """
        This class can be used to visualize a latent space. It works by 
        encoding three images with a user defined encoding. These three
        latent points (or encodings) uniquelly defines a hyper plane. The
        class then creates a grid over this hyperplane and passes the grid
        through a decoder (also user defined). This grid of latent points then
        corresponds to a grid of images which can be stich together to form
        a mosaic. This mosaic (super)image is then stored in the attribute
        "mosaic" and can either be displayed directly with the function 
        "show_mosaic" or by using the attribute externally. 
        
        The number of images to be stitched up are set by the input
        parameters "num_row" and "num_col".
    
    """
    
    def __init__(self, encoder, decoder, images, indeces = (0,1,2),
                 num_row=10, num_col=10):
        """
            Parameters
            ----------
            encoder : obj/function
                The encoding function which can be an object if it is callable.
                The encoding function should take in a set of images and
                return the latent points. The latent points need to be of
                rank 1. The output needs to be a numpy array or a 
                tensorflow tensor.
            decoder : obj/function
                The decoding function which can be an object if it is callable.
                The function should take the latent points of rank 1 and 
                return the images. The images can be zero, or multi-channeled.
            images : array
                The images needs to be stored in the shape (N,h,w) or 
                (N,w,h), where N is the number of images and h,w,c are 
                the height, whith, and the number of channels respectivly.
            indeces : tuple
                Indicates which of the images whould be interpolated 
                inbetween.
            num_row : int
                Number of images stiched together in each row of the mosaic.
            num_col : int
                Number of images stiched together in each col of the mosaic.
        """
        # Initiate the user input
        self.encoder = lambda x: np.array(encoder(x))
        self.decoder = lambda x: np.array(decoder(x))
        # Adds a channel to the images if they don't already have one.
        self.images = np.array(images)
        if images.ndim == 3: 
            images = np.expand_dims(images, axis = -1)
        
        self.mosaic = self.set_mosaic(indeces, num_row, num_col)
        
    
    def set_mosaic(self, indeces, num_row, num_col):
        # Get point vectors in latent space
        latent_origin  = self.encoder(self.images[indeces[0:1],:])
        latent_vectors = self.encoder(self.images[indeces[1:3],:])-latent_origin      
        # Convert to numpy
        if tf.is_tensor(latent_origin):
            latent_origin = latent_origin.numpy()
        if tf.is_tensor(latent_vectors):
            latent_vectors = latent_vectors.numpy()
        
        # create grid in graph space
        x = np.linspace(0,1,num_col)
        y = np.linspace(0,1,num_row)
        xgrid, ygrid = np.meshgrid(x,y)
        coeff = np.array([xgrid.flatten(),
                          ygrid.flatten()])
        
        # Decode grid
        latent_space = latent_origin + coeff.T @ latent_vectors
        reconstructions = self.decoder(latent_space)
        
        # Reshape and stich together the decoded images.
        if tf.is_tensor(reconstructions):
            reconstructions = reconstructions.numpy()
        if reconstructions.ndim == 3:
            reconstructions = reconstructions.expand_dims(axis = -1)
        _, im_height, im_width, im_channels = reconstructions.shape
        
        mosaic = reconstructions.reshape((num_row, num_col, im_height,
                                         im_width, im_channels))
        mosaic = mosaic.swapaxes(1,2)
        mosaic = mosaic.reshape((num_row*im_height, num_col*im_width, im_channels))
        return mosaic
         
    def show_mosaic(self):
        """ Plots the mosaic in the current axis """
        plt.imshow(self.mosaic)
        plt.show()




if __name__ == '__main__':
    from tensorflow import keras
    from variational_autoencoder import VariationalAutoencoder
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
    mnist_labels = np.concatenate([y_train, y_test], axis=0)
    input_shape = mnist_digits.shape[1:]
    
    VA = VariationalAutoencoder(input_shape = input_shape,
                                latent_dim = 2,
                                )
    
    
    LatentInterpolationMosaic(VA.encode, VA.decoder, images = mnist_digits).show_mosaic()
