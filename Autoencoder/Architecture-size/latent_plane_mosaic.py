# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 18:13:27 2021

@author: joelw
"""



# -*- coding: utf-8 -*-

"""
Created on Fri Jul  2 14:15:42 2021

@author: joelw
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class LatentPlaneMosaic():
    """
    Given a latent space and a decoder which maps points in the latents space
    to images, this class shows a mosaic of these decodings from a slice of
    the latent space. The latent space slice is defined by the the 
    translation vector "latent_origin", and the direction vectors 
    "latent_vectors". The number of images used in the mosaic is given by
    "num_row" and "num_col".
    
    """
    
    def __init__(self, decoder,
                 latent_dim = None,
                 latent_vectors = None,
                 latent_origin = None,
                 num_row = 10,
                 num_col = 10):
        """
            
        Latent_dim or Latent_vectors need to be passed. If no latent_dim is
        passed it will be taken from the latent_vectors. If no latent_vectors
        is passed, the latent_vectors will be set to the unit vectors in the
        first two dimensions. If no latent_origin is passed it will be set to
        be zero in the latent_dim.
            
        
            Parameters
            ----------
            latent_dim : int
                The dimension of the latent space
            latent_vectors : array (2,latent_dim)
                The directional vectors for which defines the slice ot be
                investigated. The slice is taken between -1, 1 times these
                vectors.
            latent_origin : array (latent_dim,) 
                the origin of the latent plane.
            num_row : int
                Number of rows in the mosaic
            num_col : int
                Number of columns in the mosaic.      
        """
        # Handle input structure
        if latent_dim is None and latent_vectors is None:
            raise ValueError('latent dim or latent_vectors need to be passed')
        if latent_dim is None:
            latent_dim = latent_vectors.shape[-1]
        if latent_vectors is None:
            latent_vectors = np.eye(2,latent_dim)
        if latent_origin is None:
            latent_origin = np.zeros((latent_dim,))
        
        
        
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.latent_vectors = latent_vectors
        self.latent_origin= latent_origin        
        self.mosaic = self.set_mosaic(num_row, num_col)
        
    
    def set_mosaic(self, num_row, num_col):

        # create grid in graph space
        x = np.linspace(-1,1,num_col)
        y = np.linspace(-1,1,num_row)
        xgrid, ygrid = np.meshgrid(x,y)
        coeff = np.array([xgrid.flatten(),
                          ygrid.flatten()])
        
        # Decode
        latent_space = self.latent_origin + coeff.T @ self.latent_vectors
        reconstructions = self.decoder(latent_space)
        
        # Reshape and stich together
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
    
    
    LatentPlaneMosaic(VA.decoder, latent_dim = 2).show_mosaic()



