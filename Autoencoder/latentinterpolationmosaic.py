# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 14:15:42 2021

@author: joelw
"""
import numpy as np
import matplotlib.pyplot as plt


class LatentInterpolationMosaic():
    
    
    def __init__(self, encoder, decoder, images, indeces,
                 num_row=10, num_col=5,
                 variational = False):
        
        self.encoder = encoder
        self.decoder = decoder
        self.images = images
        self.variational = variational
       
        self.fig, self.axis = plt.subplots()
        
        self.plotInterpolation(indeces, num_row, num_col)
        
    
    def plotInterpolation(self, indeces, num_row, num_col):
        # Get point vectors in latent space
        if self.variational:
            latent_means, _, _ = self.encoder.predict(self.images[indeces])
        else:
            latent_means = self.encoder.predict(self.images[indeces])
            
        # z are points in the latent space
        z00, zn0, z0n = latent_means
        
        # Get transfrer vectors in latent space
        vx = z0n-z00
        vy = zn0-z00
        
        # create grid in graph space
        x = np.linspace(0,1,num_col)
        y = np.linspace(0,1,num_row)
        xgrid, ygrid = np.meshgrid(x,y)
        
        # initiate bigger image
        im_height, im_width, im_channels = self.images[0].shape
        mosaic = np.zeros((num_row, num_col, im_height,im_width, im_channels))
        
        # loop through the graph space and calculate reconstructions
        # from the points in latent space. Attach them to the mosaic
        for row in range(num_row):
            for col in range(num_col):
                z = z00+vx*xgrid[row,col]+vy*ygrid[row,col]
                mosaic[row, col] = self.decoder(np.array([z]))[0]
        
        # Reshape the mosaic to one image with kernel 2
        mosaic = mosaic.swapaxes(1,2).reshape((num_row*im_height,
                                               num_col*im_width,
                                               im_channels))
        
        self.axis.imshow(mosaic)
        





