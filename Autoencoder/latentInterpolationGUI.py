# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 14:15:42 2021

@author: joelw
"""
import numpy as np
import matplotlib.pyplot as plt

class LatentInterpolationGUI():
    
    
    def __init__(self, encoder, decoder, images):
        self.encoder = encoder
        self.decoder = decoder
        self.images = images
        
    
    def plotInterpolation(self, indeces, num_row, num_col):
        # Get point vectors in latent space
        zm, _, _ = self.encoder.predict(self.images[indeces])
        z00, zn0, z0n = zm
        # Get transfrer vectors in latent space
        vx = z0n-z00
        vy = zn0-z00
        
        # create grid
        x = np.linspace(0,1,num_col)
        y = np.linspace(0,1,num_row)
        
        xgrid, ygrid = np.meshgrid(x,y)
        
        # initiate bigger image
        im_height, im_width, im_channels = self.images[0].shape
        mosaic = np.zeros((num_row, num_col, im_height,im_width, im_channels))
        
        # loop through grid and calculate image
        for row in range(num_row):
            for col in range(num_col):
                z = z00+vx*xgrid[row,col]+vy*ygrid[row,col]
                print(z)
                mosaic[row, col] = self.decoder(np.array([z]))[0]
        
        mosaic = mosaic.swapaxes(1,2).reshape((num_row*im_height, num_col*im_width, im_channels))
        
        #plt.figure()
        ##plt.imshow(self.decoder(z0n))
        #plt.figure()
        #plt.imshow(self.decoder(zn0))
        #lt.imshow()
        plt.imshow(mosaic)
        





