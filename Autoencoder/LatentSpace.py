# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 09:37:30 2021

@author: joelw
"""

import numpy as np
import matplotlib.pyplot as plt


class LatentSpace():
    
    def __init__(self, data, labels = None):
        """
        
        Parameters
        ----------
        data : Matrix with the variables and observations. Each row contains
        one observations and each column contains one variable. 
            DESCRIPTION.
        labels : 1D-array of same number of ellements as observations, optional
            Th. The default is set to a 1-D array of zeros.

        Returns
        -------
        None.

        """
        self.data                = data
        self.num_observations    = self.data.shape[0]
        self.latent_dim          = self.data.shape[1]
        self.labels              = labels if labels is not None else np.zeros((self.latent_dim,))
        self.mean                = np.mean(self.data, axis = 0)
        self.cov                 = np.cov(self.data, rowvar = False)
        self.unique              = np.unique(self.labels)
        self.num_labels          = np.size(self.unique)
        
        self.label_mean          = np.zeros((self.num_labels,*self.mean.shape))
        self.label_cov           = np.zeros((self.num_labels,*self.cov.shape))

        # Set the mean and covariance of each label
        for l_i,label in enumerate(self.unique):
            self.label_mean[l_i] = np.mean(self.data[self.labels == label], axis = 0)
            self.label_cov[l_i] = np.cov(self.data[self.labels == label,:], rowvar = False)

    
    def draw_total_boxplot(self, ax = None):
        if ax is None:
            ax = plt.axes()
        ax.boxplot(self.data)
    
    def draw_label_boxplot(self, figure_shape = (3,4)):
        """
        This functions plots the boxplots of each latent dimension for every 
        label-class. Each figure is devided according to "figure_shape" into a
        number of axises. The total number of axises are equal to the number
        of latent dimesnions.

        Parameters
        ----------
        figure_shape : TYPE, optional
            DESCRIPTION. The default is (3,4).

        Returns
        -------
        None.

        """
        num_ax_per_fig = np.prod(figure_shape).astype('int')
        #num_fig = np.ceil(self.latent_dim / num_ax_per_fig).astype('int')
        
        # Go through all dimensions and plot all the labels as boxplot for each dimension
        for lat_i in range(self.latent_dim):
            
            # Reset the axis counter everytime we have filled up the figure
            if lat_i % num_ax_per_fig == 0:
                fig, axs = plt.subplots(*figure_shape)
                a_i = 0
                row = 0
                col = 0
            
            # Plot the boxplots
            for lab_i, label in enumerate(self.unique):
                axs[row,col].boxplot(self.data[self.labels == label, lat_i],
                                     positions = [lab_i],
                                     labels = [label])
            
            # Update axis values
            a_i = a_i + 1
            row = a_i // figure_shape[1]
            col = a_i % figure_shape[1]
            
    def mahalanobis_distance(self):
        MD = np.zeros((self.num_labels,self.num_labels))
        for i,label_i in enumerate(self.unique):
            for j,label_j in enumerate(self.unique):
                d = self.label_mean[i] - self.label_mean[j]
                S_inv = np.linalg.inv(self.label_cov[i])
                MD[i,j] = np.sqrt(d.T@S_inv@d)
        return MD
                
                
                
                
                
                
      
if __name__ == '__main__':
    
        M = np.array([[1,3,5,5],[3,2,6,5], [1,2,5,4],[6,7,4,5],[4,1,2,3],[5,6,7,8],[2,3,4,3],[6,5,4,2]])
        label = np.array([1,2,2,1,1,2,1,2])
        from numpy import random as rnd
        
        S1 = np.array([[1., 0, 0, 0],
                       [0  ,2, 0, 0],
                       [0  ,0, 3, 0],
                       [0,  0, 0, 4]])
        M1 = rnd.multivariate_normal(np.zeros((S1.shape[0],)), S1 ,(1000,))
        label1 = np.zeros((M1.shape[0],))
        
        S2  = np.array([[2., 0, 0, 0],
                       [0  ,4, 0,  0],
                       [0  ,0, 0.1, 0],
                       [0,  0, 0,  1]])
        M2 = rnd.multivariate_normal(np.ones((S2.shape[0],)), S2 ,(1000,))
        label2 = np.ones((M1.shape[0],))       
        
        M = np.vstack((M1,M2))
        label = np.hstack((label1,label2))
        
        L = LatentSpace(M, label)
        
        MD = L. mahalanobis_distance()




