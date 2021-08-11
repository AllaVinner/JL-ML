# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 20:23:44 2021

@author: joelw
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 13:15:08 2021

@author: joelw
"""
import numpy as np
import matplotlib.pyplot as plt


class SampleScatterGUI():
    
    def __init__(self,latent_projection,labels, images):
        """
        This GUI is sutible for when we have labeled images and where each
        image have be concentrated down into a latent space. Here the images
        are stored in 'images', labels in 'labels' and latent representation in 'latent_projection'.
        Note that latent_projection has to be of the diemensions Nx2, if not, t-sne or pca are
        recommended. The GUI then plot two axis. The first one is a scatter
        plot of the latent points latent_projection. The secound axis contains an image. If 
        one click somewhere in the first axis, the GUI will find the latent
        point closes to the click and display the image that corresponds to
        that plot in the second axis.
        
        
        Parameters
        ----------
        latent_projection : numpy array (N,2)
            Numpy array consisting of N samples of dimesion 2.
        labels : numpy array (N,)
            label of each sampel in latent_projection.
        Returns
        -------
        None.

        """
        self.latent_projection = latent_projection
        self.labels = labels
        self.images = images

        self.fig = plt.figure()
        self.axs = self.fig.subplots(1,2)
            
        self.unique = np.unique(self.labels)
        self.num_labels = self.unique.size
        self.fig.canvas.mpl_connect('button_press_event', self.button_press_callback)


        self.current_index = 0
        self.draw()
        
    def draw(self):
        # Clear axeses
        self.axs[0].cla()   
        self.axs[1].cla()
        
        # Draw scatter
        for i in range(self.num_labels):
            x = self.latent_projection[self.labels == self.unique[i],0]
            y = self.latent_projection[self.labels == self.unique[i],1]
            self.axs[0].scatter(x,y,label = self.unique[i])
        
        # Draw current index
        self.axs[0].scatter(self.latent_projection[self.current_index,0], self.latent_projection[self.current_index,1],
                       label = "Current label")
        self.axs[1].imshow(self.images[self.current_index])
        self.axs[0].legend()
        self.fig.canvas.draw_idle()
        
    def button_press_callback(self, event):
        x = np.array([event.xdata, event.ydata])
        self.current_index = self.get_closest_index(x)
        self.draw()
        
        
    
    def get_closest_index(self,x):
        """
        Finds and returns the index of the point in X which is euclidean closest
        to x.

        Parameters
        ----------
        x : numpy array (2,)
            Point to which distances are suppose to be calculated and the index
            of the sample with the smalest distance is returned.

        Returns
        -------
        idx : int
            index of the sample which is closest to x.

        """
        d = np.sum(np.power(self.latent_projection-x,2),1)
        idx = np.argmin(d)
        return idx




