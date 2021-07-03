# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 13:15:08 2021

@author: joelw
"""
import numpy as np
import matplotlib.pyplot as plt


class SampleScatterGUI():
    
    def __init__(self,X,y, images):
        """
        This GUI is sutible for when we have labeled images and where each
        image have be concentrated down into a latent space. Here the images
        are stored in 'images', labels in 'y' and latent representation in 'X'.
        Note that X has to be of the diemensions Nx2, if not, t-sne or pca are
        recommended. The GUI then plot two axis. The first one is a scatter
        plot of the latent points X. The secound axis contains an image. If 
        one click somewhere in the first axis, the GUI will find the latent
        point closes to the click and display the image that corresponds to
        that plot in the second axis.
        
        
        Parameters
        ----------
        X : numpy array (N,2)
            Numpy array consisting of N samples of dimesion 2.
        y : numpy array (N,)
            label of each sampel in X.
        Returns
        -------
        None.

        """
        self.X = X
        self.y = y
        self.images = images
        self.fig , self.axs = plt.subplots(1,2)
        self.unique = np.unique(self.y)
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
            x = self.X[self.y == self.unique[i],0]
            y = self.X[self.y == self.unique[i],1]
            self.axs[0].scatter(x,y,label = self.unique[i])
        
        # Draw current index
        self.axs[0].scatter(self.X[self.current_index,0], self.X[self.current_index,1],
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
        d = np.sum(np.power(self.X-x,2),1)
        idx = np.argmin(d)
        return idx

if __name__ == '__main__':
    
    X = np.array([[1,2],[0,3],[1,4], [2,4]])
    y = np.array([0,1,1,0])
    images = np.random.multivariate_normal([0,0], np.eye(2), (4,2))
    scatter = SampleScatterGUI(X, y, images)



