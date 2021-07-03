# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 15:21:14 2021

@author: joelw
"""
import numpy as np


class LatentImage:
    def __init__(self, decoder, fig,
                              latent_boundary = (-1., 1., -1., 1.),
                              latent_dim = None,
                              latent_vectors = None,
                              original_point = None):
        """
        This GUI is made for decoder which maps an n-dimensional input to an
        image. The GUI has two axis. The first contains a plane where each
        point corresponds to point in the latent space. This latent point is
        then passed through the decoder and the image is displayed on the 
        right axis. 
        The latent space is often larger then two dimension. To reduce the 
        dimensions a slice is taken from this latent space. By default this
        slize corresponds to the two first dimensions in the latent space. The
        slice may be specified with the parameters 'latent_vectors' and 
        'original_point', where the former dictates in which direction the
        slice shall be taken and the latter specify where the origin is set.
        As an example, if we look at a point (1,2) in the graph, this 
        corresponds to the latent point z = z0+1*v1 + 2*v2, where v1 and v2
        together make up the latent_vector input and z0 represents the
        original point.
        
        Operate:
            The graph point is initiated in the origin. The point can then 
            either be draged around with the mouse or discretly moved by 
            pressing 'space' which will move the point to where ever the
            mouse is.
        
        
        Parameters
        ----------
        decoder : object with function with a predict function which output an
            array of the predicted outputs. 
        latent_boundary (4,) : sequencial whith the limits of the latent space.
                                e.i. (min_v1, max_v1, min_v2, max_v2).
        Latent_sh
        Returns
        -------
        None.
        """
        

        self.component_point = np.zeros((2,))
        self.latent_dim = latent_dim if latent_dim is not None else decoder.input.shape[1]
        self.latent_vectors = latent_vectors if latent_vectors is not None else np.eye(2,self.latent_dim)
        self.original_point = original_point if original_point is not None else np.zeros(self.latent_dim)
        self.latent_point = self.original_point
        
        self.decoder = decoder
        self.image = self.decoder.predict(np.array([self.latent_point]))[0]
        
        self.fig = fig
        self.fig.subplots(1,2)
        self.latent_boundary = latent_boundary
        self.setup_latent_space()
        
        self.click_tol = 0.05
        self.draw()
        self.fig.canvas.mpl_connect('button_press_event'  , self.button_press_callback)
        self.fig.canvas.mpl_connect('motion_notify_event' ,self.motion_notify_callback)
        self.fig.canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press_callback)
        self.point_selected = False        
    
    def setup_latent_space(self):
        self.fig.axes[0].set_xlim(self.latent_boundary[0], self.latent_boundary[1])
        self.fig.axes[0].set_ylim(self.latent_boundary[2], self.latent_boundary[3])       
        
    
    def draw(self):
        self.fig.axes[0].cla()
        self.setup_latent_space()
        self.fig.axes[0].scatter(self.component_point[0], self.component_point[1])
        self.fig.axes[1].imshow(self.image)
        #self.fig.colorbar()
        self.fig.canvas.draw_idle()
    

    def update_point_select(self, event):
        click_point = np.array([event.xdata, event.ydata])
        dist = np.linalg.norm(click_point-self.component_point)
        self.point_selected =  dist < self.click_tol
    
    def update_point(self, event):
        self.component_point = np.array([event.xdata, event.ydata])

    def update_digit(self):
        self.latent_point = self.original_point + self.latent_vectors.T @ self.component_point
        self.image = self.decoder.predict(np.array([self.latent_point]))[0]

    def motion_notify_callback(self, event):
        if not self.point_selected: return
        self.update_point(event)
        self.update_digit()
        self.draw()
 
    def button_press_callback(self, event):
        self.update_point_select(event)
       
 
    def button_release_callback(self, event):
        self.point_selected = False
        
    def key_press_callback(self, event):
        if not event.key == " ": return
        self.update_point(event)
        self.update_digit()
        self.draw()
