# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 15:34:33 2021

@author: joelw
"""

import tensorflow as tf

def kl_normal_divergence(distribution, **kwargs):
    """
    

    Parameters
    ----------
    distribution : Tensor, shape = (None, 2, None)
        Tensor which defines a multi-dimensional normal distribution.
        The first dimension is the batch size and he last is the space
        dimension. Each element in the batch is hence a (2,dim) tensor. The
        first vector represents the mean point of the distribution and the 
        second represents the log-variance in that dimension.
        
    **kwargs : TYPE
        Will be ignored.

    Returns
    -------
    loss : Tensor shape (None, None)
        The .

    """
    z_mean, z_log_var = tf.unstack(distribution, axis = 1)
    loss = 0.5 * (tf.exp(z_log_var) - z_log_var -1 + tf.square(z_mean))
    loss = tf.reduce_mean(loss)
    return loss




if __name__ == "__main__":
    #Test if it is working
    import numpy as np
    M = np.array([[[0,0,0,0],
                   [0,0,0,0]],
                  
                  [[1,1,1,1],
                   [0,0,0,0]],
                  
                  [[0,0,0,0],
                   [1,1,1,1]],
                  
                  [[1,1,1,1],
                   [1,1,1,1]]], dtype = "float32")
    
    print("M shape: ", M.shape)
    
    true_losses = np.array([0, 2, 2*np.e-4, 2*np.e-2])
    
    indeces = [0,1,2,3]
    
    pred_loss = kl_normal_divergence(M[indeces])
    true_loss = np.mean(true_losses[indeces])
    err = (pred_loss - true_loss)**2
    assert(err < 1e-10)
    
    print("Made it")
    
    
    
    
    
    
    
    
    