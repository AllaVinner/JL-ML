# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 09:55:50 2021

@author: joelw
"""
import numpy as np


class Centroid():

    def __init__(self, mu = None, Sigma = None, data = None, labels = None, name = None):
        """
        A centroid is defined by its location (mu) and its spread (Sigma).
        mu is a 1D-array of dimesnion n, which is the dimension of the space.
        Sigma is a 2D-array with diemnsion nxn, where Sigma[i,j] represent the
        covariance between the spread is the i and j direction.
        These defining variables can be either defiened explicitly or implicitly
        by sending in a data variable. This variable should be of dimension 
        Nxn where N is the number of samples. This means that each row 
        (axis = 1) is a sample with same shape as mu.
        
        Parameters
        ----------
        mu : numpy array (n,), optional
            Define the center position of the centroid. The default is None.
        Sigma : numpy array (n,n), optional
            Defines the covariance between the spread over the different
            dimensions. The default is None.
        data : numpy array (N,n), optional
            Contains N samples of data points which the mean (mu) and variance
            (Sigma) is calculated from. The data itself is not saved.
            The default is None.

        Returns
        -------
        Centroid instance.

        """
        
        # If mu and Sigma is NOT given but data are, then estimate the former
        if mu is None and Sigma is None and data is not None:
            mu = np.mean(data, axis = 0)
            Sigma  = np.cov(data,rowvar = False)
        
        # Initiate values
        self.mu = mu
        self.Sigma = Sigma
        self.Sigma_ = np.linalg.inv(Sigma)
        self.n = self.mu.shape[0]
        self.labels = labels if labels is not None else np.arange(self.n)
        self.name = name
        
        self.D, self.V = np.linalg.eig(self.Sigma)

        
    def distance_KL(self, centroid):
        """
        Measures the Kullback Leibler divergence from a centroid to self. 
        
        ----------
        centroid : centroid object
            
        Returns
        -------
        The KL-distance from centroid to self.

        """
        m0 = centroid.mu
        m1 = self.mu
        S0 = centroid.Sigma
        S1 = self.Sigma
        S1_ = np.linalg.inv(S1) 
        
        SD = np.trace(S1_@S0) - np.log(np.linalg.det(S1_@S0))-self.n
        MD = (m1-m0).T@S1_@(m1-m0)
        D_KL = 1/2*(SD+MD)
        return D_KL
        
    def distance_MD(self, centroid):
        """
        Measures the Mahalanobis distance from a centroid to self. 
        
        ----------
        centroid : centroid object
            

        Returns
        -------
        The MD-distance from centroid to self.

        """
        m0 = centroid.mu
        m1 = self.mu
        
        S1 = self.Sigma
        S1_ = np.linalg.inv(S1) 
        
        MD = np.sqrt( (m1-m0).T@S1_@(m1-m0) )
        return MD      
        
        
        
            
if __name__ == '__main__':
    """
        Test that if we sample from a multinormal distribution with Sigma our 
        centroid estimate a similar sigma.
    """
    print("Test if Sigma and mu estimations are ccurate")
    mu = np.array([1,2,3,4])
    Sigma = np.array([[1, 0, .4, 0],
                      [0, 3, 0, .5],
                      [.4, 0, 2, 0],
                      [0, 0.5, 0, 1]])
    data = np.random.multivariate_normal(mu, Sigma, (100000,))
    
    C = Centroid(data = data)
    mu_err = np.linalg.norm(C.mu - mu)
    Sigma_err = np.linalg.norm(C.Sigma - Sigma)
    print("Error of Sigma and estimated Sigma", mu_err)
    print("Error of mu and estimated mu", Sigma_err)
    print()
        
    """
        Test that if the distance measures work
    """
    print("Test if distance measures are accurate")
    m0 = np.array([0,0])
    m1 = np.array([6,8])
    S0 = np.array([[1, 0],
                   [0, 1]])
    S1 = np.array([[4, 0],
                   [0, 4]])

    C0 = Centroid(mu = m0, Sigma = S0)
    C1 = Centroid(mu = m1, Sigma = S1)
    
    dist = np.zeros((2,2))
    dist[0,0] = C0.distance_MD(C0)
    dist[0,1] = C1.distance_MD(C0)
    dist[1,0] = C0.distance_MD(C1)
    dist[1,1] = C1.distance_MD(C1)
     
    true_dist = np.array([[0, 5],[10,0]])    
    print("(MD)Measuerd distance \n", dist, "\n and true distances \n", true_dist)
    
    dist = np.zeros((2,2))
    dist[0,0] = C0.distance_KL(C0)
    dist[0,1] = C1.distance_KL(C0)
    dist[1,0] = C0.distance_KL(C1)
    dist[1,1] = C1.distance_KL(C1)
     
    true_dist = np.array([[0, 5],[10,0]])    
    print("(KL)Measuerd distance \n", dist)
    
      
          
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    