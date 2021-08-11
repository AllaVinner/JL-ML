# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 15:41:24 2021

@author: joelw
"""

import tensorflow as tf



def cont_bern_loss(y_true, y_pred):
    # Calculates Bernoulli reconstruction loss between y_true and y_pred
    # The normalization constant diverges towards 0 and 1, and hence the 
    # tensor is cliped.
    y_pred = tf.clip_by_value(y_pred, 1e-4, 1 - 1e-4)
    # Normalizing constant
    log_norm = _cont_bern_log_norm(y_pred)
    log_p_all = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred) + log_norm)
    
    # Sum over all axis except batch

    #return tf.reduce_mean(log_p_all_reduced)
    return tf.reduce_mean(log_p_all)



def _cont_bern_log_norm(x, l_lim=0.48, u_lim=0.52):
        # returns the cont. bernoulli log normalizing constant for x in (0, l_lim) U (u_lim, 1)
        # and a Taylor approximation in [l_lim, u_lim].
        # lower & upper limit needed for numerical stability, as the function diverges for x -> 0.
        # credit: https://github.com/cunningham-lab/cb_and_cc/blob/master/cb/utils.py
        
        # if x is outside out limit, leave it be, else clip at lower lim
        cut_x = tf.where(tf.logical_or(tf.less(x, l_lim), tf.greater(x, u_lim)), x, l_lim * tf.ones_like(x))
        
        # log C(lambda)
        log_norm = tf.math.log(tf.math.abs(2.0 * tf.math.atanh(1 - 2.0 * cut_x))) - tf.math.log(tf.math.abs(1 - 2.0 * cut_x))
        
        # 4th order taylor approx around 0.5?
        taylor = tf.math.log(2.0) + 4.0 / 3.0 * tf.math.pow(x - 0.5, 2) + 104.0 / 45.0 * tf.math.pow(x - 0.5, 4)
        
        # return log norm outside interval, taylor approx inside
        return tf.where(tf.logical_or(tf.less(x, l_lim), tf.greater(x, u_lim)), log_norm, taylor)










