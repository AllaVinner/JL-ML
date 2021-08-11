import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import numpy as np

def cb_vae_loss(y_true, y_pred, mean, logvar, z_latent, precision=1e-4):
    """
    Returns the Continuous Bernoulli ELBO, https://arxiv.org/abs/1907.06845
    For example used on MNIST with Variational Autoencoders.
    
    Args:
        y_true: true values of the data, e.g true pixel values.
        shape = [batch_size, d0, .. dN]
        
        y_pred: predicted outputs of the network, e.g reconstructed pixel values.
        shape = [batch_size, d0, .. dN]
        
        Both are assumed to contain values in the range [0, 1],
        and have compatible shapes.
        
        mean: learned gaussian means used to sample z
        shape = [batch_size, latent_dim]

        logvar: learned gaussian (log) variances
        shape = [batch_size, latent_dim]
        
        z_latent: sampled lantent variable
        shape = [batch_size, latent_dim]
 
    Returns:
         Loss float tensor
        
    Each pixel is assumed to follow a continuous bernoulli distribution.
    This is similar to the regular (binarized) bernoulli ELBO,
    but with an added regularization term.
    
    """ 
    
    if y_true is None:
        raise ValueError("targets must not be None.")
    if y_pred is None:
        raise ValueError("predictions must not be None.")
    
    y_true_flat, y_pred_flat = _flatten_and_convert(y_true, y_pred)
    
    # Clip outputs to avoid numerical instability
    y_pred_flat = tf.clip_by_value(y_pred_flat, precision, 1-precision)
    
    # calculate normalizing constant
    # log_norm_const = _cont_bern_log_norm(y_pred_flat)
    
    # reconstruction loss
    # log_p_all = y_true_flat * tf.math.log(y_pred_flat) + (1 - y_true_flat) * tf.math.log(1 - y_pred_flat) + log_norm_const
    # log_px_z = tf.reduce_mean(tf.reduce_sum(log_p_all, axis=1))
    
    log_px_z = reconstruction_loss(y_true_flat, y_pred_flat)
    
    # KL divergence
    log_pz = _log_normal_pdf(z_latent, 0., 0.)
    log_qz_x = _log_normal_pdf(z_latent, mean, logvar)
    kl_loss = tf.reduce_mean(log_pz - log_qz_x)
    
    return -log_px_z - kl_loss


def cb_autoencoder_loss(y_true, y_pred, precision=1e-4):
    """
    Modified VAE Continuous Bernoulli loss for regular Autoencoder
    (KL divergence in latent space has been removed)

    Args:
        y_true: true values of the data, e.g true pixel values.
        shape = [batch_size, d0, .. dN]
        
        y_pred: predicted outputs of the network, e.g reconstructed pixel values.
        shape = [batch_size, d0, .. dN]
        
        Both are assumed to contain values in the range [0, 1],
        and have compatible shapes.
        
        z_latent: sampled lantent variable
        shape = [batch_size, latent_dim]
 
    Returns:
         Loss float tensor
        
    Each pixel is assumed to follow a continuous bernoulli distribution.
    This is similar to the regular (binarized) bernoulli ELBO,
    but with an added regularization term.
    
    -- Example usage: --
    
    model = Autoencoder(latent_dim)
    model.compile(optimizer='Adam', loss = cb_autoencoder_loss)
    model.fit(...)
    
    """ 
    
    if y_true is None:
        raise ValueError("targets must not be None.")
    if y_pred is None:
        raise ValueError("predictions must not be None.")
    
    y_true_flat, y_pred_flat = _flatten_and_convert(y_true, y_pred)
    
    # Clip outputs to avoid numerical instability
    y_pred_flat = tf.clip_by_value(y_pred_flat, precision, 1-precision)
    
    return -reconstruction_loss(y_true_flat, y_pred_flat)


def reconstruction_loss(y_true, y_pred):
    # Calculates Bernoulli reconstruction loss between y_true and y_pred
    
    # Normalizing constant
    log_norm = _cont_bern_log_norm(y_pred)
    
    log_p_all = y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred) + log_norm
    return tf.reduce_mean(tf.reduce_sum(log_p_all, axis=1))


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


# Returns logarithm of normal pdf w. 
# mean, logvar for each value in 'sample'
def _log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)

    # log(norm_pdf(sample; mean, var))
    values = -.5 * (logvar + log2pi + 
    (sample - mean)**2 * tf.exp(-logvar))

    # 'values' consists of log pdf in both latent x- and y-directions
    # 'reduce_sum' sums these up to one value
    return tf.reduce_sum(values, axis=raxis)


def _flatten_and_convert(y_true, y_pred):
    # makes sure params are in tf formats?
    y_true = ops.convert_to_tensor(y_true)
    y_pred = ops.convert_to_tensor(y_pred)
    
    # check shape compatibility
    y_pred.get_shape().assert_is_compatible_with(y_true.get_shape())
    
    # cast to same dtype
    y_true = math_ops.cast(y_true,  y_pred.dtype)
    
    # Flatten. Potential problem: assumes y is in shape [batch_size, d0, .., dN, 1]
    # and turns into shape [batch_size, prod(d0,..,dN), 1].
    # Does not work if y has shape [batch_size, d0, .., dN] (change?)

    # y_pred.shape gives None so not compatible with eager execution?
    # tf.shape needed instead?
    y_pred_len = tf.shape(y_pred)[0]
    y_true_len = tf.shape(y_true)[0]
    
    y_pred_flat = tf.reshape(y_pred, [y_pred_len, -1, 1])
    y_true_flat = tf.reshape(y_true, [y_true_len, -1, 1])
    
    return y_true_flat, y_pred_flat