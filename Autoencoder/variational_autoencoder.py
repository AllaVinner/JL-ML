
import tensorflow as tf
from tensorflow import keras

class VariationalAutoencoder(keras.Model):
    """
        This class 
    
    """
    def __init__(self, encoder, decoder, sampler = None, **kwargs):
        """
        A varaiational autoencode aims at creating a latent representation of
        a data set. A normal autoencoder works by having a encoder which
        projects the data down to the latent space and a decoder which aims
        at projecting the representation back to the original data. The
        difference between the original data and reconstruction is then summed
        up in a loss which is backpropagated though the model and tunes the 
        parameters in the encoder and decoder to eventually end up with a model
        which approximates the identity function. After the training step we 
        can then pass the data though just the encoder to obtain the latent
        representation of our data, and since we now there exist an inverse 
        (the decoder) no (or very litle) information is lost. The variational 
        autoencoder works similarly but instead of the encoder uotputting
        latent points, it outputs latent distirbutions (effectivly giving a
        vlume to the latent points). This distribution is represented by two
        vectors: the mean, and the variance, and are assumed to be gauessian.
        The distribution given by the encoding is passsed to a sampling layer 
        which samples latent points which is then passed to the decoder.
        
        
        
        
        
        Parameters
        ----------
        encoder : Layer/Model
            The enocder. Takes an images and encodes it into a latent space. 
            The output array consists of two vectors stacked. The first vector
            symbolices the mean of the 
            The encoder should be callable with the argument training.
        decoder : Layer/Model
            Decodes a 
        sampler : Layer/Model (None)
        
        Note:
            The output of the encoder must match the input of the decoder.
            E.g. if the latent dim is l, then the output of the encdoer should
            be (2,l), and the input of the decoder should be (l,). 
        """
        super(VariationalAutoencoder, self).__init__(**kwargs)
        # Set up metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.normalizing_loss_tracker = keras.metrics.Mean(
            name="normalizing_loss"
            )
        
        # Initiate model structure 
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler if sampler is not None\
                       else NormalSamplingLayer()
        
    def call(self, inputs, training = False, **kwargs):
        x = inputs
        x = self.encoder(x, training = training)
        x = self.sampler(x, training = training)
        x = self.decoder(x, training = training)
        return x
    
    @property
    def latent_dim(self):
        return self.decoder.input_shape[-1]
    
    def encode(self, inputs, **kwargs):
        latent = self.encoder(inputs, training = False)
        z_mean = self.sampler(latent, training = False)
        return z_mean
    
    def get_config(self):
        config = {"encoder": self.encoder,
                  "decoder": self.decoder}
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
    def train_step(self, inputs, **kwargs):
        """
        Performs a single step of training on the data given. 
        
        A self defined train_step function usually consists of the forward
        pass, loss calculation, backpropagation, and metric updates.
            
        Parameters
        ----------
        data : array
            Input data
        
        """
        with tf.GradientTape() as tape:
            # Forward propagation
            latent            = self.encoder(inputs, training = True)
            latent_sample     = self.sampler(latent, training= True)
            reconstruction    = self.decoder(latent_sample, training= True)
            # Model overflow oterwise
            reconstruction    = tf.clip_by_value(reconstruction, 1e-4, 1 - 1e-4)
            z_mean, z_log_var = tf.unstack(latent, axis = 1)
            # Calculate losses
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(inputs, reconstruction), axis=(1, 2)
                )
            )
            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            #normalizing_loss = -self._cont_bern_log_norm(reconstruction)
            #normalizing_loss = tf.reduce_mean(tf.reduce_sum(normalizing_loss, axis=1))
            #total_loss = reconstruction_loss + kl_loss + normalizing_loss
            total_loss = reconstruction_loss + kl_loss
            normalizing_loss = -1
        
        # Calculate and apply gradient
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.normalizing_loss_tracker.update_state(normalizing_loss)
        
        # Return training message
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "Normalizing_loss": self.normalizing_loss_tracker.result(),
        }
    
        
    def _cont_bern_log_norm(self,lam, l_lim=0.49, u_lim=0.51):
        # computes the log normalizing constant of a continuous Bernoulli distribution in a numerically stable way.
        # returns the log normalizing constant for lam in (0, l_lim) U (u_lim, 1) and a Taylor approximation in
        # [l_lim, u_lim].
        # cut_y below might appear useless, but it is important to not evaluate log_norm near 0.5 as tf.where evaluates
        # both options, regardless of the value of the condition.
        cut_lam = tf.where(tf.logical_or(tf.less(lam, l_lim), tf.greater(lam, u_lim)), lam, l_lim * tf.ones_like(lam))
        log_norm = tf.math.log(tf.abs(2.0 * tf.math.atanh(1 - 2.0 * cut_lam))) - tf.math.log(tf.abs(1 - 2.0 * cut_lam))
        taylor = tf.math.log(2.0) + 4.0 / 3.0 * tf.pow(lam - 0.5, 2) + 104.0 / 45.0 * tf.pow(lam - 0.5, 4)
        return tf.where(tf.logical_or(tf.less(lam, l_lim), tf.greater(lam, u_lim)), log_norm, taylor)
        
class NormalSamplingLayer(tf.keras.layers.Layer):
    """
        This class inherits frow keras klass Layer. It takes in a vector of
        means and a vector of log-variances. The input should be anle to be
        unpacked. Then for each input, we will sample from a normal
        distribution with the given mean and varaices. This sample is what is 
        fed forward.
    """
    
    def __init__(self, **kwargs):
        super(NormalSamplingLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        pass
        

    def call(self, inputs, training = False, **kwargs):
        """
            Defines the arcitechture with the input and sends the signal
            forward.
            
            The input consists of the mean and log-variances, and we then 
            sample a new sample from the corresponding normal distribution
            which is then passed forward.
            
            Parameters
            ----------
            inputs : tuple/list (None,2,None)
                Containimg the array of means and log variances.
        """
        z_mean, z_log_var = tf.unstack(inputs, axis = 1)
        if not training: return z_mean
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z_samp = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z_samp

    def get_config(self):
        config = super(NormalSamplingLayer,self).get_config()
        return config      

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    
if __name__ == '__main__':
    import numpy as np
    import numpy.random as rnd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow import keras
    from variational_autoencoder import VariationalAutoencoder
    
    # For dimension reduction
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    # for visualization and evaluation
    from latent_plane_mosaic import LatentPlaneMosaic
    from latent_interpolation_mosaic import LatentInterpolationMosaic
    from sample_scatter_gui import SampleScatterGUI
    
    
    latent_dim = 10
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
    mnist_labels = np.concatenate([y_train, y_test], axis=0)
    input_shape = mnist_digits.shape[1:]
    input_shape
    
    
    # Variational autoencoder
    encoder = keras.Sequential([
        keras.Input(input_shape),
        keras.layers.Conv2D(4, (3,3), padding = "same", activation = "relu"),
        keras.layers.Conv2D(4, (3,3), padding = "same", activation = "relu"),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Conv2D(8, (3,3), padding = "same", activation = "relu"),
        keras.layers.Conv2D(8, (3,3), padding = "same", activation = "relu"),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Conv2D(16, (3,3), padding = "same", activation = "relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(2*latent_dim),
        keras.layers.Reshape((2,latent_dim)),
    ], name = "Encoder" )
    
    decoder = keras.Sequential([
                keras.Input((latent_dim,)),
                keras.layers.Dense(16*7*7, activation = "relu"),
                keras.layers.Reshape((7,7,16)),
                keras.layers.Conv2DTranspose(16, (3,3), activation = "relu", padding = "same"),
                keras.layers.Conv2DTranspose(16, (3,3), activation = "relu", padding = "same"),
                keras.layers.UpSampling2D((2,2)),
                keras.layers.Conv2DTranspose(8, (3,3), activation = "relu", padding = "same"),
                keras.layers.Conv2DTranspose(8, (3,3), activation = "relu", padding = "same"),
                keras.layers.UpSampling2D((2,2)),
                keras.layers.Conv2DTranspose(8, (3,3), activation = "relu", padding = "same"),
                keras.layers.Conv2DTranspose(1, (3,3), activation = "sigmoid", padding = "same"),
    ], name = "Decoder")
    va = VariationalAutoencoder(encoder, decoder)
    va.compile(optimizer = "adam")
    
    
    
    #va.train_step(mnist_digits[0:10])
    
    va.fit(mnist_digits,
          epochs = 1,
          batch_size = 256)
    
    
    
    
    

    