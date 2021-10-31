import tensorflow as tf

class NormalSamplingLayer(tf.keras.layers.Layer):
    """
        This class inherits frow keras klass Layer. It takes in a vector of
        means and a vector of log-variances. The input should be anle to be
        unpacked. Then for each input, we will sample from a normal
        distribution with the given mean and varaices. This sample is what is 
        fed forward.
    """
    
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

