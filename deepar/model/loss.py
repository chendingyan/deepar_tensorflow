import tensorflow as tf
import math
from tensorflow.keras.losses import Loss, Reduction
from tensorflow.keras.activations import softplus
import numpy as np

def build_tf_lookup(scale_data):
    """
    create and return tf hash table mapping scale keys to scale values 
        :param scale_data: pd series where index = scale keys and values = scale values 
    """
    return tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(scale_data.index.values, dtype = tf.int32),
            values=tf.constant(scale_data.values, dtype = tf.float32),
        ),
        default_value=tf.constant(-1, dtype = tf.float32),
    )

def unscale(mu, scale, scale_keys, hash_table):
    """
    unscales predictions (mu and scale) using scale_keys and values
        :param mu:
        :param scale:
        :param scale_keys: needs to be same shape as mu and theta
        :hash_table: tf hash table mapping scale keys to scale values
    """

    # unscale mu and sigma
    scale_values = tf.expand_dims(hash_table.lookup(scale_keys), 1)
    scaled_mu = tf.multiply(mu, scale_values)
    scaled_sigma = tf.divide(scale, tf.sqrt(scale_values))
    return scaled_mu, scaled_sigma

class GaussianLogLikelihood(Loss):
    """
    Custom GaussianLogLikelihood loss function
      :param mask_value: value in y_true that should be masked in loss (missing tgt in training set)
      :param name:
    """
    def __init__(self, mask_value = -10000, name='gaussian_log_likelihood'):
        
        super(GaussianLogLikelihood, self).__init__(reduction=Reduction.AUTO, name = name)
        self.mask_value = mask_value

    def _mask_loss(self, loss_term, y_true, mask_value):
        """
        util function
            mask loss tensor (y_true) according to mask_value locations
        """

        # mask loss with mask_value
        mask = tf.not_equal(y_true, tf.constant(mask_value, dtype=tf.float32))
        mask = tf.dtypes.cast(mask, tf.float32)
        return tf.multiply(loss_term, mask)

    ## TODO check numerical stability!!
    def call(self, y_true, y_pred_bundle):
        """
        return: loss (mask, scaling inversed)
        """

        mu, sigma = y_pred_bundle
        batch_size = mu.shape[0] 

        # loss
        loss_term = 0.5*tf.math.log(sigma) + 0.5*tf.divide(tf.square(y_true - mu), sigma)
        
        # mask
        masked_loss_term = self._mask_loss(loss_term, y_true, self.mask_value)

        # divide by batch size bc auto reduction will sum over batch size
        return (masked_loss_term + 1e-6 + 6) / batch_size


class NegativeBinomialLogLikelihood(Loss):
    """
    Custom NegativeBinomialLogLikelihood loss function
      :param mask_value: value in y_true that should be masked in loss (missing tgt in training set)
    """
    def __init__(self, mask_value = -10000, name='negative_binomial_log_likelihood'):
        
        super(NegativeBinomialLogLikelihood, self).__init__(reduction=Reduction.AUTO, name = name)
        self.mask_value = mask_value

    def _mask_loss(self, loss_term, y_true, mask_value):
        """
        util function
            mask loss tensor (y_true) according to mask_value locations
        """

        # mask loss with mask_value
        mask = tf.not_equal(y_true, tf.constant(mask_value, dtype=tf.float32))
        mask = tf.dtypes.cast(mask, tf.float32)
        return tf.multiply(loss_term, mask)

    def call(self, y_true, y_pred_bundle):
        """
        return: loss (mask, scaling inversed)
        """

        mu, alpha = y_pred_bundle
        batch_size = mu.shape[0] 

        # loss
        alpha_y_pred = tf.multiply(alpha, mu)
        alpha_div = tf.divide(1.0, alpha)
        denom = tf.math.log(1 + alpha_y_pred)
        log_loss = \
            tf.math.lgamma(y_true + alpha_div) - tf.math.lgamma(y_true + 1.0) - tf.math.lgamma(alpha_div)
        loss_term = \
            log_loss - tf.divide(denom, alpha) + tf.multiply(y_true, tf.math.log(alpha_y_pred) - denom)
        loss_term = -loss_term

        # mask
        masked_loss_term = self._mask_loss(loss_term, y_true, self.mask_value)

        # divide by batch size bc auto reduction will sum over batch size
        return (masked_loss_term + 1e-6 + 6) / batch_size
