
import tensorflow as tf
from tensorflow.keras import layers
from tensorlfow.keras.backend import batch_flatten
from data import *
from models import *
import os
import time
import numpy as np

#loss definitions

# Couldn't figure how to do some simple per pixel mse... smh tensorlfow!
@tf.function
def mse(label, prediction):
    #flatten the tensors, maintaining batch dim
    return tf.losses.MSE(batch_flatten(label), batch_flatten(prediction))

@tf.function
def compute_loss(model, x, mode, test=False):
    mean, logvar = model.encode(x, percep=True)
    z = model.reparameterize(mean, logvar)
    x_r = model.decode(z)
    rv = {}

    # Regularization term (KL divergence)
    kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=-1)
    rv['kl_loss']=kl_loss

    # Different losses for different trianing modes.
    if mode == 'vae':
        # Reconstruction loss
        rc_loss = mse(x, x_r)
        rv['rc_loss']=rc_loss
        # Average over mini-batch and balance the losses
        total_loss = tf.reduce_mean(rc_loss*1e4 + kl_loss)

    if mode == 'dfc':
        # get deep features
        outputs = model.get_features(x)
        outputs_r = model.get_features(x_r)
        # Perceptual loss
        perceptual_losses = [mse(original, reconstructed) for original, reconstructed in zip(outputs, outputs_r)]
        for layer, loss in zip(model.selected_layers, perceptual_losses):
            rv[layer]=loss
        percep_loss = sum(perceptual_losses)
        rv['percep_loss']=percep_loss
        total_loss = tf.reduce_mean(percep_loss + kl_loss)

    if mode == 'combo':
        outputs = model.get_features(x)
        outputs_r = model.get_features(x_r)
        perceptual_losses = [mse(original, reconstructed) for original, reconstructed in zip(outputs, outputs_r)]
        for layer, loss in zip(model.selected_layers, perceptual_losses):
            rv[layer]=loss
        percep_loss = sum(perceptual_losses)
        rv['percep_loss']=percep_loss
        rc_loss = mse(x, x_r)
        rv['rc_loss']=rc_loss
        total_loss = tf.reduce_mean(percep_loss + rc_loss + kl_loss)

    rv['total_loss']=total_loss

    if test:
        rv['x']=x
        rv['x_r']=x_r

    return rv

@tf.function
def train_step(batch, model, optimizer, mode):
    with tf.GradientTape() as tape:
        loss_dict = compute_loss(model, batch, mode)
    gradients=tape.gradient(loss_dict['total_loss'], model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_dict

# Use a class to create tf.variables on call for AutoGraph
class test:
    def __init__(self, loss_dict):
        #testing metrics
        metric_dict = {key: tf.metrics.Mean() for key in loss_dict}

    @tf.function
    def __call__(self, model, test_set, step, mode):
        for batch in test_set:
            losses_dict = compute_loss(model, batch, mode, test=True)
            for loss, value in losses_dict.items():
                self.metric_dict[loss].update_state(value)
        rv = losses_dict['total_loss'].result()
        for loss, metric in self.metric_dict.items():
            tf.summary.scalar(loss, metric.result())
            metric.reset_states()
        tf.summary.image('input', losses_dict['x'], step = step, max_outputs=3)
        tf.summary.image('output', losses_dict['x_r'], step = step, max_outputs=3)
        return rv
