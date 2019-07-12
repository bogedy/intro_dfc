
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.backend import batch_flatten
from data import *
from model import *
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
def compute_loss(model, x, mode, scales, test=False):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_r = model.decode(z)
    rv = {}

    # Regularization term (KL divergence)
    kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=-1)
    if 'kl_loss' in scales.keys(): kl_loss *= scales['kl_loss']
    rv['kl_loss']=kl_loss

    # Different losses for different trianing modes.
    if mode == 'vae':
        # Reconstruction loss
        rc_loss = mse(x, x_r)
        if 'rc_loss' in scales.keys(): rc_loss *= scales['rc_loss']
        rv['rc_loss']=rc_loss
        # Average over mini-batch and balance the losses
        total_loss = tf.reduce_mean(rc_loss + kl_loss)

    if mode == 'dfc':
        # get deep features
        outputs = model.get_features(x)
        outputs_r = model.get_features(x_r)
        # Perceptual loss
        perceptual_losses = [mse(original, reconstructed) for original, reconstructed in zip(outputs, outputs_r)]
        for layer, loss in zip(model.selected_layers, perceptual_losses):
            if layer in scales.keys(): loss*=scales[layer]
            rv[layer]=loss
        percep_loss = sum([rv[layer] for layer in model.selected_layers])
        if 'percep_loss' in scales.keys(): percep_loss *= scales['percep_loss']
        rv['percep_loss']=percep_loss
        total_loss = tf.reduce_mean(percep_loss + kl_loss)

    if mode == 'combo':
        outputs = model.get_features(x)
        outputs_r = model.get_features(x_r)
        perceptual_losses = [mse(original, reconstructed) for original, reconstructed in zip(outputs, outputs_r)]
        for layer, loss in zip(model.selected_layers, perceptual_losses):
            if layer in scales.keys(): loss*=scales[layer]
            rv[layer]=loss
        percep_loss = sum(perceptual_losses)
        if 'percep_loss' in scales.keys(): percep_loss *= scales['percep_loss']
        rv['percep_loss']=percep_loss
        rc_loss = mse(x, x_r)
        if 'rc_loss' in scales.keys(): rc_loss *= scales['rc_loss']
        rv['rc_loss']=rc_loss
        total_loss = tf.reduce_mean(percep_loss + rc_loss + kl_loss)

    rv['total_loss']=total_loss

    if test:
        rv['x']=x
        rv['x_r']=x_r
    return rv

@tf.function
def train_step(batch, model, optimizer, opt2, opt3, mode, scales):
    with tf.GradientTape(persistent=True) as tape:
        loss_dict = compute_loss(model, batch, mode, scales)
    inf_gradients = tape.gradient(loss_dict['total_loss'], model.inference_net.trainable_variables)
    gen_gradients = tape.gradient(loss_dict['total_loss'], model.generative_net.trainable_variables)
    optimizer.apply_gradients(zip(inf_gradients, model.inference_net.trainable_variables))
    opt2.apply_gradients(zip(gen_gradients, model.generative_net.trainable_variables))
    if mode == 'dfc' or mode == 'combo':
        opt3.apply_gradients(zip(inf_gradients, model.percep_net.trainable_variables))
    return loss_dict

# Use a class to create tf.variables on call for AutoGraph
class test:
    def __init__(self, loss_dict, image_size):
        #testing metrics
        self.metric_dict = {key: tf.metrics.Mean() for key in loss_dict}
        self.losses_dict = loss_dict
        self.losses_dict['x']=tf.zeros(shape=(loss_dict['kl_loss'].shape[0], image_size, image_size, 3))
        self.losses_dict['x_r']=tf.zeros(shape=(loss_dict['kl_loss'].shape[0], image_size, image_size, 3))

    @tf.function
    def __call__(self, model, test_set, step, mode, scales):
        with tf.device('/gpu:0'):
            for batch in test_set:
                self.losses_dict = compute_loss(model, batch, mode, scales, test=True)
                for loss, metric in self.metric_dict.items():
                    metric.update_state(self.losses_dict[loss])
        rv = self.metric_dict['total_loss'].result()
        for loss, metric in self.metric_dict.items():
            tf.summary.scalar(loss, metric.result(), step=step)
            metric.reset_states()
        with tf.device('/cpu:0'):
            tf.summary.image('input', self.losses_dict['x'], step = step, max_outputs=3)
            tf.summary.image('output', self.losses_dict['x_r'], step = step, max_outputs=3)
        return rv
