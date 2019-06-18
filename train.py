
import tensorflow as tf
from tensorflow.keras import layers
from data import *
from models import *
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import imageio


#loss definitions

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

mse=tf.losses.MeanSquaredError()

@tf.function
def compute_loss(model, x, test=False):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_r = model.decode(z)

    # Reconstruction loss
    rc_loss = mse(x, x_r)

    # Regularization term (KL divergence)
    kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=-1)

    # Average over mini-batch
    total_loss = tf.reduce_mean(rc_loss + kl_loss)

    if test:
        return rc_loss, kl_loss, total_loss, x, x_r
    else:
        return rc_loss, kl_loss, total_loss


@tf.function
def train_step(batch, model, optimizer):
    with tf.GradientTape() as tape:
        rc_loss, kl_loss, loss = compute_loss(model, batch)
    gradients=tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return rc_loss, kl_loss, loss

@tf.function
def test(model, test_set, step):
    rcmetric = tf.metrics.Mean()
    klmetric = tf.metrics.Mean()
    totalmetric  = tf.metrics.Mean()
    for batch in test_set:
        rc_loss, kl_loss, total_loss, x, x_r = compute_loss(model, batch, test=True)
        rcmetric.update_state(rc_loss)
        klmetric.update_state(kl_loss)
        totalmetric.update_state(total_loss)
    tf.summary.scalar('rc_loss', rcmetric.result(), step = step)
    tf.summary.scalar('kl_loss', klmetric.result(), step = step)
    tf.summary.scalar('total_loss', totalmetric.result(), step = step)
    tf.summary.image('input', x, step = optimizer.iterations, max_outputs=3)
    tf.summary.image('output', x_r, step = optimizer.iterations, max_outputs=3)
    return totalmetric.result()

if __name__ == "__main__":
    #folder to save weights and images
    DIR='train1'

    #input the celeb faces directory relative to the cwd
    image_dir='../img_align_celeba'

    all_image_paths=get_paths(image_dir)
    image_count=len(all_image_paths)

    train_paths=all_image_paths[:-20000]
    test_paths=all_image_paths[-20000:]

    BATCH_SIZE = 128

    train_set= from_path_to_tensor(train_paths, BATCH_SIZE)
    test_set=from_path_to_tensor(test_paths, BATCH_SIZE)

    train_dir='./{}/train'.format(DIR)
    test_dir='./{}/test'.format(DIR)

    # check if I'm about to overwrite event files
    train_exists = os.path.exists(train_dir) and len(os.listdir(train_dir))!=0
    test_exists = os.path.exists(test_dir) and len(os.listdir(test_dir))!=0
    assert (not train_exists), "You are going to overwrite your train event files."
    assert (not test_exists), "You are going to overwrite your test event files."

    # Tensorboard logdirs
    train_summary_writer = tf.summary.create_file_writer(train_dir)
    test_summary_writer = tf.summary.create_file_writer(test_dir)

    epochs = 10
    latent_dim = 50
    num_examples_to_generate = 4

    optimizer=tf.optimizers.Adam(1e-4)

    # to be used for checking progress.
    random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim])

    log_freq=10

    model = VAE(latent_dim)

    for epoch in range(1,epochs+1):
        start_time = time.time()
        rcmetric = tf.metrics.Mean()
        klmetric = tf.metrics.Mean()
        totalmetric  = tf.metrics.Mean()
        for batch in train_set:
            rc_loss, kl_loss, loss = train_step(batch, model, optimizer)
            rcmetric.update_state(rc_loss)
            klmetric.update_state(kl_loss)
            totalmetric.update_state(total_loss)
            if tf.equal(optimizer.iterations % log_freq, 0):
                with train_summary_writer:
                    tf.summary.scalar('rc_loss', rcmetric.result(), step = step)
                    tf.summary.scalar('kl_loss', klmetric.result(), step = step)
                    tf.summary.scalar('total_loss', totalmetric.result(), step = step)
                rcmetric.reset_states()
                klmetric.reset_states()
                totalmetric.reset_states()
        with test_summary_writer:
            avg_loss = test(model, test_set, optimizer.iterations)
            print('Epoch: {}, test set average loss: {},'.format(epoch, avg_loss),
                'time elapse for current epoch {}'.format(time.time() - start_time))
        tf.saved_model.save(model, './{}/{}'.format(DIR,epoch))
