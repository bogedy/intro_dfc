
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
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_r = model.decode(z)

    # Reconstruction loss
    rc_loss = mse(x, x_r)

    # Regularization term (KL divergence)
    kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=-1)

    # Average over mini-batch
    return tf.reduce_mean(rc_loss + kl_loss)

#generate_and_save_images(model, 0, random_vector_for_generation)

def generate_and_save_images(model, epoch, batch, test_input):
    predictions = model.sample(test_input)
    fig = plt.figure(figsize=(2,2))

    for i in range(predictions.shape[0]):
        plt.subplot(2, 2, i+1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig(TRAINING_DIR+'/image_at_epoch_{:04d}_batch_{:05d}.png'.format(epoch, batch))
    #plt.show()

@tf.function
def train_step(batch, model, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, batch)
    gradients=tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function
def test()


train_summary_writer = tf.summary.create_file_writer(TRAINING_DIR+'/summaries/train')
test_summary_writer = tf.summary.create_file_writer(TRAINING_DIR+'/summaries/test')


#todo: write train function and separate test function. I guess there's no
#reason to have this in a function?


def train_vae(model, optimizer, epochs, dataset, save_interval, log_freq=10):
    summary_writer = tf.summary.create_file_writer(DIR)
    for epoch in range(1, epochs + 1):
        avg_loss=tf.metrics.Mean(name='loss', dtype=tf.float32)
        start_time = time.time()
        batch_start_time=start_time
        for step, batch in enumerate(dataset):
            loss_x = train_step(batch, model, optimizer)
            avg_loss.update_state(loss_x)
            if tf.equal(optimizer.iterations % log_freq, 0):
                tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)
                avg_loss.reset_states()
            if step+1 % print_interval ==0:
                print('Batch',i,'done.', 'avg. batch time: {}s'.format((time.time()-batch_start_time)/print_interval))
                batch_start_time=time.time()
            if step+1 % save_interval ==0:
                generate_and_save_images(model, epoch, step+1, random_vector_for_generation)
                model.weight_saver(TRAINING_DIR, epoch, i)
                end_time = time.time()

        if epoch % 1 == 0:
            loss = tf.zeros(20000//BATCH_SIZE+1)
            j=0
            for test_x in test_set:
                loss[j]=compute_loss(model, test_x)
                j+=1
                elbo = -np.mean(loss)
            #display.clear_output(wait=False)
            print('Epoch: {}, Test set ELBO: {},'.format(epoch, elbo),
                'time elapse for current epoch {}'.format(epoch,elbo,end_time - start_time))

    generate_and_save_images(model, epoch,0, random_vector_for_generation)

#folder to save weights and images

TRAINING_DIR='train1'

##input the celeb faces directory relative to the cwd

DIR='../img_align_celeba'

all_image_paths=get_paths(DIR)
image_count=len(all_image_paths)

train_paths=all_image_paths[:-20000]
test_paths=all_image_paths[-20000:]

BATCH_SIZE = 128
#BUFFER_SIZE=image_count//9

train_set= from_path_to_tensor(train_paths, BATCH_SIZE)
test_set=from_path_to_tensor(test_paths, BATCH_SIZE)


epochs = 10
latent_dim = 50
num_examples_to_generate = 4

p_interval=100

s_interval=500

optimizer=tf.train.AdamOptimizer(1e-4)

# to be used for checking progress.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])


model = VAE(latent_dim)
with train_summary_writer.as_default():
    train_vae(model, optimizer, epochs, p_interval, s_interval)
