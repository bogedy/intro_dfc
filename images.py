from model.py import *
from train_ops import *
import tensorflow as tf

model = VAE(latent_dim, image_size, mode, kernelsize, loader='experiment18/50')

rand_im = tf.random.normal(shape=(1,192,192,3))

_ = compute_loss(model, rand_im, 'vae', {})

tf.saved_model.save(model, './savedmodel')
