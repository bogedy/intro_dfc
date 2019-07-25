from model import *
from train_ops import *
import tensorflow as tf

model = VAE(50, 192, 'vae', 3, loader='experiment18/50')

rand_im = tf.random.normal(shape=(1,192,192,3))

_ = compute_loss(model, rand_im, 'vae', {})

tf.saved_model.save(model, './savedmodel')
