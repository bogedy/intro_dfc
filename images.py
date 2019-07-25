import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

from model import *
from train_ops import *
import tensorflow as tf
from matplotlib import pyplot as plt
'''

rand_im = tf.random.normal(shape=(1,192,192,3))

_ = compute_loss(model, rand_im, 'vae', {})

tf.saved_model.save(model, './savedmodel')
'''
im='noise.png'
imr = tf.io.read_file(im)
tens = tf.image.decode_image(imr, dtype=tf.float32)
tens=tf.expand_dims(tens, 0)

model = VAE(50, 192, 'vae', 3, loader='experiment18/50')

mean, logvar = model.encode(tens)

out = model.decode(mean)

out=tf.image.convert_image_dtype(out, tf.uint8)


enc=tf.image.encode_jpeg(out[0])
tf.io.write_file('out.jpg', enc)
