import tensorflow as tf


# got some model ideas from here https://medium.com/@jonathan_hui/gan-dcgan-deep-convolutional-generative-adversarial-networks-df855c438f
# http://karpathy.github.io/2019/04/25/recipe/
# https://towardsdatascience.com/deciding-optimal-filter-size-for-cnns-d6f7b56f9363

class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(input_shape=(192, 192, 3)),
            tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=(2, 2), use_bias=False, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=(2, 2), use_bias=False, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
            filters=16, kernel_size=3, strides=(2, 2), use_bias=False, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.generative_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=24*24*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(24, 24, 32)),
            tf.keras.layers.Conv2DTranspose(
            filters=16,
            kernel_size=3,
            strides=(2, 2),
            use_bias=False,
            padding="SAME",
            activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(
            filters=32,
            kernel_size=3,
            strides=(2, 2),
            use_bias=False,
            padding="SAME",
            activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(
            filters=64,
            kernel_size=3,
            strides=(2, 2),
            use_bias=False,
            padding="SAME",
            activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # No activation
            tf.keras.layers.Conv2DTranspose(
            filters=3,
            kernel_size=3, strides=(1, 1), padding="SAME", activation='sigmoid'),
        ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(4, self.latent_dim))
        return self.decode(eps)

    @tf.function
    def encode(self, x):
        mean, logvar = tf.split(
            self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def decode(self, z):
        return self.generative_net(z)
