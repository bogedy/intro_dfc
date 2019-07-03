import tensorflow as tf


# got some model ideas from here https://medium.com/@jonathan_hui/gan-dcgan-deep-convolutional-generative-adversarial-networks-df855c438f
# http://karpathy.github.io/2019/04/25/recipe/
# https://towardsdatascience.com/deciding-optimal-filter-size-for-cnns-d6f7b56f9363

# image dim must be divisible by 8
class VAE(tf.keras.Model):
    def __init__(self, latent_dim, image_dim=192, kernelsize=3, selected_layers = None):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(input_shape=(image_dim, image_dim, 3)),
            tf.keras.layers.Conv2D(
            filters=64, kernel_size=kernelsize, strides=(2, 2), use_bias=False, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
            filters=32, kernel_size=kernelsize, strides=(2, 2), use_bias=False, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
            filters=16, kernel_size=kernelsize, strides=(2, 2), use_bias=False, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.generative_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=((image_dim//8)**2)*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(image_dim//8, image_dim//8, 32)),
            tf.keras.layers.Conv2DTranspose(
            filters=16,
            kernel_size=kernelsize,
            strides=(2, 2),
            use_bias=False,
            padding="SAME",
            activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(
            filters=32,
            kernel_size=kernelsize,
            strides=(2, 2),
            use_bias=False,
            padding="SAME",
            activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(
            filters=64,
            kernel_size=kernelsize,
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

        # if no layers are specififed, use the first two convolution layers
        if selected_layers:
            self.selected_layers = selected_layers
        else:
            self.selected_layers = [layer.name for layer in self.inference_net.layers if layer.name.startswith('conv')][:2]

    @tf.function
    def encode(self, x, percep=False):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def decode(self, z):
        return self.generative_net(z)

    @tf.function
    def get_features(self, x_r):
        rv = []
        for layer in self.inference_net.layers:
                # We do not want to apply updates for the inference pass
                x_r=tf.stop_gradient(layer(x_r))
                if layer.name in self.selected_layers:
                    rv.append(x_r)
                if len(rv) == len(self.selected_layers):
                    return rv
