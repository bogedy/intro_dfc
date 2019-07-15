import tensorflow as tf


# got some model ideas from here https://medium.com/@jonathan_hui/gan-dcgan-deep-convolutional-generative-adversarial-networks-df855c438f
# http://karpathy.github.io/2019/04/25/recipe/
# https://towardsdatascience.com/deciding-optimal-filter-size-for-cnns-d6f7b56f9363

# image dim must be divisible by 8
class VAE(tf.keras.Model):
    def __init__(self, latent_dim, image_dim, mode, kernelsize=3, selected_layers = None, loader = None):
        super(VAE, self).__init__()
        if loader == None:
            self.inference_net = tf.keras.Sequential(
                [
                tf.keras.layers.Conv2D(
                filters=64, kernel_size=kernelsize, strides=(2, 2), use_bias=False, activation='relu', input_shape=(image_dim, image_dim, 3)),
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
                tf.keras.layers.Dense(units=((image_dim//8)**2)*32, activation=tf.nn.relu, input_shape=(latent_dim,)),
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

        if loader:
            self.inference_net = tf.keras.models.load_model(loader+'/inf')
            self.generative_net = tf.keras.models.load_model(loader+'/gen')

        # Create a separate network for calculating loss, it receives the same updates
        # as the inference net. They should always be equal.
        if mode == 'dfc' or mode == 'combo':
            self.percep_net = tf.keras.models.clone_model(self.inference_net)
            self.percep_net.set_weights(self.inference_net.get_weights())

        # if no layers are specififed, use the first two convolution layers
        if selected_layers:
            self.selected_layers = selected_layers
        else:
            self.selected_layers = [layer.name for layer in self.inference_net.layers if layer.name.startswith('conv')][:2]

    @tf.function
    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def decode(self, z):
        return self.generative_net(z)

    @tf.function
    def get_features(self, x):
        rv = []
        for layer in self.percep_net.layers:
                # We do not want to apply updates for the inference pass
                x=layer(x)
                if layer.name in self.selected_layers:
                    rv.append(x)
                if len(rv) == len(self.selected_layers):
                    return rv

    def saver(self, DIR, tag):
        directory = './{}/{}'.format(DIR, tag)
        if not os.path.exists(directory):
            os.mkdir(directory)
        self.inference_net.save(directory+'/inf', save_format='h5')
        self.generative_net.save(directory+'/gen', save_format='h5')
