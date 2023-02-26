import glob
import numpy as np
import tensorflow as tf
import time


# import tensorflow_probability as tfp

class AVE:

    def __init__(self):

        self.model = None
        self.input_dim = None
        self.test_dataset = []
        self.train_dataset = []

        self.train_size = 60000
        self.batch_size = 32
        self.test_size = 10000

        self.epochs = 10
        self.latent_dim = 3

    def set_data(self, data, test2train=0.7):

        self.input_dim = data[0].shape

        split = int(len(data) * test2train)

        train_data = data[:split]
        test_data = data[split:]

        self.train_dataset = (
            tf.data.Dataset.from_tensor_slices(train_data).shuffle(self.train_size).batch(self.batch_size))

        self.test_dataset = (
            tf.data.Dataset.from_tensor_slices(test_data).shuffle(self.test_size).batch(self.batch_size))

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

    def compute_loss(self, model, x):
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    @tf.function
    def train_step(self, model, x, optimizer):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(model, x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    def create_model(self, latent):

        self.latent_dim = latent
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.random_vector_for_generation = tf.random.normal(
            shape=[100, self.latent_dim])
        self.model = self.NN(self.input_dim, self.latent_dim)
        print(self.model.encoder.summary())
        #self.model.build( input_shape= self.input_dim, output_shape=self.input_dim)
        #self.model.summary()
    def train(self):

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            for train_x in self.train_dataset:
                self.train_step(self.model, train_x, self.optimizer)
            end_time = time.time()

            loss = tf.keras.metrics.Mean()
            for test_x in self.test_dataset:
                loss(self.compute_loss(self.model, test_x))
            elbo = -loss.result()
            #print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                  #.format(epoch, elbo, end_time - start_time))

    def test(self, number=10):

        print(self.model.encoder.summary())

        for i, test in enumerate(self.test_dataset):

            if i > number:
                return;
            mean, logvar = self.model.encode(test)
            print(mean.shape, logvar.shape)

    class NN(tf.keras.Model):
        """variational autoencoder."""

        def __init__(self, input_dim, latent_dim):
            super(AVE.NN, self).__init__()
            self.latent_dim = latent_dim
            self.encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=input_dim),
                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                    tf.keras.layers.Dense(256, activation=tf.nn.relu),
                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                    tf.keras.layers.Flatten(),
                    # No activation
                    tf.keras.layers.Dense(latent_dim + latent_dim),
                ]
            )

            self.decoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
                    tf.keras.layers.Dense(256),
                    tf.keras.layers.Dense(input_dim[0]),
                    tf.keras.layers.Reshape(target_shape=input_dim),

                ]
            )

        @tf.function
        def sample(self, eps=None):
            if eps is None:
                eps = tf.random.normal(shape=(100, self.latent_dim))
            return self.decode(eps, apply_sigmoid=True)

        def encode(self, x):
            mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
            return mean, logvar

        def reparameterize(self, mean, logvar):
            eps = tf.random.normal(shape=mean.shape)
            return eps * tf.exp(logvar * .5) + mean

        def decode(self, z, apply_sigmoid=False):
            logits = self.decoder(z)
            if apply_sigmoid:
                probs = tf.sigmoid(logits)
                return probs
            return logits
