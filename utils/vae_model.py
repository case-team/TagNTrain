import os
import matplotlib.pyplot as plt
import pathlib

import tensorflow as tf

from .losses import *


class VAE( object ):

    def __init__(self, run=0, input_shape = (32,32,1), log_dir="", model_dir=""):
        # network parameters
        self.input_shape = input_shape
        self.batch_size = 256
        self.kernel_size = 3
        self.filter_n = 6
        self.z_size = 10
        self.encoder = None
        self.decoder = None
        self.model = None
        self.run = run
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.regularizer = None # regularizers.l2()

    # adding keras mse as dummy loss, because training loss in function closure not (easily) accessible and model won't load without all custom function references
    def load( self ):
        self.encoder = tf.keras.models.load_model(os.path.join(self.model_dir, 'encoder.h5'), custom_objects={'mse_kl_loss': mse_kl_loss, 'mse_loss': mse_loss, 'kl_loss': kl_loss, 'sampling' : self.sampling})
        self.decoder = tf.keras.models.load_model(os.path.join(self.model_dir, 'decoder.h5'), custom_objects={'mse_kl_loss': mse_kl_loss, 'mse_loss': mse_loss, 'kl_loss': kl_loss})
        self.model = tf.keras.models.load_model(os.path.join(self.model_dir, 'vae.h5'), custom_objects={'mse_kl_loss': mse_kl_loss, 'mse_loss': mse_loss, 'kl_loss': kl_loss, 'loss': tf.keras.losses.mse, 'sampling' : self.sampling})


    def build( self ):

        inputs = tf.keras.layers.Input(shape=self.input_shape, name='encoder_input')
        self.encoder = self.build_encoder( inputs )
        self.decoder = self.build_decoder( )
        outputs = self.decoder( self.encoder(inputs)[-1] )  # link encoder output to decoder
        # instantiate VAE model
        vae = tf.keras.Model(inputs, outputs, name='vae')
        vae.summary()
        self.compile( vae )
        self.model = vae


    def compile(self, model):
        model.compile(optimizer='adam', loss=mse_kl_loss(self.z_mean, self.z_log_var), metrics=[mse_loss, kl_loss_for_metric(self.z_mean,self.z_log_var)])  # , metrics=loss_metrics monitor mse and kl terms of loss 'rmsprop'

    # ***********************************
    #               encoder
    # ***********************************
    def build_encoder(self, inputs):

        x = inputs
        for i in range(3):
            x = tf.keras.layers.Conv2D(filters=self.filter_n, kernel_size=self.kernel_size, activation='relu', kernel_regularizer=self.regularizer)(x)
            self.filter_n += 4

        x = tf.keras.layers.AveragePooling2D()(x)
        # x = MaxPooling2D( )( x )

        # shape info needed to build decoder model
        self.shape_convolved = x.get_shape().as_list()

        # 3 dense layers
        x = tf.keras.layers.Flatten()(x)
        self.size_convolved = x.get_shape().as_list()
        x = tf.keras.layers.Dense(self.size_convolved[1] // 17, activation='relu',kernel_regularizer=self.regularizer)(x)  # reduce convolution output
        x = tf.keras.layers.Dense(self.size_convolved[1] // 42, activation='relu',kernel_regularizer=self.regularizer)(x)  # reduce again
        #x = Dense(8, activation='relu')(x)

        # *****************************
        #         latent space
        # generate latent vector Q(z|X)

        self.z_mean = tf.keras.layers.Dense(self.z_size, name='z_mean', kernel_regularizer=self.regularizer)(x)
        self.z_log_var = tf.keras.layers.Dense(self.z_size, name='z_log_var', kernel_regularizer=self.regularizer)(x)

        # use reparameterization trick to push the sampling out as input
        z = tf.keras.layers.Lambda(self.sampling, output_shape=(self.z_size,), name='z')([self.z_mean, self.z_log_var])

        # instantiate encoder model
        encoder = tf.keras.Model(inputs, [self.z_mean, self.z_log_var, z], name='encoder')
        encoder.summary()
        # plot_model(encoder, to_file=CONFIG['plotdir']+'vae_cnn_encoder.png', show_shapes=True)
        return encoder


    # ***********************************
    #           decoder
    # ***********************************
    def build_decoder(self):

        latent_inputs = tf.keras.layers.Input(shape=(self.z_size,), name='z_sampling')
        x = tf.keras.layers.Dense(self.size_convolved[1] // 42, activation='relu',kernel_regularizer=self.regularizer)(latent_inputs)  # inflate to input-shape/200
        x = tf.keras.layers.Dense(self.size_convolved[1] // 17, activation='relu',kernel_regularizer=self.regularizer)(x)  # double size
        x = tf.keras.layers.Dense(self.shape_convolved[1] * self.shape_convolved[2] * self.shape_convolved[3], activation='relu',kernel_regularizer=self.regularizer)(x)
        x = tf.keras.layers.Reshape((self.shape_convolved[1], self.shape_convolved[2], self.shape_convolved[3]))(x)

        x = tf.keras.layers.UpSampling2D()(x)

        for i in range(3):
            self.filter_n -= 4
            x = tf.keras.layers.Conv2DTranspose(filters=self.filter_n, kernel_size=self.kernel_size, activation='relu',kernel_regularizer=self.regularizer)(x)

        outputs_decoder = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=self.kernel_size, activation='relu', kernel_regularizer=self.regularizer, padding='same', name='decoder_output')(x)

        # instantiate decoder model
        decoder = tf.keras.Model(latent_inputs, outputs_decoder, name='decoder')
        decoder.summary()
        # plot_model(decoder, to_file=CONFIG['plotdir'] + 'vae_cnn_decoder.png', show_shapes=True)
        return decoder


    def fit( self, t_data, epochs=3, verbose=2, validation_data = None):
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1),tf.keras.callbacks.TerminateOnNaN(),
                     ] #TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        self.history = self.model.fit(t_data, epochs=epochs, verbose=verbose, callbacks=callbacks, validation_data = validation_data)
        return self.history


    def predict(self, x):
        return self.model.predict( x, batch_size=self.batch_size )


    def predict_with_latent(self,x):
        z_mean, z_log_var, z = self.encoder.predict(x, batch_size=self.batch_size)
        reco = self.decoder.predict(z, batch_size=self.batch_size)
        return [ reco, z_mean, z_log_var ]


    # ***********************************
    #       reparametrization trick
    # ***********************************
    def sampling( self, args ):
        """
        instead of sampling from Q(z|X),
        sample eps = N(0,I), then z = z_mean + sqrt(var)*eps

        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)

        # Returns:
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        # by default, random_normal has mean=0 and std=1.0
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


    def save_model(self):
        print('saving model to {}'.format(self.model_dir))
        self.encoder.save(os.path.join(self.model_dir, 'encoder.h5'))
        self.decoder.save(os.path.join(self.model_dir,'decoder.h5'))
        self.model.save(os.path.join(self.model_dir,'vae.h5'))


    def plot_training(self, fig_dir=""):
        plt.figure()
        plt.semilogy(self.history.history['loss'])
        plt.semilogy(self.history.history['val_loss'])
        plt.title('training and validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training','validation'], loc='upper right')
        plt.savefig(os.path.join(fig_dir,'loss.png'))
        plt.close()

    def sample_pixels_from_dist(self,dist):
        return np.random.exponential(1. / dist)  # numpy exponential dist takes 1/k param instead of k param
