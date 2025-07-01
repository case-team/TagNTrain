import tensorflow as tf
from .vae_model import *
#from tensorflow.keras.layers import Dense, tf.keras.layers.Conv2D, tf.keras.layers.MaxPooling2D, Dropout, Input, Flatten, tf.keras.layers.Activation, Reshape, UpSampling2D
#from tensorflow.keras.models import Model, Sequential, load_model


def cwbh_net(input_shape, drop_rate = 0.2):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.BatchNormalization(input_shape = (input_shape,)))
    model.add(tf.keras.layers.Dense(64, input_dim=input_shape,use_bias=True,
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/64))))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    model.add(tf.keras.layers.Dropout(drop_rate))
    model.add(tf.keras.layers.Dense(32, use_bias=True, activation='elu',
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/32))))
    model.add(tf.keras.layers.Dropout(drop_rate))
    model.add(tf.keras.layers.Dense(16, use_bias=True, activation='elu',
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/16))))
    model.add(tf.keras.layers.Dense(4, use_bias=True, activation='elu',
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/4))))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model


def dense_net(input_shape, drop_rate = 0.2):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.BatchNormalization(input_shape = (input_shape,)))
    model.add(tf.keras.layers.Dense(64, input_dim=input_shape,use_bias=True,
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/64))))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    model.add(tf.keras.layers.Dropout(drop_rate))
    model.add(tf.keras.layers.Dense(128, use_bias=True, activation='elu',
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/128))))
    model.add(tf.keras.layers.Dropout(drop_rate))
    model.add(tf.keras.layers.Dense(128, use_bias=True, activation='elu',
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/128))))
    model.add(tf.keras.layers.Dropout(drop_rate))
    model.add(tf.keras.layers.Dense(32, use_bias=True, activation='elu',
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/32))))
    model.add(tf.keras.layers.Dropout(drop_rate))
    model.add(tf.keras.layers.Dense(16, use_bias=True, activation='elu',
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/16))))
    model.add(tf.keras.layers.Dense(8, use_bias=True, activation='elu',
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/8))))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

def dense_med_net(input_shape, drop_rate = 0.2):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.BatchNormalization(input_shape = (input_shape,)))
    model.add(tf.keras.layers.Dense(16, input_dim=input_shape,use_bias=True,
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/16))))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    model.add(tf.keras.layers.Dropout(drop_rate))
    model.add(tf.keras.layers.Dense(32, use_bias=True, activation='elu',
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/32))))
    model.add(tf.keras.layers.Dropout(drop_rate))
    model.add(tf.keras.layers.Dense(32, use_bias=True, activation='elu',
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/32))))
    model.add(tf.keras.layers.Dropout(drop_rate))
    model.add(tf.keras.layers.Dense(16, use_bias=True, activation='elu',
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/16))))
    model.add(tf.keras.layers.Dense(8, use_bias=True, activation='elu',
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/8))))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model



def dense_small_net(input_shape, drop_rate = 0.2):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.BatchNormalization(input_shape = (input_shape,)))
    model.add(tf.keras.layers.Dense(16, input_dim=input_shape,use_bias=True,
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/16))))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    model.add(tf.keras.layers.Dropout(drop_rate))
    model.add(tf.keras.layers.Dense(16, use_bias=True, activation='elu',
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/16))))
    model.add(tf.keras.layers.Dropout(drop_rate))
    model.add(tf.keras.layers.Dense(16, use_bias=True, activation='elu',
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/16))))
    model.add(tf.keras.layers.Dense(8, use_bias=True, activation='elu',
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/8))))
    model.add(tf.keras.layers.Dense(4, use_bias=True, activation='elu',
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/4))))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model




def CNN(input_shape):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.BatchNormalization(input_shape = input_shape))
    model.add(tf.keras.layers.Conv2D(32, (4, 4), input_shape=input_shape, activation ='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(16, (4, 4), activation = 'relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding = 'same'))

    model.add(tf.keras.layers.Conv2D(8, (3, 3), activation = 'relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(4, (3, 3), activation = 'relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding = 'same'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation = 'relu'))
    model.add(tf.keras.layers.Dense(16, activation = 'relu'))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))
    return model

def CNN_large(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.BatchNormalization(input_shape = input_shape))
    model.add(tf.keras.layers.Conv2D(64, (4, 4), input_shape=input_shape, activation = 'relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(32, (4, 4), activation = 'relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding = 'same'))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation = 'relu'))
    model.add(tf.keras.layers.Dense(128, activation = 'relu'))
    model.add(tf.keras.layers.Dense(16, activation = 'relu'))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))
    return model


def dense_auto_encoder(input_shape, compressed_size = 2):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.BatchNormalization(input_shape = (input_shape,)))
    model.add(tf.keras.layers.Dense(64, input_dim=input_shape,use_bias=True, activation = 'elu',
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/64))))
    model.add(tf.keras.layers.Dense(64, use_bias=True, activation='elu',
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/64))))
    model.add(tf.keras.layers.Dense(16, use_bias=True, activation='elu',
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/16))))
    model.add(tf.keras.layers.Dense(compressed_size, use_bias=True, activation='elu',
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/16))))
    model.add(tf.keras.layers.Dense(16, use_bias=True, activation='elu',
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/16))))
    model.add(tf.keras.layers.Dense(64, use_bias=True, activation='elu',
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/16))))
    model.add(tf.keras.layers.Dense(64, use_bias=True, activation='elu',
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/16))))
    model.add(tf.keras.layers.Dense(input_shape, use_bias=True, activation='elu',
                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/16))))
    return model


def auto_encoder(input_shape, compressed_size=6):
    npix = input_shape[0]
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same',input_shape=input_shape, activation ='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(tf.keras.layers.Conv2D(4, (3, 3), padding='same', activation='relu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(16, activation = 'relu'))

    #compressed layer
    model.add(tf.keras.layers.Dense(compressed_size, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation = 'relu'))
    mini_size = npix//4
    model.add(tf.keras.layers.Dense((mini_size*mini_size) * 4, activation = 'relu'))

    model.add(tf.keras.layers.Reshape((mini_size,mini_size,4)))
    model.add(tf.keras.layers.Conv2D(4, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.UpSampling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.UpSampling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(1, (3, 3), padding='same'))
    model.add(tf.keras.layers.Reshape((1,npix*npix)))
    
    model.add(tf.keras.layers.Activation('softmax'))
    model.add(tf.keras.layers.Reshape((npix,npix,1)))

    return model

def auto_encoder_large(input_shape, compressed_size=6):
    npix = input_shape[0]
    mini_size = npix//8

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same',input_shape=input_shape, activation ='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(tf.keras.layers.Conv2D(24, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation = 'relu'))
    model.add(tf.keras.layers.Dense(32, activation = 'relu'))

    #compressed layer
    model.add(tf.keras.layers.Dense(compressed_size, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation = 'relu'))
    model.add(tf.keras.layers.Dense(128, activation = 'relu'))
    model.add(tf.keras.layers.Dense((mini_size*mini_size) * 16, activation = 'relu'))

    model.add(tf.keras.layers.Reshape((mini_size,mini_size,16)))
    model.add(tf.keras.layers.UpSampling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.UpSampling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(24, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.UpSampling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(1, (3, 3), padding='same'))
    model.add(tf.keras.layers.Reshape((1,npix*npix)))
    
    model.add(tf.keras.layers.Activation('softmax'))
    model.add(tf.keras.layers.Reshape((npix,npix,1)))

    return model

def auto_encoder_like_VAE(input_shape, z_size = 10):

    inputs = tf.keras.layers.Input(input_shape, name='encoder_input')
    kernel_size = 3
    filter_n = 6
    x = inputs
    for i in range(3):
        x = tf.keras.layers.Conv2D(filters=filter_n, kernel_size=kernel_size, activation='relu' )(x)
        filter_n += 4

    x = tf.keras.layers.AveragePooling2D()(x)
    # x = MaxPooling2D( )( x )

    # shape info needed to build decoder model
    shape_convolved = x.get_shape().as_list()

    # 3 dense layers
    x = tf.keras.layers.Flatten()(x)
    size_convolved = x.get_shape().as_list()
    x = tf.keras.layers.Dense(size_convolved[1] // 17, activation='relu')(x)  # reduce convolution output
    x = tf.keras.layers.Dense(size_convolved[1] // 42, activation='relu')(x)  # reduce again
    #x = Dense(8, activation='relu')(x)

    x = tf.keras.layers.Dense(z_size, name='z_mean')(x)

    # use reparameterization trick to push the sampling out as input
    x = tf.keras.layers.Dense(size_convolved[1] // 42, activation='relu')(x)  # inflate to input-shape/200
    x = tf.keras.layers.Dense(size_convolved[1] // 17, activation='relu')(x)  # double size
    x = tf.keras.layers.Dense(shape_convolved[1] * shape_convolved[2] * shape_convolved[3], activation='relu')(x)
    x = tf.keras.layers.Reshape((shape_convolved[1], shape_convolved[2], shape_convolved[3]))(x)

    x = tf.keras.layers.UpSampling2D()(x)

    for i in range(3):
        filter_n -= 4
        x = tf.keras.layers.Conv2DTranspose(filters=filter_n, kernel_size=kernel_size, activation='relu')(x)

    outputs = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=kernel_size, activation='relu',  padding='same', name='decoder_output')(x)

    ae = tf.keras.Model(inputs, outputs, name = 'AE')
    return ae
