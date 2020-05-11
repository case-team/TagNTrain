from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Input, Flatten, Activation, Reshape, UpSampling2D
from keras.models import Model, Sequential


def cwbh_net(input_shape, drop_rate = 0.2):
    model = Sequential()
    model.add(Dense(64, input_dim=input_shape,use_bias=True,
                    bias_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.5)))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(Dropout(drop_rate))
    model.add(Dense(32, use_bias=True, activation='elu',
                    bias_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.2)))
    model.add(Dropout(drop_rate))
    model.add(Dense(16, use_bias=True, activation='elu',
                    bias_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.2)))
    model.add(Dense(4, use_bias=True, activation='elu',
                    bias_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.2)))
    model.add(Dense(1, activation='sigmoid'))
    return model


def dense_net(input_shape, drop_rate = 0.2):
    model = Sequential()
    model.add(Dense(32, input_dim=input_shape,use_bias=True,
                    bias_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.5)))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(Dropout(drop_rate))
    model.add(Dense(32, use_bias=True, activation='elu',
                    bias_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.2)))
    model.add(Dropout(drop_rate))
    model.add(Dense(16, use_bias=True, activation='elu',
                    bias_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.2)))
    model.add(Dense(8, use_bias=True, activation='elu',
                    bias_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.2)))
    model.add(Dense(1, activation='sigmoid'))
    return model



def CNN(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Conv2D(8, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(4, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(16))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def auto_encoder(input_shape, compressed_size=6):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=input_shape, activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(4, (3, 3), padding='same', activation='relu'))

    model.add(Flatten())
    model.add(Dense(16, activation = 'relu'))
    model.add(Activation('relu'))

    #compressed layer
    model.add(Dense(compressed_size, activation='relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(400, activation = 'relu'))

    model.add(Reshape((10,10,4)))
    model.add(Conv2D(4, (3, 3), padding='same', activation='relu'))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(1, (3, 3), padding='same'))
    model.add(Reshape((1,1600)))
    
    model.add(Activation('softmax'))
    model.add(Reshape((40,40,1)))

    return model
