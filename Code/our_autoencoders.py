import numpy as np

from keras import metrics
from keras import callbacks
from keras import backend as K
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Lambda
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model

from sklearn import model_selection
import tensorflow as tf

if len(tf.config.list_physical_devices('GPU')) < 1:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow-gpu'])

class DAE_2l:
    def __init__(self, input_dim, batch_size, latent_dim):
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        input_img = Input(shape=(self.input_dim, ))

        # 'encoded' is the encoded representation of the input
        encoded = Dense(
            int(self.input_dim / 8),
            kernel_initializer='glorot_uniform')(input_img)

        encoded = BatchNormalization()(encoded)
        encoded = Activation('relu')(encoded)

        encoded = Dense(
            int(self.input_dim / 16),
            kernel_initializer='glorot_uniform')(encoded)

        encoded = BatchNormalization()(encoded)
        encoded = Activation('relu')(encoded)

        encoded = Dense(self.latent_dim, activation='linear')(encoded)

        # 'decoded' is the lossy reconstruction of the input
        decoded = Dense(
            int(self.input_dim / 16),
            kernel_initializer='glorot_uniform')(encoded)

        decoded = BatchNormalization()(decoded)
        decoded = Activation('relu')(decoded)

        decoded = Dense(
            int(self.input_dim / 8),
            kernel_initializer='glorot_uniform')(decoded)

        decoded = BatchNormalization()(decoded)
        decoded = Activation('relu')(decoded)

        decoded = Dense(
            self.input_dim,
            activation='sigmoid',
            kernel_initializer='glorot_uniform')(decoded)

        self.autoencoder = Model(inputs=input_img, outputs=decoded)
        self.autoencoder.compile(optimizer='Adam', loss='mse')
        self.encoder = Model(inputs=input_img, outputs=encoded)

    # return a fit deep encoder
    def fit(self, x_train, y_train):
        x_train, x_valid = model_selection.train_test_split(
            x_train,
            test_size=int(
                0.1 * x_train.shape[0] // self.batch_size * self.batch_size),
            train_size=int(
                0.9 * x_train.shape[0] // self.batch_size * self.batch_size),
            stratify=y_train)

        self.autoencoder.fit(
            x_train,
            x_train,
            epochs=999,
            batch_size=self.batch_size,
            validation_data=(x_valid, x_valid),
            verbose=0,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    min_delta=0.0001,
                    patience=10,
                    restore_best_weights=True)
            ])
        return self

    # return prediction for x
    def transform(self, x):
        prediction = self.encoder.predict(x)
        return prediction.reshape((len(prediction),
                                   np.prod(prediction.shape[1:])))
                          

class DAE_1l:
    def __init__(self, input_dim, batch_size, latent_dim):
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        input_img = Input(shape=(self.input_dim, ))

        # 'encoded' is the encoded representation of the input
        encoded = Dense(
            int(self.input_dim / 8),
            kernel_initializer='glorot_uniform')(input_img)

        encoded = BatchNormalization()(encoded)
        encoded = Activation('relu')(encoded)

        encoded = Dense(self.latent_dim, activation='linear')(encoded)

        # 'decoded' is the lossy reconstruction of the input
        decoded = Dense(
            int(self.input_dim / 8),
            kernel_initializer='glorot_uniform')(encoded)

        decoded = BatchNormalization()(decoded)
        decoded = Activation('relu')(decoded)

        decoded = Dense(
            self.input_dim,
            activation='sigmoid',
            kernel_initializer='glorot_uniform')(decoded)

        self.autoencoder = Model(inputs=input_img, outputs=decoded)
        self.autoencoder.compile(optimizer='Adam', loss='mse')
        self.encoder = Model(inputs=input_img, outputs=encoded)

    # return a fit deep encoder
    def fit(self, x_train, y_train):
        x_train, x_valid = model_selection.train_test_split(
            x_train,
            test_size=int(
                0.1 * x_train.shape[0] // self.batch_size * self.batch_size),
            train_size=int(
                0.9 * x_train.shape[0] // self.batch_size * self.batch_size),
            stratify=y_train)

        self.autoencoder.fit(
            x_train,
            x_train,
            epochs=999,
            batch_size=self.batch_size,
            validation_data=(x_valid, x_valid),
            verbose=0,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    min_delta=0.0001,
                    patience=10,
                    restore_best_weights=True)
            ])
        return self

    # return prediction for x
    def transform(self, x):
        prediction = self.encoder.predict(x)
        return prediction.reshape((len(prediction),
                                   np.prod(prediction.shape[1:])))
                                   
 