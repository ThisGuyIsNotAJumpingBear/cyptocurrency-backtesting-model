"""analyze the input data
find out the linear transformation of from a 1*8 matrix to 1*3 result"""
import numpy as np
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from machine_learning import module_weight_variable


def multi_layer_transformation(X: np.array, Y: np.array) -> Sequential:
    """analyze the data"""
    model = Sequential()
    model.add(Dense(4, input_dim=8))
    model.add(Activation('softmax'))

    model.add(Dense(4))
    model.add(Activation('softmax'))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])

    model.fit(X, Y, epochs=20, batch_size=20)

    return model


def rnn_transformation(X: np.array, Y: np.array, data_shape: int,
                       epochs: int = 100, batch_size: int = 50) -> Sequential:
    """analyze the data using rnn"""
    # early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)

    model = Sequential()
    model.add(SimpleRNN(25,
                        kernel_initializer=module_weight_variable.weight_variable,
                        input_shape=(25, data_shape)
                        ))

    model.add(Dense(10, kernel_initializer=module_weight_variable.weight_variable))
    model.add(Activation('softmax'))

    # model.add(Dense(40))
    # model.add(Activation('softmax'))
    #
    # model.add(Dense(20))
    # model.add(Activation('softmax'))
    #
    # model.add(Dense(10))
    # model.add(Activation('softmax'))

    model.add(Dense(5))
    model.add(Activation('softmax'))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    optimizer = Adam(lr=0.003, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    model.fit(X, Y,
              batch_size=batch_size,
              epochs=epochs,
              # callbacks=[early_stopping]
              )

    return model


def lstm(X: np.array, Y: np.array, data_shape: int,
         epochs: int = 100, batch_size: int = 50) -> Sequential:
    """analyze the data using rnn and lstm"""
    # early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)

    model = Sequential()
    model.add(LSTM(25,
                   kernel_initializer=module_weight_variable.weight_variable,
                   input_shape=(25, data_shape)
                   ))

    model.add(Dense(10, kernel_initializer=module_weight_variable.weight_variable))
    model.add(Activation('softmax'))

    model.add(Dense(5))
    model.add(Activation('softmax'))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    optimizer = Adam(lr=0.003, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    model.fit(X, Y,
              batch_size=batch_size,
              epochs=epochs,
              # callbacks=[early_stopping]
              )

    return model
