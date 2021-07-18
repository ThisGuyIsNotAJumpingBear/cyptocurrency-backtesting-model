"""The 2nd layer of the analysis part"""
import numpy as np
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from machine_learning import module_weight_variable


def multi_layer_transformation(X: np.array, Y: np.array) -> Sequential:
    """analyze the data"""
    model = Sequential()
    model.add(Dense(3, input_dim=3))
    model.add(Activation('softmax'))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])

    model.fit(X, Y, epochs=20, batch_size=20)

    return model


def rnn_2nd_layer(X: np.array, Y: np.array,
                  epochs: int = 100, batch_size: int = 50) -> Sequential:
    """analyze the data using rnn"""
    # early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)

    model = Sequential()
    model.add(SimpleRNN(17,
                        kernel_initializer=module_weight_variable.weight_variable,
                        input_shape=(25, 3)
                        ))

    model.add(Dense(8, kernel_initializer=module_weight_variable.weight_variable))
    model.add(Activation('softmax'))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    model.fit(X, Y,
              batch_size=batch_size,
              epochs=epochs,
              # callbacks=[early_stopping]
              )

    return model
