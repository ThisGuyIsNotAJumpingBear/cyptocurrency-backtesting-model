"""read the data and feed it"""
import numpy as np
import data_reshape
import data_analyze
import data_analysis_layer2
from keras.models import Sequential
from sklearn.utils import shuffle


def model_accuracy(current_model: Sequential, X_test: np.array, Y_test: np.array) -> float:
    """calculate the accuracy of the given model"""
    actu = [item.argmax() for item in Y_test]
    pred = current_model.predict_classes(X_test, batch_size=1)
    prob = [0, 0]
    for i in range(len(X_test)):
        if actu[i] == pred[i]:
            prob[0] += 1
        else:
            prob[1] += 1

    accuracy = prob[0] / (len(X_test))
    print(prob)
    return accuracy


def v1_model() -> Sequential:
    """return a model using traditional technique"""
    X, Y = data_reshape.read_v2_file()
    #     test_x = X[-5:]
    #     test_y = Y[-5:]
    model = data_analyze.multi_layer_transformation(X[:-5], Y[:-5])
    # pred = model.predict_classes(X, batch_size=1)
    # lm = model.evaluate(X, Y)
    return model


def v2_model() -> Sequential:
    """return a model using rnn"""
    X, Y = data_reshape.read_v1_file()
    model = data_analyze.rnn_transformation(X, Y, 1)
    return model


def v3_model(epoch: int = 100, batch: int = 50) -> Sequential:
    """return a rnn model by using the data from reshape v2"""
    X1, Y1 = data_reshape.read_v2_file()
    X2 = np.array([X1[i:i + 25] for i in range(len(X1) - 25)])
    Y2 = Y1[25:]
    model = data_analyze.rnn_transformation(X2, Y2, 8, epoch, batch)
    model.fit(
        X2[1000:], Y2[1000:],
        batch_size=batch,
        epochs=epoch
    )
    return model


def v3e1_model() -> Sequential:
    """return a two-layer rnn model"""
    X1, Y1 = data_reshape.read_v2_file()
    X2 = np.array([X1[i:i + 25] for i in range(len(X1) - 25)])
    Y2 = Y1[50:]
    prob = v3_model().predict_proba(X2, batch_size=1).reshape(len(X2), 3)
    X3 = np.array([prob[i:i+25] for i in range(len(prob)-25)])
    model = data_analysis_layer2.rnn_2nd_layer(X3, Y2)
    return model


def v4_model(epoch: int = 100, batch: int = 50) -> Sequential:
    """return a model using LSTM to interpret the given data"""
    X1, Y1 = data_reshape.read_v2_file()
    X2 = np.array([X1[i:i + 25] for i in range(len(X1) - 25)])
    Y2 = Y1[25:]
    model = data_analyze.lstm(X2, Y2, 8, epoch, batch)
    return model


if __name__ == '__main__':
    # m1 = v1_model()
    # m2 = v2_model()
    # m3 = v3_model(100, 50)
    m4 = v4_model()
