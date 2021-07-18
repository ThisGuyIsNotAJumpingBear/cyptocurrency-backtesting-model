"""a file that tests the outcome of the project on real market"""
import all_model
import numpy as np
import data_reshape
import strategies


eth_value = data_reshape.read_data()
model = all_model.v4_model(epoch=200, batch=25)
# model = all_model.v4_model()

X1, Y1 = data_reshape.read_v2_file()
X2 = np.array([X1[i:i + 25] for i in range(len(X1) - 25)])
Y2 = Y1[25:]
Y = [np.argmax(i) for i in Y2]

prob = model.predict_proba(X2, batch_size=1)
pred = model.predict_classes(X2, batch_size=1)

game = strategies.Game(10000.0, 0.0)
eth_value_edited = np.array(eth_value[52:-6])

acc = [0, 0]
for i in range(len(Y)):
    if pred[i] != 1:
        if Y[i] == pred[i]:
            acc[0] += 1
        else:
            acc[1] += 1

for i in range(len(Y)):
    act = pred[i]
    curr_price = eth_value_edited[i]
    game.next_day(curr_price)
    if act == 0:
        game.sell()
    if act == 2:
        game.buy()

