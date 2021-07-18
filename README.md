# cyptocurrency backtesting model
 This is machine learning model trained to predict the trend of cryptocurrencies, namely Ethereum.
 
Three .csv files are datasets from FRED. They contains every day's value of BTC, ETH, LTC from approximately 2016.

data_reshape.py reshapes the data into an n * 8 training set and an n * 3 comparison set. it calculates the indicators of the sequence of the data such as macd, maX (i.e. ma9, ma12, ma26, etc.) and bollinger bands, and uses them to present the data as this increases the amount of information provided.

data_analyze.py contains the original code of various models, and they are instantiated in all_model.py. 
 - model 1 uses a simple model of multiple layers of neurons. It usually reaches the accuracy of 0.4.
 - model 2 uses rnn model accepts a list of 25 continuous cryptocurrency values as its input. Its accuracy is around 0.4 while it carries smaller size of input.
 - model 3 accepts the standard output of data_reshape.py and uses it to train a rnn model. It reaches the accuracy of 0.6.
 - model 4 uses the same input as model 3 and adds a lstm layer to the rnn. it reaches the accuracy of 0.8-0.9.
 
 strategies.py creates the playground for back testing. The strategies of investment are created here and operated in the operation.py, which is the only file you should change if you want to change the way of testing.