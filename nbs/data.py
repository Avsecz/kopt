from keras.datasets import imdb
from keras.preprocessing import sequence

# 1. define the data function returning training, (validation, test) data
def data(max_features=5000, maxlen=80):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    return (x_train[:100], y_train[:100], max_features), (x_test, y_test)