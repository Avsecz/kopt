import keras.layers as kl
from keras.optimizers import Adam
from keras.models import Sequential

# 2. Define the model function returning a compiled Keras model
def model(train_data, lr=0.001,
          embedding_dims=128, rnn_units=64,
          dropout=0.2):
    # extract data dimensions
    max_features = train_data[2]

    model = Sequential()
    model.add(kl.Embedding(max_features, embedding_dims))
    model.add(kl.LSTM(rnn_units, dropout=dropout, recurrent_dropout=dropout))
    model.add(kl.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=lr),
                  metrics=['accuracy'])
    return model