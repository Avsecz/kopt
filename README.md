# kopt - Hyper-parameter optimization for Keras

[![Build Status](https://travis-ci.org/avsecz/keras-hyperop.svg?branch=master)](https://travis-ci.org/avsecz/keras-hyperop)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/avsecz/keras-hyperopt/blob/master/LICENSE)

kopt is a hyper-parameter optimization library for Keras. It is based on [hyperopt](https://github.com/hyperopt/hyperopt).

## Getting started

Here is an example of hyper-parameter optimization for the Keras IMDB
example model.

```python
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
import keras.layers as kl
from keras.optimizers import Adam
# kopt and hyoperot imports
from kopt import CompileFN, KMongoTrials, fn_test
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


# 1. define the data function returning training, (validation, test) data
def data(max_features=5000, maxlen=80):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    return (x_train[:100], y_train[:100], max_features), (x_test, y_test)


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

# Specify the optimization metrics
db_name="imdb"
exp_name="myexp1"
objective = CompileFN(db_name, exp_name,
                      data_fn=data,
                      model_fn=model,
                      loss_metric="acc", # which metric to optimize for
                      loss_metric_mode="max",  # try to maximize the metric
                      valid_split=.2, # use 20% of the training data for the validation set
                      save_model='best', # checkpoint the best model
                      save_results=True, # save the results as .json (in addition to mongoDB)
                      save_dir="./saved_models/")  # place to store the models

# define the hyper-parameter ranges
# see https://github.com/hyperopt/hyperopt/wiki/FMin for more info
hyper_params = {
	"data": {
	    "max_features": 100,
		"maxlen": 80,
	},
	"model": {
     	"lr": hp.loguniform("m_lr", np.log(1e-4), np.log(1e-2)), # 0.0001 - 0.01
	    "embedding_dims": hp.choice("m_emb", (64, 128)),
	    "rnn_units": 64,
		"dropout": hp.uniform("m_do", 0, 0.5),
	},
	"fit": {
	    "epochs": 20
	}
}

# test model training, on a small subset for one epoch
test_fn(objective, hyper_params)

# run hyper-parameter optimization sequentially (without any database)
trials = Trials()
best = fmin(objective, hyper_params, trials=trials, algo=tpe.suggest, max_evals=2)

# run hyper-parameter optimization in parallel (saving the results to MonogoDB)
# Follow the hyperopt guide:
# https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB
# KMongoTrials extends hyperopt.MongoTrials with convenience methods
trials = KMongoTrials(db_name, exp_name,
                      ip="localhost",
	                  port=22334)
trials = Trials()
best = fmin(objective, hyper_params, trials=trials, algo=tpe.suggest, max_evals=2)
```

## See also

- [nbs/imdb_example.ipynb](nbs/imdb_example.ipynb)

The documentation of `concise.hyopt` (`kopt` was ported from `concise.hyopt`):

- [Tutorial](https://i12g-gagneurweb.in.tum.de/public/docs/concise/tutorials/hyper-parameter_optimization/)
- [API documentation](https://i12g-gagneurweb.in.tum.de/public/docs/concise/hyopt/)
- [Jupyter notebook](https://github.com/gagneurlab/concise/blob/master/nbs/hyper-parameter_optimization.ipynb)
