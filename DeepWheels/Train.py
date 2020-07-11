import glob
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from DataPreparation import DataPreparation
from DataReader import DataReader
from bayes_opt import BayesianOptimization
from bayes_opt import BayesianOptimization
from datetime import datetime
from datetime import datetime
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorboard.plugins.hparams import api as hp
from tensorboard.plugins.hparams import api as hp


def optimize_hyperparameters():
    """
    hyper parameter search for model/training optimization
    """
    hyperparams = []
    optimizer = ['adam', 'nadam', 'sgd', 'rmsprop']
    activation = ['tanh', 'prelu']
    pbounds = {
        'optimizer': (0, len(optimizer)),
        'activation': (0, len(activation)),
        'dropout': (.1, .5),
        'num_layer': (0, 2),
        'num_units': (20, 300)
    }
    # hparam = []
    # for key, val in pbounds.items():
    #    hyperparams.append(hp.HParam(key, hp.Discrete(list(val))))
    # pass

    optimizer = BayesianOptimization(
        f=fit_model_with,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=3,
        n_iter=30,
    )

    print(optimizer.max)


def get_model(activation, dropout, num_layer, num_units):
    """
    :param activation: activation-function
    :param dropout: dropout-value
    :param num_layer: number of hidden layers
    :param num_units: number of units per hidden layer
    :return: model
    """
    model = Sequential()
    for _ in range(num_layer):
        model.add(LSTM(input_shape='####', units=num_units, return_sequences=True))
        model.add(Dropout(dropout))
    model.add(LSTM(input_shape='####', units=50, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(units=num_units, activation=activation))

    return model


def fit_model_with(optimizer, activation, dropout, num_layer, num_units):
    """
    Trains a model with all given databases in AllData folder.
    :param optimizer:  optimizer-function
    :param activation: activation-function
    :param dropout: dropout-value
    :param num_layer: number of hidden layers
    :param num_units: number of units per hidden layer
    :return: the accuracy
    """
    model = get_model(activation, dropout, num_layer, num_units)

    #define database list incl. paths
    databases = glob.glob('./Data/AllData/*.sqlite3')

    # define run_dir
    time_stamp = datetime.timestamp()
    run_dir = f'logs/run{time_stamp}'

    for database in databases:
        model.compile(loss='mean_squared_error', optimizer=optimizer)

        # Train the model with the train dataset.
        history = model.fit(Train.to_3D(x_train_lstm, features, TIMESTEPS), y_train,
                            epochs=3000, batch_size=16, validation_split=0.3, verbose=1,
                            callbacks=[
                                keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              min_delta=0,
                                                              patience=3,
                                                              verbose=2,
                                                              mode='min'),

                                keras.callbacks.ModelCheckpoint(model_path,
                                                                monitor='val_loss',
                                                                save_best_only=True,
                                                                mode='min',
                                                                verbose=2),
                                keras.callbacks.TensorBoard(log_dir=run_dir),
                                hp.KerasCallback(run_dir + '/hparam', {
                                    self.hparam[0]: float(int(round())),

                                }),
                           ])

    #score = model.evaluate()...
    #...

    #Return the accuracy.
    return num_layer + num_units / num_units


if __name__ == "__main__":
    optimize_hyperparameters()
    print(glob.glob('./Data/AllData/*.sqlite3'))
