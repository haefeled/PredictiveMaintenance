import glob
from copy import deepcopy
from datetime import datetime

import keras
from bayes_opt import BayesianOptimization
from keras.layers import Dense, Dropout, LSTM, PReLU
from keras.models import Sequential
from tensorboard.plugins.hparams import api as hp

from DataPreparation import DataPreparation

OPTIMIZER = ['adam', 'rmsprop', 'sgd', 'nadam']
ACTIVATION = ['relu', 'tanh', 'linear', 'elu', 'prelu'] # prelu was unkown
TIMESTEPS = 30
N_FEATURES = 520


def optimize_hyperparameters():
    """
    hyper parameter search for model/training optimization
    """
    hyperparams = []

    pbounds = {
        'optimizer': (0, len(OPTIMIZER) - 1),
        'activation': (0, len(ACTIVATION) - 1),
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
        init_points=6,
        n_iter=3,
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
    for _ in range(int(round(num_layer))):
        model.add(LSTM(input_shape=(TIMESTEPS, N_FEATURES), units=int(round(num_units)), return_sequences=True))
        model.add(Dropout(round(dropout, 1)))
    model.add(LSTM(input_shape=(TIMESTEPS, N_FEATURES), units=int(round(num_units)), return_sequences=False))
    model.add(Dropout(round(dropout, 1)))
    model.add(Dense(units=4, activation=ACTIVATION[int(round(activation))]))

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
    model_path = r".\Model\lstm_model_" + str(OPTIMIZER[int(round(optimizer))]) + "_" + str(
        ACTIVATION[int(round(activation))]) + "_" + str(round(dropout, 1)) + "_" + str(
        int(round(num_layer))) + "_" + str(int(round(num_units))
                                           ) + ".h5"
    data_prep = DataPreparation()

    # define database list incl. paths
    databases = glob.glob('./Data/AllData/*.sqlite3')

    # define run_dir
    # time_stamp = datetime.timestamp()
    # run_dir = f'logs/run{time_stamp}'

    for database in databases:
        data = data_prep.load_data(database)
        x_train, y_train = data_prep.prepare_data(data)
        model.compile(loss='mse', optimizer=OPTIMIZER[int(round(optimizer))])

        # Train the model with the train dataset.
        history = model.fit(x_train, y_train,
                            epochs=3000, batch_size=16, validation_split=0.3, verbose=2,
                            callbacks=[
                                keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              min_delta=0,
                                                              patience=10,
                                                              verbose=2,
                                                              mode='min'),

                                keras.callbacks.ModelCheckpoint(model_path,
                                                                monitor='val_loss',
                                                                save_best_only=True,
                                                                mode='min',
                                                                verbose=2)
                                # keras.callbacks.TensorBoard(log_dir=run_dir),
                                # hp.KerasCallback(run_dir + '/hparam', {
                                #     self.hparam[0]: float(int(round())),
                                #
                                # }),
                            ])

    # score = model.evaluate()...
    eval_data = data_prep.load_data(r".\Data\EvalData\ABCDFUCKU.sqlite3")
    x_eval, y_eval = data_prep.prepare_data(eval_data)
    output_diff0 = []
    output_diff1 = []
    output_diff2 = []
    output_diff3 = []
    score = []
    counter = 0
    for packet in x_eval:
        output = model.predict((packet.reshape(1, TIMESTEPS, N_FEATURES)))
        output_diff0.append(output[0][0] - y_eval[counter][0])
        output_diff1.append(output[0][1] - y_eval[counter][1])
        output_diff2.append(output[0][2] - y_eval[counter][2])
        output_diff3.append(output[0][3] - y_eval[counter][3])

    score.append(100-(100 / y_eval[0][0] * abs(sum(output_diff0) / len(output_diff0))))
    score.append(100-(100 / y_eval[0][0] * abs(sum(output_diff1) / len(output_diff1))))
    score.append(100-(100 / y_eval[0][0] * abs(sum(output_diff2) / len(output_diff2))))
    score.append(100-(100 / y_eval[0][0] * abs(sum(output_diff3) / len(output_diff3))))

    out_score = sum(score) / len(score)
    print(out_score)

    # Return the accuracy.
    return out_score


if __name__ == "__main__":
    optimize_hyperparameters()
