import glob
import math
import os
import random
from time import sleep

import keras
import progressbar

import matplotlib.pyplot as plt
import numpy as np
from bayes_opt import BayesianOptimization
from keras.layers import Dense, Dropout, LSTM, PReLU
from keras.models import Sequential
from copy import deepcopy

from DataPreparation import DataPreparation

OPTIMIZER = ['adamax', 'Nadam']
ACTIVATION = ['linear', 'swish']  # prelu was unkown
TIMESTEPS = 10
N_FEATURES = 356


def optimize_hyperparameters():
    """
    hyper parameter search for model/training optimization
    """
    hyperparams = []

    pbounds = {
        'optimizer': (0, len(OPTIMIZER) - 1),
        'activation': (0, len(ACTIVATION) - 1),
        'num_layer': (1, 3),
        'num_units': (6, 9)
    }
    # hparam = []
    # for key, val in pbounds.items():
    #    hyperparams.append(hp.HParam(key, hp.Discrete(list(val))))
    # pass

    optimizer = BayesianOptimization(
        f=fit_model_with,
        pbounds=pbounds,
        random_state=7,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=2,
    )

    print(optimizer.max)


def get_model(activation, num_layer, num_units):
    """
    :param activation: activation-function
    :param dropout: dropout-value
    :param num_layer: number of hidden layers
    :param num_units: number of units per hidden layer
    :return: model
    """

    model = Sequential()
    layer_counter = 0
    for i in range(int(round(num_layer))):
        model.add(LSTM(input_shape=(TIMESTEPS, N_FEATURES), units=(
            int(pow(2, (round(num_units) - i)))
        ), return_sequences=True))
        model.add(Dropout(round(random.uniform(0.1, 0.5), 1)))
        layer_counter += 1
    model.add(LSTM(
        input_shape=(TIMESTEPS, N_FEATURES), units=int(pow(2, (round(num_units) - layer_counter))),
        return_sequences=False
    ))
    model.add(Dropout(round(random.uniform(0.1, 0.5), 1)))
    model.add(Dense(units=4, activation=ACTIVATION[int(round(activation))]))
    # print(model.summary())

    return model


def fit_model_with(optimizer, activation, num_layer, num_units):
    """
    Trains a model with all given databases in AllData folder.
    :param batch_size: pow factor batch-size
    :param optimizer:  optimizer-function
    :param activation: activation-function
    :param dropout: dropout-value
    :param num_layer: number of hidden layers
    :param num_units: number of units per hidden layer
    :return: the accuracy
    """
    model = get_model(activation, num_layer, num_units)
    model_path = "Model/lstm_model_{}_{}_{}_{}.h5".format(
        OPTIMIZER[int(round(optimizer))],
        ACTIVATION[int(round(activation))],
        int(round(num_layer)),
        int(pow(2, round(num_units)))
    )
    data_prep = DataPreparation()
    print(model_path)

    # define database list incl. paths
    databases = glob.glob('./Data/TrainData/*.sqlite3')

    # define shape of input and output data
    x_train = np.array([[[0] * N_FEATURES for i in range(TIMESTEPS)]])
    y_train = np.array([[0] * 4])
    # load all databases
    widgets = [
        '\x1b[33mCollecting Data... \x1b[39m',
        progressbar.Percentage(),
        progressbar.Bar(marker='\x1b[32m#\x1b[39m'),
    ]
    bar = progressbar.ProgressBar(widgets=widgets, min_value=0, max_value=len(databases) * 2).start()
    bar_counter = 0
    first_database = True
    for database in databases:
        # print("##########################################################")
        # print(str(data_counter) + "/" + str(len(databases)) + " Database")
        # print("##########################################################")
        data = data_prep.load_data(database)
        bar_counter += 1
        bar.update(bar_counter)
        x, y = data_prep.prepare_data(data, shuffle=True)
        x_train = np.concatenate((x_train, deepcopy(x)))
        y_train = np.concatenate((y_train, deepcopy(y)))
        bar_counter += 1
        bar.update(bar_counter)
        del x, y, data  # memory cleanup
        # remove empty first row
        if first_database:
            x_train = np.delete(x_train, 0, axis=0)
            y_train = np.delete(y_train, 0, axis=0)
            first_database = False
    bar.finish()
    sleep(0.2)
    model.compile(loss='mean_squared_error', optimizer=OPTIMIZER[int(round(optimizer))])

    # Train the model with the train dataset.
    history = model.fit(x_train, y_train,
                        epochs=3000, batch_size=1024, validation_split=0.3, verbose=1,
                        callbacks=[
                            keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          min_delta=0,
                                                          patience=5,
                                                          verbose=0,
                                                          mode='min'),

                            keras.callbacks.ModelCheckpoint(model_path,
                                                            monitor='val_loss',
                                                            save_best_only=True,
                                                            mode='min',
                                                            verbose=0)
                            # keras.callbacks.TensorBoard(log_dir=run_dir),
                            # hp.KerasCallback(run_dir + '/hparam', {
                            #     self.hparam[0]: float(int(round())),
                            #
                            # }),
                        ])

    eval_data = data_prep.load_data("Data/EvalData/Ernoe2.sqlite3")
    x_eval, y_eval = data_prep.prepare_data(eval_data, shuffle=False)
    output_acc = [[] for i in range(4)]
    score = []
    counter = 0
    factor_dict = dict()
    with open("Data/analysis_results.txt") as f:
        content = f.readlines()
        for line in content:
            entry = line.strip().split(':')
            factor_dict[deepcopy(entry[0])] = deepcopy(float(entry[1]))
    for packet in x_eval:
        if (counter % 30) == 0:
            output = model.predict((packet.reshape(1, TIMESTEPS, N_FEATURES)))
            if counter == 0:
                # for logging
                first_pred = deepcopy(output)
                for wheel in range(len(first_pred[0])):
                    maxrul_STR = 'maxRUL' + str(wheel)
                    first_pred[0][wheel] = first_pred[0][wheel] * factor_dict[maxrul_STR]
            for wheel in range(len(output[0])):
                maxrul_STR = 'maxRUL' + str(wheel)
                diff = (output[0][wheel] - y_eval[counter][wheel]) * factor_dict[maxrul_STR]
                abs_diff = math.sqrt(diff ** 2)
                if 100 >= abs_diff >= 0:
                    output_acc[wheel].append(100 - abs_diff)
                else:
                    output_acc[wheel].append(0)

        counter += 1
    print("start: {}, {}, {}, {} min".format(
        first_pred[0][0],
        first_pred[0][1],
        first_pred[0][2],
        first_pred[0][3],
    ))
    # denormalize for readable output
    # print(output[0])

    for i in range(len(output[0])):
        maxrul_STR = 'maxRUL' + str(i)
        output[0][i] = output[0][i] * factor_dict[maxrul_STR]
    print("end: {}, {}, {}, {} min".format(
        output[0][0],
        output[0][1],
        output[0][2],
        output[0][3],
    ))

    model_name = str(OPTIMIZER[int(round(optimizer))]) + "_" + str(
        ACTIVATION[int(round(activation))]) + "_" + str(
        int(round(num_layer))) + "_" + str(
        int(pow(2, round(num_units))))
    if not os.path.exists("Data/Plots/" + model_name):
        os.mkdir("Data/Plots/" + model_name)
    plt.figure(figsize=(10, 8), dpi=90)
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.savefig("Data/Plots/{}/loss_{}.png".format(model_name, model_name))
    plt.close()

    plt.figure(figsize=(10, 8), dpi=90)
    plt.plot(output_acc[0], label='RL')
    plt.plot(output_acc[1], label='RR')
    plt.plot(output_acc[2], label='FL')
    plt.plot(output_acc[3], label='FR')
    plt.xlabel('seconds')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig("Data/Plots/{}/acc_{}.png".format(model_name, model_name))
    plt.close()

    # get average acc
    for wheel in range(len(output[0])):
        score.append(np.average(np.array(output_acc[wheel])))
    out_score = sum(score) / len(score)

    # logging into file
    with open("Data/train_results_mac.txt", "a") as out_file:
        out_file.write(
            "Score:{} Optimizer:{} Activation:{} Layer:{} Units:{} \n".format(
                out_score,
                OPTIMIZER[int(round(optimizer))],
                ACTIVATION[int(round(activation))],
                round(num_layer),
                int(pow(2, round(num_units)))
            )
        )
    # Return the accuracy.
    return out_score


if __name__ == "__main__":
    optimize_hyperparameters()
