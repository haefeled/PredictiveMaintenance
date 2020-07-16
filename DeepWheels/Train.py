import glob
import os
import random
import keras
import progressbar

import matplotlib.pyplot as plt
import numpy as np

from bayes_opt import BayesianOptimization
from keras.layers import Dense, Dropout, LSTM, PReLU
from keras.models import Sequential
from copy import deepcopy



from DataPreparation import DataPreparation

OPTIMIZER = ['adamax', 'adam']
ACTIVATION = ['swish', 'linear']  # prelu was unkown
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
        'num_layer': (2, 3),
        'num_units': (6, 7)
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
        init_points=12,
        n_iter=0,
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
            int(pow(2, (round(num_units)-i)))
        ), return_sequences=True))
        model.add(Dropout(round(random.uniform(0.1, 0.5), 1)))
        layer_counter += 1
    model.add(LSTM(
        input_shape=(TIMESTEPS, N_FEATURES), units=int(pow(2, (round(num_units)-layer_counter))), return_sequences=False
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
    model_path = "Model/lstm_model_" + str(OPTIMIZER[int(round(optimizer))]) + "_" + str(
        ACTIVATION[int(round(activation))]) + "_" + str(
        int(round(num_layer))) + "_" + str(int(pow(2, round(num_units)))) + ".h5"
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
    bar = progressbar.ProgressBar(widgets=widgets, min_value=0, max_value=len(databases)*2).start()
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
        x_train = np.concatenate((x_train, x))
        y_train = np.concatenate((y_train, y))
        bar_counter += 1
        bar.update(bar_counter)
        # remove empty first row
        if first_database:
            x_train = np.delete(x_train, 0, axis=0)
            y_train = np.delete(y_train, 0, axis=0)
            first_database = False
    print()
    model.compile(loss='mse', optimizer=OPTIMIZER[int(round(optimizer))])

    # Train the model with the train dataset.
    history = model.fit(x_train, y_train,
                        epochs=3000, batch_size=4096, validation_split=0.3, verbose=1,
                        callbacks=[
                            keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          min_delta=0,
                                                          patience=5,
                                                          verbose=1,
                                                          mode='min'),

                            keras.callbacks.ModelCheckpoint(model_path,
                                                            monitor='val_loss',
                                                            save_best_only=True,
                                                            mode='min',
                                                            verbose=1)
                            # keras.callbacks.TensorBoard(log_dir=run_dir),
                            # hp.KerasCallback(run_dir + '/hparam', {
                            #     self.hparam[0]: float(int(round())),
                            #
                            # }),
                        ])

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
    plt.savefig("Data/Plots/" + model_name + "/loss_" + model_name + ".png")
    plt.close()

    eval_data = data_prep.load_data("Data/EvalData/Ernoe2.sqlite3")
    x_eval, y_eval = data_prep.prepare_data(eval_data, shuffle=False)
    output_diff0 = []
    output_diff1 = []
    output_diff2 = []
    output_diff3 = []
    score = []
    counter = 0
    factor_dict = dict()
    with open("Data/analysis_results.txt") as f:
        content = f.readlines()
        for line in content:
            entry = line.strip().split(':')
            factor_dict[deepcopy(entry[0])] = deepcopy(float(entry[1]))

    widgets2 = [
        '\x1b[33Evaluating Data... \x1b[39m',
        progressbar.Percentage(),
        progressbar.Bar(marker='\x1b[32m#\x1b[39m'),
    ]
    bar2 = progressbar.ProgressBar(widgets=widgets2, min_value=0, max_value=len(x_eval)/30).start()
    bar_counter = 0
    for packet in x_eval:
        if (counter % 30) == 0:
            output = model.predict((packet.reshape(1, TIMESTEPS, N_FEATURES)))
            bar_counter += 1
            bar2.update(bar_counter)
            if counter == 0:
                # for logging
                first_pred = deepcopy(output)
                for i in range(len(first_pred[0])):
                    maxrul_STR = 'maxRUL' + str(i)
                    first_pred[0][i] = first_pred[0][i] * factor_dict[maxrul_STR]
                print("start: {}, {}, {}, {} min".format(
                    first_pred[0][0],
                    first_pred[0][1],
                    first_pred[0][2],
                    first_pred[0][3],
                ))
            output_diff0.append(abs(output[0][0] - y_eval[counter][0]))
            output_diff1.append(abs(output[0][1] - y_eval[counter][1]))
            output_diff2.append(abs(output[0][2] - y_eval[counter][2]))
            output_diff3.append(abs(output[0][3] - y_eval[counter][3]))
        counter += 1
    print()
    score.append(100 - abs(100 / y_eval[0][0] * (sum(output_diff0) / len(output_diff0))))
    score.append(100 - abs(100 / y_eval[0][0] * (sum(output_diff1) / len(output_diff1))))
    score.append(100 - abs(100 / y_eval[0][0] * (sum(output_diff2) / len(output_diff2))))
    score.append(100 - abs(100 / y_eval[0][0] * (sum(output_diff3) / len(output_diff3))))


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
    for i in range(len(output_diff0)):
        output_diff0[i] = round(output_diff0[i] * factor_dict['maxRUL0'], 2)
        output_diff1[i] = round(output_diff1[i] * factor_dict['maxRUL1'], 2)
        output_diff2[i] = round(output_diff2[i] * factor_dict['maxRUL2'], 2)
        output_diff3[i] = round(output_diff3[i] * factor_dict['maxRUL3'], 2)

    plt.figure(figsize=(10, 8), dpi=90)
    plt.plot(output_diff0, label='RL Differenz')
    plt.plot(output_diff1, label='RR Differenz')
    plt.plot(output_diff2, label='FL Differenz')
    plt.plot(output_diff3, label='FR Differenz')
    plt.xlabel('seconds')
    plt.ylabel('diff in minutes')
    plt.legend()
    plt.savefig("Data/Plots/" + model_name + "/diff_" + model_name + ".png")
    plt.close()



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
