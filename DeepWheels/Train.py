import glob
from copy import deepcopy
import random

import keras
from bayes_opt import BayesianOptimization
from keras.layers import Dense, Dropout, LSTM, PReLU
from keras.models import Sequential


from DataPreparation import DataPreparation

OPTIMIZER = ['adam', 'adamax']
ACTIVATION = ['swish', 'linear']  # prelu was unkown
TIMESTEPS = 10
N_FEATURES = 464


def optimize_hyperparameters():
    """
    hyper parameter search for model/training optimization
    """
    hyperparams = []

    pbounds = {
        'optimizer': (0, len(OPTIMIZER) - 1),
        'activation': (0, len(ACTIVATION) - 1),
        'num_layer': (0, 3),
        'num_units': (7, 11)
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
        init_points=30,
        n_iter=3,
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

    # define run_dir
    # time_stamp = datetime.timestamp()
    # run_dir = f'logs/run{time_stamp}'
    data_counter = 1
    for _ in range(1):
        for database in databases:
            print("##########################################################")
            print(str(data_counter) + "/" + str(len(databases)) + " Database")
            print("##########################################################")
            data_counter += 1
            data = data_prep.load_data(database)
            x_train, y_train = data_prep.prepare_data(data)
            model.compile(loss='mse', optimizer=OPTIMIZER[int(round(optimizer))])

            # Train the model with the train dataset.
            history = model.fit(x_train, y_train,
                                epochs=3000, batch_size=4096, validation_split=0.3, verbose=0,
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

            # score = model.evaluate()...
            eval_data = data_prep.load_data("Data/EvalData/a0466252c5ce018b.sqlite3")
            x_eval, y_eval = data_prep.prepare_data(eval_data)
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

            for packet in x_eval:
                if (counter % 100) == 0:
                    output = model.predict((packet.reshape(1, TIMESTEPS, N_FEATURES)))
                    if counter == 100:
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
                    output_diff0.append(output[0][0] - y_eval[counter][0])
                    output_diff1.append(output[0][1] - y_eval[counter][1])
                    output_diff2.append(output[0][2] - y_eval[counter][2])
                    output_diff3.append(output[0][3] - y_eval[counter][3])
                counter += 1

            score.append(100 - (100 / y_eval[0][0] * abs(sum(output_diff0) / len(output_diff0))))
            score.append(100 - (100 / y_eval[0][0] * abs(sum(output_diff1) / len(output_diff1))))
            score.append(100 - (100 / y_eval[0][0] * abs(sum(output_diff2) / len(output_diff2))))
            score.append(100 - (100 / y_eval[0][0] * abs(sum(output_diff3) / len(output_diff3))))

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
