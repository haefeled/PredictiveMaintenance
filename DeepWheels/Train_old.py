import random

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from DataPreparation import DataPreparation
from DataReader import DataReader
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bayes_opt import BayesianOptimization
from tensorboard.plugins.hparams import api as hp
from datetime import datetime


class Train:
    hparam = []
    failure_threshold = 50

    @staticmethod
    def to_3D(X, features, timesteps):
        '''
        Shapes the dataset so it can fit into LSTM's format requirement.

            :param X: DataFrame<timestamp, features> A DataFrame with timestamps as rows and features as columns.
            :param features: list<str> A list of all feature names.
            :param timesteps: int The number of timesteps to use for constructing a sequence of previous values.
            :return: DataFrame<timestamps, features, previous_values>
                    A DataFrame with timestamps as rows and features as columns and a sequence of previous values for each value.
        '''
        # Creating an empty tridimensional array
        X_trans = np.empty((X.shape[0], timesteps, 0))

        # Adjusting the shape of the data
        for feat in features:
            # Regular expressions to filter each feature and
            # drop the NaN values generated from the shift
            df_filtered = X.filter(regex=f'{feat}(_|$)')
            df_filtered = df_filtered.values.reshape(df_filtered.shape[0], timesteps, 1)
            X_trans = np.append(X_trans, df_filtered, axis=2)

        return X_trans

    @staticmethod
    def is_faulty(df, tyreID, failure_threshold):
        """
        Calculates a failure flag for every sample.

        :param df: DataFrame DataFrame object which includes tyreWear data.
        :param tyreID: int ID of the tyre to train on.
        :param failure_threshold: int tyreWear in percent.
        :return: numpy array of failure flags.
        """
        tyreIDstr = 'tyresWear' + str(tyreID)
        return np.where((df[tyreIDstr] < failure_threshold), 0, 1)

    def train(self, optimizer,
              dropout,
              num_layer,
              num_units):
        """
        Trains a model for a given database.

        :param filename: str Name of database file to train on.
        :param use_existing_model: Boolean Loads existing model for training if set to True.
        :param failure_threshold: int A percentage of tyreWear representing a failure.
        """
        # number of last timesteps to use for training
        TIMESTEPS = 10

        # fix random seed for reproducibility
        np.random.seed(7)

        data_reader = DataReader()
        data_prep = DataPreparation()
        data = data_reader.load_data_from_sqlite3(r".\Data\AllData\\" + filename)
        data = data_prep.sort_dict_into_list(data, False)
        df = data_prep.list_to_dataframe(data)

        # train for every tyre
        for i in range(4):
            # add failure flag to samples
            df['is_faulty'] = Train.is_faulty(df, i, self.failure_threshold)

            # checking if dataset contains failure
            if 1 not in pd.Series(list(df['is_faulty'])).unique():
                print('dataset does not contain any failure\naborting...')
                return
            else:
                print('dataset ok')

            # get number of rows which are intact to obtain the RUL of the first element
            failure_row_index = df.query('is_faulty == 0').is_faulty.count()
            MAX_RUL = df.loc[failure_row_index, 'sessionTime']
            print("{} maxRUL: {}".format(filename, MAX_RUL))

            # backpropagate the RUL of every row
            df['RUL'] = MAX_RUL - df['sessionTime']
            target = 'RUL'

            # Removing target and unused columns
            features = df.columns.tolist()
            features.remove('sessionTime')
            features.remove('is_faulty')
            features.remove(target)

            print(df)
            # remove rows after failure
            # df[df['is_faulty'] == 0]

            # remove unused columns
            del df['is_faulty']

            # List of shifted dataframes according to the number of TIMESTEPS
            df_list = [df[features].shift(shift_val) if (shift_val == 0)
                       else df[features].shift(-shift_val).add_suffix(f'_{shift_val}')
                       for shift_val in range(0, TIMESTEPS)]

            # Concatenating list
            df_concat = pd.concat(df_list, axis=1, sort=False)
            df_concat = df_concat.iloc[:-TIMESTEPS, :]

            # Default train_test_split - test_size=0.25
            x_train, x_test, y_train, y_test = train_test_split(df_concat, df[target].iloc[:-TIMESTEPS],
                                                                random_state=10,
                                                                shuffle=True)

            scaler = StandardScaler()

            # Scaling and transforming the array back to the dataframe format
            # Training
            x_train_lstm = pd.DataFrame(data=scaler.fit_transform(x_train), columns=x_train.columns)
            # Test
            x_test_lstm = pd.DataFrame(data=scaler.transform(x_test), columns=x_test.columns)
            time_stamp = datetime.timestamp()

            model = Sequential()
            for _ in range(num_layer):
                model.add(LSTM(input_shape=(TIMESTEPS, len(features)), units=50, return_sequences=True))
                model.add(Dropout(dropout))
            model.add(LSTM(input_shape=(TIMESTEPS, len(features)), units=50, return_sequences=False))
            model.add(Dropout(round(random.uniform(0.1, 0.5), 1)))
            # model.add(LSTM(input_shape=(TIMESTEPS, len(features)), units=80, return_sequences=False))
            # model.add(Dropout(round(random.uniform(0.1, 0.5), 1)))
            model.add(Dense(units=1, activation='relu'))

            model.compile(loss='mean_squared_error', optimizer='adam')
            print(model.summary())

            model_path = r".\Model\lstm_model" + str(i) + ".h5"
            run_dir = f'logs/run{time_stamp}'
            history = model.fit(Train.to_3D(x_train_lstm, features, TIMESTEPS), y_train,
                                epochs=3000, batch_size=32, validation_split=0.3, verbose=2,
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

            plt.figure(figsize=(10, 8), dpi=90)
            plt.plot(history.history['val_loss'], label='val_loss')
            plt.plot(history.history['loss'], label='loss')
            plt.xlabel('Epochs')
            plt.ylabel('Mean Squared Error (MSE)')
            plt.legend()
            plt.show()


            print(x_test_lstm)
            rul_pred = model.predict(Train.to_3D(x_test_lstm, features, TIMESTEPS))
            print(f"R2 Score: {round(r2_score(y_test, rul_pred), 4)}")
            print(rul_pred)

            plt.figure(figsize=(10, 8), dpi=90)
            plt.plot(y_test.iloc[:].values, label='Actual RUL')
            plt.plot(rul_pred[:], label='Pred RUL')
            plt.xlabel('time in packet-send-cycles')
            plt.ylabel('RUL in minutes')
            plt.legend()
            plt.show()

    @staticmethod
    def train_on_all_datasets(path_to_datasets, failure_threshold):
        """
        Initiates training on a series of databases.

        :param path_to_datasets: str Represents the path where all databases can be located.
        :param failure_threshold: int A percentage of tyreWear representing a failure.
        """

        # db_file_names = [f for f in listdir(path_to_datasets) if isfile(join(path_to_datasets, f))]

        db_file_names = []
        with open(r".\Data\analysis_results.txt") as f:
            content = f.readlines()
            for line in content:
                maxTyreWearsStr = line.strip().split(':')[1].strip().split('%')
                del maxTyreWearsStr[-1]
                maxTyreWears = []

                for strValue in maxTyreWearsStr:
                    maxTyreWears.append(int(float(strValue)))

                ok_values_count = 0

                for maxTyreWear in maxTyreWears:
                    if maxTyreWear >= failure_threshold:
                        ok_values_count = ok_values_count + 1

                if ok_values_count == 4:
                    db_file_names.append(line.split(' ')[0])

        for i in range(len(db_file_names)):
            if i > 0:
                Train.train(db_file_names[i], True, failure_threshold)
            else:
                Train.train(db_file_names[i], False, failure_threshold)


if __name__ == "__main__":
    train = Train()
    train.train_on_all_datasets(r".\Data\AllData", 50)
    pbounds = {
        '': (),
        '': (),
    }







