from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import DataPreparation
from Train import Train


class Predict:
    # Beispielinit
    def __init__(self):
        self.model_path = r".\Model\lstm_model.h5"
        self.model = load_model(self.model_path)
        self.model.load_weights(self.model_path)
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        
        self.df = pd.DataFrame()
        self.current_rul = []

    def predict(self, current_df, prep_writer):
        """
        Predicts RUL values for a list of a list of timestep-related features.

        :param current_df: DataFrame A DataFrame containing more than one sample.
        :return: list<float> A list of predicted RUL values.
        """
        # number of last timesteps
        TIMESTEPS = 20

        self.df.append(current_df)

        # Removing target and unused columns
        features = self.df.columns.tolist()
        features.remove('sessionTime')

        # remove unused columns
        del self.df['sessionTime']

        # List of shifted dataframes according to the number of TIMESTEPS
        df_list = [self.df[features].shift(shift_val) if (shift_val == 0)
                   else self.df[features].shift(-shift_val).add_suffix(f'_{shift_val}')
                   for shift_val in range(0, TIMESTEPS)]

        # Concatenating list
        df_concat = pd.concat(df_list, axis=1, sort=False)
        df_test = df_concat.iloc[:-TIMESTEPS, :]

        scaler = StandardScaler()
        scaler.fit(df_test)

        df_test_lstm = pd.DataFrame(data=scaler.transform(df_test), columns=df_test.columns)
        current_rul = self.model.predict(Train.to_3D(df_test_lstm, features, timesteps=TIMESTEPS))
        current_rul = current_rul[0][0] - current_df.iloc[len(current_df.index) - 1]['sessionTime'] / 60
        print('RUL = ' + current_rul)

        #prep_writer.insert_data({'rul' : current_rul})
