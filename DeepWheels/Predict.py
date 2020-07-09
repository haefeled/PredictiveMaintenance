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
    def __init__(self):
        self.model_path0 = r".\Model\lstm_model0.h5"
        self.model_path1 = r".\Model\lstm_model1.h5"
        self.model_path2 = r".\Model\lstm_model2.h5"
        self.model_path3 = r".\Model\lstm_model3.h5"
        self.model0 = load_model(self.model_path0)
        self.model1 = load_model(self.model_path1)
        self.model2 = load_model(self.model_path2)
        self.model3 = load_model(self.model_path3)
        self.model0.load_weights(self.model_path0)
        self.model1.load_weights(self.model_path1)
        self.model2.load_weights(self.model_path2)
        self.model3.load_weights(self.model_path3)
        self.model0.compile(loss='mean_squared_error', optimizer='adam')
        self.model1.compile(loss='mean_squared_error', optimizer='adam')
        self.model2.compile(loss='mean_squared_error', optimizer='adam')
        self.model3.compile(loss='mean_squared_error', optimizer='adam')

        self.df = pd.DataFrame()
        self.current_rul = []

    def predict(self, current_df, prep_writer):
        """
        Predicts RUL values for a list of a list of timestep-related features.

        :param current_df: DataFrame A DataFrame containing more than one sample.
        :return: list<float> A list of predicted RUL values.
        """
        # number of last timesteps
        TIMESTEPS = 30

        self.df = pd.concat([self.df, current_df])
        self.df.append(current_df)

        # Removing target and unused columns
        features = self.df.columns.tolist()
        features.remove('sessionTime')

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
        df_3D = Train.to_3D(df_test_lstm, features, timesteps=TIMESTEPS)
        rul_pred0 = self.model0.predict(df_3D)
        rul_pred1 = self.model1.predict(df_3D)
        rul_pred2 = self.model2.predict(df_3D)
        rul_pred3 = self.model3.predict(df_3D)

        session_time_min = self.df.iloc[len(self.df.index) - 1]['sessionTime'] / 60
        current_rul0 = (rul_pred0[0][0]/60) - session_time_min
        current_rul1 = (rul_pred1[0][0]/60) - session_time_min
        current_rul2 = (rul_pred2[0][0]/60) - session_time_min
        current_rul3 = (rul_pred3[0][0]/60) - session_time_min

        # RUL [RL, RR, FL, FR]
        current_rul_list = [current_rul0, current_rul1, current_rul2, current_rul3]

        for i in range(len(current_rul_list)):
            if current_rul_list[i] < 0:
                current_rul_list[i] = 0.0
            print("\nRUL: {} min\n".format(current_rul_list[i]))

        prep_writer.insert_data({'rul0': current_rul_list[0], 'rul1': current_rul_list[1], 'rul2': current_rul_list[2],
                                 'rul3': current_rul_list[3]})
