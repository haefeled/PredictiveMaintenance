from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import DataPreparation
from Train import to_3D

class Predict:
    # Beispielinit
    def __init__(self):
        self.model_path = r".\Model\lstm_model.h5"
        self.model = load_model(self.model_path)
        self.model.load_weights(self.model_path)
        self.model.compile(loss='mean_squared_error', optimizer='adam')

        self.current_rul = []

    def predict(self, df):
        """
        Predicts RUL values for a list of a list of timestep-related features.

        :param df: DataFrame A DataFrame containing more than one sample.
        :return: list<float> A list of predicted RUL values.
        """
        # number of last timesteps to use for training
        TIMESTEPS = 5
        
        # Removing target and unused columns
        features = df.columns.tolist()
        features.remove('sessionTime')

        # remove unused columns
        del df['sessionTime']

        # List of shifted dataframes according to the number of TIMESTEPS
        df_list = [df[features].shift(shift_val) if (shift_val == 0) 
                                    else df[features].shift(-shift_val).add_suffix(f'_{shift_val}') 
                                    for shift_val in range(0,TIMESTEPS)]

        # Concatenating list
        df_concat = pd.concat(df_list, axis=1, sort=False)
        df_test = df_concat.iloc[:-TIMESTEPS,:]

        scaler = StandardScaler()
        scaler.fit(df_test)

        df_test_lstm = pd.DataFrame(data=scaler.transform(df_test), columns=df_test.columns)
        current_rul = self.model.predict(to_3D(df_test_lstm,features, TIMESTEPS=TIMESTEPS))
        self.current_rul = current_rul

        return current_rul