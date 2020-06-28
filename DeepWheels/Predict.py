from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Predict:
    # Beispielinit
    def __init__(self):
        self.model_path = r".\Model\lstm_model.h5"
        self.model = load_model(self.model_path)
        self.model.load_weights(self.model_path)
        self.model.compile(loss='mean_squared_error', optimizer='adam')

        self.current_rul = []

    def predict(self, data_frame):
        """
        Predicts RUL values for a list of a list of timestep-related features.

        :param data_frame:
        :return: list<float> A list of predicted RUL values.
        """
        ### Hier predicten!

        # return current_rul
