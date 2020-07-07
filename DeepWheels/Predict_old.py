from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import socket

from DataReader import DataReader
from DataPreparation import DataPreparation
from Train import Train

data_reader = DataReader()
data_prep = DataPreparation()

def predict(data_list):
    """
    Predicts RUL values for a list of a list of timestep-related features.

    :param data: list<packet> List of timestep-related packets.
    :return: list<float> A list of predicted RUL values.
    """
    # number of last timesteps
    TIMESTEPS = 5

    data = data_prep.sort_dict_into_list(data_list, False)
    df = data_prep.list_to_dataframe(data)

    # Removing target and unused columns
    features = df.columns.tolist()
    features.remove('sessionTime')

    # List of shifted dataframes according to the number of TIMESTEPS
    df_list = [df[features].shift(shift_val) if (shift_val == 0)
               else df[features].shift(-shift_val).add_suffix(f'_{shift_val}')
               for shift_val in range(0, TIMESTEPS)]

    # Concatenating list
    df_concat = pd.concat(df_list, axis=1, sort=False)
    df_test = df_concat.iloc[:-TIMESTEPS, :]

    scaler = StandardScaler()
    scaler.fit(df_test)

    df_test_lstm = pd.DataFrame(data=scaler.transform(df_test), columns=df_test.columns)
    df_3D = Train.to_3D(df_test_lstm, features, timesteps=TIMESTEPS)
    rul_pred0 = model0.predict(df_3D)
    rul_pred1 = model1.predict(df_3D)
    rul_pred2 = model2.predict(df_3D)
    rul_pred3 = model3.predict(df_3D)

    session_time_min = df.iloc[len(df.index) - 1]['sessionTime'] / 60
    current_rul0 = rul_pred0[0][0] - session_time_min
    current_rul1 = rul_pred1[0][0] - session_time_min
    current_rul2 = rul_pred2[0][0] - session_time_min
    current_rul3 = rul_pred3[0][0] - session_time_min

    current_rul_list = [current_rul0, current_rul1, current_rul2, current_rul3]

    for current_rul in current_rul_list:
        if current_rul < 0:
            current_rul = 0
        print("\nRUL: {} min\n".format(current_rul))
        #plt.figure(figsize=(10, 8), dpi=90)
        #plt.plot(rul_pred[:], label='Pred RUL')
        #plt.xlabel('time in packet-send-cycles')
        #plt.ylabel('RUL in minutes')
        #plt.legend()
    return current_rul_list


# load models
model_path0 = r".\Model\lstm_model0.h5"
model_path1 = r".\Model\lstm_model1.h5"
model_path2 = r".\Model\lstm_model2.h5"
model_path3 = r".\Model\lstm_model3.h5"
model0 = load_model(model_path0)
model1 = load_model(model_path1)
model2 = load_model(model_path2)
model3 = load_model(model_path3)
model0.load_weights(model_path0)
model1.load_weights(model_path1)
model2.load_weights(model_path2)
model3.load_weights(model_path3)
model0.compile(loss='mean_squared_error', optimizer='adam')
model1.compile(loss='mean_squared_error', optimizer='adam')
model2.compile(loss='mean_squared_error', optimizer='adam')
model3.compile(loss='mean_squared_error', optimizer='adam')

# apply prediction function to each dataset
udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
udp_socket.bind(('', 20777))
udp_socket.setblocking(False)
data_reader.apply_to_live_data(udp_socket, predict, buffer_time_in_seconds=1)
plt.show()
