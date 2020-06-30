from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import socket

import DataReader
import DataPreparation
from Train import to_3D, is_faulty

timer_start = datetime.datetime.now()


def predict(data_list):
    """
    Predicts RUL values for a list of a list of timestep-related features.

    :param data: list<packet> List of timestep-related packets.
    :return: list<float> A list of predicted RUL values.
    """
    # number of last timesteps to use for training
    TIMESTEPS = 5

    data = DataPreparation.sort_dict_into_list(data_list, False)
    df = DataPreparation.list_to_dataframe(data)

    # Removing target and unused columns
    features = df.columns.tolist()
    features.remove('sessionTime')

    # remove unused columns
    del df['sessionTime']

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
    rul_pred = model.predict(to_3D(df_test_lstm, features, TIMESTEPS=TIMESTEPS))
    # print(df_test_lstm)
    time_since_start = datetime.datetime.now() - timer_start
    minutes_since_start = time_since_start.seconds / 60
    current_rul = rul_pred[0][0] - minutes_since_start
    if current_rul < 0:
        current_rul = 0
    current_rul_min = int(current_rul)
    current_rul_sec = int((current_rul - int(current_rul)) * 60)
    print("\nRUL: {}min {}s\n".format(current_rul_min, current_rul_sec))

    plt.figure(figsize=(10, 8), dpi=90)
    plt.plot(rul_pred[:], label='Pred RUL')
    plt.xlabel('time in packet-send-cycles')
    plt.ylabel('RUL in minutes')
    plt.legend()

    return current_rul


# load model from single file
model_path = r".\Model\lstm_model.h5"
model = load_model(model_path)
model.load_weights(model_path)
model.compile(loss='mean_squared_error', optimizer='adam')

timer_start = datetime.datetime.now()

# apply prediction function to each dataset
udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
udp_socket.bind(('', 20777))
udp_socket.setblocking(False)
DataReader.apply_to_live_data(udp_socket, predict, buffer_time_in_seconds=2)
plt.show()
