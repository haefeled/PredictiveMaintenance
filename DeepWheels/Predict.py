from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import read
from Train import to_3D

features = [
            'tyresWear0',
            'tyresWear1',
            'tyresWear2',
            'tyresWear3',
            ]
timesteps = 5

def predict(data):
    df = pd.DataFrame(data, columns = features)

    # List of shifted dataframes according to the number of timesteps
    df_list = [df[features].shift(shift_val) if (shift_val == 0) 
                                else df[features].shift(-shift_val).add_suffix(f'_{shift_val}') 
                                for shift_val in range(0,timesteps)]

    # Concatenating list
    df_concat = pd.concat(df_list, axis=1, sort=False)
    df_test = df_concat.iloc[:-timesteps,:]

    scaler = StandardScaler()
    scaler.fit(df_test)

    df_test_lstm = pd.DataFrame(data=scaler.transform(df_test), columns=df_test.columns)
    rul_pred = model.predict(to_3D(df_test_lstm,features, timesteps=timesteps))
    #print(df_test_lstm)
    print(rul_pred)

    plt.figure(figsize = (10,8), dpi=90)
    plt.plot(rul_pred[:],label='Pred RUL')
    plt.xlabel('time in packet-send-cycles')
    plt.ylabel('RUL in minutes')
    plt.legend()

# load model from single file
model_path = r".\Data\AllData\lstm_model.h5"
model = load_model(model_path)
model.load_weights(model_path)
model.compile(loss='mean_squared_error', optimizer='adam')

# apply prediction function to each dataset
read.apply_to_live_data(predict, buffer_time_in_seconds = 3)
plt.show()