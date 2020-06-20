from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import read
from Train import to_3D

feature_col_names = [
            'sessionTime',
            'tyresWear0',
            'tyresWear1',
            'tyresWear2',
            'tyresWear3',
            'is_faulty'
            ]
timesteps = 5

timer_start = datetime.datetime.now()

def predict(data):
    df = pd.DataFrame(data, columns = feature_col_names)
    
    # Removing target and unused columns
    features = df.columns.tolist()
    features.remove('sessionTime')
    features.remove('is_faulty')

    # remove unused columns
    del df['sessionTime']
    del df['is_faulty']

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
    time_since_start = datetime.datetime.now() - timer_start
    minutes_since_start = time_since_start.seconds / 60
    current_rul = rul_pred[0][0] - minutes_since_start
    current_rul_min = int(current_rul)
    current_rul_sec = int((current_rul - int(current_rul)) * 60)
    print("\n{}min {}s\n".format(current_rul_min, current_rul_sec))

    plt.figure(figsize = (10,8), dpi=90)
    plt.plot(rul_pred[:],label='Pred RUL')
    plt.xlabel('time in packet-send-cycles')
    plt.ylabel('RUL in minutes')
    plt.legend()

# load model from single file
model_path = r".\Model\lstm_model.h5"
model = load_model(model_path)
model.load_weights(model_path)
model.compile(loss='mean_squared_error', optimizer='adam')

timer_start = datetime.datetime.now()

# apply prediction function to each dataset
read.apply_to_live_data(predict, buffer_time_in_seconds = 1)
plt.show()