from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
import keras
import numpy as np

import read

def to_3D(X, features, timesteps=5):
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

def main():
    # fix random seed for reproducibility
    np.random.seed(7)

    features_col_names = [
                            'sessionTime',
                            'tyresWear0',
                            'tyresWear1',
                            'tyresWear2',
                            'tyresWear3',
                            'is_faulty']

    df = pd.DataFrame(read.database_as_list(r".\Data\AllData\example.sqlite3"), columns = features_col_names)

    # Removing target and unused columns
    features = df.columns.tolist()
    features.remove('sessionTime') # Unused column
    features.remove('is_faulty') # Unused column

    print(features)

    # get number of rows which are intact to obtain the RUL of the first element
    failure_row_index = df.query('is_faulty == 0').is_faulty.count()
    MAX_RUL = df.loc[failure_row_index, 'sessionTime']
    # backpropagate the RUL of every row
    df['RUL'] = MAX_RUL - df['sessionTime']
    target = 'RUL'

    print(df)
    #remove rows after failure
    df[df['is_faulty'] == 0]

    # remove unused columns
    del df['sessionTime']
    del df['is_faulty']

    timesteps = 5

    # List of shifted dataframes according to the number of timesteps
    df_list = [df[features].shift(shift_val) if (shift_val == 0) 
                                    else df[features].shift(-shift_val).add_suffix(f'_{shift_val}') 
                                    for shift_val in range(0,timesteps)]

    # Concatenating list
    df_concat = pd.concat(df_list, axis=1, sort=False)
    df_concat = df_concat.iloc[:-timesteps,:]

    # Default train_test_split - test_size=0.25
    x_train, x_test, y_train, y_test = train_test_split(df_concat, df[target].iloc[:-timesteps], random_state=10, shuffle=True)

    scaler = StandardScaler()

    # Scaling and transforming the array back to the dataframe format
    # Training
    x_train_lstm = pd.DataFrame(data=scaler.fit_transform(x_train), columns=x_train.columns)
    # Test
    x_test_lstm = pd.DataFrame(data=scaler.transform(x_test), columns=x_test.columns)

    model = Sequential()
    model.add(LSTM(input_shape=(timesteps, len(features)), units=15, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(input_shape=(timesteps,len(features)), units=10, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation = 'relu'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())

    model_path = r".\Model\lstm_model.h5"

    history = model.fit(to_3D(x_train_lstm, features), y_train, 
                        epochs=1000, batch_size= 8, validation_split=0.2, verbose=1, 
                        callbacks = [
                                keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            min_delta=0, 
                                                            patience=200,
                                                            verbose=1,
                                                            mode='min'),

                                keras.callbacks.ModelCheckpoint(model_path, 
                                                                monitor='val_loss',
                                                                save_best_only=True, 
                                                                mode='min',
                                                                verbose=1)])

    plt.figure(figsize=(10,8), dpi=90)
    plt.plot(history.history['val_loss'],label='val_loss')
    plt.plot(history.history['loss'],label='loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()

    model.load_weights(model_path)
    model.compile(loss='mean_squared_error', optimizer='adam')

    print(x_test_lstm)
    rul_pred = model.predict(to_3D(x_test_lstm,features))
    print(f"R2 Score: {round(r2_score(y_test, rul_pred),4)}")
    print(rul_pred)

    plt.figure(figsize = (10,8), dpi=90)
    plt.plot(y_test.iloc[:].values,label = 'Actual RUL')
    plt.plot(rul_pred[:],label='Pred RUL')
    plt.xlabel('time in packet-send-cycles')
    plt.ylabel('RUL in minutes')
    plt.legend()
    plt.show()

if __name__ == "__main__":
   # code is only run when module is not called via 'import'
   main()