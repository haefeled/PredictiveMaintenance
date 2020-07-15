from copy import deepcopy
from DataPreparation import DataPreparation
from DataReader import DataReader
import matplotlib.pyplot as plt
from keras.models import load_model


class Predict:
    def __init__(self):
        self.TIMESTEPS = 10
        self.N_FEATURES = 464

    def predict(self, current_df):  # prep_writer
        """
        Predicts RUL values for a list of a list of timestep-related features.

        :param current_df: DataFrame A DataFrame containing more than one sample.
        :return: list<float> A list of predicted RUL values.
        """

        data_prep = DataPreparation()
        X_predict = data_prep.prepare_data(current_df, False)

        # predict

        model = load_model('Model/lstm_model_adam_swish_0_256.h5')
        model.load_weights('Model/lstm_model_adam_swish_0_256.h5')
        model.compile(loss='mse', optimizer='adam')
        pred = model.predict((X_predict.reshape(1, self.TIMESTEPS, self.N_FEATURES)))

        # Denormalize
        factor_dict = dict()
        with open("Data/analysis_results.txt") as f:
            content = f.readlines()
            for line in content:
                entry = line.strip().split(':')
                factor_dict[deepcopy(entry[0])] = deepcopy(float(entry[1]))
        for i in range(len(pred[0])):
            maxrul_STR = 'maxRUL' + str(i)
            pred[0][i] = pred[0][i] * factor_dict[maxrul_STR]

        # # RUL [RL, RR, FL, FR]
        # current_rul_list = [current_rul0, current_rul1, current_rul2, current_rul3]
        #
        # for i in range(len(current_rul_list)):
        #     if current_rul_list[i] < 0:
        #         current_rul_list[i] = 0.0
        #     print("\nRUL: {} min\n".format(current_rul_list[i]))
        #
        # prep_writer.insert_data({'rul0': current_rul_list[0], 'rul1': current_rul_list[1], 'rul2': current_rul_list[2],
        #                          'rul3': current_rul_list[3]})
        return pred[0]


if __name__ == "__main__":
    # just for testing
    data_pred = Predict()
    data_reader = DataReader()
    data_prep = DataPreparation()
    data = data_reader.load_data_from_sqlite3('Data/EvalData/Ernoe1.sqlite3')
    data = data_prep.sort_dict_into_list(data, False)
    df = data_prep.list_to_dataframe(data)

    maxrul_list = [df.loc[df.query('tyresWear0 < 50').tyresWear0.count()-1, 'sessionTime'] / 60,
                   df.loc[df.query('tyresWear1 < 50').tyresWear1.count()-1, 'sessionTime'] / 60,
                   df.loc[df.query('tyresWear2 < 50').tyresWear2.count()-1, 'sessionTime'] / 60,
                   df.loc[df.query('tyresWear3 < 50').tyresWear3.count()-1, 'sessionTime'] / 60]

    compare0 = []
    compare1 = []
    compare2 = []
    compare3 = []
    tmp_list = []
    output = []
    counter = 0
    for paket in data:
        if counter == 0:
            tmp_list.append(paket)
        elif counter % 30 == 0:
            if counter % 120 == 0:
                compare0.append((maxrul_list[0] - df.loc[counter - 1, 'sessionTime']) / 60)
                compare1.append((maxrul_list[1] - df.loc[counter - 1, 'sessionTime']) / 60)
                compare2.append((maxrul_list[2] - df.loc[counter - 1, 'sessionTime']) / 60)
                compare3.append((maxrul_list[3] - df.loc[counter - 1, 'sessionTime']) / 60)
                tmp_list.append(paket)
                tmp_df = data_prep.list_to_dataframe(tmp_list)
                output.append(data_pred.predict(tmp_df))
            tmp_list.clear()
        else:
            tmp_list.append(paket)
        counter += 1

    output0 = []
    output1 = []
    output2 = []
    output3 = []

    for i in range(len(output)):
        output0.append(output[i][0])
        output1.append(output[i][1])
        output2.append(output[i][2])
        output3.append(output[i][3])

    output_list = [output0, output1, output2, output3]
    compare_list = [compare0, compare1, compare2, compare3]
    for i in range(4):
        plt.figure(figsize=(10, 8), dpi=90)
        plt.title('wheel' + str(i), loc='center')
        plt.plot(compare_list[i], label='Actual RUL')
        plt.plot(output_list[i], label='Pred RUL')
        plt.xlabel('time in packet-send-cycles')
        plt.ylabel('RUL')
        plt.legend()
        plt.show()
