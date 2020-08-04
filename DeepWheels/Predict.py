import progressbar
from copy import deepcopy
from keras.models import load_model
from DataPreparation import DataPreparation
from DataReader import DataReader
import matplotlib.pyplot as plt




class Predict:
    def __init__(self):
        self.TIMESTEPS = 10
        self.N_FEATURES = 356

    def predict(self, current_df):  # prep_writer
        """
        Predicts RUL values for a list of a list of timestep-related features.

        :param current_df: DataFrame A DataFrame containing more than one sample.
        :return: list<float> A list of predicted RUL values.
        """

        data_prep = DataPreparation()
        X_predict = data_prep.prepare_data(current_df, False)

        # predict

        model = load_model('Model/save/lstm_model_adamax_swish_3_128.h5')
        model.load_weights('Model/lstm_model_adamax_swish_3_128.h5')
        model.compile(loss='mean_squared_error', optimizer='adamax')
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
        return pred[0]


if __name__ == "__main__":
    # just for testing
    data_pred = Predict()
    data_reader = DataReader()
    data_prep = DataPreparation()
    data = data_reader.load_data_from_sqlite3('Data/EvalData/Ernoe2.sqlite3')
    data = data_prep.sort_dict_into_list(data, active_progressbar=False)
    df = data_prep.list_to_dataframe(data)

    df['sessionTime'] = df['sessionTime'] / 60

    maxrul_list = [df.loc[df.query('tyresWear0 < 50').tyresWear0.count()-1, 'sessionTime'],
                   df.loc[df.query('tyresWear1 < 50').tyresWear1.count()-1, 'sessionTime'],
                   df.loc[df.query('tyresWear2 < 50').tyresWear2.count()-1, 'sessionTime'],
                   df.loc[df.query('tyresWear3 < 50').tyresWear3.count()-1, 'sessionTime']]

    compare0 = []
    compare1 = []
    compare2 = []
    compare3 = []
    tmp_list = []
    output = []
    counter = 0
    widgets = [
        '\x1b[33mPrediction in progress... \x1b[39m',
        progressbar.Percentage(),
        progressbar.Bar(marker='\x1b[32m#\x1b[39m'),
    ]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=len(data)).start()
    for paket in data:
        if counter == 0:
            tmp_list.append(paket)
        elif counter % 30 == 0:
            if counter % 120 == 0:
                compare0.append((maxrul_list[0] - df.loc[counter - 1, 'sessionTime']))
                compare1.append((maxrul_list[1] - df.loc[counter - 1, 'sessionTime']))
                compare2.append((maxrul_list[2] - df.loc[counter - 1, 'sessionTime']))
                compare3.append((maxrul_list[3] - df.loc[counter - 1, 'sessionTime']))
                tmp_list.append(paket)
                tmp_df = data_prep.list_to_dataframe(tmp_list)
                output.append(data_pred.predict(tmp_df))
            tmp_list.clear()
        else:
            tmp_list.append(paket)
        counter += 1
        bar.update(counter)
    bar.finish()
    output0 = []
    output1 = []
    output2 = []
    output3 = []

    for i in range(len(output)):
        output0.append(output[i][0])
        output1.append(output[i][1])
        output2.append(output[i][2])
        output3.append(output[i][3])
        if output[i][0] < 0:
            output[i][0] = 0
        if output[i][1] < 0:
            output[i][1] = 0
        if output[i][2] < 0:
            output[i][2] = 0
        if output[i][3] < 0:
            output[i][3] = 0


    output_list = [output0, output1, output2, output3]
    compare_list = [compare0, compare1, compare2, compare3]
    for i in range(4):
        plt.figure(figsize=(10, 8), dpi=90)
        plt.title('wheel' + str(i), loc='center')
        plt.plot(compare_list[i], label='Actual RUL')
        plt.plot(output_list[i], label='Pred RUL')
        plt.xlabel('time in seconds')
        plt.ylabel('RUL')
        plt.legend()
        plt.savefig("Data/Plots/acctest_wheel" + str(i) + ".png")
        plt.close()
