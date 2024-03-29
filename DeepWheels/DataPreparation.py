import ctypes
from copy import deepcopy

import numpy as np
import pandas
import progressbar
from DataReader import DataReader
from f1_2019_telemetry.packets import PackedLittleEndianStructure, PacketHeader


class DataPreparation:
    def __init__(self):
        self.TIMESTEPS = 10
        self.n_features = 0

    def filter_entries(self, entries, data):
        """
        This function is the third step to transform all received packets of f1_2019_telemetry.
        The function is responsible to handle the _fields of PackedLittleEndianStructure classes.
        In case of a PackedLittleEndianStructure as _field a callback to filter_object() is done.
        In all other cases the features are gathered and returned as dictionary.

        :param entries: _fields of the handled PackedLittleEndianStructure class
        :param data: PackedLittleEndianStructure Object
        :return: dict<string, (int, float)
        """
        tmp_dict_second = dict()
        for fname in entries:
            if not fname.startswith('_'):
                value = getattr(data, fname)
                if isinstance(value, (int, float, bytes)):
                    tmp_dict_second[fname] = value
                elif isinstance(value, PackedLittleEndianStructure):
                    tmp = dict(self.filter_objects(value))
                    for key in tmp.keys():
                        tmp_dict_second[key] = deepcopy(tmp[key])
                elif isinstance(value, ctypes.Array) and len(value) != 20:
                    wheel = 0
                    for e in value:
                        str_key = fname + '%i' % wheel
                        tmp_dict_second[str_key] = deepcopy(e)
                        wheel += 1
                elif isinstance(value, ctypes.Array) and len(value) == 20:
                    tmp = dict(self.filter_objects(value[data.header.playerCarIndex]))
                    for key in tmp.keys():
                        tmp_dict_second[key] = deepcopy(tmp[key])
        return tmp_dict_second

    def filter_objects(self, data):
        """
        This function is the second step to transform all received packets of f1_2019_telemetry.
        The function is responsible to view at PackedLittleEndianStructure classes. Furthermore
        the _fields of the class are collected from the function filter_entries() and returned
        as dictionary.

        :param data: PackedLittleEndianStructure variable
        :return:    dict<string, (int, float)>
        """
        tmp_dict = dict()
        if isinstance(data, PackedLittleEndianStructure) and not isinstance(data, PacketHeader):
            attr = dir(data)
            if attr.__contains__('header'):
                ident = data.header.packetId
                if ident == 0 or ident == 2 or ident == 6 or ident == 7:
                    tmp_dict = self.filter_entries(attr, data)
            else:
                tmp_dict = self.filter_entries(attr, data)
        elif isinstance(data, PackedLittleEndianStructure) and isinstance(data, PacketHeader):
            tmp_dict['sessionTime'] = getattr(data, 'sessionTime')
        return tmp_dict

    def sort_dict_into_list(self, data, active_progressbar):
        """
        This function is the first step to transform all received packets of f1_2019_telemetry.
        This function has the task to split the list and put the results of the transform into a list.
        At the end there is a validity test.
        In case of training, a progressbar show the state of progress.

        :param train_flag:  Boolean variable to indicate a training session
        :param data:    list<PackedLittleEndianStructure>, a list with all received packets
        :return:    list<dict>  A list containing the dictionaries including the filtered features
        """
        result = []
        if active_progressbar:
            widgets = [
                '\x1b[33mFilter Data... \x1b[39m',
                progressbar.Percentage(),
                progressbar.Bar(marker='\x1b[32m#\x1b[39m'),
            ]
            bar = progressbar.ProgressBar(widgets=widgets, max_value=len(data)).start()
            bar_counter = 0
        for paket in data:
            tmp = dict(self.filter_objects(paket))
            if tmp:
                if result and result[-1]['sessionTime'] == (tmp['sessionTime']):
                    for key in tmp.keys():
                        result[-1][key] = deepcopy(tmp[key])
                else:
                    result.append(deepcopy(tmp))
            if active_progressbar:
                bar_counter += 1
                bar.update(bar_counter)
        if active_progressbar:
            bar.finish()
        result = self.check_dict_in_list(result)
        return result

    def check_dict_in_list(self, data):
        """
        This function is a validity test. The Input is the list with all filtered features.
        The function tests the length of dictionaries inside a list and removes a dictionary
        if it hasn't 130 entries.

        :param data: list<dict>
        :return: list<dict>
        """
        list_of_removable = []
        for tmp_dict in data:
            if len(tmp_dict) != 130 and isinstance(data, list):
                list_of_removable.append(data.index(tmp_dict))
        for removable_index in list_of_removable:
            data.remove(data[removable_index])
        return data

    def list_to_dataframe(self, list_data):
        """
        Converts a list of maps to a pandas DataFrame.

        :param list_data: list<dict<number>> A list of maps which contain non-collection data.
        :return: pandas.DataFrame A pandas DataFrame.
        """
        outer_list = []
        for row in list_data:
            outer_list.append(list(row[key] for key in row.keys()))
        return pandas.DataFrame(outer_list, columns=list(list_data[0].keys()))

    def load_data(self, filepath):
        """
        load the data with proper order for training

        :param filepath: String     path to the file
        :return: pandas.DataFrame
        """
        data_reader = DataReader()
        data_prep = DataPreparation()
        data = data_reader.load_data_from_sqlite3(filepath)
        data = data_prep.sort_dict_into_list(data, False)
        return data_prep.list_to_dataframe(data)

    def prepare_data(self, df, trainflag=True, shuffle=False):
        """
        Prepare the Data to a proper format for the LSTM

        :param shuffle: Boolean to enable data shuffling
        :param trainflag: Boolean to distinguish between training mode and prediction mode
        :param df: pandas.DataFrame Input Data as DataFrame
        :return: X Array with an Array of 30 Pakets inside, y the RUL of the 30 Pakets
        """
        # convert to minutes
        df['sessionTime'] = df['sessionTime'] / 60

        # define MAX_RUL for each tyre
        if trainflag:
            maxrul_list = [df.loc[df.query('tyresWear0 < 50').tyresWear0.count() - 1, 'sessionTime'],
                           df.loc[df.query('tyresWear1 < 50').tyresWear1.count() - 1, 'sessionTime'],
                           df.loc[df.query('tyresWear2 < 50').tyresWear2.count() - 1, 'sessionTime'],
                           df.loc[df.query('tyresWear3 < 50').tyresWear3.count() - 1, 'sessionTime']]
        else:
            maxrul_list = [0, 0, 0, 0]

        # normalize data
        norm_df, factor_dict = self.normalize_data(df)

        # define output
        output_seq = []
        for i in range(len(maxrul_list)):
            tmp_list = []
            maxrul_STR = 'maxRUL' + str(i)
            for j in range(df.shape[0]):
                tmp_list.append((maxrul_list[i] - df.loc[j, 'sessionTime']) / (factor_dict[maxrul_STR]))
            output_seq.append(deepcopy(tmp_list))
            del tmp_list
        output_seq = np.array(output_seq)

        # filter unnecessary data
        filtered_norm_df = self.filter_norm_data(norm_df)
        del norm_df  # memory cleanup

        # define input array for each tyre
        input_seq0 = np.array(filtered_norm_df)
        input_seq1 = np.array(filtered_norm_df)
        input_seq2 = np.array(filtered_norm_df)
        input_seq3 = np.array(filtered_norm_df)

        # transpose output
        tmp_array = [[0 for i in range(len(output_seq))] for j in range(len(output_seq[0]))]
        for i in range(len((output_seq))):
            for j in range(len(output_seq[0])):
                tmp_array[j][i] = output_seq[i][j]
        output_seq = np.array(deepcopy(tmp_array))
        del tmp_array  # memory cleanup

        # horizontally stack columns
        dataset = np.hstack((input_seq0, input_seq1, input_seq2, input_seq3, output_seq))
        self.n_features = dataset.shape[1] - 4
        if trainflag:
            # convert into input/output
            X, y = self.split_sequences(dataset, self.TIMESTEPS, shuffle=shuffle)

            return X, y
        else:
            X = dataset[df.shape[0] - self.TIMESTEPS:, :self.n_features]
            return np.array(X)

    def split_sequences(self, sequences, n_steps, shuffle=False):
        """
        This function divides the data packets according to the number of timesteps and shuffles them if desired.

        :param shuffle: Boolean to enable data shuffling
        :param sequences: Dataset for the training including RUL for each Tyre for each paket
        :param n_steps: Integer number of Timesteps
        :return: X Array with an Array of 30 Pakets inside, y the RUL of the 30 Pakets
        """
        X, y = list(), list()
        counter = np.array([i for i in range(int(len(sequences)))])
        if shuffle:
            np.random.shuffle(counter)
        for i in range(int(len(sequences))):
            end_ix = counter[i] + n_steps
            if end_ix < len(sequences):
                # gather input and output parts of the pattern
                seq_x, seq_y = sequences[counter[i]:end_ix, :self.n_features], sequences[end_ix, self.n_features:]
                X.append(seq_x)
                y.append(seq_y)
        return np.array(X), np.array(y)

    def normalize_data(self, df):
        """
        This function normalize each value with its proper global maximum.

        :param df: data as dataframe
        :return: normalized data
        """
        # MinMax normalization (from -1 to 1) and convert all to float
        factor_dict = dict()
        with open(r".\Data\analysis_results.txt") as f:
            content = f.readlines()
            for line in content:
                entry = line.strip().split(':')
                factor_dict[deepcopy(entry[0])] = deepcopy(float(entry[1]))
        colums = df.columns.tolist()
        norm_df = deepcopy(df)
        for colum in colums:
            if factor_dict[colum] == 0:
                norm_df[colum] = 0
            else:
                norm_df[colum] = df[colum] / factor_dict[colum]
        return norm_df, factor_dict

    def filter_norm_data(self, norm_df):
        """
        This function removes unnecessary data which is not required for our use case.

        :param norm_df: data as dataframe
        :return: filtered dataframe
        """
        # filter some data
        del norm_df['sessionTime']
        # filter CarMotionData
        del norm_df['worldVelocityX']
        del norm_df['worldVelocityY']
        del norm_df['worldVelocityZ']
        del norm_df['worldForwardDirX']
        del norm_df['worldForwardDirY']
        del norm_df['worldForwardDirZ']
        del norm_df['worldRightDirX']
        del norm_df['worldRightDirY']
        del norm_df['worldRightDirZ']
        # filter LapData
        del norm_df['lastLapTime']
        del norm_df['currentLapTime']
        del norm_df['bestLapTime']
        del norm_df['sector1Time']
        del norm_df['sector2Time']
        del norm_df['safetyCarDelta']
        del norm_df['carPosition']
        del norm_df['pitStatus']
        del norm_df['sector']
        del norm_df['currentLapInvalid']
        del norm_df['penalties']
        del norm_df['gridPosition']
        del norm_df['driverStatus']
        del norm_df['resultStatus']
        # filter CarTelemetryData
        del norm_df['buttonStatus']
        del norm_df['clutch']
        del norm_df['drs']
        del norm_df['revLightsPercent']
        # filter CarStatusData
        del norm_df['tractionControl']
        del norm_df['antiLockBrakes']
        del norm_df['pitLimiterStatus']
        del norm_df['fuelCapacity']
        del norm_df['maxRPM']
        del norm_df['maxGears']
        del norm_df['drsAllowed']
        del norm_df['vehicleFiaFlags']
        del norm_df['ersStoreEnergy']
        del norm_df['ersDeployMode']
        del norm_df['ersHarvestedThisLapMGUH']
        del norm_df['ersHarvestedThisLapMGUK']
        del norm_df['ersDeployedThisLap']

        return norm_df


if __name__ == "__main__":
    # example for gathering data for training
    data_prep = DataPreparation()
    data = data_prep.load_data(r".\Data\AllData\0ff7c436c1c21664.sqlite3")
    x_train, y_train = data_prep.prepare_data(data)
    print(x_train[0], y_train[0])

    # # live
    # print('starting test')
    # quit_flag = True
    # udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    # udp_socket.bind(('', 20777))
    # time_for_receiving_cycles = []
    # time_for_filter = []
    # t_start = time.time()
    # counter = 0
    # for i in range(0, (1800*5)):
    #     t_begin = time.time()
    #     # print('collecting data')
    #     data = DataReader.listen_udp(udp_socket, 1)
    #     t1 = time.time()
    #     # print('got %i packets in %.3f ms' % (len(data), ((t1-t_begin)*1000)))
    #     time_for_receiving_cycles.append(((t1-t_begin)*1000))
    #     result = []
    #     result = sort_dict_into_list(data, False)
    #     t2 = time.time()
    #     # print('data filtered in %.3f ms' % ((t2-t1)*1000))
    #     time_for_filter.append(((t2-t1)*1000))
    #     # print('got %i cycles with relevant data' % len(result))
    #     # print(result)
    #     t_end = time.time()
    #     # print("total time: %.3f ms" % ((t_end-t_begin)*1000))
    #     # print('hier wird dann die Vorhersage gestartet')
    #     counter += 1
    # t_finish = time.time()
    # print('max receiving time: %.5f ms' % max(time_for_receiving_cycles))
    # print('min receiving time: %.5f ms' % min(time_for_receiving_cycles))
    # print('average receiving time: %.5f ms' % (sum(time_for_receiving_cycles) / len(time_for_receiving_cycles)))
    # print('max filter time: %.5f ms' % max(time_for_filter))
    # print('min filter time: %.5f ms' % min(time_for_filter))
    # print('average filter time: %.5f ms' % (sum(time_for_filter) / len(time_for_filter)))
    # print('time elapsed: %.3f s' % (t_finish-t_start))
    # print('total cycles received: %i' % counter)
    # print('average received cycles per second: %.5f' % (counter / (t_finish - t_start)))
    #
    # # list_to_dataframe test
    # data = DataReader.load_data_from_sqlite3(r".\Data\AllData\example.sqlite3")
    # data = sort_dict_into_list(data, False)
    # print(list_to_dataframe(data))
