import subprocess
import threading
from copy import deepcopy

import pandas as pd

from DataWriter import DataWriter
from DataPreparation import DataPreparation
from DataReader import DataReader
from multiprocessing import Process
from Predict import Predict


class DeepWheels:
    def __init__(self):
        self.data_reader = DataReader()
        self.data_prep = DataPreparation()
        self.data_writer = DataWriter("live_data")
        self.last_session_writer = DataWriter("last_session_data")
        self.prep_writer = DataWriter("prep_data")
        self.data_predict = Predict()
        self.prep_list = pd.DataFrame()

    def predict(self):
        """
        This function is the main of the project. It collects the live data and starts the prediction process.

        :return:
        """
        # example
        # data = self.data_reader.load_data_from_sqlite3(r".\Data\AllData\example.sqlite3")
        tmp_list = []
        for i in range(0, 30):
            data = self.data_reader.listen_udp(1)
            data = self.data_prep.sort_dict_into_list(data, False)
            self.data_writer.insert_data(data[0])
            tmp_list.append(deepcopy(data[0]))
        tmp_df = self.data_prep.list_to_dataframe(tmp_list)
        self.prep_list = deepcopy(tmp_df)
        predict_process = threading.Thread(target=Predict.predict, args=(self.data_predict, self.prep_list, self.prep_writer))
        predict_process.start()
        predict_process.join()

        # self.data_writer.print_data(data[i])


if __name__ == '__main__':
    deepwheels = DeepWheels()
    while True:
        deepwheels.predict()

