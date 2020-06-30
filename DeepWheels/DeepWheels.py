from DataWriter import DataWriter
from DataPreparation import DataPreparation
from DataReader import DataReader
from multiprocessing import Process


class DeepWheels:
    def __init__(self):
        self.data_reader = DataReader()
        self.data_prep = DataPreparation()
        self.data_writer = DataWriter("live_data")
        # self.predict_process = Process(target=Predict.predict, args=('bob',))

    def predict(self):
        """

        :return:
        """
        # example
        data = self.data_reader.load_data_from_sqlite3(r".\Data\AllData\example.sqlite3")
        data = self.data_prep.sort_dict_into_list(data, True)
        for i in range(0, 100):
            self.data_writer.insert_data(data[i])
            self.data_writer.print_data(data[i]['sessionTime'])


if __name__ == '__main__':
    DeepWheels().predict()
    while True:
        x = 1
