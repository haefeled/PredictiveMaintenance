import InfluxManager
import DataPreparation
import DataReader

from influxdb import InfluxDBClient


class DataWriter:
    def __init__(self, table_name):
        self.influx_manager = InfluxManager.InfluxManager()
        self.table_name = table_name
        self.client = InfluxDBClient(host='localhost', port=8086)
        self.create_table()


    def create_table(self):
        self.client.create_database(self.table_name)
        self.client.switch_database(self.table_name)

    def insert_data(self, data):
        features = data[0].keys()
        print(features)


if __name__ == '__main__':
    data = DataReader.load_data_from_sqlite3(r".\Data\AllData\example.sqlite3")
    data = DataPreparation.sort_dict_into_list(data, True)
    data_writer = DataWriter("test")
    data_writer.insert_data(data)