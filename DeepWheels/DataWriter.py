from datetime import time

from InfluxManager import InfluxManager
from DataPreparation import DataPreparation
from DataReader import DataReader
from InfluxManager import InfluxManager
from influxdb import InfluxDBClient


class DataWriter:
    def __init__(self, table_name):
        self.influx_manager = InfluxManager()
        self.database_name = table_name
        self.client = InfluxDBClient(host='localhost', port=8086)
        self.create_table()

    def create_table(self):
        self.client.create_database(self.database_name)
        self.client.switch_database(self.database_name)

    def insert_data(self, data):
        metrics = {}
        metrics['measurement'] = "kartoffel"
        metrics['tags'] = {}
        metrics['fields'] = {}
        for feature in data[0].keys():
            metrics['fields'][feature] = data[0][feature]
        self.client.write_points([metrics])

    def print_data(self):
        loginRecords = self.client.query('select * from kartoffel;')

        # Print the time series query results

        print(loginRecords)
        bla = self.client.get_list_database()
        print(bla)


if __name__ == '__main__':
    data = DataReader().load_data_from_sqlite3(r".\Data\AllData\example.sqlite3")
    data = DataPreparation().sort_dict_into_list(data, True)
    data_writer = DataWriter("testboom")
    data_writer.insert_data(data)
    data_writer.print_data()
