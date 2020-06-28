import InfluxManager
from influxdb import InfluxDBClient

class DataWriter:
    def __init__(self, table_name):
        self.influx_manager = InfluxManager()
        self.table_name = table_name
        self.client = InfluxDBClient(host='localhost', port=8086)
        self.create_table()


    def create_table(self):
        self.client.create_database(self.table_name)
        self.client.switch_database(self.table_name)

if __name__ == '__main__':

