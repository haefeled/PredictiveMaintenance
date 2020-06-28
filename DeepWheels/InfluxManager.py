import os
import subprocess
import psutil

from multiprocessing import Process
from influxdb import InfluxDBClient


class InfluxManager:
    def __init__(self):
        Process(self.start_influx_server(), args=(self,)).start()
        self.db_client = InfluxDBClient(host='localhost', port=8086)

    def start_influx_server(self):
        if not self.check_if_server_running():

            subprocess.Popen([r".\Data\InfluxDB\InfluxServer\influxd.exe"],
                             stdout=subprocess.PIPE,
                             universal_newlines=True)

        elif self.check_if_server_running():
            print('Server already running.')
        else:
            print('Something Bad happened')

    def check_if_server_running(self):
        process_name = 'influxd.exe'
        for proc in psutil.process_iter():
            try:
                # Check if process name contains the given name string.
                if process_name.lower() in proc.name().lower():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return False

    def close_server(self):
        process_name = 'influxd.exe'
        for proc in psutil.process_iter():
            try:
                # Check if process name contains the given name string.
                if process_name.lower() in proc.name().lower():
                    os.kill(proc.pid, 9)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return False

    def __del__(self):
        self.close_server()


if __name__ == '__main__':
    dw = InfluxManager()
    print('Hallo')
    dw.close_server()
    dwe = InfluxManager()
