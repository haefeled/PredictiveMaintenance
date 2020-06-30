import os
import subprocess

import psutil


class GrafanaManager:
    def __init__(self):
        self.start_grafana_server()

    def start_grafana_server(self):
        if not self.check_if_server_running():
            #TODO: ADD -config
            subprocess.Popen([r".\GrafanaServer\bin\grafana-server.exe",  "-homepath", r".\GrafanaServer"],
                             stdout=subprocess.PIPE,
                             universal_newlines=True)

        elif self.check_if_server_running():
            print('Server already running.')
        else:
            print('Something Bad happened')

    def check_if_server_running(self):
        process_name = 'grafana-server.exe'
        for proc in psutil.process_iter():
            try:
                # Check if process name contains the given name string.
                if process_name.lower() in proc.name().lower():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return False

    def close_server(self):
        process_name = 'grafana-server.exe'
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

    dw = GrafanaManager()
    print('Hallo')

