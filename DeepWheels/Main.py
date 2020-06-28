import subprocess

def startInfluxDB():

    process = subprocess.Popen([r'C:\Users\fabianaust\PycharmProjects\PredictiveMaintenance\DeepWheels\Data\InfluxDB\InfluxServer\influxd.exe',
                                '-config', r'C:\Users\fabianaust\PycharmProjects\PredictiveMaintenance\DeepWheels\Data\InfluxDB\InfluxServer\influxdb.conf'],
                               stdout=subprocess.PIPE,
                               universal_newlines=True)

    while True:
        output = process.stdout.readline()
        print(output.strip())
        # Do something else
        return_code = process.poll()
        if return_code is not None:
            print('RETURN CODE', return_code)
            # Process has finished, read rest of the output
            for output in process.stdout.readlines():
                print(output.strip())
            break


if __name__ == "__main__":
    startInfluxDB()