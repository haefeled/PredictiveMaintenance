import subprocess


def startInfluxDBServer():
    process = subprocess.Popen([r".\Data\InfluxDB\InfluxServer\influxd.exe"],
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
    startInfluxDBServer()
