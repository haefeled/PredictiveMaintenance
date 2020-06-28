from influxdb import InfluxDBClient

def createInfluxDB():

    json_body = [
        {
            "measurement": "cpu_load_short",
            "tags": {
                "host": "server01",
                "region": "us-west"
            },
            "time": "2009-11-10T23:00:00Z",
            "fields": {
                "value": 0.64
            }from influxdb import InfluxDBClient
        }
    ]

    client = InfluxDBClient('localhost', 8086, 'root', 'root', 'example')

    client.create_database('example')

    client.write_points(json_body)

    result = client.query('select value from cpu_load_short;')

    print("Result: {0}".format(result))

if __name__ == "__main__":
    createInfluxDB()