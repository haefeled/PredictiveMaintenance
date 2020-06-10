import sqlite3
import progressbar

from f1_2019_telemetry.packets import unpack_udp_packet


# return list with all packets
def load_data_from_sqlite3():
    # establish connection
    conn = sqlite3.connect(r".\Data\AllData\example.sqlite3")
    cursor = conn.cursor()
    query = "SELECT timestamp, packet FROM packets ORDER BY pkt_id;"
    query_count_rows = "SELECT COUNT(pkt_ID) FROM packets;"
    cursor.execute(query_count_rows)
    max_rows = cursor.fetchone()
    max_rows = max_rows[0]
    cursor.execute(query)

    # logging info
    print("Reading started.")

    # progressbar
    widgets = [
        '\x1b[33mCollecting Data... \x1b[39m',
        progressbar.Percentage(),
        progressbar.Bar(marker='\x1b[32m#\x1b[39m'),
    ]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=max_rows).start()
    data_list = []
    for i in range(1, max_rows):
        # Collecting data
        timestamped_packet = cursor.fetchone()
        if timestamped_packet is not None:
            (timestamp, packet) = timestamped_packet
            packet = unpack_udp_packet(packet)
            data_list.append(packet)
        bar.update(i + 1)
    # sort data by sessionTime
    # sorted_data_collection = dict(sorted(data_collection.items()))
    bar.finish()
    cursor.close()
    conn.close()
    return data_list

# TODO Add Functions of reader.py

if __name__ == "__main__":
    # for testing
    data = load_data_from_sqlite3()
