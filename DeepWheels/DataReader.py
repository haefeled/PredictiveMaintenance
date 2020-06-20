
import sqlite3
from copy import deepcopy

import progressbar
import socket

from f1_2019_telemetry.packets import unpack_udp_packet


def load_data_from_sqlite3(path_to_db):
    """
    The function will return all packets saved inside a sqlite3 database.

    :param path_to_db:  A string variable for the database path.
    :return: list<packets>  A list containing all packets
    """
    # establish connection
    conn = sqlite3.connect(path_to_db)
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
    #bar = progressbar.ProgressBar(widgets=widgets, max_value=max_rows).start()
    data_list = []
    for i in range(1, max_rows):
        # Collecting data
        timestamped_packet = cursor.fetchone()
        if timestamped_packet is not None:
            (timestamp, packet) = timestamped_packet
            packet = unpack_udp_packet(packet)
            data_list.append(packet)
        #bar.update(i + 1)
    # sort data by sessionTime
    # sorted_data_collection = dict(sorted(data_collection.items()))
    #bar.finish()
    cursor.close()
    conn.close()
    return data_list


def listen_udp(udp_conn, nmb_of_cycles):
    """
    The function will receive a bunch of packages depended on the number of cycles you want to listen.
    In this case a cycle contains all packets with the same timestamp of the session.
    A filter is integrated which only adds the packets, if they build a group of 4 with the same timestamp.
    Reason: it could be possible, that the entry point is not the beginning of the session.

    :param udp_conn:    already connected udp socket
    :param nmb_of_cycles:   An Integer variable which defines the number of cycles to listen.
    :return: list<packet>   a list containing all received packages with the ID 0, 2, 6, and 7
    """
    data_list = []
    for j in range(0, nmb_of_cycles):
        # gather packet 0, 2, 6 and 7 of the same cycle
        counter = 0
        tmp_list = []
        last_sessionTime = 0
        while counter/4 < 1:
            try:
                udp_packet = udp_conn.recv(2048)
                packet = unpack_udp_packet(udp_packet)
                ident = packet.header.packetId
                current_sessionTime = packet.header.sessionTime
                if ident == 0 or ident == 2 or ident == 6 or ident == 7:
                    if current_sessionTime != last_sessionTime:
                        counter = 0
                        tmp_list.clear()
                    tmp_list.append(packet)
                    counter += 1
                    last_sessionTime = current_sessionTime
            except socket.error:
                print("no udp packets received")
        tmp_list = deepcopy(tmp_list)
        for entry in tmp_list:
            data_list.append(entry)
    return data_list


if __name__ == "__main__":
    # just for testing
    path_to_db = r".\Data\AllData\example.sqlite3"
    data = load_data_from_sqlite3(path_to_db)


    # # just for testing
    # udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    # udp_socket.bind(('', 20777))
    # last_sessiontime = 0.0
    # current_sessiontime = 0.0
    # counter = 0
    # bottom = 0.0
    # top = 1.0
    # for i in range(0, 1000):
    #     packets = listen_udp(udp_socket, 1)
    #     for paket in packets:
    #         # print("%i %f" % (paket.header.packetId, paket.header.sessionTime))
    #         current_sessiontime = paket.header.sessionTime
    #         # print("%f ms since last cycle received" % ((current_sessiontime-last_sessiontime)*1000))
    #         last_sessiontime = current_sessiontime
    #     if current_sessiontime >= top:
    #         bottom = current_sessiontime
    #         top = bottom + 1.0
    #         print("%i received cycles per second (received Hz)" % (counter))    # should be 60???
    #         counter = 0
    #     counter += 1
