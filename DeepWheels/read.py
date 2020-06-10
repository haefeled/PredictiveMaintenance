import os
import sys
import socket
import datetime
from array import array
from threading import Thread

import f1_2019_telemetry.cli.player
from f1_2019_telemetry.packets import unpack_udp_packet, PacketID, PacketHeader

#if data should be played back standalone from a database file, run: python -m f1_2019_telemetry.cli.player -r 100 database.sqlite3

def replay_database(filename):
    """reads F1 2019 telemetry packets stored in a SQLite3 database file and sends them out over UDP, effectively replaying a session of the F1 2019 game"""
    os.system('python -m f1_2019_telemetry.cli.player -r 100 ' + filename)

def is_faulty(wear_data):
    return 1 if wear_data >= 6.0 else 0

def database_as_list(filename):
    """reads F1 2019 telemetry packets stored in a SQLite3 database file and returns the data as a list as soon as reading is completed"""
    packet_already_received = False
    var_changed = True

    udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    udp_socket.bind(('', 20777))
    udp_socket.setblocking(0)

    start_packet_missing_time = datetime.datetime.now()

    tyresWear = array('I', [0, 0, 0, 0])
    tyresDamage = array('I', [0, 0, 0, 0])

    dataset_raw = []

    thread = Thread(target = replay_database, args = [filename])
    thread.start()

    while True:
        try:
            udp_packet = udp_socket.recv(2048)
            packet = unpack_udp_packet(udp_packet)
            if packet.header.packetId == 7:
                packet_already_received = True
                start_packet_missing_time = datetime.datetime.now()
                playerCarIndex = packet.header.playerCarIndex
                data = packet.carStatusData[playerCarIndex]
                for i in range(4):
                    if tyresWear[i] != data.tyresWear[i]:
                        tyresWear[i] = data.tyresWear[i]
                        var_changed = True
                    if tyresDamage[i] != data.tyresDamage[i]:
                        tyresDamage[i] = data.tyresDamage[i]
                        var_changed = True
                if var_changed:
                    var_changed = False
                    #add new row to dataset
                    dataset_raw.append(
                        [
                        (packet.header.sessionTime / 60),
                        #data.fuelInTank,
                        #data.fuelRemainingLaps,
                        data.tyresWear[0],
                        #data.tyresDamage[0],
                        data.tyresWear[1],
                        #data.tyresDamage[1],
                        data.tyresWear[2],
                        #data.tyresDamage[2],
                        data.tyresWear[3],
                        #data.tyresDamage[3],
                        is_faulty(data.tyresWear[0])
                        ])
        #if no packets were received
        except socket.error:
            if packet_already_received == True:
                end_packet_missing_time = datetime.datetime.now()
                packet_missing_duration = end_packet_missing_time - start_packet_missing_time
                if packet_missing_duration.seconds > 5:
                    return dataset_raw

def apply_to_live_data(func, buffer_time_in_seconds = 3):
    """reads live F1 2019 telemetry packets, converts them into lists and applies a given function on each list"""
    packet_already_received = False
    var_changed = True

    udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    udp_socket.bind(('', 20777))
    udp_socket.setblocking(0)

    tyresWear = array('I', [0, 0, 0, 0])
    tyresDamage = array('I', [0, 0, 0, 0])

    live_buffer = []

    #only for testing purposes
    thread = Thread(target = replay_database, args = [r".\Data\AllData\example.sqlite3"])
    thread.start()

    start_packet_missing_time = datetime.datetime.now()
    start_buffer_time = datetime.datetime.now()

    while True:
        try:
            udp_packet = udp_socket.recv(2048)
            packet = unpack_udp_packet(udp_packet)
            if packet.header.packetId == 7:
                packet_already_received = True
                start_packet_missing_time = datetime.datetime.now()
                playerCarIndex = packet.header.playerCarIndex
                data = packet.carStatusData[playerCarIndex]
                for i in range(4):
                    if tyresWear[i] != data.tyresWear[i]:
                        tyresWear[i] = data.tyresWear[i]
                        var_changed = True
                    if tyresDamage[i] != data.tyresDamage[i]:
                        tyresDamage[i] = data.tyresDamage[i]
                        var_changed = True
                if var_changed:
                    var_changed = False
                    live_buffer.append([
                        #data.fuelInTank,
                        #data.fuelRemainingLaps,
                        data.tyresWear[0],
                        #data.tyresDamage[0],
                        data.tyresWear[1],
                        #data.tyresDamage[1],
                        data.tyresWear[2],
                        #data.tyresDamage[2],
                        data.tyresWear[3],
                        #data.tyresDamage[3]
                        ])
                    current_buffer_time = datetime.datetime.now() - start_buffer_time
                    if current_buffer_time.seconds > buffer_time_in_seconds:
                        #apply function on live data buffer after every buffer interval
                        func(live_buffer)  
                        start_buffer_time = datetime.datetime.now()     
        #if no packets were received
        except socket.error:
            if packet_already_received == True:
                end_packet_missing_time = datetime.datetime.now()
                packet_missing_duration = end_packet_missing_time - start_packet_missing_time
                if packet_missing_duration.seconds > 5:
                    return