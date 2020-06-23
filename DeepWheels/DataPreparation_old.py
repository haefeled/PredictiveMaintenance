import DataReader
import progressbar

from f1_2019_telemetry.packets import unpack_udp_packet

# function for data analyse
# maybe enough to pass to NN
# output dict with key=feature value=value or
# output dict (key=sessionTime, value=dict(key=feature, value=value))
# reason: maybe easier to give packets grouped by sessionTime and resolve dict.keys() as features for NN



def all_data_to_list():
    """
    Reads and filters all telemetry packets of some database.

    :return: list<dict('h' : dict, '0' : dict, '2' : dict, '6' : dict, '7' : dict)> 
             A list of hashmaps representing header data and packets 0, 2, 6, 7. 
             Each list item includes data for one timestep.
    """
    data = DataReader.load_data_from_sqlite3(r".\Data\AllData\example.sqlite3")
    data = filter_data(data)

    # for testing
    print(data[0])
    print(data[len(data) - 1])

    return data

def filter_data(data_list):
    """
    Filters a list of incoming packets and convert packet objects to dictionaries.

    :param data_list: A list of telemetry packets including headers.
    :return: list<dict('h' : dict, '0' : dict, '2' : dict, '6' : dict, '7' : dict)> 
             A list of hashmaps representing header data and packets 0, 2, 6, 7. 
             Each list item includes data for one timestep.
    """
    rel_data_dict = []
    counter = 0
    # counts header and packets 0, 2, 6, 7
    rel_data_counter = 0
    prev_session_time = 0.0
    max_len = len(data_list)

    # progressbar
    widgets = [
        '\x1b[33mFilter Data... \x1b[39m',
        progressbar.Percentage(),
        progressbar.Bar(marker='\x1b[32m#\x1b[39m'),
    ]
    #bar = progressbar.ProgressBar(widgets=widgets, max_value=max_len).start()
    
    for i in range(max_len):
        packet = data_list[i]
        session_time = packet.header.sessionTime
        player_car_index = packet.header.playerCarIndex

        # indexing
        if i == 0:
            counter = 0
        elif session_time != prev_session_time:
            counter = counter + 1

        if i == 0 or session_time != prev_session_time:
            rel_data_counter = rel_data_counter + 1
            # convert packet object to dictionary
            rel_data_dict.append({'h' : {}, '0' : {}, '2': {}, '6' : {}, '7' : {}})
            data = packet.header
            data_dict = dict((name, getattr(data, name)) for name in dir(data) if not name.startswith('_'))
            for key in data_dict:
                rel_data_dict[counter]['h'][key] = data_dict[key]

        packet_id = packet.header.packetId

        # sort
        if packet_id == 0:
            data = packet.carMotionData[player_car_index]
        elif packet_id == 2:
            data = packet.lapData[player_car_index]
        elif packet_id == 6:
            data = packet.carTelemetryData[player_car_index]
        elif packet_id == 7:
            data = packet.carStatusData[player_car_index]

        if packet_id == 0 or packet_id == 2 or packet_id == 6 or packet_id == 7:
            rel_data_counter = rel_data_counter + 1
            # convert packet object to dictionary
            data_dict = dict((name, getattr(data, name)) for name in dir(data) if not name.startswith('_'))
            for key in data_dict:
                    # if value is an array with one value for each tire
                    if type(data_dict[key]) is not float and type(data_dict[key]) is not int:
                        for i in range(len(data_dict[key])):
                            rel_data_dict[counter][str(packet_id)][key + str(i)] = data_dict[key][i]
                    else:
                        rel_data_dict[counter][str(packet_id)][key] = data_dict[key]

        # remove timestamp-related packets if one packet is missing
        if rel_data_counter == 5:
            current_data = rel_data_dict[counter]
            if not current_data['h'] or not current_data['0'] or not current_data['2'] or not current_data['6'] or not current_data['7']:
                del rel_data_dict[counter]
                counter = counter - 1
                rel_data_counter = 0

        prev_session_time = session_time
        #bar.update(i)
    #bar.finish()

    # remove timestamp-related packets if one packet is missing
    if rel_data_counter % 5 != 0:
        del rel_data_dict[counter]

    return rel_data_dict

if __name__ == "__main__":
    all_data_to_list()