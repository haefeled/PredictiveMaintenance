import DataReader
import progressbar


# function for data analyse
# maybe enough to pass to NN
# output dict with key=feature value=value or
# output dict (key=sessionTime, value=dict(key=feature, value=value))
# reason: maybe easier to give packets grouped by sessionTime and resolve dict.keys() as features for NN
def all_data_to_list():
    # @TODO implement data from load_data_from_sqlite3 to dict
    data = DataReader.load_data_from_sqlite3()


# @TODO caution maybe discard or change this function filter_data()!!!!!
# input list of unpacked packets
# return list of relevant data dependent on player's carindex
# format [[header], [packetId, data], [packetId, data], ...]
# caution weather data is a dictonary inside an array
def filter_data(data_list):
    rel_data_dict = []
    counter = 0
    prev_session_time = 0.0
    max_len = len(data_list)

    # progressbar
    widgets = [
        '\x1b[33mFilter Data... \x1b[39m',
        progressbar.Percentage(),
        progressbar.Bar(marker='\x1b[32m#\x1b[39m'),
    ]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=max_len).start()
    for i in range(max_len):
        packet = data_list[i]
        session_time = packet.header.sessionTime
        player_car_index = packet.header.playerCarIndex

        # indexing
        if i == 0:
            counter = 0
            rel_data_dict.append([packet.header])
        elif session_time != prev_session_time:
            counter = counter + 1
            rel_data_dict.append([packet.header])

        # sort
        if packet.header.packetId == 0:
            rel_data_dict[counter].append([
                0,
                packet.carMotionData[player_car_index]
            ])
        elif packet.header.packetId == 1:
            rel_data_dict[counter].append([0, [dict(
                weather=packet.weather,
                tracktemperature=packet.trackTemperature,
                airTemperature=packet.airTemperature,
                trackLength=packet.trackLength,
                sessionTimeLeft=packet.sessionTimeLeft,
                sessionDuration=packet.sessionDuration
            )]])
        elif packet.header.packetId == 2:
            rel_data_dict[counter].append([2, packet.lapData[player_car_index]])
        elif packet.header.packetId == 6:
            rel_data_dict[counter].append([6, packet.carTelemetryData[player_car_index]])
        elif packet.header.packetId == 7:
            rel_data_dict[counter].append([7, packet.carStatusData[player_car_index]])

        prev_session_time = session_time
        bar.update(i)
    bar.finish()
    return rel_data_dict


if __name__ == "__main__":
    all_data_to_list()
