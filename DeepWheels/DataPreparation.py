import ctypes
import socket
import time
import progressbar
import pandas
import DataReader

from f1_2019_telemetry.packets import PackedLittleEndianStructure, PacketHeader
from copy import deepcopy


def filter_entries(entries, data):
    """
    This function is the third step to transform all received packets of f1_2019_telemetry.
    The function is responsible to handle the _fields of PackedLittleEndianStructure classes.
    In case of a PackedLittleEndianStructure as _field a callback to filter_object() is done.
    In all other cases the features are gathered and returned as dictionary.

    :param entries: _fields of the handled PackedLittleEndianStructure class
    :param data: PackedLittleEndianStructure Object
    :return: dict<string, (int, float)
    """
    tmp_dict_second = dict()
    for fname in entries:
        if not fname.startswith('_'):
            value = getattr(data, fname)
            if isinstance(value, (int, float, bytes)):
                tmp_dict_second[fname] = value
            elif isinstance(value, PackedLittleEndianStructure):
                tmp = dict(filter_objects(value))
                for key in tmp.keys():
                    tmp_dict_second[key] = deepcopy(tmp[key])
            elif isinstance(value, ctypes.Array) and len(value) != 20:
                wheel = 0
                for e in value:
                    str_key = fname + '%i' % wheel
                    tmp_dict_second[str_key] = deepcopy(e)
                    wheel += 1
            elif isinstance(value, ctypes.Array) and len(value) == 20:
                tmp = dict(filter_objects(value[data.header.playerCarIndex]))
                for key in tmp.keys():
                    tmp_dict_second[key] = deepcopy(tmp[key])
    return tmp_dict_second


def filter_objects(data):
    """
    This function is the second step to transform all received packets of f1_2019_telemetry.
    The function is responsible to view at PackedLittleEndianStructure classes. Furthermore
    the _fields of the class are collected from the function filter_entries() and returned
    as dictionary.

    :param data: PackedLittleEndianStructure variable
    :return:    dict<string, (int, float)>
    """
    tmp_dict = dict()
    if isinstance(data, PackedLittleEndianStructure) and not isinstance(data, PacketHeader):
        attr = dir(data)
        if attr.__contains__('header'):
            ident = data.header.packetId
            if ident == 0 or ident == 2 or ident == 6 or ident == 7:
                tmp_dict = filter_entries(attr, data)
        else:
            tmp_dict = filter_entries(attr, data)
    elif isinstance(data, PackedLittleEndianStructure) and isinstance(data, PacketHeader):
        tmp_dict['sessionTime'] = getattr(data, 'sessionTime')
    return tmp_dict


def sort_dict_into_list(data, train_flag):
    """
    This function is the first step to transform all received packets of f1_2019_telemetry.
    This function has the task to split the list and put the results of the transform into a list.
    At the end there is a validity test.
    In case of training, a progressbar show the state of progress.

    :param train_flag:  Boolean variable to indicate a training session
    :param data:    list<PackedLittleEndianStructure>, a list with all received packets
    :return:    list<dict>  A list containing the dictionaries including the filtered features
    """
    result = []
    if train_flag:
        widgets = [
            '\x1b[33mFilter Data... \x1b[39m',
            progressbar.Percentage(),
            progressbar.Bar(marker='\x1b[32m#\x1b[39m'),
        ]
        bar = progressbar.ProgressBar(widgets=widgets, max_value=len(data)).start()
        bar_counter = 0
    for paket in data:
        tmp = dict(filter_objects(paket))
        if tmp:
            if result and result[-1]['sessionTime'] == (tmp['sessionTime']):
                for key in tmp.keys():
                    result[-1][key] = deepcopy(tmp[key])
            else:
                result.append(deepcopy(tmp))
        if train_flag:
            bar_counter += 1
            bar.update(bar_counter)
    if train_flag:
        bar.finish()
    result = check_dict_in_list(result)
    return result


def check_dict_in_list(data):
    """
    This function is a validity test. The Input is the list with all filtered features.
    The function tests the length of dictionaries inside a list and removes a dictionary
    if it hasn't 130 entries.

    :param data: list<dict>
    :return: list<dict>
    """
    list_of_removable = []
    for tmp_dict in data:
        if len(tmp_dict) != 130 and isinstance(data, list):
            list_of_removable.append(data.index(tmp_dict))
    for removable_index in list_of_removable:
        data.remove(data[removable_index])
    return data


def list_to_dataframe(list_data):
    '''
    Converts a list of maps to a pandas DataFrame.

    :param list_data: list<dict<number>> A list of maps which contain non-collection data.
    :return: pandas.DataFrame A pandas DataFrame.
    '''
    outer_list = []
    for row in list_data:
        outer_list.append(list(row[key] for key in row.keys()))
    return pandas.DataFrame(outer_list, columns = list(list_data[0].keys()))


if __name__ == "__main__":
    # # train
    # data = DataReader.load_data_from_sqlite3(r".\Data\AllData\example.sqlite3")
    # # sleep just for ide terminal output, else progressbar not properly displayed
    # time.sleep(0.2)
    # print('got %i packets' % len(data))
    # result = sort_dict_into_list(data, True)
    # # sleep just for ide terminal output, else progressbar not properly displayed
    # time.sleep(0.2)
    # print('data filtered!')
    # print('got %i cycles with relevant data' % len(result))

    # live
    print('starting test')
    quit_flag = True
    udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    udp_socket.bind(('', 20777))
    time_for_receiving_cycles = []
    time_for_filter = []
    t_start = time.time()
    counter = 0
    for i in range(0, (1800*5)):
        t_begin = time.time()
        # print('collecting data')
        data = DataReader.listen_udp(udp_socket, 1)
        t1 = time.time()
        # print('got %i packets in %.3f ms' % (len(data), ((t1-t_begin)*1000)))
        time_for_receiving_cycles.append(((t1-t_begin)*1000))
        result = []
        result = sort_dict_into_list(data, False)
        t2 = time.time()
        # print('data filtered in %.3f ms' % ((t2-t1)*1000))
        time_for_filter.append(((t2-t1)*1000))
        # print('got %i cycles with relevant data' % len(result))
        # print(result)
        t_end = time.time()
        # print("total time: %.3f ms" % ((t_end-t_begin)*1000))
        # print('hier wird dann die Vorhersage gestartet')
        counter += 1
    t_finish = time.time()
    print('max receiving time: %.5f ms' % max(time_for_receiving_cycles))
    print('min receiving time: %.5f ms' % min(time_for_receiving_cycles))
    print('average receiving time: %.5f ms' % (sum(time_for_receiving_cycles) / len(time_for_receiving_cycles)))
    print('max filter time: %.5f ms' % max(time_for_filter))
    print('min filter time: %.5f ms' % min(time_for_filter))
    print('average filter time: %.5f ms' % (sum(time_for_filter) / len(time_for_filter)))
    print('time elapsed: %.3f s' % (t_finish-t_start))
    print('total cycles received: %i' % counter)
    print('average received cycles per second: %.5f' % (counter / (t_finish - t_start)))

    # list_to_dataframe test
    data = DataReader.load_data_from_sqlite3(r".\Data\AllData\example.sqlite3")
    data = sort_dict_into_list(data, False)
    print(list_to_dataframe(data))
    '''
    '''