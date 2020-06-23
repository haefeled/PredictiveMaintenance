import ctypes
import socket
import time
import progressbar
import DataReader

from f1_2019_telemetry.packets import PackedLittleEndianStructure, PacketHeader
from copy import deepcopy


def filter_entries(attr_list, data):
    """
    
    :param attr_list: 
    :param data: 
    :return: 
    """
    tmp_dict_second = dict()
    for fname in attr_list:
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
            elif isinstance(value, ctypes.Array):
                for e in value:
                    tmp = dict(filter_objects(e))
                    for key in tmp.keys():
                        tmp_dict_second[key] = deepcopy(tmp[key])
    return tmp_dict_second

def filter_objects(data):
    """

    :param data:
    :return:
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

    :param train_flag:
    :param data:
    :return:
    """
    result = []
    dic = dict()
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

    :param data:
    :return:
    """
    list_of_removable = []
    for tmp_dict in data:
        if len(tmp_dict) != 130 and isinstance(data, list):
            list_of_removable.append(data.index(tmp_dict))
    for removable_index in list_of_removable:
        data.remove(data[removable_index])
    return data



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
    udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    udp_socket.bind(('', 20777))
    for i in range(0, 20):
        t_begin = time.time()
        print('collecting data')
        data = DataReader.listen_udp(udp_socket, 1)

        t1 = time.time()
        print('got %i packets in %.3f ms' % (len(data), ((t1-t_begin)*1000)))
        result = []
        result = sort_dict_into_list(data, False)
        t2 = time.time()
        print('data filtered in %.3f ms' % ((t2-t1)*1000))
        print('got %i cycles with relevant data' % len(result))
        print(result)
        t_end = time.time()
        print("total time: %.3f ms" % ((t_end-t_begin)*1000))
        print('hier wird dann die Vorhersage gestartet')
