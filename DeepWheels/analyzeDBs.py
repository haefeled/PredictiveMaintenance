from copy import deepcopy

import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

from DataReader import DataReader
from DataPreparation import DataPreparation

data_reader = DataReader()
data_prep = DataPreparation()

def analyze(filename, isFirst):
    print("read {}".format(filename))
    data = data_reader.load_data_from_sqlite3(r".\Data\AllData\\" + filename)
    data = data_prep.sort_dict_into_list(data, False)
    df = data_prep.list_to_dataframe(data)

    maxrul_list = [df.loc[df.query('tyresWear0 < 50').tyresWear0.count(), 'sessionTime'],
                   df.loc[df.query('tyresWear1 < 50').tyresWear1.count(), 'sessionTime'],
                   df.loc[df.query('tyresWear2 < 50').tyresWear2.count(), 'sessionTime'],
                   df.loc[df.query('tyresWear3 < 50').tyresWear3.count(), 'sessionTime']]

    for i in range(len(maxrul_list)):
        maxrul_STR = 'maxRUL' + str(i)
        df[maxrul_STR] = maxrul_list[i]

    colums = df.columns.tolist()
    compare_dict = dict()
    if not isFirst:
        with open(r".\Data\analysis_results.txt") as f:
            content = f.readlines()
            for line in content:
                entry = line.strip().split(':')
                compare_dict[deepcopy(entry[0])] = deepcopy(float(entry[1]))


    with open(r".\Data\analysis_results.txt", "w") as out_file:
        for colum in colums:
            if abs(df[colum].max()) >= abs(df[colum].min()):
                factor = abs(df[colum].max())
            else:
                factor = abs(df[colum].min())

            if not isFirst:
                if factor < compare_dict[colum]:
                    out_file.write(
                        "{}:{}\n".format(colum, compare_dict[colum])
                    )
                else:
                    out_file.write(
                        "{}:{}\n".format(colum, factor)
                    )
            else:
                out_file.write(
                    "{}:{}\n".format(colum, factor)
                )



def analyze_all_datasets(path_to_datasets):
    """
    Initiates training on a series of databases.

    :param path_to_datasets: str Represents the path where all databases can be located.
    """
    db_file_names = [f for f in listdir(path_to_datasets) if isfile(join(path_to_datasets, f))]
    for i in range(len(db_file_names)):
        if i == 0:
            analyze(db_file_names[i], True)
        else:
            analyze(db_file_names[i], False)

def main():
    analyze_all_datasets(r".\Data\AllData")


if __name__ == "__main__":
    # code is only run when module is not called via 'import'
    main()
