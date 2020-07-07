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

    max_vals = [df['tyresWear0'].max(), df['tyresWear1'].max(), df['tyresWear2'].max(), df['tyresWear3'].max()]

    write_mode = "a"
    if isFirst:
        write_mode = "w"

    with open(r".\Data\analysis_results.txt", write_mode) as out_file:
            out_file.write("{} maxTyreWear: {}%{}%{}%{}%\n".format(filename, max_vals[0], max_vals[1], max_vals[2], max_vals[3]))

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
