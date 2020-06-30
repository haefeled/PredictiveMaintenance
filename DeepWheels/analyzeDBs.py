import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

from DataReader import load_data_from_sqlite3
from DataPreparation import sort_dict_into_list, list_to_dataframe

def analyze(filename, isFirst):
    print("read {}".format(filename))
    data = load_data_from_sqlite3(r".\Data\AllData\\" + filename)
    data = sort_dict_into_list(data, False)
    df = list_to_dataframe(data)

    max_vals = [df['tyresWear0'].max(), df['tyresWear1'].max(), df['tyresWear2'].max(), df['tyresWear3'].max()]

    if isFirst:
        with open(r".\Data\analysis_results.txt", "w") as out_file:
            out_file.write("{} maxTyreWear: {}%\n".format(filename, max(max_vals)))
    else:
        with open(r".\Data\analysis_results.txt", "a") as out_file:
            out_file.write("{} maxTyreWear: {}%\n".format(filename, max(max_vals)))

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
