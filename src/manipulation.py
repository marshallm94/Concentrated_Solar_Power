import pandas as pd
import numpy as np

def count_nans(df, verbose=True):
    """
    Calculates NaN percentages per column in a pandas DataFrame.

    Parameters:
        df: (Pandas DataFrame)
        verbose: (Boolean) Prints column names and NaN percentage if True

    Output:
        col_nans: List containing tuples of column names and percentage NaN for that column.
    """
    col_nans = []
    for col in df.columns:
        percent_nan = pd.isnull(df[col]).sum()/len(pd.isnull(df[col]))
        col_nans.append((col, percent_nan))
        if verbose:
            print("{} | {:.2f}% NaN".format(col, percent_nan*100))

    return col_nans

def import_data(root, start, stop):
    """
    Import multiple csv files from one directory assuming there is consistency in the naming of the files.

    Parameters:
        root: (str) The relative or absolute path to the directory containing all files desired.
        start: (str) The name of the file to import first, in this case a data formatted as YYYYMMDD
        stop: (str) The name of last file to import, in this case a data formatted as YYYYMMDD

    Return:
        df: (Pandas DataFrame) One dataframe for all data from all files contained in the date range specified by start and stop parameters.
    """
    extension = ".csv"
    out = root + start + extension
    print(out)



if __name__ == "__main__":

    root = "../data/solar_measurements/"

    df = import_data(root, "20060318", "20061231")

    col_names = list(pd.read_csv("../data/solar_measurements/header.csv"))

    test = pd.read_csv("../data/solar_measurements/20180413.csv", header=None, names=col_names)
