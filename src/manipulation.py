import pandas as pd
import numpy as np
import os
from datetime import datetime

def build_master_csv(root, filename):
    """
    Reads all files in filepath specified by root parameter and
    concatenates into one Pandas DataFrame, which is then saved as a
    separate file specified by filename parameter

    Parameters:
        root: (str) The relative or absolute path to the directory containing all files desired to be concatenated.
        filename: (str) The filename where the final data object should be saved.

    Returns:
        None
    """
    all_files = os.listdir(path=root)
    all_files.remove("header.csv")
    col_names = list(pd.read_csv("../data/solar_measurements/header.csv"))
    for x, file in enumerate(all_files):
        current_file = root + file
        if x == 0:
            df = pd.read_csv(current_file, header=None, names=col_names)
        else:
            current_df = pd.read_csv(current_file, header=None, names=col_names)
            df = pd.concat([df, current_df], axis=0)

        print(f"Final DataFrame Shape: {df.shape}")

    df.to_csv(filename)
    print(f"\nAll files concatenated to one data object and saved to {filename}\nProcess Complete.")


def check_data_count(year_list, df, verbose=True):
    """
    Checks to ensure that every yeat in year_list has 355 or 366 days
    and that every day has 1440 observations (1 observation per
    minute)

    Parameters:
        year_list: (list) List of years (formatted as ints) to check
        df: (pandas dataframe)
        verbose: (boolean, default=True) Will print output if True

    Return:
    """
    for year in year_list:
        subset = df[df["Year"] == year]
        print("\n\n", str(year), "\n\n")
        print((subset.groupby("DOY").count() == 1440).sum())

def make_dates(year_list, df):
    """
    Sets the index of df to be a datetime object

    Parameters:
        df: (pandas dataframe) Dataframe to be changed

    Return:
        updated_df: (pandas dataframe) Dataframe with datetime as index
    """
    for year in year_list:
        mask = df['Year'] == 2014
        days = np.unique(df[mask]['DOY'])
        for day in days:
            mask2 = df[mask]['DOY'] == 45
            hours = df[mask & mask2].reset_index().index // 60
            for hour in np.unique(hours):
                np.where(hours == hour, True, False)
                mask3 = df[mask & mask2]['']
                minute_subset = minute_subset.reset_index()
                minute_subset["Minute"] == minute_subset.index



    for year in year_list:
        subset = df[df['Year'] == year]
        days = np.unique(subset['DOY'])
        for day in days:
            nested_subset = subset[subset['DOY'] == day]
            nested_subset = nested_subset.reset_index
            nested_subset["Hour"] = nested_subset.index // 60
            hours = np.unique(nested_subset['Hour'])
            for hour in hours:
                minute_subset = nested_subset[nested_subset['Hour'] == hour]
                minute_subset = minute_subset.reset_index()
                minute_subset["Minute"] == minute_subset.index


    df["string_DOY"] = df["Year"].astype(str) + " " +  df["DOY"].astype(str)
    df["Date"] = pd.to_datetime(df["string_DOY"], format="%Y %j")
    hour = 0



    return df

if __name__ == "__main__":

    # build_master_csv("../data/solar_measurements/", "../data/ivanpah_measurements.csv")

    df = pd.read_csv("../data/ivanpah_measurements.csv")

    df.drop('Unnamed: 0.1', axis=1, inplace=True)

    years = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]

    check_data_count(years, df)

    df = make_dates(df)
