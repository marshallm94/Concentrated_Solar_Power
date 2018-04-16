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

    minutes = np.arange(60)
    minute_array = np.array([np.copy(minutes) for i in range(24)]).ravel()

    for x, small_file in enumerate(all_files):
        current_file = root + small_file
        date = small_file[:-4]
        year, month, day = date[:4], date[4:6], date[6:]
        date = year + "-" + month + "-" + day
        if x == 0:
            df = pd.read_csv(current_file, header=None, names=col_names)
            df["Hour"] = df.index // 60
            df['Minute'] = minute_array
            df["Date"] = date
            df['final_date'] = df['Date'].astype(str) + " " + df['Hour'].astype(str) + ":" + df['Minute'].astype(str)
            df['final_date'] = pd.to_datetime(df['final_date'])
            df.set_index('final_date', inplace=True)
        else:
            current_df = pd.read_csv(current_file, header=None, names=col_names)
            current_df["Hour"] = current_df.index // 60
            current_df['Minute'] = minute_array
            current_df["Date"] = date
            current_df['final_date'] = current_df['Date'].astype(str) + " " + current_df['Hour'].astype(str) + ":" + current_df['Minute'].astype(str)
            current_df['final_date'] = pd.to_datetime(current_df['final_date'])
            current_df.set_index('final_date', inplace=True)
            df = pd.concat([df, current_df], axis=0)

        print(f"Final DataFrame Shape: {df.shape}")

    df.to_csv(filename)
    print(f"\nAll files concatenated to one data object and saved to {filename}\nProcess Complete.")


def check_data_count(year_list, df):
    """
    Checks to ensure that every yeat in year_list has 355 or 366 days
    and that every day has 1440 observations (1 observation per
    minute)

    Parameters:
        year_list: (list) List of years (formatted as ints) to check
        df: (pandas dataframe)
        verbose: (boolean, default=True) Will print output if True

    Returns:
        list of tuples where the first element in the tuple is the year and the second element is the number of days in that year
    """
    out = []

    for year in year_list:
        subset = df[df["Year"] == year]
        days = (subset.groupby("DOY").count() == 1440).sum()[0]
        out.append((year, days))

    return out


def make_dates(year_list, df):
    """
    Sets the index of df to be a datetime object

    Parameters:
        df: (pandas dataframe) Dataframe to be changed

    Return:
        updated_df: (pandas dataframe) Dataframe with datetime as index
    """
    year_day_list = check_data_count(year_list, df)

    for pair in year_day_list:
        for day in range(pair[1]):
            mask = df['Year'] == pair[0]
            mask2 = df['DOY'] == day
            df[mask & mask2]['Hour'] = df[mask & mask2].reset_index().index // 60

    # for year in year_list:
    #     mask = df['Year'] == 2014
    #     mask2 = df[mask]['DOY'] == 45
    #
    #     for day in days:
    #         mask2 = df[mask]['DOY'] == 50
    #         df['Hour'] = df[mask & mask2].reset_index().index // 60
            # hours = df[mask & mask2].reset_index().index // 60
    return df

            # for hour in np.unique(hours):
            #     mask3 = np.where(hours == 12, True, False)
            #     df[mask & mask2 & mask3]
            #     minute_subset = minute_subset.reset_index()
            #     minute_subset["Minute"] == minute_subset.index



    # for year in year_list:
    #     subset = df[df['Year'] == year]
    #     days = np.unique(subset['DOY'])
    #     for day in days:
    #         nested_subset = subset[subset['DOY'] == day]
    #         nested_subset = nested_subset.reset_index
    #         nested_subset["Hour"] = nested_subset.index // 60
    #         hours = np.unique(nested_subset['Hour'])
    #         for hour in hours:
    #             minute_subset = nested_subset[nested_subset['Hour'] == hour]
    #             minute_subset = minute_subset.reset_index()
    #             minute_subset["Minute"] == minute_subset.index

if __name__ == "__main__":

    df = build_master_csv("../data/solar_measurements/", "../data/ivanpah_measurements.csv")

    # df = pd.read_csv("../data/ivanpah_measurements.csv")
    #
    # df.drop('Unnamed: 0.1', axis=1, inplace=True)

    # years = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
    #
    # year_days = check_data_count(years, df)
    #
    # df = make_dates(years, df)
