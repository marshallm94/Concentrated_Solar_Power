import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('ggplot')

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
            df['Minute'] = minute_array[:df.shape[0]]
            df["Date"] = date
            df['final_date'] = df['Date'].astype(str) + " " + df['Hour'].astype(str) + ":" + df['Minute'].astype(str)
            df['final_date'] = pd.to_datetime(df['final_date'])
            df.set_index('final_date', inplace=True)
        else:
            current_df = pd.read_csv(current_file, header=None, names=col_names)
            current_df["Hour"] = current_df.index // 60
            current_df['Minute'] = minute_array[:current_df.shape[0]]
            current_df["Date"] = date
            current_df['final_date'] = current_df['Date'].astype(str) + " " + current_df['Hour'].astype(str) + ":" + current_df['Minute'].astype(str)
            current_df['final_date'] = pd.to_datetime(current_df['final_date'])
            current_df.set_index('final_date', inplace=True)
            df = pd.concat([df, current_df], axis=0)

        print(f"Final DataFrame Shape: {df.shape}")

    df.drop('Unnamed: 0', axis=1, inplace=True)
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


def get_master_df(filename):

    df = pd.read_csv(filename)
    df['final_date'] = pd.to_datetime(df['final_date'])
    df['Time'] = df['final_date'].hour + df['final_date'].minute
    df.set_index('final_date', inplace=True, drop=False)
    return df


def plot_day(date, hour_range, variable):
    """
    Plots the value of the variable specified on a given day within
    a given range of time.


    Parameters:
        date: (str) The date to be plotted, in the format YYYY-MM-DD
        variable: (str) The variable to be plotted across the day specified
    """
    mask = df['Date'] == date
    mask2 = df['Hour'] >= hour_range[0]
    mask3 = df['Hour'] <= hour_range[1]
    subset = df[mask & mask2 & mask3]
    plt.plot(subset['final_date'], subset[variable])
    plt.xticks(rotation=75)
    plt.show()


if __name__ == "__main__":

    build_master_csv("../data/solar_measurements/", "../data/ivanpah_measurements.csv")

    df = get_master_df("../data/ivanpah_measurements.csv")
    #
    # subset = df[df['Date'] == '2006-06-15']
    #
    # plot_day('2006-06-15',(4, 20), 'Direct Normal [W/m^2]')
    #
    # pd.to_datetime(subset['final_date'], format="%H:%M")
