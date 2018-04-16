import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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


def get_master_df(filename):

    df = pd.read_csv(filename)
    df['final_date'] = pd.to_datetime(df['final_date'])
    df["Time"] = df["Hour"].astype(str) + df["Minute"].astype(str)

    df.set_index('final_date', inplace=True, drop=False)
    return df


def plot_day(date, hour_range, variables, df):
    """
    Plots the value of the variable specified on a given day within
    a given hour range.


    Parameters:
        date: (str) The date to be plotted, in the format YYYY-MM-DD
        hour_range: (tuple) Tuple of integers for the start and stop hours during the day (in 24 hour format i.e. 11pm = 23)
        variables: (list) list of variables to be plotted across the day specified
        df: (DataFrame) Pandas DataFrame containing variables

    Returns:
        None
    """
    mask = df['Date'] == date
    mask2 = df['Hour'] >= hour_range[0]
    mask3 = df['Hour'] <= hour_range[1]
    subset = df[mask & mask2 & mask3]
    hours = mdates.HourLocator()
    hour_formater = mdates.DateFormatter('%I')
    fig, ax = plt.subplots()
    for variable in variables:
        ax.plot(subset[variable], label=variable)
        ax.xaxis.set_major_locator(hours)
        ax.xaxis.set_major_formatter(hour_formater)
    plt.xlabel("Time of Day", fontweight="bold", fontsize=16)
    plt.legend()
    plt.suptitle(datetime.strptime(date, "%Y-%m-%d").strftime("%B, %d %Y"))
    plt.show()


if __name__ == "__main__":

    # build_master_csv("../data/solar_measurements/", "../data/ivanpah_measurements.csv")
    #
    # df = get_master_df("../data/ivanpah_measurements.csv")

    plot_day('2006-06-15', (4, 20), ['Direct Normal [W/m^2]'], df)
