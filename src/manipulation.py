import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from pandas.plotting import scatter_matrix
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
    df.set_index('final_date', inplace=True, drop=False)
    df["Month"] = df.index.month
    df["Year"]
    return df


def plot_day(date, hour_range, variables, df, savefig=False):
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
    hour_formatter = mdates.DateFormatter('%I')
    fig, ax = plt.subplots()
    for variable in variables:
        ax.plot(subset[variable], label=variable)
        ax.xaxis.set_major_locator(hours)
        ax.xaxis.set_major_formatter(hour_formatter)
    plt.xlabel("Time of Day", fontweight="bold", fontsize=16)
    plt.legend()
    plt.suptitle(datetime.strptime(date, "%Y-%m-%d").strftime("%B, %d %Y"), fontweight='bold', fontsize=18)
    plt.show()
    if savefig:
        plt.savefig(savefig)


def plot_daterange_DNI(start_date, end_date, hour_range, groupby, variables, ylab, df, savefig=False):
    mask = df['Date'] >= start_date
    mask2 = df['Date'] <= end_date
    day_subset = df[mask & mask2]
    mask3 = df['Hour'] >= hour_range[0]
    mask4 = df['Hour'] <= hour_range[1]
    subset = day_subset[mask3 & mask4]
    subset = subset.groupby(groupby).mean()[variables]
    hours = mdates.HourLocator()
    fig, ax = plt.subplots(figsize=(12,8))
    for variable in variables:
        ax.plot(subset[variable], '-o', label=variable)
    plt.xlabel(groupby, fontweight="bold", fontsize=16)
    plt.ylabel(ylab, fontweight="bold", fontsize=16)
    plt.legend()
    start_title = datetime.strptime(start_date, "%Y-%m-%d").strftime("%B, %Y")
    end_title = datetime.strptime(end_date, "%Y-%m-%d").strftime("%B, %Y")
    plt.suptitle(str(start_title + " to " + end_title), fontweight='bold', fontsize=18)
    plt.show()
    if savefig:
        plt.savefig(savefig)


def heatmap(df, filename=False):
    """
    Creates a heatmap of the correlation matrix of df (Pandas DataFrame).
    Inputs:
        df: (Pandas DataFrame)
        filename: (str) - the path to which you would like to save the image.
    Output:
        None (displays figure and saves image)
    """
    corr = df.corr()
    ylabels = ["{} = {}".format(col, x + 1) for x, col in enumerate(list(corr.columns))]
    xlabels = [str(x + 1) for x in range(len(ylabels))]
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 6))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, xticklabels=xlabels, yticklabels=ylabels, vmax=0.3, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    plt.suptitle("Correlation Between Attributes", fontweight="bold", fontsize=16)
    if filename:
        plt.savefig(filename, orientation='landscape')


if __name__ == "__main__":

    # build_master_csv("../data/solar_measurements/", "../data/ivanpah_measurements.csv")
    #
    # df = get_master_df("../data/ivanpah_measurements.csv")

    correlation_df = df[['DOY',
                         'Direct Normal [W/m^2]',
                         'Global Horiz [W/m^2]',
                         'Global UVA [W/m^2]',
                         'Global UVE [W/m^2]',
                         'Dry Bulb Temp [deg C]',
                         'Avg Wind Speed @ 30ft [m/s]',
                         'Avg Wind Direction @ 30ft [deg from N]',
                         'Peak Wind Speed @ 30ft [m/s]',
                         'UVSAET Temp [deg C]',
                         'Logger Temp [deg C]',
                         'Logger Battery [VDC]',
                         'Wind Chill Temp [deg C]',
                         'Diffuse Horiz (calc) [W/m^2]',
                         'Zenith Angle [degrees]',
                         'Azimuth Angle [degrees]',
                         'Airmass']]

    heatmap(correlation_df, "../images/correlation_plot.png")

    plot_daterange_DNI('2017-01-01', '2017-12-31', (6, 16), 'Hour', ['Direct Normal [W/m^2]', 'Zenith Angle [degrees]', 'Global Horiz [W/m^2]','Diffuse Horiz (calc) [W/m^2]'], "Avg Across Days", df, "../images/avg_hourly_irradiance.png")

    plot_day('2006-06-15', (4, 20), ['Direct Normal [W/m^2]','Global Horiz [W/m^2]','Zenith Angle [degrees]'], df)
