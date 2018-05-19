import pandas as pd
import numpy as np
from manipulation import heatmap
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pandas.plotting import scatter_matrix
plt.style.use('ggplot')


def format_nrel_dataframe(filepath):
    '''
    Properly formats a csv file created by running the script 'nrel_api.py'.

    Parameters:
    ----------
    filepath : (str)
        Absolute or relative location of the .csv file

    Returns:
    ----------
    df : (Pandas DataFrame)
    '''
    df = pd.read_csv(filepath)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df['final_date'] = pd.to_datetime(df['Unnamed: 0.1'])
    df.drop('Unnamed: 0.1', axis=1, inplace=True)
    df.set_index('final_date', inplace=True, drop=False)
    df['Date'] = pd.to_datetime(df['final_date'].values).strftime("%Y-%m-%d")
    df['DOY'] = pd.to_datetime(df['final_date'].values).dayofyear
    return df


def plot_day(date, hour_range, variables, xlab, ylab, df, savefig=False):
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
    fig, ax = plt.subplots(figsize=(12,8))
    for variable in variables:
        ax.plot(subset[variable], '-o', label=variable)
        ax.xaxis.set_major_locator(hours)
        ax.xaxis.set_major_formatter(hour_formatter)
    plt.xlabel(xlab, fontweight="bold", fontsize=16)
    plt.ylabel(ylab, fontweight="bold", rotation=0, fontsize=16)
    ax.yaxis.set_label_coords(-0.105,0.5)
    plt.legend()
    plt.suptitle(datetime.strptime(date, "%Y-%m-%d").strftime("%B, %d %Y"), fontweight='bold', fontsize=18)
    plt.show()
    if savefig:
        plt.savefig(savefig)


def distribution_plot(df, column_name, target_column, xlab, ylab, title, filename=False, plot_type="box", order=None):
    """
    Create various plot types leverage matplotlib.
    Inputs:
        df: (Pandas DataFrame)
        column_name: (str) - A column in df that you want to have on the x-axis
        target_column: (str) - A column in df that you want to have on the y_axis
        xlab, ylab, title: (all str) - Strings for the x label, y label and title of the plot, respectively.
        filename: (str) - the relative path to which you would like to save the image
        plot_type: (str) - "box", "violin" or "bar"
        order: (None (default) or list) - the ordering of the variable on the x-axis
    Output:
        None (displays figure and saves image)
    """
    fig = plt.figure(figsize=(13,6))
    ax = fig.add_subplot(111)
    if plot_type == "box":
        ax = sns.boxplot(df[column_name], df[target_column], order=order)
    elif plot_type == "violin":
        ax = sns.violinplot(df[column_name], df[target_column])
    elif plot_type == "bar":
        ax = sns.barplot(df[column_name], df[target_column], palette="Greens_d", order=order)
    ax.set_xlabel(xlab, fontweight="bold", fontsize=14)
    ax.set_ylabel(ylab, fontweight="bold", fontsize=14)
    plt.xticks(rotation=75)
    plt.suptitle(title, fontweight="bold", fontsize=16)
    if filename:
        plt.savefig(filename)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


if __name__ == "__main__":

    df = format_nrel_dataframe("../data/2003_2016.csv")

    heatmap(df, "../images/heatmap.png")

    plot_day("2016-12-05", (4, 20), ['DNI', 'Solar Zenith Angle','Pressure', 'Relative Humidity'], r"$Hour$", r"$\frac{Watts}{Meter^2}$", df, "../images/irradiance_20161205")
