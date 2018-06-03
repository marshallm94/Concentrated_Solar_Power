import pandas as pd
import numpy as np
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
    df['Cloud Type'] = np.nan_to_num(df['Cloud Type'].values, copy=False)
    return df


def plot_day(date, hour_range, variables, xlab, ylab, df, savefig=False):
    '''
    Plots the value of the variable specified on a given day within
    a given hour range.


    Parameters:
    ----------
    date : (str)
        The date to be plotted, in the format YYYY-MM-DD
    hour_range : (tuple)
        Tuple of integers for the start and stop hours during the day (in 24
        hour format i.e. 11pm = 23)
    variables : (list)
        List of variables to be plotted across the day specified (Note that
        even if only one variable is specified it must be contained in a list)
    df : (Pandas DataFrame)
        Contains variables specified

    Returns:
    ----------
    None
    '''
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
    plt.xlabel(xlab, fontweight='bold', fontsize=19)
    plt.ylabel(ylab, fontweight='bold', rotation=0, fontsize=19)
    ax.tick_params(axis='both', labelcolor='black', labelsize=15.0)
    ax.yaxis.set_label_coords(-0.105,0.5)
    plt.legend()
    plt.suptitle(datetime.strptime(date, "%Y-%m-%d").strftime("%B %d, %Y"), fontweight='bold', fontsize=21)
    if savefig:
        plt.savefig(savefig)
    else:
        plt.show()


def heatmap(df, filename=False):
    '''
    Creates a heatmap of the correlation matrix of a DataFrame.

    Parameters:
    ----------
    df : (Pandas DataFrame)
    filename : (str)
        The path to which you would like to save the image.

    Returns:
    ----------
    None
    '''
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
    else:
        plt.show()


if __name__ == "__main__":

    df = format_nrel_dataframe("../data/2003_2016.csv")

    heatmap(df, "../images/heatmap.png")

    plot_day("2016-12-05", (4, 20), ['DNI', 'Solar Zenith Angle','Pressure', 'Relative Humidity'], r"$Hour$", r"$\frac{Watts}{Meter^2}$", df, "../images/irradiance_20161205")

    plot_day("2016-07-04", (4, 20), ['DNI', 'Solar Zenith Angle','Pressure', 'Relative Humidity'], r"$Hour$", r"$\frac{Watts}{Meter^2}$", df, "../images/irradiance_20160704")
