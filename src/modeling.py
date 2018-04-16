import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from manipulation import get_master_df

def engineer_lagged_DNI_features(df):
    """
    Creates 15 new columns
    """
    lagged_steps = np.arange(1, 16)
    feature_names = [f"DNI_T_minus{i}" for i in lagged_steps]

    for x, i in enumerate(lagged_steps):
        base = df['Direct Normal [W/m^2]'].copy().values
        values = np.insert(base, np.repeat(0,i), np.repeat(0,i))
        df[feature_names[x]] = values[:-i]

    return df


if __name__ == "__main__":

    df = get_master_df("../data/ivanpah_measurements.csv")

    mask = df["Year"] == 2017
    mask2 = df['Date'] == "2017-07-04"
    subset = df[mask & mask2]

    test = engineer_lagged_DNI_features(subset)

    test[['Direct Normal [W/m^2]','DNI_T_minus1','DNI_T_minus2','DNI_T_minus3']]
