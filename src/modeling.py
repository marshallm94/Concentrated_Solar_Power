import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from manipulation import get_master_df

def engineer_lagged_DNI_features(num_lagged_features, df):
    """
    Creates new variables that have the Direct Normal [W/m^2] value
    from the time stamp X minutes ago.

    Example:
        The value of df[1, 'DNI_T_minus1'] will be equal to
        df[0, 'Direct Normal [W/m^2]']. Similarly, the value of
        df[2, 'DNI_T_minus2'] will be equal to the value of
        df[0, 'Direct Normal [W/m^2]'].

    Parameters:
        num_lagged_features: (int) The number of minutes each
                             observation should look back (will be
                             the number of new columns created as
                             well)
        df: (pandas dataframe)

    Return:
        df: (pandas dataframe) df with additional columns.
    """
    lagged_steps = np.arange(1, num_lagged_features + 1)
    feature_names = [f"DNI_T_minus{i}" for i in lagged_steps]

    for x, i in enumerate(lagged_steps):
        base = df['Direct Normal [W/m^2]'].copy().values
        values = np.insert(base, np.repeat(0,i), np.repeat(0,i))
        df[feature_names[x]] = values[:-i]

    dni_copy = df['Direct Normal [W/m^2]'].copy().values
    target = np.insert(dni_copy, np.repeat(dni_copy.shape[0], num_lagged_features), np.repeat(0, num_lagged_features))
    print(dni_copy[-15:], target[-15:])
    df[f'DNI_T_plus{num_lagged_features}'] = target[num_lagged_features:]

    return df


def create_day_subset(date, df):
    """
    Creates a subset of df where only dates equal to date are included

    Parameters:
        date: (str) The date for the beginning of the date range,
                    in YYYY-MM-DD
        df: (pandas dataframe)

    Returns:
        df: (pandas dataframe) A subset of df
    """
    out = df[df['Date'] == date]

    y = out.pop('Direct Normal [W/m^2]').values

    X = out[['Year',
            'Month',
            'DOY',
            'Hour',
            'Minute',
            'DNI_T_minus1',
            'DNI_T_minus2',
            'DNI_T_minus3',
            'DNI_T_minus4',
            'DNI_T_minus5',
            'DNI_T_minus6',
            'DNI_T_minus7',
            'DNI_T_minus8',
            'DNI_T_minus9',
            'DNI_T_minus10',
            'DNI_T_minus11',
            'DNI_T_minus12',
            'DNI_T_minus13',
            'DNI_T_minus14',
            'DNI_T_minus15']].values

    return X, y

if __name__ == "__main__":

    df = get_master_df("../data/ivanpah_measurements.csv")

    subset = df[df['Date'] == '2007-07-04']

    test = engineer_lagged_DNI_features(15, subset)

    (test['Direct Normal [W/m^2]'] == test['DNI_T_plus15']).sum()
    # df.to_csv("../data/ivanpah_measurements_modeling.csv")
    #
    # df = pd.read_csv("../data/ivanpah_measurements_modeling.csv")
    #
    # X, y = create_day_subset('2007-07-04', df)
    #
    # rf = RandomForestRegressor()
    # rf.fit(X, y)
    # y_hat = rf.predict(X)
    # print(np.sqrt(mean_squared_error(y_hat, y)))
