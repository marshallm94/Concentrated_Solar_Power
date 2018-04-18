import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from manipulation import get_master_df
import datetime

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
    df[f'DNI_T_plus{num_lagged_features}'] = target[num_lagged_features:]

    return df


def create_X_y(df):
    """
    Creates a subset of df where only dates equal to date are included

    Parameters:
        date: (str) The date for the beginning of the date range,
                    in YYYY-MM-DD
        df: (pandas dataframe)

    Returns:
        df: (pandas dataframe) A subset of df
    """

    y = df.pop('DNI_T_plus15').values

    X = df[['Year',
            'Month',
            'DOY',
            'Hour',
            'Minute',
            'Direct Normal [W/m^2]',
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


def cross_validate(model, cv_iter, df):
    """
    A custom cross_validation function for predicting DNI 15 minutes
    out from the current time. The model is trained on 90 days worth
    of data, 1440 observations per day (1 per minute). The model is
    then tested on the next 30 days immediately after the training
    days.

    cv_iter random dates will be selected from the df provided. For
    each of those dates, the training dataset will be that date and
    the previous 89 days. The testing dataset will be the 30 days
    immediately following the random date selected (not including
    that day.)

    Parameters:
        model: (sklearn model object) A model object that implements
               fit() and predict() methods
        cv_iter: (int) Number of iterations to do (also number of
                 random days that will be selected)
        df: (pandas dataframe)

    Returns:
        rmses: (list) List of Root Mean Squared Errors, one for each
                iteration.

    """
    days = np.unique(df['Date'])
    CV_subset = np.random.choice(days, cv_iter)

    rmses = []
    test_periods = []
    train_periods = []
    for day in CV_subset:
        # cross validation training set
        train_start = pd.to_datetime(day) + pd.Timedelta("-90 days")
        train_end = pd.to_datetime(day)
        mask1 = df['final_date'] >= pd.to_datetime(train_start)
        mask2 = df['final_date'] < pd.to_datetime(train_end)
        x_train, y_train = create_X_y(df[mask1 & mask2])

        # cross_validation testing set
        test_end = pd.to_datetime(day) + pd.Timedelta("+31 days")
        test_start = pd.to_datetime(day) + pd.Timedelta("+1 days")
        mask3 = df['final_date'] >= pd.to_datetime(test_start)
        mask4 = df['final_date'] < pd.to_datetime(test_end)
        x_test, y_test = create_X_y(df[mask3 & mask4])

        model.fit(x_train, y_train)
        y_hat = model.predict(x_test)
        print("Test RMSE: {}".format(np.sqrt(mean_squared_error(y_hat, y_test))))
        rmses.append(np.sqrt(mean_squared_error(y_hat, y_test)))
        test_periods.append((test_start, test_end))
        train_periods.append((train_start, train_end))

    print("Average RMSE over {} splits: {}".format(cv_iter, np.mean(rmses)))

    return rmses, test_periods, train_periods


if __name__ == "__main__":

    df = get_master_df("../data/ivanpah_measurements.csv")

    df = engineer_lagged_DNI_features(15, df)

    train = df[df['Year'] < 2017]
    test = df[df['Year'] >= 2017]

    rf = RandomForestRegressor()
    cv_errors, cv_test_periods, cv_train_periods = cross_validate(rf, 10, train)

    # Persistence Model
    np.mean(np.sqrt(mean_squared_error(train['DNI_T_plus15'].values, train['Direct Normal [W/m^2]'].values)))

    # df.to_csv("../data/ivanpah_measurements_modeling.csv")
    #
    # df = get_master_df("../data/ivanpah_measurements_modeling.csv")

    # create display data
    mask = train['Date'] == '2015-06-11'
    mask2 = train['Hour'] > 8
    mask3 = train['Hour'] <= 18
    display_data = train[mask2 & mask3][['final_date','Direct Normal [W/m^2]','DNI_T_plus15']].head(61)
    display_data.to_csv("../data/display_data.csv")
