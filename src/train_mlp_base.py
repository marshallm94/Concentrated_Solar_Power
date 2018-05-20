import pandas as pd
import numpy as np
import calendar
import random
from manipulation import get_master_df, plot_day
from datetime import datetime
from modeling import build_neural_network
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
from keras.objectives import MSE, MAE
from keras.models import Sequential
from keras.layers.core import Activation, Dense


def create_X_y(df, columns, target, date, num_units, units, same=True):
    """
    Creates a subset of df for training model

    Parameters:
        df: (pandas dataframe) Master dataframe
        columns: (list) A list of strings specifying which columns
                 should be included in the X matrix as predictive
                 attributes
        target: (str) Target column within df
        date: (pandas._libs.tslib.Timestamp)
        num_units: (int) Specifies the number of units to used
        units: (str) Units that specify the time-period
                     for the data set to
                     option are:
                        'years',
                        'months',
                        'weeks',
                        'days',
                        'hours'
        same: (boolean - default = True) If True, units will go back incrementally by years.
                example: if date = "2008-08-13", num_units = 2, units = "months" and same = True, then the data set will be composed of data from August (8th month) of 2007 and 2006.

    Returns:
        X: (Numpy array) A 2 dimensional numpy array with the values
           of the columns specified
        y: (Numpy array) A 1 dimensional numpy array with the values
           of the target column
    """
    date_dt = datetime.strptime(datetime.strftime(date, "%Y-%m-%d"), "%Y-%m-%d")

    if units == 'years':
        available_years = set(df['Year'])
        if date.year - num_units not in available_years:
            print("\nStart year not in data set")
            return None, None
        else:
            train_start = date.replace(year=date.year - num_units)

            mask1 = df['final_date'] >= pd.to_datetime(train_start)
            mask2 = df['final_date'] < date

            y = df[mask1 & mask2].pop(target)
            X = df[mask1 & mask2][columns]
            return X, y

    elif units == 'months':
        if same:
            start_year = date.year - num_units

            mask1 = df['Year'] >= start_year
            mask2 = df['Year'] <= date.year
            mask3 = df['Month'] == date.month
            mask4 = df['final_date'] < date

            y = df[mask1 & mask2 & mask3 & mask4].pop(target)
            X = df[mask1 & mask2 & mask3 & mask4][columns]
            return X, y

        if not same:
            if date_dt.month - num_units < 1:
                start_month = date_dt.month + 12 - num_units
                start_year = date.year - 1
                start_date = date_dt.replace(month=start_month, year=start_year)
                start_date = pd.to_datetime(start_date)
                print(start_date)
                mask1 = df['final_date'] >= start_date
                mask2 = df['final_date'] < date

                y = df[mask1 & mask2].pop(target)
                X = df[mask1 & mask2][columns]
                return X, y
            else:
                start_month = date_dt.month - num_units
                start_date = date_dt.replace(month=start_month)
                start_date = pd.to_datetime(start_date)

                mask1 = df['final_date'] >= start_date
                mask2 = df['final_date'] < date

                y = df[mask1 & mask2].pop(target)
                X = df[mask1 & mask2][columns]
                return X, y

    elif units == 'weeks':
        if same:
            start_date = date + pd.Timedelta("-3 days")
            end_date = date + pd.Timedelta("+3 days")
            days = list(pd.date_range(start_date, end_date))

            if start_date.year < end_date.year:
                years = [(year, year + 1) for year in range(start_date.year - num_units, end_date.year)]
                days2 = []
                for start_year, end_year in years:
                    for day in days:
                        if day.month == 12:
                            new_day = day.replace(year=start_year)
                            days2.append(new_day)
                        else:
                            new_day = day.replace(year=end_year)
                            days2.append(new_day)
                days2 = [datetime.strftime(i, "%Y-%m-%d") for i in days2]

            elif start_date.year > end_date.year:
                years = [(year, year + 1) for year in range(start_date.year - num_units, end_date.year)]
                days2 = []
                for start_year, end_year in years:
                    for day in days:
                        if day.month == 12:
                            new_day = day.replace(year=start_year)
                            days2.append(new_day)
                        else:
                            new_day = day.replace(year=end_year)
                            days2.append(new_day)
                days2 = [datetime.strftime(i, "%Y-%m-%d") for i in days2]
            else:
                years = [year for year in range(date.year - num_units, date.year)]
                for year in years:
                    start_date = (date + pd.Timedelta("-3 days")).replace(year=year)
                    end_date = (date + pd.Timedelta("+3 days")).replace(year=year)
                    date_range = pd.date_range(start_date, end_date)
                    for day in date_range:
                        days.append(day)
                days2 = [datetime.strftime(i, "%Y-%m-%d") for i in days]
            frames = []
            for day in days2:
                frame = df[df['Date'] == day]
                frames.append(frame)
            final = pd.concat(frames)
            final = final[final['final_date'] < date]
            y = final.pop(target)
            X = final[columns]
            return X, y

        if not same:
            train_start = pd.to_datetime(date) + pd.Timedelta("-{} days".format(num_units * 7))

            mask1 = df['final_date'] >= pd.to_datetime(train_start)
            mask2 = df['final_date'] < date

            y = df[mask1 & mask2].pop(target)
            X = df[mask1 & mask2][columns]
            return X, y

    elif units == 'days':
        if same:

            dates = []
            for year in range(1, num_units + 1):
                new_date = date.replace(year=date.year - year)
                dates.append(new_date.strftime("%Y-%m-%d"))

            frames = []
            for day in dates:
                frame = df[df['Date'] == day]
                frames.append(frame)

            final = pd.concat(frames)
            final = final[final['final_date'] < date]

            y = final.pop(target)
            X = final[columns]
            return X, y

        if not same:
            train_start = date + pd.Timedelta(f"-{num_units} days")
            mask1 = df['final_date'] >= train_start
            mask2 = df['final_date'] < date

            y = df[mask1 & mask2].pop(target)
            X = df[mask1 & mask2][columns]
            return X, y

    elif units == 'hours':

        start = date + pd.Timedelta(f"-{num_units} hours")
        mask1 = df['final_date'] >= pd.to_datetime(start)
        mask2 = df['final_date'] < pd.to_datetime(date)

        y = df[mask1 & mask2].pop(target)
        X = df[mask1 & mask2][columns]
        return X, y


def create_lagged_DNI_features(num_lagged_features, df):
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


columns = ['Year',
        'Month',
        'DOY',
        'Hour',
        'Minute',
        'Direct Normal [W/m^2]',
        'Zenith Angle [degrees]',
        'Azimuth Angle [degrees]',
        'Airmass',
        'Avg Wind Speed @ 30ft [m/s]',
        'Avg Wind Direction @ 30ft [deg from N]',
        'Peak Wind Speed @ 30ft [m/s]',
        'Wind Chill Temp [deg C]',
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
        'DNI_T_minus15']

# stop_criteria = EarlyStopping(monitor='val_loss', min_delta=0.00001)

hidden_layer_neurons = [10, 40]

NN_dict = {'epochs': 38,
           'batch_size': 17,
           'shuffle': True,
           'validation_split': 0.2,
           # 'callback': stop_criteria
}

mlp = build_neural_network(len(columns), hidden_layer_neurons)

test_dates = get_random_test_dates(5, 2017, (8, 18), 2)

def train_mlp(X, y, model=mlp, NN_dict=NN_dict):
    """
    Trains a Muli-Layer Perceptron and returns the training RMSE.

    Parameters:
    ----------
    X: (numpy array) 2 Dimensional matrix
    y: (numpy array) response values for training data
    model: (Keras Neural Network Model)
    NN_dict: (dict) Neural Network training criteria

    Returns:
    -------
    RMSE: (float) Training RMSE
    """
    model.fit(X,
              y,
              epochs=NN_dict['epochs'],
              batch_size=NN_dict['batch_size'],
              shuffle=NN_dict['shuffle'],
              validation_split=NN_dict['validation_split'],
              # callbacks=[NN_dict['callback']],
              verbose=1)
    y_hat = model.predict(X)
    return np.sqrt(mean_squared_error(y, y_hat))
