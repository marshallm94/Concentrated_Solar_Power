import pandas as pd
import numpy as np
import calendar
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from manipulation import get_master_df, plot_day
import train_mlp_base as tmb
from eda import format_nrel_dataframe
from datetime import datetime
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.objectives import MSE, MAE
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
import matplotlib.dates as mdates
import random


def create_X_y(df, columns, target, date, num_units, units, same=True):
    '''
    Creates a subset of df for training a model.

    Parameters:
    ----------
    df : (Pandas DataFrame)
    columns : (list)
        A list of strings specifying which columns should be included in the X
        matrix as predictive attributes
    target : (str)
        Target column within df
    date : (pandas._libs.tslib.Timestamp)
    num_units : (int)
        Specifies the number of units to used
    units : (str)
        Units that specify the time-period for the data set to option are:
            'years',
            'months',
            'weeks',
            'days',
            'hours'
    same : (bool)
        If True, units will go back incrementally by years.
        example: if date = "2008-08-13", num_units = 2, units = "months" and
        same = True, then the data set will be composed of data from August
        (8th month) of 2007 and 2006.

    Returns:
    ----------
    X : (Pandas DataFrame)
        A DataFrame with the values of the columns specified
    y : (Pandas Series)
        A 1 dimensional Seriess with the values of the target column
    '''
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


def create_lagged_features(df, columns, shift, row_time_steps):
    '''
    Creates new variables that are current variables "shifted," allowing
    sequential information to be contained in a DataFrame that can be passed to
    more than time-series models.

    Note that the dimensions of the returned DataFrame will be:

        N rows X (df.shape[1] + (len(columns) * shift)) columns

    Parameters:
    ----------
    df : (Pandas DataFrame)
        Must contain all columns specified by the columns parameter
    columns : (list)
        A list of columns for which to create lagged variables. (Note that even
        if only one column is specified, it must be contained within a list)
    shift : (int)
        The number of observations to shift forward
    row_time_steps : (int)
        The time difference (in minutes) from one observation to the next in
        the DataFrame specified

    Returns:
    ----------
    df : (Pandas DataFrame)
        DataFrame with new columns added
    '''
    out = df.copy()
    for col in columns:
        feature_names = [f"{col}_T_minus_{row_time_steps*i}" for i in range(1, shift + 1)]
        base = out[col].copy().values
        for x, new_col in enumerate(feature_names):
            x += 1
            values = np.insert(base, np.repeat(0, x), np.repeat(0, x))
            out[new_col] = values[:-x]

    return out


def create_future_target(df, column, shift, row_time_steps):
    '''
    Creates a target column for modeling that is the future values of the column
    specified. The number of the new column specifies the change in minutes.

    Example:
        [1]: new = create_future_target(df, 'DNI', 5, 10)
         # This is saying that there is a 10 minute difference from one
           one observation to the next. The target column that will be created
           will be called 'DNI_T_plus50', meaning that for an observation at
           time-stamp X, the value of 'DNI_T_plus50' is the value of 'DNI' 50
           minutes from time-stamp X.

    Parameters:
    ----------
    df : (Pandas DataFrame)
        Must contain column specified by the columns parameter
    column : (str)
        Name of the column whose values will be shifted to create a new target
        column
    shift : (int)
        The number of observations to shift backward
    row_time_steps : (int)
        The time difference (in minutes) from one observation to the next in
        the DataFrame specified

    Returns:
    ----------
    df : (Pandas DataFrame)
        DataFrame with new column added
    '''
    out = df.copy()
    base = out[column].copy().values
    target = np.insert(base, np.repeat(base.shape[0], shift), np.repeat(0, shift))
    target_column_name = f"{column}_T_plus{shift * row_time_steps}"
    out[target_column_name] = target[shift:]
    return out


def sequential_impute(df, col):
    '''
    Replaces NaN values with the value most recently seen in the data set.
    Assumes DataFrame has a sequential aspect to it.

    Parameters:
    ----------
    df : (Pandas DataFrame)
        DataFrame containing col
    col : (str)
        The column name within df for which sequential imputing will be
        performed

    Returns:
    ----------
    df : (Pandas DataFrame)
    '''
    out = df.copy()
    values = out[col].values


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


def test_model(model, X, y):
    '''
    Evaluates the model specified using 5-fold cross validation and tests model
    on unseen data.

    Parameters:
    ----------
    model : (object)
        Machine Learning object that implements both .fit() and .predict()
    X : (Pandas DataFrame)
        Contains attributes on which the model will be built.
    y : (Pandas Series)
        Target variable

    Returns:
    ----------
    mae : (float)
        Testing Mean Absolute Error
    scores : (dict)
        Cross validation scores
    '''
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    scores = cross_validate(model, x_train, y_train, scoring='neg_mean_absolute_error', verbose=0, n_jobs=-1, cv=5)
    model.fit(x_train, y_train)
    y_hat = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_hat)
    return mae, scores


def error_plot(y_dict, colors, title, xlab, ylab, savefig=False):
    """
    Plots the errors associated with the output of test_model()

    Parameters:
        y_dict: (dict) Keys equal the name of the error array
                     (i.e. Random Forest, Persistence, True)
        colors: (list) List of strings (must same length as y_dict)
        title: (str) Title for plot
        xlab: (str) Label for X axis
        ylab: (str) Label for Y axis
        savefig: (boolean/str) Will show the figure if False
                 (default), will save the figure if a string (str).

    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=(12,8))
    counter = 0

    for name, array in y_dict.items():
        ax.plot(array, c=colors[counter], label=f"{name}")
        counter +=1

    plt.xlabel(xlab, fontweight='bold', fontsize=16)
    plt.ylabel(ylab, fontweight='bold', rotation=0, fontsize=16)
    ax.yaxis.set_label_coords(-0.105,0.5)
    plt.suptitle(title, fontweight='bold', fontsize=18)
    ax.legend()
    if savefig:
        plt.savefig(savefig)
    else:
        plt.show()


def build_neural_network(n_predictors, hidden_layer_neurons):
    """
    Builds a Multi-Layer-Perceptron utilizing Keras.

    Parameters:
        x_train: (2D numpy array) A n x p matrix, with n observations
                 and p features
        y_train: (1D numpy array) A numpy array of length n with the
                 target training values.
        hidden_layer_neurons: (list) List of ints for the number of
                              neurons in each hidden layer.

    Returns:
        model: A MLP with 2 hidden layers
    """
    model = Sequential()
    input_layer_neurons = n_predictors

    model.add(Dense(units=hidden_layer_neurons[0],
                    input_dim=input_layer_neurons,
                    kernel_initializer='uniform',
                    activation='relu'))

    model.add(Dense(units=hidden_layer_neurons[1],
                    kernel_initializer='uniform',
                    activation='relu'))

    model.add(Dense(units=1))

    model.compile(optimizer='rmsprop',
                  loss='mean_squared_error')

    return model


if __name__ == "__main__":

    df = format_nrel_dataframe("../data/2003_2016.csv")
    lag_features = ['Temperature', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'Cloud Type','Dew Point','DHI','DNI','Fill Flag', 'GHI','Relative Humidity','Solar Zenith Angle','Surface Albedo','Pressure','Precipitable Water','Wind Direction','Wind Speed']
    df = create_lagged_features(df, lag_features, 4, 30)
    df = create_future_target(df, 'DNI', 1, 30)

    test_dates = tmb.get_random_test_dates(5, 2017, (6,18), 2)

    rf = RandomForestRegressor()

    X, y = create_X_y(df, df.columns, 'DNI_T_plus30', test_dates[0], 3, 'months', same=True)

    X.drop(['final_date','Date'], axis=1, inplace=True)
    # X.set_index(np.arange(X.shape[0]), inplace=True)
    mae, scores = test_model(rf, X, y)
