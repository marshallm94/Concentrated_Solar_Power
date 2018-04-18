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
        rmses: (list) List of Root Mean Squared Errors of model
                predictions, one value for each iteration.

        test_periods: (list) List of tuples, element 0 being the
                        start date of the test period, element 1
                        being the end date of the test period.

        train_periods: (list) List of tuples, element 0 being the
                        start date of the training period, element
                        1 being the end date of the training period.

        persistence_model: (list) List of Root Mean Squared Errors
                            of the Persistence Model, one value for
                            each iteration.
    """
    days = np.unique(df['Date'])
    CV_subset = np.random.choice(days, cv_iter)

    rmses = []
    test_periods = []
    train_periods = []
    persistence_model = []
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

        pers_mod_y_hat_test = df[mask3 & mask4]['Direct Normal [W/m^2]']

        model.fit(x_train, y_train)
        y_hat = model.predict(x_test)

        rmse_test_error = np.sqrt(mean_squared_error(y_test, y_hat))
        pm_test_error = np.sqrt(mean_squared_error(y_test, pers_mod_y_hat_test))

        print("{} RMSE: {:.4f} | Persistence Model RMSE: {:.4f}".format(model.__class__.__name__, rmse_test_error, pm_test_error))

        rmses.append(rmse_test_error)
        persistence_model.append(pm_test_error)
        test_periods.append((test_start, test_end))
        train_periods.append((train_start, train_end))

    print("Average RMSE over {} splits: {:.4f}".format(cv_iter, np.mean(rmses)))

    return rmses, test_periods, train_periods, persistence_model


def error_plot(y_dict, colors, title, xlab, ylab, savefig=False):
    """
    Plots the errors associated with the output of cross_validate()

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

def get_persistence_model_predictions(df):

    return df['Direct Normal [W/m^2]']

if __name__ == "__main__":

    df = get_master_df("../data/ivanpah_measurements.csv")

    df = engineer_lagged_DNI_features(15, df)

    train = df[df['Year'] < 2017]
    test = df[df['Year'] >= 2017]

    # set seed
    np.random.seed(10000)

    # base model
    rf = RandomForestRegressor()
    cv_errors, cv_test_periods, cv_train_periods, pm_errors = cross_validate(rf, 10, train)

    base_model_evaluation = zip(cv_train_periods, cv_test_periods, cv_errors, pm_errors)
    base_model_overview = []
    for triplet in base_model_evaluation:
        base_model_overview.append([str(triplet[0][0]), str(triplet[0][1]), str(triplet[1][0]), str(triplet[1][1]), triplet[2], triplet[3]])
    base_model_df = pd.DataFrame(base_model_overview, columns=['Train_Start','Train_End','Test_Start','Test_End','Test_Error','Persistent_Model_error'])
    base_model_df.to_csv("../data/random_forest_base_model_cv.csv")

    # Persistence Model
    np.mean(np.sqrt(mean_squared_error(train['DNI_T_plus15'].values, train['Direct Normal [W/m^2]'].values)))

    # create display data
    mask = train['Date'] == '2015-06-11'
    mask2 = train['Hour'] > 10
    mask3 = train['Hour'] <= 11
    display_data = train[mask & mask2 & mask3][['final_date','Direct Normal [W/m^2]','DNI_T_plus15']].head(61)
    display_data = display_data.set_index(np.arange(display_data.shape[0]))

    # used in Predicting_DNI.md
    top = display_data.loc[:6,:]
    bottom = display_data.loc[15:21,]
    final_display = pd.concat([top, bottom])

    # plot CV errors
    error_dict = {"Random Forest Error": base_model_df["Test_Error"].values,
                  "Persistence Model Error": base_model_df["Persistent_Model_error"].values
    }
    error_plot(error_dict, ['red','orange'], "Random Forest vs. Persistence Model Errors", "Cross Validation Period", r"$\frac{Watts}{Meter^2}$", "../images/cross_validation_plot.png")
