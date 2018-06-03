import pandas as pd
import numpy as np
from modeling_base import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.objectives import MSE, MAE
from keras.models import Sequential
from keras import metrics as met
from keras.layers.core import Activation, Dense


def build_neural_network(n_predictors, hidden_layer_neurons, loss='mean_absolute_error'):
    '''
    Builds a Multi-Layer-Perceptron utilizing Keras.

    Parameters:
    ----------
    n_predictors : (int)
        The number of attributes that will be used as input for the model
        (i.e. the number of columns in a dataframe or array)
    hidden_layer_neurons : (list)
        List (length 2) of ints for the number of neurons in each hidden layer.
    loss : (str)
        The loss function for which the network will be optimized. Options
        are 'mean_squared_error' or 'mean_absolute_error'

    Returns:
    ----------
    model : (keras model object)
        A Multi-Layer Perceptron with 2 hidden layers
    '''
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
                  loss=loss,
                  metrics=['mse','mae'])

    return model


hidden_layer_neurons = [10, 40]

NN_dict = {'epochs': 38,
           'batch_size': 17,
           'shuffle': True,
           'validation_split': 0.2,
}

max_year = 2016

test_dates = get_random_test_dates(5, max_year, (4, 20), 2)

def test_nn_model(model, X, y, fit_params=NN_dict):
    '''
    Evaluates the model specified using 5-fold cross validation and tests model
    on unseen data.

    Parameters:
    ----------
    model : (object)
        Machine Learning object that implements both .fit() and .predict()
    X : (Pandas DataFrame)
        Contains attributes on which the model will be trained
    y : (Pandas Series)
        Target variable
    fit_params : (dictionary)
        Parameters to pass to the fit method

    Returns:
    ----------
    mae : (float)
        Testing Mean Absolute Error
    rmse : (float)
        Testing Root Mean Squared Error
    pm_mae : (float)
        Persistence model Mean Absolute Error
    pm_rmse : (float)
        Persistence model Root Mean Squared Error
    train_size : (int)
        Number of observations in the training dataset
    test_size : (int)
        Number of observations in the testing dataset
    '''
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    history = model.fit(x_train.values,
              y_train.values,
              epochs=fit_params['epochs'],
              batch_size=fit_params['batch_size'],
              shuffle=fit_params['shuffle'],
              validation_split=fit_params['validation_split'],
              verbose=1)

    y_hat = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_hat)
    rmse = np.sqrt(mean_squared_error(y_test, y_hat))

    pm_rmse = np.sqrt(mean_squared_error(y_test, x_test['DNI'].values))
    pm_mae = mean_absolute_error(y_test, x_test['DNI'].values)

    train_size = x_train.shape[0]
    test_size = x_test.shape[0]

    return mae, rmse, pm_mae, pm_rmse, train_size, test_size

def iterative_nn_testing(model, df, target_col, test_dates, num_units, units, fit_params=NN_dict, same=True):
    '''
    Iteratively tests model using test_model() for every date in test_dates

    Parameters:
    ----------
    model : (object)
        Machine Learning object that implements both .fit() and .predict()
    df : (Pandas DataFrame)
        DataFrame containing attributes that will be used to predict
        target column
    target_col : (str)
        The target column to be removed from the DataFrame and predicted on
    test_dates : (list)
        List containing dates in pandas._libs.tslib.Timestamp format
    num_units : (int)
        Used in create_X_y(). See docstring for create_X_y()
    units : (str)
        Used in create_X_y(). See docstring for create_X_y()
    n_jobs : (int)
        number of cores to use for cross validation
    fit_params : (dictionary)
        Parameters to pass to the fit method
    same : (bool)
        Used in create_X_y(). See docstring for create_X_y()

    Returns:
    ----------
    errors : (dictionary)
        Dictionary with 5 key value pairs:

            date - date used in create_X_y()
            model MAE - model Mean Absolute Error
            model RMSE - model Root Mean Squared Error
            Persistence Model MAE - Persistence Model Mean Absolute Error
            Persistence Model RMSE - Persistence Model Root Mean Squared Error
            training observations - Number of observations in training dataset
            testing observations - Number of observations in testing dataset

        The element at index X of each list
        corresponds to the same training and testing period.
    '''
    errors = {'date': [],
              f'{model.__class__.__name__} MAE': [],
              f'{model.__class__.__name__} RMSE': [],
              'Persistence Model MAE': [],
              'Persistence Model RMSE': [],
              'training observations': [],
              'testing observations': []
    }

    cols = list(df.columns)
    cols.remove(target_col)

    pm_mae_cache = []
    pm_rmse_cache = []
    model_mae_cache = []
    model_rmse_cache = []

    for date in test_dates:
        X, y = create_X_y(df, cols, target_col, date, num_units, units, same=same)
        X.drop(['final_date','Date'], axis=1, inplace=True)

        mae, rmse, pm_mae, pm_rmse, train_size, test_size = test_nn_model(model, X, y, fit_params=fit_params)

        print("{} Testing MAE | {:.4f}".format(model.__class__.__name__, mae))
        print("Persistence Model MAE | {:.4f}".format(pm_mae))
        print("{} Testing RMSE | {:.4f}".format(model.__class__.__name__, rmse))
        print("Persistence Model RMSE | {:.4f}".format(pm_rmse))

        if len(pm_mae_cache) == 1:
            pm_mae = np.mean((pm_mae_cache[0], pm_mae))
            errors['Persistence Model MAE'].append(pm_mae)
            pm_mae_cache = []
        elif len(pm_mae_cache) == 0:
            pm_mae_cache.append(pm_mae)

        if len(pm_rmse_cache) == 1:
            pm_rmse = np.mean((pm_rmse_cache[0], pm_rmse))
            errors['Persistence Model RMSE'].append(pm_rmse)
            pm_rmse_cache = []
        elif len(pm_rmse_cache) == 0:
            pm_rmse_cache.append(pm_rmse)

        if len(model_mae_cache) == 1:
            mae = np.mean((model_mae_cache[0], mae))
            errors[f'{model.__class__.__name__} MAE'].append(mae)
            model_mae_cache = []
        elif len(model_mae_cache) == 0:
            model_mae_cache.append(mae)

        if len(model_rmse_cache) == 1:
            rmse = np.mean((model_rmse_cache[0], rmse))
            errors[f'{model.__class__.__name__} RMSE'].append(rmse)
            model_rmse_cache = []
        elif len(model_rmse_cache) == 0:
            model_rmse_cache.append(rmse)


        errors['date'].append(date)

    return errors
