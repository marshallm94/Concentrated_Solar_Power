# The single purpose of this script is to determine the optimal architecture
# for the Multi-Layer Perceptron that will serve as the final model for
# predicting DNI.
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from manipulation import get_master_df
from modeling import create_X_y, test_model, create_lagged_DNI_features
import datetime
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.objectives import MSE, MAE
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import RMSprop


def build_neural_network(n_predictors=28, hidden_layer_neurons=8, hidden_layer_neurons1=12):
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

    model.add(Dense(units=hidden_layer_neurons,
                    input_dim=input_layer_neurons,
                    kernel_initializer='uniform',
                    activation='linear'))

    model.add(Dense(units=hidden_layer_neurons1,
                    kernel_initializer='uniform',
                    activation='linear'))

    model.add(Dense(units=1))

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    model.compile(optimizer=optimizer,
                  loss='mse')

    return model


if __name__ == "__main__":

    df = get_master_df("../data/ivanpah_measurements.csv")

    # drop un-used columns
    df.drop(['PST',
             'Global Horiz [W/m^2]',
             'Global UVA [W/m^2]',
             'Global UVE [W/m^2]',
             'Global UVE [Index]',
             'UVSAET Temp [deg C]',
             'Logger Temp [deg C]',
             'Logger Battery [VDC]',
              'Diffuse Horiz (calc) [W/m^2]'], axis=1, inplace=True)

    print("\nData successfully loaded")

    df = create_lagged_DNI_features(15, df)

    print("\n15 new features successfully engineered")

    df = df[df['Direct Normal [W/m^2]'] > -10]

    print("\nDataFrame limited to observation with DNI >= -10")

    columns = ['Year',
            'Month',
            'DOY',
            'Hour',
            'Minute',
            'Direct Normal [W/m^2]',
            'Zenith Angle [degrees]',
            'Azimuth Angle [degrees]',
            'Airmass',
            'Wind Chill Temp [deg C]',
            'Avg Wind Speed @ 30ft [m/s]',
            'Avg Wind Direction @ 30ft [deg from N]',
            'Peak Wind Speed @ 30ft [m/s]',
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

    n_predictors = len(columns)
    hidden_layer_neurons = [8, 12]

    seed = 3
    np.random.seed(seed)

    # create parameter lists for GridSearchCV
        # Results:
            # Bach Size = 17
            # Epochs = 38

    model = KerasRegressor(build_fn=build_neural_network, epochs=38, batch_size=17)

    x = list(np.arange(10, 101, 10))
    x1 = list(np.arange(10, 101, 10))

    neural_net_grid_dict = {'hidden_layer_neurons': x,
                            "hidden_layer_neurons1": x1}

    neural_net_grid = GridSearchCV(estimator=model,
                                   param_grid=neural_net_grid_dict,
                                   scoring='neg_mean_squared_error',
                                   verbose=1,
                                   n_jobs=-1,
                                   cv=3)



    mask = df['Date'] == np.random.choice(np.unique(df['Date']))
    X, y = create_X_y(df[mask], columns)

    grid_result = neural_net_grid.fit(X, y)

    print(f"\nBest Score: {grid_result.best_score_}")
    print(f"\nBest Parameters: {grid_result.best_params_}")
    stop_criteria = EarlyStopping(monitor='val_loss', min_delta=0.001)

    network_dict = {'epochs': 38,
                    'batch_size': 17,
                    'shuffle': True,
                    'validation_split': 0.25,
                    'callback': stop_criteria
    }

    mlp = build_neural_network(len(columns), 50, 1)

    cv_errors, cv_test_periods, cv_train_periods, pm_errors = test_model(mlp, columns, 10, 90, 31, df, network_dict)

    print("\vScript complete")
