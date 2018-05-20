import os
import sys
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

from modeling_base import *
from eda import format_nrel_dataframe
from mlp_base import *


def train_same_weeks_range(df, num_weeks):
    results = []
    for day in test_dates:
        X, y = create_X_y(df=df, columns=columns, target='DNI_T_plus15', date=day, num_units=num_weeks, units='weeks', same=True)
        train_rmse = train_mlp(X.values, y.values)
        print("\nMLP Training RMSE | {:.4f}\n".format(train_rmse))
        results.append((day, train_rmse))
    return results


def train_weeks_range(df, num_weeks):
    results = []
    for day in test_dates:
        X, y = create_X_y(df=df, columns=columns, target='DNI_T_plus15', date=day, num_units=num_weeks, units='weeks', same=False)
        train_rmse = train_mlp(X.values, y.values)
        print("\nMLP Training RMSE | {:.4f}\n".format(train_rmse))
        results.append((day, train_rmse))
    return results


if __name__ == "__main__":

    df = format_nrel_dataframe("../data/2003_2016.csv")
    lag_features = ['Temperature', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'Cloud Type','Dew Point','DHI','DNI','Fill Flag', 'GHI','Relative Humidity','Solar Zenith Angle','Surface Albedo','Pressure','Precipitable Water','Wind Direction','Wind Speed']
    df = create_lagged_features(df, lag_features, 4, 30)
    df = create_future_target(df, 'DNI', 1, 30)

    mlp = build_neural_network(len(df.columns) - 3, [10, 40])    

    one_week_one_year = train_same_weeks_range(df, 1)
    one_week_two_years = train_same_weeks_range(df, 2)
    one_week_three_years = train_same_weeks_range(df, 3)
    one_week_four_years = train_same_weeks_range(df, 4)
    one_week_five_years = train_same_weeks_range(df, 5)

    one_week_diff = train_weeks_range(df, 1)
    two_week_diff = train_weeks_range(df, 2)
    three_week_diff = train_weeks_range(df, 3)
    four_week_diff = train_weeks_range(df, 4)
    five_week_diff = train_weeks_range(df, 5)
