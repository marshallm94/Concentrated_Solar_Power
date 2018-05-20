import os
import sys
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

from mlp_base import *
from manipulation import get_master_df


def train_same_days_range(df, num_days):
    results = []
    for day in test_dates:
        X, y = create_X_y(df=df, columns=columns, target='DNI_T_plus15', date=day, num_units=num_days, units='days', same=True)
        train_rmse = train_mlp(X.values, y.values)
        print("\nMLP Training RMSE | {:.4f}\n".format(train_rmse))
        results.append((day, train_rmse))
    return results


def train_days_range(df, num_days):
    results = []
    for day in test_dates:
        X, y = create_X_y(df=df, columns=columns, target='DNI_T_plus15', date=day, num_units=num_days, units='days', same=False)
        train_rmse = train_mlp(X.values, y.values)
        print("\nMLP Training RMSE | {:.4f}\n".format(train_rmse))
        results.append((day, train_rmse))
    return results


if __name__ == "__main__":
    df = get_master_df("../../data/ivanpah_measurements.csv")
    df['Direct Normal [W/m^2]'] = np.where(df['Direct Normal [W/m^2]'] < 0, 0, df['Direct Normal [W/m^2]'])
    df = create_lagged_DNI_features(15, df)

    same_day_2_years = train_same_days_range(df, 2)
    same_day_3_years = train_same_days_range(df, 3)
    same_day_4_years = train_same_days_range(df, 4)
    same_day_5_years = train_same_days_range(df, 5)
    same_day_6_years = train_same_days_range(df, 6)
    same_day_7_years = train_same_days_range(df, 7)
    same_day_8_years = train_same_days_range(df, 8)
    same_day_9_years = train_same_days_range(df, 9)
    same_day_10_years = train_same_days_range(df, 10)

    days_5 = train_days_range(df, 5)
    days_10 = train_days_range(df, 10)
    days_15 = train_days_range(df, 15)
    days_20 = train_days_range(df, 20)
