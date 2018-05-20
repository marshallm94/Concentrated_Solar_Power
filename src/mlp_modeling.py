from modeling_base import *

import train_mlp_base as tmb
from eda import format_nrel_dataframe

df = format_nrel_dataframe("../data/2003_2016.csv")
lag_features = ['Temperature', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'Cloud Type','Dew Point','DHI','DNI','Fill Flag', 'GHI','Relative Humidity','Solar Zenith Angle','Surface Albedo','Pressure','Precipitable Water','Wind Direction','Wind Speed']
df = create_lagged_features(df, lag_features, 4, 30)
df = create_future_target(df, 'DNI', 1, 30)

test_dates = tmb.get_random_test_dates(5, 2017, (6, 18), 2)
