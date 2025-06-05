# -*- coding: utf-8 -*-
"""new_york_taxi_fare_prediction """
import opendatasets as od
import numpy as np
dataset_url = 'https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/overview'
od.download(dataset_url)
data_dir = './new-york-city-taxi-fare-prediction'

import pandas as pd
import random

sample_frac = 0.01

selected_cols = 'fare_amount,pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count'.split(',')
dtypes = {
 'fare_amount': 'float32',
 'pickup_longitude': 'float32',
 'pickup_latitude': 'float32',
 'dropoff_longitude': 'float32',
 'dropoff_latitude': 'float32',
 'passenger_count': 'uint8'
}

def skip_row(row_idx):
  if row_idx == 0:
    return False
  return random.random() > sample_frac

df = pd.read_csv("/content/new-york-city-taxi-fare-prediction/train.csv", usecols=selected_cols, parse_dates=['pickup_datetime'], dtype=dtypes, nrows=10000000 )

df.head()

test_df = pd.read_csv("/content/new-york-city-taxi-fare-prediction/test.csv", dtype=dtypes, parse_dates=['pickup_datetime'] )

test_df.head()

df.describe()

test_df.describe()

df['pickup_datetime'].min(), df['pickup_datetime'].max()

'''
Data de 2009 até 2015
Possíveis erros de longitude e latitude
passenger_count - 0 até 208


Podemos nos livrar de tudo que esteja fora do range do test_df
'''

from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

len(train_df), len(val_df)

#remove na
train_df = train_df.dropna()
val_df = val_df.dropna()

train_df.columns

input_cols = ['pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
target_col = 'fare_amount'

train_input = train_df[input_cols]
train_target = train_df[target_col]

val_input = val_df[input_cols]
val_target = val_df[target_col]

test_input = test_df[input_cols]

from sklearn.metrics import mean_squared_error
def rmse(target, pred):
    return np.sqrt(mean_squared_error(target, pred))

def predict_and_submit(model, fname, test_input):
  test_preds = model.predict(test_input)
  sub_df = pd.read_csv('/content/new-york-city-taxi-fare-prediction/sample_submission.csv')
  sub_df['fare_amount'] = test_preds
  sub_df.to_csv(fname, index=None)
  return sub_df

def add_dateparts(df, col):
  df[col + '_year'] = df[col].dt.year
  df[col + '_month'] = df[col].dt.month
  df[col + '_day'] = df[col].dt.day
  df[col + '_weekday'] = df[col].dt.weekday
  df[col + '_hour'] = df[col].dt.hour

col = 'pickup_datetime'
add_dateparts(train_df, col)
add_dateparts(test_df, col)
add_dateparts(val_df, col)

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the distance between two points on the earth (specified in decimal degrees) using the Haversine formula.

    Args:
        lon1 (numpy.ndarray or float): Longitude of the first point(s) in decimal degrees.
        lat1 (numpy.ndarray or float): Latitude of the first point(s) in decimal degrees.
        lon2 (numpy.ndarray or float): Longitude of the second point(s) in decimal degrees.
        lat2 (numpy.ndarray or float): Latitude of the second point(s) in decimal degrees.

    Returns:
        numpy.ndarray or float: Distance between the two points in kilometers.
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def add_trip_distance(df):
  df['trip_distance'] = haversine_np(df['pickup_longitude'], df['pickup_latitude'],
                                     df['dropoff_longitude'], df['dropoff_latitude'])

add_trip_distance(train_df)
add_trip_distance(val_df)
add_trip_distance(test_df)

'''JFK Airport
LGA Airport
EWR Airport
Times Square
Met Meuseum
World Trade Center
We'll add the distance from drop location.'''

jfk_lonlat = -73.7781, 40.6413
lga_lonlat = -73.8740, 40.7769
ewr_lonlat = -74.1745, 40.6895
met_lonlat = -73.9632, 40.7794
wtc_lonlat = -74.0099, 40.7126

def add_landmark_dropoff_distance(df, landmark_name, landmark_lonlat):
    lon, lat = landmark_lonlat
    df[landmark_name + '_drop_distance'] = haversine_np(lon, lat, df['dropoff_longitude'], df['dropoff_latitude'])

for a_df in [train_df, val_df, test_df]:
    for name, lonlat in [('jfk', jfk_lonlat), ('lga', lga_lonlat), ('ewr', ewr_lonlat), ('met', met_lonlat), ('wtc', wtc_lonlat)]:
        add_landmark_dropoff_distance(a_df, name, lonlat)

def remove_outliers(df):
  return df [
      (df['fare_amount'] >= 1.) &
      (df['fare_amount'] <= 500.) &
      (df['pickup_longitude'] >= -75) &
      (df['pickup_longitude'] <= -72) &
      (df['dropoff_longitude'] >= -75) &
      (df['dropoff_longitude'] <= -72) &
      (df['pickup_latitude'] >= 40) &
      (df['pickup_latitude'] <= 42) &
      (df['dropoff_latitude'] >= 40) &
      (df['dropoff_latitude'] <= 42) &
      (df['passenger_count'] >= 1 )&
      (df['passenger_count'] <= 6  )
  ]

train_df = remove_outliers(train_df)
val_df = remove_outliers(val_df)

train_df[train_df['fare_amount'] > 500]

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

input_cols = [ 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'passenger_count',
       'pickup_datetime_year', 'pickup_datetime_month', 'pickup_datetime_day',
       'pickup_datetime_weekday', 'pickup_datetime_hour', 'trip_distance',
       'jfk_drop_distance', 'lga_drop_distance', 'ewr_drop_distance',
       'met_drop_distance', 'wtc_drop_distance']

target_col ='fare_amount'

train_inputs = train_df[input_cols]
train_target = train_df[target_col]

val_inputs = val_df[input_cols]
val_target = val_df[target_col]

test_inputs = test_df[input_cols]

def evaluate(model, train_inputs, val_inputs):
  train_preds = model.predict(train_inputs)
  train_rmse = mean_squared_error(train_target, train_preds)
  val_preds = model.predict(val_inputs)
  val_rmse = mean_squared_error(val_target, val_preds)
  return np.sqrt(train_rmse), np.sqrt(val_rmse), train_preds, val_preds


from xgboost import XGBRegressor

def test_params(ModelClass, **params):
  model = ModelClass(**params).fit(train_inputs, train_target)
  train_rmse = rmse(model.predict(train_inputs), train_target)
  val_rmse = rmse(model.predict(val_inputs), val_target)
  return train_rmse, val_rmse

best_params = {
    'objective':'reg:squarederror',
    'random_state' : 42,
    'n_jobs':-1,
    'learning_rate': 0.1,
    'n_estimators': 800,
    'max_depth': 7

}

xgb_model_final = XGBRegressor(learning_rate=0.1, max_depth=7, n_estimators=800, n_jobs=-1, objective='reg:squarederror',
    random_state=42
  )

xgb_model_final.fit(train_inputs, train_target)

evaluate(xgb_model_final, train_inputs, val_inputs)

predict_and_submit(xgb_model_final, 'xgb_model_final.csv',test_inputs)