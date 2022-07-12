import numpy as np
import pandas as pd
import datetime


def split_data(pd_dataframe):
    # split it ~90% on train + validation (11 months) and ~10% on test (1 month)
    pd_educational = pd_dataframe.copy()
    pd_train = pd.concat([pd_educational.loc[pd.to_datetime(pd_educational["timestamp"]) < datetime.datetime(2016, 6, 1)] , pd_educational.loc[pd.to_datetime(pd_educational["timestamp"]) > datetime.datetime(2016, 6, 30)]]) 
    pd_test = pd_educational.loc[pd.to_datetime(pd_educational["timestamp"]) > datetime.datetime(2016, 6, 1)]  
    pd_test = pd_test[pd.to_datetime(pd_educational["timestamp"]) < datetime.datetime(2016, 7, 1)] 
    timestamp_train = pd_train.pop('timestamp')  
    timestamp_test = pd_test.pop('timestamp') 
    
    return pd_train, pd_test, timestamp_train, timestamp_test

def prepare_data_simple_encoding(pd_dataframe):
  # Timestamp feature encoding:
    ## Looking at the data during data exploration, data shows clear dependence on
    ## time of the day and weekend/weekday. There is also some month variation
    ## of the meter_reading data. I will thus split the timestamp data into month,
    ## weekday and hour that I will then encode using regular numeric encoding (normalize)
    pd_educational = pd_dataframe.copy()
    pd_educational['month'] = pd.to_datetime(pd_educational["timestamp"]).dt.month
    pd_educational['weekday'] = pd.to_datetime(pd_educational["timestamp"]).dt.weekday
    pd_educational['hour'] = pd.to_datetime(pd_educational["timestamp"]).dt.hour
    pd_train, pd_test, timestamp_train, timestamp_test = split_data(pd_educational)
    
    # Normalize the data before we send it to linear regression model to learn this data:
    ## Note: we want to normalize test data using same mean and std as train!
    train_mean = pd_train.mean()
    train_std = pd_train.std()
    
    pd_train = (pd_train - train_mean) / train_std
    pd_test = (pd_test- train_mean) / train_std
    
    train_features = pd_train.copy()
    test_features = pd_test.copy()
    
    train_labels = train_features.pop('meter_reading')
    test_labels = test_features.pop('meter_reading')
    
    return (train_features,train_labels), (test_features,test_labels), timestamp_train, timestamp_test


def prepare_data_polynomial_encoding(pd_dataframe):
    # Wind feature encoding:
    pd_educational = pd_dataframe.copy()
    wind_speed = pd_educational.pop('wind_speed') 
    wind_direction = pd_educational.pop('wind_direction')*np.pi/180 # wind direction in radians 
    
    pd_educational['wind_x'] = wind_speed * np.cos(wind_direction) 
    pd_educational['wind_y'] = wind_speed * np.sin(wind_direction) 
    
    # Timestamp feature encoding:
    timestamp = pd.to_datetime(pd_educational["timestamp"]).map(datetime.datetime.timestamp) 
    day = 24*60*60 
    week = 7*day 
    month = 30*day 
    year = (365.2425)*day 
    pd_educational['day_sin'] = np.sin(timestamp * (2 * np.pi / day)) 
    pd_educational['day_cos'] = np.cos(timestamp * (2 * np.pi / day)) 
    
    pd_educational['week_sin'] = np.sin(timestamp * (2 * np.pi / week)) 
    pd_educational['week_cos'] = np.cos(timestamp * (2 * np.pi / week)) 
    
    pd_educational['year_sin'] = np.sin(timestamp * (2 * np.pi / year)) 
    pd_educational['year_cos'] = np.cos(timestamp * (2 * np.pi / year)) 
    
    pd_educational['is_weekday'] = (pd.to_datetime(pd_educational["timestamp"]).dt.weekday.values < 5).astype(int) 
    
    # Year_built feature encoding:
    ## Looking at the year_built histogram, there seems to be 3 waves of building 
    ## '1900-1940', '1940-1980', '1980-today'. I will split this column in those  
    ## 3 features and I will assign them values 0, 1 and 2:  
    year_built = pd_educational.pop('year_built') 
    pd_educational['year_built_range'] = ((year_built.values-1900)/40).astype(int) 
    
    pd_train, pd_test, timestamp_train, timestamp_test = split_data(pd_educational) 
    
    # Normalize the data before we send it to linear regression model to learn this data: 
    ## Note: we want to normalize test data using same mean and std as train! 
    train_mean = pd_train.mean() 
    train_std = pd_train.std() 
    
    pd_train = (pd_train - train_mean) / train_std 
    pd_test = (pd_test- train_mean) / train_std 
    
    train_features = pd_train.copy() 
    test_features = pd_test.copy() 
    
    train_labels = train_features.pop('meter_reading') 
    test_labels = test_features.pop('meter_reading') 
    
    return (train_features,train_labels), (test_features,test_labels), timestamp_train, timestamp_test
