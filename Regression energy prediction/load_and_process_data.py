### Imports
import numpy as np
import pandas as pd
import os
from os.path import join
import glob

def process_data(datafolder):
    """
    This function takes folder name where we keep data for
    Building Energy consumption project and returns a pandas
    dataframe with all the data that was processed for the NA
    values.
    """
    
    # IMPORT DATA:
    # Import weather train data, train data and building metadata
    pd_weather_train = pd.DataFrame(pd.read_csv(datafolder + 'weather_train.csv'))
    pd_train = pd.DataFrame(pd.read_csv(datafolder + 'train.csv'))
    pd_bld_metadata = pd.DataFrame(pd.read_csv(datafolder + 'building_metadata.csv'))
    #---------------------------------------------------------------------------------
    # FIX NA VALUES:
    
    # Fix weather train data NA by taking the average of neighbouring cells:
    pd_weather_train.loc[:, pd_weather_train.columns != 'timestamp'] = (
        (
            pd_weather_train.loc[:, pd_weather_train.columns != 'timestamp']
              .fillna(method = 'ffill')
            +
            pd_weather_train.loc[:, pd_weather_train.columns != 'timestamp']
              .fillna(method = 'bfill')
         )/2
    )
    pd_weather_train = pd_weather_train.fillna(method = 'ffill').fillna(method='bfill')
    
    # Fix building metadata NA by substituting it with the mean its primary_use group:
    pd_bld_metadata[['year_built','floor_count']] = (
        pd_bld_metadata.groupby('primary_use')[['year_built','floor_count']]
          .transform(lambda x: x.fillna((x.mean())))
    )
    pd_bld_metadata.year_built = pd_bld_metadata.year_built.fillna(pd_bld_metadata.year_built.mean())
    pd_bld_metadata.floor_count = pd_bld_metadata.floor_count.fillna(pd_bld_metadata.floor_count.mean())
    pd_bld_metadata = round(pd_bld_metadata)
    
    # Train data seems to have no NA values so I won't fix those
    #---------------------------------------------------------------------------------
    # COMBINE ALL DATA:
    # Combine all data into one pandas dataframe:
    pd_weather_building = pd.merge(pd_bld_metadata, pd_weather_train, on = ['site_id'])
    pd_all = pd.merge(pd_weather_building, pd_train, on = ['building_id', "timestamp"])
    
    return pd_all