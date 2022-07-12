import numpy as np
import pandas as pd

def load_electricity_education(pd_all):
    pd_educational = pd_all.copy()
    pd_educational = pd_educational.loc[(pd_educational["meter"]==0)]
    pd_educational = pd_educational.loc[(pd_educational["primary_use"]=='Education')]
    # Load all building_ids that have bad meter_reading and remove this data from our DataFrame:
    drop_ids = np.loadtxt('dropids.txt')
    pd_educational = pd_educational[~pd_educational.building_id.isin(drop_ids)]
    # Choose features we care about:
    features = ['square_feet','year_built','floor_count','timestamp','air_temperature','cloud_coverage','dew_temperature',
            'precip_depth_1_hr','sea_level_pressure','wind_direction','wind_speed','meter_reading']
    pd_educational = pd_educational[features]
    return pd_educational

