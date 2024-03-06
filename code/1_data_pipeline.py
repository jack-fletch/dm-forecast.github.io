
# Dependencies -----------------------------------------------------------------

### libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pylab import rcParams
from statsforecast import StatsForecast
from statsmodels.tsa.seasonal import seasonal_decompose
from statsforecast.models import ARIMA, AutoARIMA

from sklearn.preprocessing import OneHotEncoder

### data

cwd = os.getcwd()

os.chdir('../')

sales   = pd.read_csv('data/train.csv')
stores  = pd.read_csv('data/stores.csv')

os.chdir(cwd)

### helper functions

execfile('z_utils.py')

# Clean core series ------------------------------------------------------------

### clean up main series
 
sales_clean = (
    sales
        .assign(
            date = lambda x: pd.to_datetime(x['date']),
            id = lambda x: pd.Categorical(x['store_nbr'].astype(str) + '_' + x['family']),
            store_nbr = lambda x: x['store_nbr'].astype('category'),
            family = lambda x: x['family'].astype('category')
        )
        .rename({
            'onpromotion': 'promo'
            },
            axis = 1)
        .set_index(['date', 'id'])
)

# Create target variable -------------------------------------------------------

sales_clean = (
    sales_clean
        .assign(
            sales_1_week_out = lambda x: x.groupby(['id'])['sales'].shift(-7),
            sales_2_week_out = lambda x: x.groupby(['id'])['sales'].shift(-14),
            sales_3_week_out = lambda x: x.groupby(['id'])['sales'].shift(-21),
            sales_4_week_out = lambda x: x.groupby(['id'])['sales'].shift(-28)
        )
)

# Add lags & rolling window features -------------------------------------------

### lags

def LagSales(df, lag):

    new_cols = {
        'lag_' + str(lag) + '_sales': lambda x: x.groupby('id')['sales'].shift(lag)
    }

    return df.assign(**new_cols)

for i in [7, 14, 21, 28]:
    sales_clean = LagSales(sales_clean, i)


### window aggregations

def RollingMean(df, window_size):

    new_cols = {
        'rolling_' + str(window_size) + '_sales_mean': lambda x: x.groupby('id')['sales'].rolling(window = window_size).mean().reset_index(0, drop=True)
    }

    return df.assign(**new_cols)

def RollingSD(df, window_size):

    new_cols = {
        'rolling_' + str(window_size) + '_sales_sd': lambda x: x.groupby('id')['sales'].rolling(window = window_size).std().reset_index(0, drop=True)
    }

    return df.assign(**new_cols)

for i in [7, 14, 21, 28]:
    sales_clean = RollingMean(sales_clean, i)

for i in [7, 14, 21, 28]:
    sales_clean = RollingSD(sales_clean, i)

# Add difference features ------------------------------------------------------

def FD(df, lag):

    new_cols = {
        'fd_' + str(lag) + '_sales': lambda x: x['sales'] - x.groupby('id')['sales'].shift(lag)
    }

    return df.assign(**new_cols)

for i in [7, 14, 21, 28]:
    sales_clean = FD(sales_clean, i)


# Add date features ------------------------------------------------------------

### day, day of week, month features

sales_clean = (
    sales_clean
        .assign(
            month = lambda x: x.index.get_level_values('date').month,
            day = lambda x: x.index.get_level_values('date').day,
            day_of_week = lambda x: x.index.get_level_values('date').dayofweek
        )
)

# Add store metadata -----------------------------------------------------------

### OHE store data (type and cluster)

enc = OneHotEncoder(sparse_output = False).set_output(transform = 'pandas')

stores_cat = stores[['type', 'cluster']]

stores_cat = enc.fit_transform(stores_cat)

stores_enc = pd.concat([stores, stores_cat], axis = 1).drop(['city', 'state', 'type', 'cluster'], axis = 1)

### join to core series

sales_clean = (
    sales_clean.reset_index()
        .merge(
            stores_enc,
            how = 'left',
            on = 'store_nbr'
        )
)

# RUN THIS LINE IF YOU WANT TO RUN z_paper_plots_R.R
# sales_clean.to_csv('sales_clean.csv')

