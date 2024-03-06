# Dependencies -----------------------------------------------------------------

### libraries

import numpy as np
import pandas as pd

from statsforecast import StatsForecast
from statsforecast.models import ARIMA, AutoARIMA

# Prep data for modeling -------------------------------------------------------

### aggregate to family level

sales_agg = (
    sales_clean
        .groupby(['date', 'family'])
        .agg(
            sales = ('sales', 'sum'),
            promo = ('promo', 'sum'),
        )
        .reset_index()
        .rename({
            'family': 'id',
            'date': 'ds',
        },
        axis = 1)
)

### recalculate a few features at the aggregated level

# day, day of week, month features

sales_agg = (
    sales_agg
        .assign(
            month = lambda x: pd.to_datetime(x['ds']).dt.month,
            day = lambda x: pd.to_datetime(x['ds']).dt.day,
            day_of_week = lambda x: pd.to_datetime(x['ds']).dt.dayofweek
        )
)

### rolling window aggregations

for i in [7, 14, 21, 28]:
    sales_agg = RollingMean(sales_agg, i)

for i in [7, 14, 21, 28]:
    sales_agg = RollingSD(sales_agg, i)

### split into training and validation

sales_agg = sales_agg.dropna()

sales_agg = sales_agg.rename({'id': 'unique_id', 'sales': 'y'},axis = 1)

train = sales_agg[sales_agg.ds < pd.Timestamp('2017-06-01')]
val1  = sales_agg[(sales_agg.ds >= pd.Timestamp('2017-06-01')) & (sales_agg.ds <= pd.Timestamp('2017-06-28'))]
val2  = sales_agg[(sales_agg.ds >= pd.Timestamp('2017-07-01')) & (sales_agg.ds <= pd.Timestamp('2017-07-28'))]

# Train models (SARIMAX) -------------------------------------------------------

a_sarimax = AutoARIMA(
    season_length = 7,
    trace = True,
    alias = 'auto_sarimax'
)

sf = StatsForecast(
    models = [
        a_sarimax,
    ],
    freq = 'D',
    verbose = True,  
)

sf.fit(train)

### validation 1

X_val1 = val1.drop('y', axis = 1)

preds_val1 = sf.predict(X_df = X_val1, h=28, level = [.95])

preds_df_val1 = preds_val1.reset_index()[['unique_id', 'ds', 'auto_sarimax']]

preds_df_val1.columns = ['unique_id', 'ds', 'y_pred']

rmsse_id_val1_sarimax = ReturnRMSSE(
    val_df = val1,
    training_df = train,
    preds_df = preds_df_val1
)
rmsse_id_val1_sarimax['wrmsse_id'].sum()
rmsse_id_val1_sarimax['rmsse_id'].mean()

### validation 2

# combine training & validation 1 sets, refit model
train = pd.concat([train, val1])

sf.fit(train)

# predict on validation 2 set
X_val2 = val2.drop('y', axis = 1)

preds_val2 = sf.predict(X_df = X_val2, h=28, level = [.95])

preds_df_val2 = preds_val2.reset_index()[['unique_id', 'ds', 'auto_sarimax']]
preds_df_val2.columns = ['unique_id', 'ds', 'y_pred']

rmsse_id_val2_sarimax = ReturnRMSSE(
    val_df = val2,
    training_df = train,
    preds_df = preds_df_val2
)
rmsse_id_val2_sarimax['wrmsse_id'].sum()
rmsse_id_val2_sarimax['rmsse_id'].mean()

# Train models (ARIMA) ---------------------------------------------------------

a_arima = AutoARIMA(
    season_length = 7,
    trace = True,
    alias = 'auto_arima',
    seasonal = False
)

sf = StatsForecast(
    models = [
        a_arima,
    ],
    freq = 'D',
    verbose = True,  
)

train_no_x = train[['ds', 'unique_id', 'y']]

sf.fit(train_no_x)

### validation 1

preds_val1 = sf.predict(h=28, level = [.95])

preds_df_val1 = preds_val1.reset_index()[['unique_id', 'ds', 'auto_arima']]

preds_df_val1.columns = ['unique_id', 'ds', 'y_pred']

rmsse_id_val1_arima = ReturnRMSSE(
    val_df = val1,
    training_df = train,
    preds_df = preds_df_val1
)
rmsse_id_val1_arima['wrmsse_id'].sum()
rmsse_id_val1_arima['rmsse_id'].mean()

### validation 2

# combine training & validation 1 sets, refit model
train_no_x = pd.concat([train_no_x, val1[['ds', 'unique_id', 'y']]])

sf.fit(train_no_x)

# predict on validation 2 set

preds_val2 = sf.predict(h=28, level = [.95])

preds_df_val2 = preds_val2.reset_index()[['unique_id', 'ds', 'auto_arima']]
preds_df_val2.columns = ['unique_id', 'ds', 'y_pred']

rmsse_id_val2_arima = ReturnRMSSE(
    val_df = val2,
    training_df = train,
    preds_df = preds_df_val2
)
rmsse_id_val2_arima['wrmsse_id'].sum()
rmsse_id_val2_arima['rmsse_id'].mean()

# Train models (SARIMA) ---------------------------------------------------------

a_sarima = AutoARIMA(
    season_length = 7,
    trace = True,
    alias = 'auto_sarima'
)

sf = StatsForecast(
    models = [
        a_sarima,
    ],
    freq = 'D',
    verbose = True,  
)

train_no_x = train[['ds', 'unique_id', 'y']]

sf.fit(train_no_x)

### validation 1

preds_val1 = sf.predict(h=28, level = [.95])

preds_df_val1 = preds_val1.reset_index()[['unique_id', 'ds', 'auto_sarima']]

preds_df_val1.columns = ['unique_id', 'ds', 'y_pred']

rmsse_id_val1_sarima = ReturnRMSSE(
    val_df = val1,
    training_df = train,
    preds_df = preds_df_val1
)
rmsse_id_val1_sarima['wrmsse_id'].sum()
rmsse_id_val1_sarima['rmsse_id'].mean()

### validation 2

# combine training & validation 1 sets, refit model
train_no_x = pd.concat([train_no_x, val1[['ds', 'unique_id', 'y']]])

sf.fit(train_no_x)

# predict on validation 2 set

preds_val2 = sf.predict(h=28, level = [.95])

preds_df_val2 = preds_val2.reset_index()[['unique_id', 'ds', 'auto_sarima']]
preds_df_val2.columns = ['unique_id', 'ds', 'y_pred']

rmsse_id_val2_sarima = ReturnRMSSE(
    val_df = val2,
    training_df = train,
    preds_df = preds_df_val2
)
rmsse_id_val2_sarima['wrmsse_id'].sum()
rmsse_id_val2_sarima['rmsse_id'].mean()

# Plot Example Forecasts -------------------------------------------------------

train_df_plot = train[['ds', 'unique_id', 'y']].assign(ds = lambda x: pd.to_datetime(x['ds']))
val1_df_plot = val1[['ds', 'unique_id', 'y']].assign(ds = lambda x: pd.to_datetime(x['ds']))
val2_df_plot = val2[['ds', 'unique_id', 'y']].assign(ds = lambda x: pd.to_datetime(x['ds']))

train_df_plot = pd.concat([train_df_plot[train_df_plot.ds > pd.Timestamp('2017-04-01')], val1_df_plot, val2_df_plot])
y_hat_plot = pd.concat([preds_df_val1.rename({'y_pred': 'SARIMA'}, axis = 1), preds_df_val2.rename({'y_pred': 'SARIMA'}, axis = 1)])

plot_fams = ['GROCERY I', 'BEVERAGES', 'PRODUCE', 'CLEANING', 'DAIRY', 'BREAD/BAKERY', 'POULTRY', 'MEATS']

# StatsForecast.plot(train_df_plot, y_hat_plot, unique_ids = plot_fams, plot_random = False).savefig(fname = 'sarima_direct_example.png', dpi = 300)
