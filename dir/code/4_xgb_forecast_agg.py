# Prep data --------------------------------------------------------------------

### create lagged features

sales_agg_xgb = (
    sales_agg
        .rename({'unique_id': 'id', 'y': 'sales'}, axis = 1)
)

for i in [7, 14, 21, 28]:
    sales_agg_xgb = LagSales(sales_agg_xgb, i)

### create difference features

for i in [7, 14, 21, 28]:
    sales_agg_xgb = FD(sales_agg_xgb, i)

### create target for direct multi-step forecasting

sales_agg_xgb = (
    sales_agg_xgb
        .assign(
            sales_1_week_out = lambda x: x.groupby(['id'])['sales'].shift(-7),
            sales_2_week_out = lambda x: x.groupby(['id'])['sales'].shift(-14),
            sales_3_week_out = lambda x: x.groupby(['id'])['sales'].shift(-21),
            sales_4_week_out = lambda x: x.groupby(['id'])['sales'].shift(-28)
        )
)

### split into training and validation

sales_agg_xgb = sales_agg_xgb.rename({'y': 'sales', 'ds': 'date'},axis = 1)

train = sales_agg_xgb[sales_agg_xgb.date < pd.Timestamp('2017-06-01')]
val1  = sales_agg_xgb[(sales_agg_xgb.date >= pd.Timestamp('2017-06-01')) & (sales_agg_xgb.date <= pd.Timestamp('2017-06-28'))]
val2  = sales_agg_xgb[(sales_agg_xgb.date >= pd.Timestamp('2017-07-01')) & (sales_agg_xgb.date <= pd.Timestamp('2017-07-28'))]

# include only relevant features
drop_features_all = [
    'date', 'sales', 'sales_1_week_out', 'sales_2_week_out', 'sales_3_week_out', 
    'sales_4_week_out'
]
drop_features_w2 = [
    'lag_7_sales', 'rolling_7_sales_mean', 'rolling_7_sales_sd', 'fd_7_sales'
]
drop_features_w3 = [
    'lag_7_sales', 'rolling_7_sales_mean', 'rolling_7_sales_sd', 'fd_7_sales',
    'lag_14_sales', 'rolling_14_sales_mean', 'rolling_14_sales_sd', 'fd_14_sales'
]
drop_features_w4 = [
    'lag_7_sales', 'rolling_7_sales_mean', 'rolling_7_sales_sd', 'fd_7_sales',
    'lag_14_sales', 'rolling_14_sales_mean', 'rolling_14_sales_sd', 'fd_14_sales',
    'lag_21_sales', 'rolling_21_sales_mean', 'rolling_21_sales_sd', 'fd_21_sales'
]

# week 1
X_train_w1 = train.drop(drop_features_all, axis = 1)
X_val1_w1  = val1.drop(drop_features_all, axis = 1)
X_val2_w1  = val2.drop(drop_features_all, axis = 1)

# week 2
X_train_w2 = train.drop(drop_features_all + drop_features_w2, axis = 1)
X_val1_w2  = val1.drop(drop_features_all + drop_features_w2, axis = 1)
X_val2_w2  = val2.drop(drop_features_all + drop_features_w2, axis = 1)

# week 3
X_train_w3 = train.drop(drop_features_all + drop_features_w3, axis = 1)
X_val1_w3  = val1.drop(drop_features_all + drop_features_w3, axis = 1)
X_val2_w3  = val2.drop(drop_features_all + drop_features_w3, axis = 1)

# week 4
X_train_w4 = train.drop(drop_features_all + drop_features_w4, axis = 1)
X_val1_w4  = val1.drop(drop_features_all + drop_features_w4, axis = 1)
X_val2_w4  = val2.drop(drop_features_all + drop_features_w4, axis = 1)

# create 4 target vectors

y_train_w1 = train['sales_1_week_out']
y_val1_w1  = val1['sales_1_week_out']
y_val2_w1  = val2['sales_1_week_out']

y_train_w2 = train['sales_2_week_out']
y_val1_w2  = val1['sales_2_week_out']
y_val2_w2  = val2['sales_2_week_out']

y_train_w3 = train['sales_3_week_out']
y_val1_w3  = val1['sales_3_week_out']
y_val2_w3  = val2['sales_3_week_out']

y_train_w4 = train['sales_4_week_out']
y_val1_w4  = val1['sales_4_week_out']
y_val2_w4  = val2['sales_4_week_out']

# Predict on validation 1 ------------------------------------------------------

params = {
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}

### train models

mod_w1 = xgb.XGBRegressor(
    objective = 'reg:tweedie',
    enable_categorical = True,
    eval_metric = 'rmse',
    n_jobs = -1,
    **params
)
mod_w1.fit(X_train_w1, y_train_w1)

mod_w2 = xgb.XGBRegressor(
    objective = 'reg:tweedie',
    enable_categorical = True,
    eval_metric = 'rmse',
    n_jobs = -1, 
    **params
)
mod_w2.fit(X_train_w2, y_train_w2)

mod_w3 = xgb.XGBRegressor(
    objective = 'reg:tweedie',
    enable_categorical = True,
    eval_metric = 'rmse',
    n_jobs = -1,
    **params
)
mod_w3.fit(X_train_w3, y_train_w3)

mod_w4 = xgb.XGBRegressor(
    objective = 'reg:tweedie',
    enable_categorical = True,
    eval_metric = 'rmse',
    n_jobs = -1,
    **params
    
)
mod_w4.fit(X_train_w4, y_train_w4)

### predict each week

y_hat_week_1_val1 = mod_w1.predict(X_val1_w1)

y_hat_week_2_val1 = mod_w2.predict(X_val1_w2)

y_hat_week_3_val1 = mod_w3.predict(X_val1_w3)

y_hat_week_4_val1 = mod_w4.predict(X_val1_w4)

### consolidate forecast into one dataframe

y_hat_1_df_val1 = (
    pd.DataFrame(
        {
        'y_pred': y_hat_week_1_val1,
        'ds': val1['date'],
        'unique_id': val1['id']
        }
    )
    .pipe(AssignHorizon)
    .query('horizon == "week_1"')
)

y_hat_2_df_val1 = (
    pd.DataFrame(
        {
        'y_pred': y_hat_week_2_val1,
        'ds': val1['date'],
        'unique_id': val1['id']
        }
    )
    .pipe(AssignHorizon)
    .query('horizon == "week_2"')
)
y_hat_3_df_val1 = (
    pd.DataFrame(
        {
        'y_pred': y_hat_week_3_val1,
        'ds': val1['date'],
        'unique_id': val1['id']
        }
    )
    .pipe(AssignHorizon)
    .query('horizon == "week_3"')
)
y_hat_4_df_val1 = (
    pd.DataFrame(
        {
        'y_pred': y_hat_week_4_val1,
        'ds': val1['date'],
        'unique_id': val1['id']
        }
    )
    .pipe(AssignHorizon)
    .query('horizon == "week_4"')
)
y_hat_consolidated_val1 = pd.concat([y_hat_1_df_val1, y_hat_2_df_val1, y_hat_3_df_val1, y_hat_4_df_val1])

# Predict on validation 2 ------------------------------------------------------

### combine training & validation 1 sets, retrain models

X_train_w1 = pd.concat([X_train_w1, X_val1_w1])
y_train_w1 = pd.concat([y_train_w1, y_val1_w1])

X_train_w2 = pd.concat([X_train_w2, X_val1_w2])
y_train_w2 = pd.concat([y_train_w2, y_val1_w2])

X_train_w3 = pd.concat([X_train_w3, X_val1_w3])
y_train_w3 = pd.concat([y_train_w3, y_val1_w3])

X_train_w4 = pd.concat([X_train_w4, X_val1_w4])
y_train_w4 = pd.concat([y_train_w4, y_val1_w4])

mod_w1.fit(X_train_w1, y_train_w1)
mod_w2.fit(X_train_w2, y_train_w2)
mod_w3.fit(X_train_w3, y_train_w3)
mod_w4.fit(X_train_w4, y_train_w4)

### predict

y_hat_week_1_val2 = mod_w1.predict(X_val2_w1)

y_hat_week_2_val2 = mod_w2.predict(X_val2_w2)

y_hat_week_3_val2 = mod_w3.predict(X_val2_w3)

y_hat_week_4_val2 = mod_w4.predict(X_val2_w4)

# consolidate forecast into one dataframe

y_hat_1_df_val2 = (
    pd.DataFrame(
        {
        'y_pred': y_hat_week_1_val2,
        'ds': val2['date'],
        'unique_id': val2['id']
        }
    )
    .pipe(AssignHorizon)
    .query('horizon == "week_1"')
)

y_hat_2_df_val2 = (
    pd.DataFrame(
        {
        'y_pred': y_hat_week_2_val2,
        'ds': val2['date'],
        'unique_id': val2['id']
        }
    )
    .pipe(AssignHorizon)
    .query('horizon == "week_2"')
)
y_hat_3_df_val2 = (
    pd.DataFrame(
        {
        'y_pred': y_hat_week_3_val2,
        'ds': val2['date'],
        'unique_id': val2['id']
        }
    )
    .pipe(AssignHorizon)
    .query('horizon == "week_3"')
)
y_hat_4_df_val2 = (
    pd.DataFrame(
        {
        'y_pred': y_hat_week_4_val2,
        'ds': val2['date'],
        'unique_id': val2['id']
        }
    )
    .pipe(AssignHorizon)
    .query('horizon == "week_4"')
)
y_hat_consolidated_val2 = pd.concat([y_hat_1_df_val2, y_hat_2_df_val2, y_hat_3_df_val2, y_hat_4_df_val2])

# Check performance at family level --------------------------------------------

# Note: This is comperable to the ARIMA performance at the family level.

# get dfs in right format

train = train[['date', 'id', 'sales']].rename({'date': 'ds', 'sales': 'y', 'id': 'unique_id'}, axis = 1)
val1 = val1[['date', 'id', 'sales']].rename({'date': 'ds', 'sales': 'y','id': 'unique_id'}, axis = 1)
val2 = val2[['date', 'id', 'sales']].rename({'date': 'ds', 'sales': 'y','id': 'unique_id'}, axis = 1)

### validation 1

# rmsse
rmsse_val1_direct_agg_xgb = ReturnRMSSE(
    val_df = val1,
    training_df = train,
    preds_df = y_hat_consolidated_val1.rename({'y': 'y_pred'}, axis = 1)
)
print(np.sum(rmsse_val1_direct_agg_xgb['wrmsse_id']))
print(np.mean(rmsse_val1_direct_agg_xgb['rmsse_id']))

### validation 2

# rmsse
rmsse_val2_direct_agg_xgb = ReturnRMSSE(
    val_df = val2,
    training_df = pd.concat([train, val1]),
    preds_df = y_hat_consolidated_val2.rename({'y': 'y_pred'}, axis = 1)
)
print(np.sum(rmsse_val2_direct_agg_xgb['wrmsse_id']))
print(np.mean(rmsse_val2_direct_agg_xgb['rmsse_id']))

# Plot Example Forecasts -------------------------------------------------------

train_df_plot = train[['ds', 'unique_id', 'y']].assign(ds = lambda x: pd.to_datetime(x['ds']))
val1_df_plot = val1[['ds', 'unique_id', 'y']].assign(ds = lambda x: pd.to_datetime(x['ds']))
val2_df_plot = val2[['ds', 'unique_id', 'y']].assign(ds = lambda x: pd.to_datetime(x['ds']))

train_df_plot = pd.concat([train_df_plot[train_df_plot.ds > pd.Timestamp('2017-04-01')], val1_df_plot, val2_df_plot])
y_hat_plot = pd.concat([y_hat_consolidated_val1.drop('horizon', axis = 1), y_hat_consolidated_val2.drop('horizon', axis = 1)]).rename({'y_pred': 'XGB Direct'}, axis = 1)

plot_fams = ['GROCERY I', 'BEVERAGES', 'PRODUCE', 'CLEANING', 'DAIRY', 'BREAD/BAKERY', 'POULTRY', 'MEATS']

# StatsForecast.plot(train_df_plot, y_hat_plot, unique_ids = plot_fams, plot_random = False).savefig(fname = 'xgb_direct_example.png', dpi = 300)
