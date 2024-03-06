
def PlotEarlyStopping(model, plot_title):

    results = model.evals_result()
    best = model.best_iteration

    # plt.figure(figsize=(10,7))
    plt.plot(results["validation_0"]["rmse"], label="Training loss")
    plt.plot(results["validation_1"]["rmse"], label="Validation loss")
    plt.axvline(best, color="gray", label="Optimal tree number")
    plt.xlabel("Number of trees")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(plot_title)
    plt.show()


def AssignHorizon(df):
    df = (
        df
            .assign(
                rank = lambda x: x.groupby('unique_id').cumcount() + 1,
                horizon = lambda x: np.select(
                    [
                        (x['rank'] <= 7),
                        (x['rank'] > 7) & (x['rank'] <= 14),
                        (x['rank'] > 14) & (x['rank'] <= 21),
                        (x['rank'] > 21) & (x['rank'] <= 28),
                    ], [
                        'week_1', 'week_2', 'week_3', 'week_4'
                    ]
                )
            )
            .drop('rank', axis = 1)
    )
    return df


def ReturnMSE(val_df, preds_df, by_id = True):
    temp = (
        val_df
            .merge(
                preds_df,
                how = 'left',
                left_on = ['unique_id', 'ds'],
                right_on = ['unique_id', 'ds']
            )
            .assign(
                residsq = lambda x: (x['y'] - x['y_pred'])**2
            )
    )

    if by_id:
        temp = (
            temp
                .groupby('unique_id')
                .agg(sum_residsq = ('residsq', 'sum'), n = ('unique_id', 'count'))
                .reset_index()
                .assign(mse_id = lambda x: x['sum_residsq'] / x['n'])

        )
    
    else:
        temp = np.sum(temp['residsq']) / temp.shape[0]


    return (temp)


def ReturnRMSSE(training_df, val_df, preds_df):

    ### calculate Scale (MSE of naive one-step-ahead forecast)

    temp_train = (
        training_df
            .assign(cum_y = lambda x: x.groupby('unique_id')['y'].cumsum())
            .query('cum_y != 0')
            .assign(
                y_pred_naive = lambda x: x.groupby('unique_id')['y'].shift(1),
                residsq = lambda x: (x['y'] - x['y_pred_naive'])**2
            )
    )

    temp_train = (
        temp_train
            .groupby('unique_id')
            .agg(
                sum_residsq = ('residsq', 'sum'),
                count = ('unique_id', 'count')
            )
            .reset_index()
            .assign(
                scale = lambda x: x['sum_residsq'] / (x['count'] - 1)
            )
    )

    ### calculate validation MSE

    temp_val = (
            val_df
                .merge(
                    preds_df,
                    how = 'left',
                    left_on = ['unique_id', 'ds'],
                    right_on = ['unique_id', 'ds']
                )
                .assign(
                    residsq = lambda x: (x['y'] - x['y_pred'])**2
                )
        )

    temp_val = (
        temp_val
            .groupby('unique_id')
            .agg(sum_residsq = ('residsq', 'sum'), n = ('unique_id', 'count'))
            .reset_index()
            .assign(mse_id = lambda x: x['sum_residsq'] / x['n'])

    )

    ### join and calculate RMSSE for each unique id

    train_val = (
        temp_train
            .merge(temp_val, on = ['unique_id'], how = 'left')
            .assign(
                rmsse_id = lambda x: np.sqrt(x['mse_id'] / x['scale'])
            )
    )

    ### weight RMSSE by number of sales over the last 28 days of the training set

    weights = (
        training_df
            .groupby('unique_id')
            .tail(28)
            .groupby('unique_id')
            .agg(last_28_sales = ('y', 'sum'))
            .reset_index()
    )

    train_val = (
        train_val
            .merge(
                weights,
                on = ['unique_id']
            )
            .assign(
                total_sales = lambda x: np.sum(x['last_28_sales']),
                prop = lambda x: x['last_28_sales'] / x['total_sales'],
                wrmsse_id = lambda x: x['rmsse_id'] * x['prop']
            )
            .drop(['total_sales', 'prop'], axis = 1)
    )

    return train_val

