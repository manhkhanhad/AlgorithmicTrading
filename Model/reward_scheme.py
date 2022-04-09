from scipy.stats import iqr
import numpy as np

n_bins = 5 

def fix_dataset_inconsistencies(dataframe, fill_value=None):
    dataframe = dataframe.replace([-np.inf, np.inf], np.nan)

    # This is done to avoid filling middle holes with backfilling.
    if fill_value is None:
        dataframe.iloc[0,:] = \
            dataframe.apply(lambda column: column.iloc[column.first_valid_index()], axis='index')
    else:
        dataframe.iloc[0,:] = \
            dataframe.iloc[0,:].fillna(fill_value)

    return dataframe.fillna(axis='index', method='pad').dropna(axis='columns')

def estimate_outliers(data):
    return iqr(data) * 1.5

def estimate_percent_gains(data, column='close'):
    returns = get_returns(data, column=column)
    gains = estimate_outliers(returns)
    return gains

def get_returns(data, column='close'):
    return fix_dataset_inconsistencies(data[[column]].pct_change(), fill_value=0)

def precalculate_ground_truths(data, column='close', threshold=None):
    #returns = data #get_returns(data)
    returns = np.diff(data) / data[1:] #Calulate returns by percentage change

    gains = estimate_outliers(returns) if threshold is None else threshold
    binary_gains = np.where(returns > gains, 1, 0).astype(int)
    return binary_gains

def is_null(data):
    return data.isnull().sum().sum() > 0

def is_sparse(data, column='close'):
    binary_gains = precalculate_ground_truths(data, column=column)
    bins = [n * (binary_gains.shape[0] // n_bins) for n in range(n_bins)]
    bins += [binary_gains.shape[0]]
    bins = [binary_gains.iloc[bins[n]:bins[n + 1]] for n in range(n_bins)]
    return all([bin.astype(bool).any() for bin in bins])

def is_data_predictible(data, column):
    return not is_null(data) & is_sparse(data, column)

def PenalizedProfit(initial_amount, total_asset, amount, day):
    """
    A reward scheme which penalizes net worth loss and 
    decays with the time spent.
    """
    cash_penalty_proportion = 0.10
    if day > 1:
        initial_amount = initial_amount
        net_worth = total_asset
        cash_worth = amount
        cash_penalty = max(0, (net_worth * cash_penalty_proportion - cash_worth))
        net_worth -= cash_penalty
        reward = (net_worth / initial_amount) - 1
        reward /= day
        return reward
    else:
        return 0.0

def AnomalousProfit(total_assets,day):
    """
    A reward scheme which penalizes net worth loss and 
    decays with the time spent.
    """
    if day > 1:
        net_worths = np.array(total_assets)
        ground_truths = precalculate_ground_truths(net_worths,threshold=0.02)
        reward_factor = 2.0 * ground_truths - 1.0
        #return net_worths.iloc[-1] / net_worths.iloc[-min(current_step, self._window_size + 1)] - 1.0
        return (reward_factor[-1] * np.absolute(net_worths)[-1])
    else:
        return 0.0


def SimpleProfit(total_assets,day, window_size = 20):
    """
    A reward scheme which penalizes net worth loss and 
    decays with the time spent.
    """
    if day > 1:
        net_worths = np.array(total_assets)
        return net_worths[-1] / net_worths[-min(day, window_size + 1)] - 1.0
    else:
        return 0.0

