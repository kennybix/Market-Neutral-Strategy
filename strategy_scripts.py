# import libraries
# import libraries

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import linear_model
from scipy.stats import t, linregress
from scipy import stats

from sklearn.ensemble import RandomForestRegressor

from numpy.lib.stride_tricks import as_strided


import class_SeriesAnalyser, class_DataProcessor
from sklearn.manifold import TSNE
import matplotlib.cm as cm

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn import preprocessing



# temporary
# ignore warnings
import warnings
warnings.filterwarnings('ignore')



# helper functions

def check_for_NaN(data): # function to check for NaN values
    nan_values = data.isna().sum()
    # Display the count of NaN values for each column
    print(nan_values)
    return nan_values
  
  
# putting all cleaning steps in a single function
def clean_data(df,threshold_weight=0.9):
    # Remove columns with more than 'threshold' NaN values. A minimum of 90% is set for non-NaN values
    filtered_data = df.dropna(axis=1, thresh=threshold_weight*len(df))
    # Perform data imputation for each column
    filtered_data = filtered_data.fillna(method='ffill', axis=0)
    # Remove any duplicate rows - not expected 
    filtered_data.drop_duplicates(inplace=True)

    filtered_data.index =  pd.to_datetime(filtered_data.index) # convert date to datetime for easy handling

    # some stocks not listed in the first date will still be in the tables, it is best to filter them out
    filtered_data = filtered_data.loc[:, filtered_data.iloc[0].notna()] # removes '3289 JT' , '6988 JT' , '3863 JT' : not listed in Jan 2013
        
    return filtered_data
  

# helper functions

# pair selection helper functions
def cluster_size(counts): #for pair selection
    plt.figure()
    plt.barh(counts.index+1, counts.values)
    #plt.title('Cluster Member Counts')
    plt.yticks(np.arange(1, len(counts)+1, 1))
    plt.xlabel('Stocks within cluster', size=12)
    plt.ylabel('Cluster Id', size=12)

def plot_TSNE(X, clf, clustered_series_all):
    """
    This function makes use of t-sne to visualize clusters in 2d.
    """
    
    X_tsne = TSNE(learning_rate=1000, perplexity=25, random_state=1337).fit_transform(X)
    
    # visualization
    fig = plt.figure(1, facecolor='white', figsize=(15,15), frameon=True, edgecolor='black')
    plt.clf()
    
    # axis in the middle
    ax = fig.add_subplot(1, 1, 1, alpha=0.9)
    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_position('center')
    ax.spines['bottom'].set_alpha(0.3)
    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(which='major', labelsize=18)
    #plt.axis('off')

    # stocks in cluster
    labels = clf.labels_
    x = X_tsne[(labels!=-1), 0]
    y = X_tsne[(labels!=-1), 1]
    tickers = list(clustered_series_all[clustered_series_all != -1].index)
    plt.scatter(
        x,
        y,
        s=300,
        alpha=0.75,
        c=labels[labels!=-1],
        cmap=cm.Paired
    )
    for i, ticker in enumerate(tickers):
        # plt.annotate(ticker, (x[i]-20, y[i]+12), size=15)
        plt.annotate(ticker, (x[i]-2, y[i]+1), size=10)


    # x = np.array(x)
    # y = np.array(y)

    #for i, ticker in enumerate(tickers):
    #    plt.annotate(ticker, (x[i]+20, y[i]+20))#, arrowprops={'arrowstyle':'simple'})
        
    #plt.title('OPTICS clusters visualized with t-SNE', size=16);
    plt.xlabel('t-SNE Dim. 1', position=(0.92,0), size=20)
    plt.ylabel('t-SNE Dim. 2', position=(0,0.92), size=20)
    # ax.set_xticks(range(-300, 0, 60))
    # ax.set_yticks(range(-300, 0, 60))
    #plt.savefig('DBSCAN_2014_2018_eps0_15.png', bbox_inches='tight', pad_inches=0.01)
    plt.savefig('DBSCAN_2013_2015.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()


# signal evaluation metrics

def compute_hit_rate(signals, actual_returns):
    # Compute the hit rate of the signals
    predicted_directions = np.sign(signals)
    actual_directions = np.sign(actual_returns)
    num_correct = np.sum(predicted_directions == actual_directions)
    hit_rate = num_correct / len(signals)
    return hit_rate

def compute_profitability(signals, actual_returns):
    # Compute the profitability of the signals
    pnl = signals * actual_returns
    total_profit = np.sum(pnl)
    return total_profit

def compute_risk_reward_ratio(signals, actual_returns):
    # Compute the risk-to-reward ratio of the signals
    positive_pnl = signals * actual_returns
    positive_pnl = positive_pnl[positive_pnl > 0]
    negative_pnl = signals * actual_returns
    negative_pnl = negative_pnl[negative_pnl < 0]
    
    if len(negative_pnl) > 0:
        risk_reward_ratio = np.abs(np.mean(positive_pnl) / np.mean(negative_pnl))
    else:
        risk_reward_ratio = np.inf
    
    return risk_reward_ratio


def compute_win_rate(signal, returns):
    """
    Calculates the win rate of the trading signal.

    Args:
    signal (np.ndarray): The trading signal
    returns (np.ndarray): The actual returns

    Returns:
    float: The win rate of the trading signal
    """
    profits = signal * returns
    return np.mean(profits > 0)

def compute_maximum_drawdown(signal, returns):
    """
    Calculates the maximum drawdown of the trading signal.

    Args:
    signal (np.ndarray): The trading signal
    returns (np.ndarray): The actual returns

    Returns:
    float: The maximum drawdown of the trading signal
    """
    profits = signal * returns
    cum_profits = np.cumsum(profits)
    # remove nan from cum_profits
    cum_profits = [item if isinstance(item, (int, float)) and not np.isnan(item) else 0 for item in cum_profits]
    max_drawdown = np.max(np.maximum.accumulate(cum_profits) - cum_profits)
    return max_drawdown

# strategy performance metrics

APPROX_BDAYS_PER_MONTH = 21
APPROX_BDAYS_PER_YEAR = 245

MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52
QTRS_PER_YEAR = 4

DAILY = 'daily'
WEEKLY = 'weekly'
MONTHLY = 'monthly'
QUARTERLY = 'quarterly'
YEARLY = 'yearly'

ANNUALIZATION_FACTORS = {
    DAILY: APPROX_BDAYS_PER_YEAR,
    WEEKLY: WEEKS_PER_YEAR,
    MONTHLY: MONTHS_PER_YEAR,
    QUARTERLY: QTRS_PER_YEAR,
    YEARLY: 1
}


# slower numpy
nanmean = np.nanmean
nanstd = np.nanstd
nansum = np.nansum
nanmax = np.nanmax
nanmin = np.nanmin
nanargmax = np.nanargmax
nanargmin = np.nanargmin


def roll(*args, **kwargs): # Calculates a given statistic across a rolling time period.

    func = kwargs.pop('function')
    window = kwargs.pop('window')
    if len(args) > 2:
        raise ValueError("Cannot pass more than 2 return sets")

    if len(args) == 2:
        if not isinstance(args[0], type(args[1])):
            raise ValueError("The two returns arguments are not the same.")

    if isinstance(args[0], np.ndarray):
        return _roll_ndarray(func, window, *args, **kwargs)
    return _roll_pandas(func, window, *args, **kwargs)


def up(returns, factor_returns, **kwargs): # Calculates a given statistic filtering only positive factor return periods.

    func = kwargs.pop('function')
    returns = returns[factor_returns > 0]
    factor_returns = factor_returns[factor_returns > 0]
    return func(returns, factor_returns, **kwargs)


def down(returns, factor_returns, **kwargs): # Calculates a given statistic filtering only negative factor return periods.


    func = kwargs.pop('function')
    returns = returns[factor_returns < 0]
    factor_returns = factor_returns[factor_returns < 0]
    return func(returns, factor_returns, **kwargs)


def _roll_ndarray(func, window, *args, **kwargs):
    data = []
    for i in range(window, len(args[0]) + 1):
        rets = [s[i-window:i] for s in args]
        data.append(func(*rets, **kwargs))
    return np.array(data)


def _roll_pandas(func, window, *args, **kwargs):
    data = {}
    index_values = []
    for i in range(window, len(args[0]) + 1):
        rets = [s.iloc[i-window:i] for s in args]
        index_value = args[0].index[i - 1]
        index_values.append(index_value)
        data[index_value] = func(*rets, **kwargs)
    return pd.Series(data, index=type(args[0].index)(index_values))


def get_utc_timestamp(dt):

    dt = pd.to_datetime(dt)
    try:
        dt = dt.tz_localize('UTC')
    except TypeError:
        dt = dt.tz_convert('UTC')
    return dt


def rolling_window(array, length, mutable=False):
    if not length:
        raise ValueError("Can't have 0-length window")

    orig_shape = array.shape
    if not orig_shape:
        raise IndexError("Can't restride a scalar.")
    elif orig_shape[0] < length:
        raise IndexError(
            "Can't restride array of shape {shape} with"
            " a window length of {len}".format(
                shape=orig_shape,
                len=length,
            )
        )

    num_windows = (orig_shape[0] - length + 1)
    new_shape = (num_windows, length) + orig_shape[1:]

    new_strides = (array.strides[0],) + array.strides

    out = as_strided(array, new_shape, new_strides)
    out.setflags(write=mutable)
    return out



def _adjust_returns(returns, adjustment_factor):
    if isinstance(adjustment_factor, (float, int)) and adjustment_factor == 0:
        return returns
    return returns - adjustment_factor


def annualization_factor(period, annualization):
    if annualization is None:
        try:
            factor = ANNUALIZATION_FACTORS[period]
        except KeyError:
            raise ValueError(
                "Period cannot be '{}'. "
                "Can be '{}'.".format(
                    period, "', '".join(ANNUALIZATION_FACTORS.keys())
                )
            )
    else:
        factor = annualization
    return factor


def simple_returns(prices): # Compute simple returns from a timeseries of prices.
    if isinstance(prices, (pd.DataFrame, pd.Series)):
        out = prices.pct_change().iloc[1:]
    else:
        # Assume np.ndarray
        out = np.diff(prices, axis=0)
        np.divide(out, prices[:-1], out=out)

    return out


def cum_returns(returns, starting_value=0, out=None): # Compute cumulative returns from simple returns.
    if len(returns) < 1:
        return returns.copy()

    nanmask = np.isnan(returns)
    if np.any(nanmask):
        returns = returns.copy()
        returns[nanmask] = 0

    allocated_output = out is None
    if allocated_output:
        out = np.empty_like(returns)

    np.add(returns, 1, out=out)
    out.cumprod(axis=0, out=out)

    if starting_value == 0:
        np.subtract(out, 1, out=out)
    else:
        np.multiply(out, starting_value, out=out)

    if allocated_output:
        if returns.ndim == 1 and isinstance(returns, pd.Series):
            out = pd.Series(out, index=returns.index)
        elif isinstance(returns, pd.DataFrame):
            out = pd.DataFrame(
                out, index=returns.index, columns=returns.columns,
            )

    return out


def cum_returns_final(returns, starting_value=0): # Compute total returns from simple returns.
    if len(returns) == 0:
        return np.nan

    if isinstance(returns, pd.DataFrame):
        result = (returns + 1).prod()
    else:
        result = np.nanprod(returns + 1, axis=0)

    if starting_value == 0:
        result -= 1
    else:
        result *= starting_value

    return result


def aggregate_returns(returns, convert_to): # Aggregates returns by week, month, or year.


    def cumulate_returns(x):
        return cum_returns(x).iloc[-1]

    if convert_to == WEEKLY:
        grouping = [lambda x: x.year, lambda x: x.isocalendar()[1]]
    elif convert_to == MONTHLY:
        grouping = [lambda x: x.year, lambda x: x.month]
    elif convert_to == QUARTERLY:
        grouping = [lambda x: x.year, lambda x: int(np.ceil(x.month/3.))]
    elif convert_to == YEARLY:
        grouping = [lambda x: x.year]
    else:
        raise ValueError(
            'convert_to must be {}, {} or {}'.format(WEEKLY, MONTHLY, YEARLY)
        )

    return returns.groupby(grouping).apply(cumulate_returns)


def max_drawdown(returns, out=None): # Determines the maximum drawdown of a strategy.
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 1:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    returns_array = np.asanyarray(returns)

    cumulative = np.empty(
        (returns.shape[0] + 1,) + returns.shape[1:],
        dtype='float64',
    )
    cumulative[0] = start = 100
    cum_returns(returns_array, starting_value=start, out=cumulative[1:])

    max_return = np.fmax.accumulate(cumulative, axis=0)

    nanmin((cumulative - max_return) / max_return, axis=0, out=out)
    if returns_1d:
        out = out.item()
    elif allocated_output and isinstance(returns, pd.DataFrame):
        out = pd.Series(out)

    return out


def annual_return(returns, period=DAILY, annualization=None): #Determines the mean annual growth rate of returns. This is equivilent
    # to the compound annual growth rate.

    if len(returns) < 1:
        return np.nan

    ann_factor = annualization_factor(period, annualization)
    num_years = len(returns) / ann_factor
    # Pass array to ensure index -1 looks up successfully.
    ending_value = cum_returns_final(returns, starting_value=1)

    return ending_value ** (1 / num_years) - 1


def cagr(returns, period=DAILY, annualization=None):
    return annual_return(returns, period, annualization)


def annual_volatility(returns,
                      period=DAILY,
                      alpha=2.0,
                      annualization=None,
                      out=None):
    # Determines the annual volatility of a strategy.
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 2:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    ann_factor = annualization_factor(period, annualization)
    nanstd(returns, ddof=1, axis=0, out=out)
    out = np.multiply(out, ann_factor ** (1.0 / alpha), out=out)
    if returns_1d:
        out = out.item()
    return out


def calmar_ratio(returns, period=DAILY, annualization=None):
    max_dd = max_drawdown(returns=returns)
    if max_dd < 0:
        temp = annual_return(
            returns=returns,
            period=period,
            annualization=annualization
        ) / abs(max_dd)
    else:
        return np.nan

    if np.isinf(temp):
        return np.nan

    return temp



def sharpe_ratio(returns,
                 risk_free=0,
                 period=DAILY,
                 annualization=None,
                 out=None):
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    return_1d = returns.ndim == 1

    if len(returns) < 2:
        out[()] = np.nan
        if return_1d:
            out = out.item()
        return out

    returns_risk_adj = np.asanyarray(_adjust_returns(returns, risk_free))
    ann_factor = annualization_factor(period, annualization)

    np.multiply(
        np.divide(
            nanmean(returns_risk_adj, axis=0),
            nanstd(returns_risk_adj, ddof=1, axis=0),
            out=out,
        ),
        np.sqrt(ann_factor),
        out=out,
    )
    if return_1d:
        out = out.item()

    return out


def sortino_ratio(returns,
                  required_return=0,
                  period=DAILY,
                  annualization=None,
                  out=None,
                  _downside_risk=None):
   
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    return_1d = returns.ndim == 1

    if len(returns) < 2:
        out[()] = np.nan
        if return_1d:
            out = out.item()
        return out

    adj_returns = np.asanyarray(_adjust_returns(returns, required_return))

    ann_factor = annualization_factor(period, annualization)

    average_annual_return = nanmean(adj_returns, axis=0) * ann_factor
    annualized_downside_risk = (
        _downside_risk
        if _downside_risk is not None else
        downside_risk(returns, required_return, period, annualization)
    )
    np.divide(average_annual_return, annualized_downside_risk, out=out)
    if return_1d:
        out = out.item()
    elif isinstance(returns, pd.DataFrame):
        out = pd.Series(out)

    return out


def downside_risk(returns,
                  required_return=0,
                  period=DAILY,
                  annualization=None,
                  out=None):
    # Determines the downside deviation below a threshold

    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 1:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    ann_factor = annualization_factor(period, annualization)

    downside_diff = np.clip(
        _adjust_returns(
            np.asanyarray(returns),
            np.asanyarray(required_return),
        ),
        np.NINF,
        0,
    )

    np.square(downside_diff, out=downside_diff)
    nanmean(downside_diff, axis=0, out=out)
    np.sqrt(out, out=out)
    np.multiply(out, np.sqrt(ann_factor), out=out)

    if returns_1d:
        out = out.item()
    elif isinstance(returns, pd.DataFrame):
        out = pd.Series(out, index=returns.columns)
    return out


SIMPLE_STAT_FUNCS = [
    cum_returns_final,
    annual_return,
    annual_volatility,
    sharpe_ratio,
    calmar_ratio,
    max_drawdown,
    sortino_ratio,
    stats.skew,
    stats.kurtosis,
    cagr
]


def calculate_rsi(values, period=14):
    """
    Calculates the Relative Strength Index (RSI) for a given list of values.

    Args:
        values (list, np.ndarray): List or numpy array of values.
        period (int): Period for calculating the RSI. Default is 14.

    Returns:
        np.ndarray: Numpy array of RSI values.
    """
    if len(values) < period:
        raise ValueError("Number of values is less than the specified period.")

    deltas = np.diff(values)
    gains = np.where(deltas > 0, deltas, 0)
    losses = -np.where(deltas < 0, deltas, 0)
    avg_gain = np.convolve(gains, np.ones(period) / period, mode='valid')
    avg_loss = np.convolve(losses, np.ones(period) / period, mode='valid')
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # Pad the resulting array to match the original length
    rsi = np.pad(rsi, (values.size - rsi.size, 0), mode='constant', constant_values=np.nan)
    rsi = (pd.Series(rsi)).tolist()
    rsi = [item if item is not None else 0 for item in rsi]
    return np.array(rsi)


def calculate_sma(values, period):
    """
    Calculates the Simple Moving Average (SMA) for a given list of values and period.

    Args:
        values (list, np.ndarray): List or numpy array of values.
        period (int): Period for which to calculate the SMA.

    Returns:
        np.ndarray: Numpy array of SMA values.
    """
    if len(values) < period:
        raise ValueError("Number of values is less than the specified period.")

    sma_values = np.convolve(values, np.ones((period,))/period, mode='valid')
    sma_values = np.concatenate(([None]*(period-1), sma_values))
    sma_values = [item if item is not None else 0 for item in sma_values]
    return np.array(sma_values)

def calculate_ema(values, period):
    """
    Calculates the Exponential Moving Average (EMA) for a given list of values and period.

    Args:
        values (list, np.ndarray): List or numpy array of values.
        period (int): Period for which to calculate the EMA.

    Returns:
        np.ndarray: Numpy array of EMA values.
    """
    if len(values) < period:
        raise ValueError("Number of values is less than the specified period.")

    alpha = 2 / (period + 1)
    ema_values = [None]
    for i in range(1, len(values)):
        if ema_values[-1] is None:
            ema = values[i]
        else:
            ema = alpha * values[i] + (1 - alpha) * ema_values[-1]
        ema_values.append(ema)
    ema_values = [item if item is not None else 0 for item in ema_values]
    return np.array(ema_values)



def zscore_with_zero_exempt(series):
    # we make the healthy assumption that the ratio of the price or moving averages
    # of two stocks can never be zero
    non_zero_elements = [x for x in series if x != 0]
    mean = np.mean(non_zero_elements)
    std = np.std(non_zero_elements)

    def zscore_element(x):
        return (x - mean) / std if x != 0 else 0

    return [zscore_element(x) for x in series]

# a = [0, 0, 0, 0, 0, 1, 2, 3, 4]
# result = zscore_with_zero_exempt(a)
# print(result)
# [0, 0, 0, 0, 0, -1.3416407864998738, -0.4472135954999579, 0.4472135954999579, 1.3416407864998738]


# calculate z-score
def zscore(series):
    return (series - series.mean()) / np.std(series)

class Backtesting:
    def __init__(self, data, asset1, asset2,allocation,signal_effectiveness_metric='hit_rate',use_percentile=False):
        self.data = data # data = (train_data, test_data)
        self.train_data = data['train_data']
        self.test_data = data['test_data']
        # self.whole_data = data['whole_data']
        
        self.asset1 = asset1
        self.asset2 = asset2
        self.allocation = allocation
        self.signal_effectiveness_metric = signal_effectiveness_metric
        
        self.portfolio = pd.DataFrame()
        self.portfolio['asset1'] = self.test_data[self.asset1] 
        self.portfolio['asset2'] = self.test_data[self.asset2]

        self.use_percentile = use_percentile

    def extract_features(self,data):   
        # metric is a dataframe containing all the metrics at each time step
        sma_5_ratio = calculate_sma(data['asset1'], period=5) / calculate_sma(data['asset2'], period=5)
        sma_10_ratio = calculate_sma(data['asset1'], period=10) / calculate_sma(data['asset2'], period=10)
        sma_30_ratio = calculate_sma(data['asset1'], period=30) / calculate_sma(data['asset2'], period=30)
        sma_50_ratio = calculate_sma(data['asset1'], period=50) / calculate_sma(data['asset2'], period=50)
        ema_5_ratio = calculate_ema(data['asset1'], period=5) / calculate_ema(data['asset2'], period=5)
        ema_10_ratio = calculate_ema(data['asset1'], period=10) / calculate_ema(data['asset2'], period=10)
        ema_30_ratio = calculate_ema(data['asset1'], period=30) / calculate_ema(data['asset2'], period=30)
        ema_50_ratio = calculate_ema(data['asset1'], period=50) / calculate_ema(data['asset2'], period=50)
        ema_95_ratio = calculate_ema(data['asset1'], period=95) / calculate_ema(data['asset2'], period=95)
        rsi_14_ratio = calculate_rsi(data['asset1'], period=14) / calculate_rsi(data['asset1'], period=14)

        # clean the extracts
        sma_5_ratio = [item if isinstance(item, (int, float)) and not np.isnan(item) else 0 for item in sma_5_ratio]
        sma_10_ratio = [item if isinstance(item, (int, float)) and not np.isnan(item) else 0 for item in sma_10_ratio]
        sma_30_ratio = [item if isinstance(item, (int, float)) and not np.isnan(item) else 0 for item in sma_30_ratio]
        sma_50_ratio = [item if isinstance(item, (int, float)) and not np.isnan(item) else 0 for item in sma_50_ratio]
        ema_5_ratio = [item if isinstance(item, (int, float)) and not np.isnan(item) else 0 for item in ema_5_ratio]
        ema_10_ratio = [item if isinstance(item, (int, float)) and not np.isnan(item) else 0 for item in ema_10_ratio]
        ema_30_ratio = [item if isinstance(item, (int, float)) and not np.isnan(item) else 0 for item in ema_30_ratio]
        ema_50_ratio = [item if isinstance(item, (int, float)) and not np.isnan(item) else 0 for item in ema_50_ratio]
        ema_95_ratio = [item if isinstance(item, (int, float)) and not np.isnan(item) else 0 for item in ema_95_ratio]
        rsi_14_ratio = [item if isinstance(item, (int, float)) and not np.isnan(item) else 0 for item in rsi_14_ratio]

        data['price_ratio'] = data['asset1']/data['asset2']
        data['sma_5_ratio'] =  sma_5_ratio
        data['sma_10_ratio'] =  sma_10_ratio
        data['sma_30_ratio'] = sma_30_ratio
        data['sma_50_ratio'] =  sma_50_ratio
        data['ema_5_ratio'] =  ema_5_ratio
        data['ema_10_ratio'] =  ema_10_ratio
        data['ema_30_ratio'] = ema_30_ratio
        data['ema_50_ratio'] =  ema_50_ratio
        data['ema_95_ratio'] = ema_95_ratio
        data['rsi_14_ratio'] = rsi_14_ratio
        
        return data

    def estimate_signal_feature_importance(self,feature_name):
        # access the feature from self.feature
        feature_data = self.features[feature_name]
        # get z score of feature to create signal
        feature_zscore = zscore_with_zero_exempt(feature_data)
        feature_zscore_without_zeroes = [y for y in feature_zscore if y != 0]
        # create signal based on the z score
        if self.use_percentile == True:
            z_upper_limit = np.percentile(feature_zscore_without_zeroes, 75) # single value that can be shared for test data
            z_lower_limit = np.percentile(feature_zscore_without_zeroes, 25)
        else:
            z_upper_limit = np.mean(feature_zscore_without_zeroes) + np.std(feature_zscore_without_zeroes) # single value that can be shared for test data
            z_lower_limit = np.mean(feature_zscore_without_zeroes) - np.std(feature_zscore_without_zeroes)

        signals1 = np.select([feature_zscore> \
                                        z_upper_limit, feature_zscore < z_lower_limit], [-1, 1], default=0)
        
        actual_return1 = self.features['asset1'].pct_change()

        # also evaluate for signal2 and compute average of the two as the total signal metric
        signals2 = -signals1
        actual_return2 = self.features['asset2'].pct_change()


        # evaluate the performance
        computed_signal_metric = 0
        if self.signal_effectiveness_metric == 'hit_rate':
            hit_rate_1 = compute_hit_rate(signals1, actual_return1)
            hit_rate_2 = compute_hit_rate(signals2, actual_return2)
            computed_signal_metric = (hit_rate_1 + hit_rate_2) / 2

        elif self.signal_effectiveness_metric == 'profitability':
            profitability_1 = compute_profitability(signals1, actual_return1)
            profitability_2 = compute_profitability(signals2, actual_return2)
            computed_signal_metric = (profitability_1 + profitability_2) / 2

        elif self.signal_effectiveness_metric == 'risk_reward':
            risk_reward_ratio_1 = compute_risk_reward_ratio(signals1, actual_return1)
            risk_reward_ratio_2 = compute_risk_reward_ratio(signals2, actual_return2)
            computed_signal_metric = (risk_reward_ratio_1 + risk_reward_ratio_2) / 2
    
        elif self.signal_effectiveness_metric == 'maximum_drawdown':
            maximum_drawdown_1 = compute_maximum_drawdown(signals1, actual_return1)
            maximum_drawdown_2 = compute_maximum_drawdown(signals2, actual_return2)
            computed_signal_metric = (maximum_drawdown_1 + maximum_drawdown_2) / 2

        elif self.signal_effectiveness_metric == 'win_rate':
            win_rate_1 = compute_win_rate(signals1, actual_return1)
            win_rate_2 = compute_win_rate(signals2, actual_return2)
            computed_signal_metric = (win_rate_1 + win_rate_2) / 2

        return computed_signal_metric,signals1



    def generate_trading_signal(self):      
        # create a dataframe for trading signals

        # signals should be made on training data

        self.signals = pd.DataFrame()
        self.signals['asset1'] = self.train_data[self.asset1] 
        self.signals['asset2'] = self.train_data[self.asset2]

        # try to optimize signal generation here
        features = pd.DataFrame()
        features['asset1'] = self.signals['asset1']
        features['asset2'] = self.signals['asset2']
        features = self.extract_features(features)
        # self.features = features.drop(columns=['asset1', 'asset2'])
        self.features = features
        # print(features.head(10))

        # next, be able to select the feature that optimizes the chosen objective
        metric_dict = {}
        signal1_dict = {}
        ratio_features = [x for x in self.features.columns if x not in ['asset1', 'asset2']]


        for ratio in ratio_features:
            importance,signal1 = self.estimate_signal_feature_importance(ratio)
            metric_dict[ratio] = importance
            signal1_dict[ratio] = signal1


        metric_dict = {key: value for key, value in metric_dict.items() if not np.isinf(value)}
        metric_dict = {key: value for key, value in metric_dict.items() if value != 0.0}



        print(metric_dict)

        # select the feature with the highest importance
        self.optimal_feature = max(metric_dict, key=metric_dict.get)

        print(self.optimal_feature)

        # fetch the ratio from the features dataframe
        ratios = self.features[self.optimal_feature]



        # ratios = self.signals.asset1 / self.signals.asset2
        self.train_input = ratios



        # build model based on the ratio and the z scores

        # Create the Random Forest Regressor model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

        # print(list(signal1_dict[self.optimal_feature])) # debug

        # calculate z-scores
        self.signals['z'] = zscore_with_zero_exempt(ratios)
    


        train_x = self.train_input
        train_y = self.signals['z']

        

        # clean train_x and train_y just before training
        train_x = [x for x in train_x if x != 0]
        train_y = [y for y in train_y if y != 0]

        # print(train_y)

        # define upper and lower threshold
        if self.use_percentile == True:
            self.signals['z upper limit'] = np.percentile(train_y, 75) # single value that can be shared for test data
            self.signals['z lower limit'] = np.percentile(train_y, 25) # train_y is signals['z'] without the zeroes
        else:
            self.signals['z upper limit'] = np.mean(train_y) + np.std(train_y) # single value that can be shared for test data
            self.signals['z lower limit'] = np.mean(train_y) - np.std(train_y) # train_y is signals['z'] without the zeroes


        # Train the model on the training data
        train_x = np.array(train_x)
        train_y = np.array(train_y)

        train_x = train_x.reshape(-1,1)
        train_y = train_y.reshape(-1,1)
        self.model.fit(train_x, train_y)


        # print(self.signals['z'])# ignore the zeroes in the limits computation

        # create signal - short if z-score is greater than upper limit else long
        self.signals['signals1'] = 0
        self.signals['signals1'] = np.select([self.signals['z'] > \
                                        self.signals['z upper limit'], self.signals['z'] < self.signals['z lower limit']], [-1, 1], default=0)

        # we take the first order difference to obtain portfolio position in that stock
        self.signals['positions1'] = self.signals['signals1'].diff()
        self.signals['signals2'] = -self.signals['signals1']
        self.signals['positions2'] = self.signals['signals2'].diff()

        # print(list(self.signals['signals1']))
        # print(self.signals.head(10))
        # print(self.signals.tail(10))

    def plot_signals(self,chart_name = ""):
        # visualize trading signals and position
        # use signals for training data
        # use positions for testing data

        fig=plt.figure(figsize=(14,6))
        bx = fig.add_subplot(111)   
        bx2 = bx.twinx()

        #plot two different assets
        l1, = bx.plot(self.signals['asset1'], c='#4abdac')
        l2, = bx2.plot(self.signals['asset2'], c='#907163')
        u1, = bx.plot(self.signals['asset1'][self.signals['positions1'] == 1], lw=0, marker='^', markersize=8, c='g',alpha=0.7)
        d1, = bx.plot(self.signals['asset1'][self.signals['positions1'] == -1], lw=0,marker='v',markersize=8, c='r',alpha=0.7)
        u2, = bx2.plot(self.signals['asset2'][self.signals['positions2'] == 1], lw=0,marker=2,markersize=9, c='g',alpha=0.9, markeredgewidth=3)
        d2, = bx2.plot(self.signals['asset2'][self.signals['positions2'] == -1], lw=0,marker=3,markersize=9, c='r',alpha=0.9,markeredgewidth=3)
        bx.set_ylabel(self.asset1,)
        bx2.set_ylabel(self.asset2, rotation=270)
        bx.yaxis.labelpad=15
        bx2.yaxis.labelpad=15
        bx.set_xlabel('Date')
        bx.xaxis.labelpad=15
        plt.legend([l1,l2,u1,d1,u2,d2], [self.asset1, self.asset2,'LONG {}'.format(self.asset1),
                'SHORT {}'.format(self.asset1),
                'LONG {}'.format(self.asset2),
                'SHORT {}'.format(self.asset2)], loc ='best')
        plt.title('Pair Trading')
        plt.xlabel('Date')
        plt.grid(True)

        plt.tight_layout()
        if chart_name == '':
            plt.savefig('results/signal/'+str(self.asset1)+'_'+str(self.asset2)+'_signal_chart', dpi=300)
        else:
            plt.savefig(chart_name, dpi=300)

    def evaluate_signals(self):
        # calculate actual return for both stocks
        self.signals['actual_return1'] = self.signals['asset1'].pct_change()
        self.signals['actual_return2'] = self.signals['asset2'].pct_change()

        print("=================== Evaluation of signal for "+str(self.asset1) + "    =================")
        hit_rate = compute_hit_rate(self.signals['signals1'], self.signals['actual_return1'])
        profitability = compute_profitability(self.signals['signals1'], self.signals['actual_return1'])
        risk_reward_ratio = compute_risk_reward_ratio(self.signals['signals1'], self.signals['actual_return1'])

        maximum_drawdown = compute_maximum_drawdown(self.signals['signals1'], self.signals['actual_return1'])
        win_rate = compute_win_rate(self.signals['signals1'], self.signals['actual_return1'])

        print("Hit Rate:", hit_rate)
        print("Profitability:", profitability)
        print("Risk-to-Reward Ratio:", risk_reward_ratio)

        print("Maximum Drawdown:", maximum_drawdown)
        print("Win Rate:", win_rate)

        print('\n')

        print("=================== Evaluation of signal for "+str(self.asset2) + "    =================")
        hit_rate = compute_hit_rate(self.signals['signals2'], self.signals['actual_return2'])
        profitability = compute_profitability(self.signals['signals2'], self.signals['actual_return2'])
        risk_reward_ratio = compute_risk_reward_ratio(self.signals['signals2'], self.signals['actual_return2'])

        maximum_drawdown = compute_maximum_drawdown(self.signals['signals2'], self.signals['actual_return2'])
        win_rate = compute_win_rate(self.signals['signals2'], self.signals['actual_return2'])

        print("Hit Rate:", hit_rate)
        print("Profitability:", profitability)
        print("Risk-to-Reward Ratio:", risk_reward_ratio)
    
        print("Maximum Drawdown:", maximum_drawdown)
        print("Win Rate:", win_rate)
        print('\n')



    def backtest(self):

        # use the signal model to create the signal on the test set
        # show another set of plots

        # need to adjust self.signals['positions1'] , self.signals['positions2'] 
        # also need to set self.signals['asset1'] and self.signals['asset2'] to the test set

        # combine both training and testing sets, then extract features on the whole data
        # then extract the test set again

        # create a data frame with the prices of the two stocks
        new_features = pd.DataFrame()
        new_features['asset1'] = list(self.train_data[self.asset1]) + list(self.test_data[self.asset1])
        new_features['asset2'] = list(self.train_data[self.asset2]) + list(self.test_data[self.asset2])

        # extract features 
        new_features = self.extract_features(new_features)

        # obtain the data of the optimal feature
        whole_optimal_feature_data = new_features[self.optimal_feature]
        self.test_input = whole_optimal_feature_data[-len(self.test_data[self.asset1]):]


        # Make predictions on the test data
        test_x = np.array(self.test_input)
        test_x = test_x.reshape(-1, 1)
        test_y = self.model.predict(test_x)

        self.z_score_pred = test_y.tolist()

        # print(self.z_score_pred)

        self.portfolio['z'] = self.z_score_pred # this is where I will make predictions using the model trained with the training data
 
        self.portfolio['z upper limit'] = (self.signals['z upper limit'])[0] # setting the limit to any value in the upper limit column
        self.portfolio['z lower limit'] = (self.signals['z lower limit'])[0]

        # create signal - short if z-score is greater than upper limit else long
        self.portfolio['signals1'] = 0
        self.portfolio['signals1'] = np.select([self.portfolio['z'] > \
                                        self.portfolio['z upper limit'], self.portfolio['z'] < self.portfolio['z lower limit']], [-1, 1], default=0)

        # we take the first order difference to obtain portfolio position in that stock
        self.portfolio['positions1'] = self.portfolio['signals1'].diff()
        self.portfolio['signals2'] = -self.portfolio['signals1']
        self.portfolio['positions2'] = self.portfolio['signals2'].diff()

        # print(self.portfolio.head(10))

        # self.portfolio.to_csv('test_portfolio.csv')

        # initial capital to calculate the actual pnl
        initial_capital = self.allocation / 2

        # shares to buy for each position
        positions1 = initial_capital// max(self.portfolio['asset1'])
        positions2 = initial_capital// max(self.portfolio['asset2'])

        # since there are two assets, we calculate each asset Pnl 
        # separately and in the end we aggregate them into one portfolio


        self.portfolio['holdings1'] = self.portfolio['positions1'].cumsum() * self.portfolio['asset1'] * positions1
        self.portfolio['cash1'] = initial_capital - (self.portfolio['positions1'] * self.portfolio['asset1'] * positions1).cumsum()
        self.portfolio['total asset1'] = self.portfolio['holdings1'] + self.portfolio['cash1']
        self.portfolio['return1'] = self.portfolio['total asset1'].pct_change()

        # pnl for the 2nd asset
        self.portfolio['holdings2'] = self.portfolio['positions2'].cumsum() * self.portfolio['asset2'] * positions2
        self.portfolio['cash2'] = initial_capital - (self.portfolio['positions2'] * self.portfolio['asset2'] * positions2).cumsum()
        self.portfolio['total asset2'] = self.portfolio['holdings2'] + self.portfolio['cash2']
        self.portfolio['return2'] = self.portfolio['total asset2'].pct_change()


        # total pnl and z-score
        self.portfolio['total asset'] = self.portfolio['total asset1'] + self.portfolio['total asset2']
        self.portfolio = self.portfolio.dropna()

        

    def plot_portfolio_performance(self, chart_name = ""):
        # plot the asset value change of the portfolio and pnl along with z-score
        # first plot the signal with the test set
        # then provide the backtesting results
        # if possible, have both plots in one figure
        

        fig=plt.figure(figsize=(14,6))
        ax = fig.add_subplot(111)   
        ax2 = ax.twinx()

        #plot two different assets
        l1, = ax.plot(self.portfolio['asset1'], c='#4abdac')
        l2, = ax2.plot(self.portfolio['asset2'], c='#907163')
        u1, = ax.plot(self.portfolio['asset1'][self.portfolio['positions1'] == 1], lw=0, marker='^', markersize=8, c='g',alpha=0.7)
        d1, = ax.plot(self.portfolio['asset1'][self.portfolio['positions1'] == -1], lw=0,marker='v',markersize=8, c='r',alpha=0.7)
        u2, = ax2.plot(self.portfolio['asset2'][self.portfolio['positions2'] == 1], lw=0,marker=2,markersize=9, c='g',alpha=0.9, markeredgewidth=3)
        d2, = ax2.plot(self.portfolio['asset2'][self.portfolio['positions2'] == -1], lw=0,marker=3,markersize=9, c='r',alpha=0.9,markeredgewidth=3)
        ax.set_ylabel(self.asset1,)
        ax2.set_ylabel(self.asset2, rotation=270)
        ax.yaxis.labelpad=15
        ax2.yaxis.labelpad=15
        ax.set_xlabel('Date')
        ax.xaxis.labelpad=15
        ax.legend([l1,l2,u1,d1,u2,d2], [self.asset1, self.asset2,'LONG {}'.format(self.asset1),
                'SHORT {}'.format(self.asset1),
                'LONG {}'.format(self.asset2),
                'SHORT {}'.format(self.asset2)], loc ='best')
        plt.title('Pair Trading')
        plt.grid(True)


        if chart_name == '':
            plt.savefig('results/backtesting/'+str(self.asset1)+'_'+str(self.asset2)+'_portfolio_signal_chart', dpi=300)
        else:
            plt.savefig(chart_name, dpi=300)      


        # Plot the second plot    
        fig2 = plt.figure(figsize=(14,6),)
        ax3 = fig2.add_subplot(111)
        ax4 = ax3.twinx()
        l3, = ax3.plot(self.portfolio['total asset'], c='g')
        l4, = ax4.plot(self.portfolio['z'], c='black', alpha=0.3)
        b2 = ax4.fill_between(self.portfolio.index,self.portfolio['z upper limit'],\
                        self.portfolio['z lower limit'], \
                        alpha=0.2,color='#ffb48f')
        ax3.set_ylabel('Asset Value')
        ax4.set_ylabel('Z Statistics',rotation=270)
        ax3.yaxis.labelpad=15
        ax4.yaxis.labelpad=15
        ax3.set_xlabel('Date')
        ax3.xaxis.labelpad=15
        plt.title('Portfolio Performance with Profit and Loss')
        
        if chart_name == '':
            plt.savefig('results/backtesting/'+str(self.asset1)+'_'+str(self.asset2)+'_portfolio_perf_chart', dpi=300)
        else:
            plt.savefig(chart_name, dpi=300)

    def obtain_performance(self):
        # calculate CAGR
        final_portfolio = self.portfolio['total asset'].iloc[-1]
        # delta = (self.portfolio.index[-1] - self.portfolio.index[0]).days
        # print('Number of days = ', delta)


        total_returns = self.portfolio['total asset'].pct_change()
        self.total_returns = total_returns
        self.total_assets_value = final_portfolio

        mdd = max_drawdown(total_returns) # I will need to convert to percentage myself
        shp_rat = sharpe_ratio(total_returns, annualization = 252)
        ann_vol = annual_volatility(total_returns)
        sortino = sortino_ratio(total_returns)
        cagr_estimate = cagr(total_returns)


        print("=======Performance Metrics ("+str(self.asset1)+'_'+str(self.asset2)+'_pair)'+"=======")
        print('Compound Annual Growth Rate: {0: .4%}'.format(cagr_estimate))
        print('Maximum Drawdown: {0: .4%}'.format(mdd))
        print('Annual Volatility: {0: .4%}'.format(ann_vol))
        print('Sharpe Ratio: ', shp_rat)
        print('Sortino Ratio: ', sortino)
        print('\n')
        print('\n')

    

  

# ================= Main code starts here ==========
# 1. DATA EXPLORATION

# fetch the data using pandas
technical = pd.read_csv('data/data.csv')
print('Data shape:', technical.shape)# explore the datasets

# explore the technical data
print(technical.head(5))
print(technical.describe())

# 2. DATA MASSAGING AND CLEANING

# For better handling of the data, I will use pivot tables to transform the technical data

# Pivot the DataFrame
pivot_df = technical.pivot_table(index='date', columns='ticker')

# Then proceed to obtaining the price/volume tables
price_df = pivot_df['last']
volume_df = pivot_df['volume']


# check the first 3 data in each table
print(price_df.head(3))
print(volume_df.head(3))

# check the last 3 data in each table
print(price_df.tail(3))
print(volume_df.tail(3))

# Check for duplicate values
duplicate_dates = price_df.index.duplicated().sum()
if duplicate_dates > 0:
    print(f'There are {duplicate_dates} duplicate dates in the dataset.')
else:
    print('There are no duplicate dates in the dataset.')



'''
COMMENT:
After my checks, there are many NaN in the technical tables i.e price/volume tables
'''


'''
!ACTION:
Use the clean_data function to 
1. drop stocks (columns) that do not have up to 90% of their data to be non-NaN
2. Use forward filling method for data imputation
3. Convert the date in each table to datetime for easy handling
'''
filtered_price_df = clean_data(price_df)
filtered_volume_df = clean_data(volume_df)


# check the filtered volume
print(filtered_volume_df.describe())


'''
COMMENT:
The different tables have different number of stocks, next, I will collate the common stocks in all the tables
'''

# get the column names of the different datasets and collate them
# get the column names of the different datasets and collate them
filtered_price_df_cols = (filtered_price_df.columns).tolist()
filtered_volume_df_cols = (filtered_volume_df.columns).tolist()

collated_cols = ([x for x in filtered_price_df_cols if all(x in lst for lst in [filtered_volume_df_cols])])
collated_cols = sorted(collated_cols)

print(len(collated_cols))


# obtain the collated data with the 205 stocks
collated_price_df = filtered_price_df[collated_cols]
collated_volume_df = filtered_volume_df[collated_cols]

'''
NEXT:
Separate the data into training and test sets

training set: 2013, 2014, 2015 data

test set: 2016, 2017, 2018, 2019, 2020, 2021 data
'''

# divide data into training and testing sets
# convert index to datetime
collated_price_df.index = pd.to_datetime(collated_price_df.index)
collated_volume_df.index = pd.to_datetime(collated_volume_df.index)

# separate the data
train_collated_price_df = collated_price_df[(collated_price_df.index).year <= 2015]
test_collated_price_df = collated_price_df[(collated_price_df.index).year > 2015]


train_collated_volume_df = collated_volume_df[(collated_volume_df.index).year <= 2015]
test_collated_volume_df = collated_volume_df[(collated_volume_df.index).year > 2015]

# manual check
print(train_collated_price_df.head(3))
print(train_collated_price_df.tail(3))


print(test_collated_price_df.head(3))
print(test_collated_price_df.tail(3))



# 3. PAIR SELECTION
'''
A robust and multi criteria pair selection algorithm will be used here
'''

series_analyser = class_SeriesAnalyser.SeriesAnalyser()
data_processor = class_DataProcessor.DataProcessor()
training_returns = data_processor.get_return_series(train_collated_price_df)
training_returns.head()

print('Total number of possible pairs: ', len(training_returns.columns)*(len(training_returns.columns)-1)/2)

average_volume_info = pd.DataFrame(train_collated_volume_df.mean(axis=0), columns=['volume']) # compute average volume of each stock using training data

print(average_volume_info.head(5))

np.random.seed(42)
N_PRIN_COMPONENTS = 50
pca = PCA(n_components=N_PRIN_COMPONENTS)
pca.fit(training_returns)

# adding volume information to improve clustering performance
X = np.hstack(
    (pca.components_.T,
    average_volume_info['volume'][training_returns.columns].values[:,np.newaxis])
)

X = preprocessing.StandardScaler().fit_transform(X)

clustered_series_all, clustered_series, counts, clf = series_analyser.apply_DBSCAN(3.5,
                                                                                   3,
                                                                                   X,
                                                                                   training_returns)


plt.figure(1,figsize=(16,12))
cluster_size(counts)


print(counts)
print('Average cluster size: ', np.mean(counts))


for clust in range(len(counts)):
    symbols = list(clustered_series[clustered_series==clust].index)
    means = np.log(train_collated_price_df[symbols].mean())
    series = np.log(train_collated_price_df[symbols]).sub(means)
    series.plot(figsize=(10,5))#title='ETFs Time Series for Cluster %d' % (clust+1))
    #plt.ylabel('Normalized log prices', size=12)
    #plt.xlabel('Date', size=12)
    plt.savefig('results/clustering_outcome/cluster_{}.png'.format(str(clust+1)), bbox_inches='tight', pad_inches=0.1)


subsample = 1000 # 2500
min_half_life = 1 #78 # number of points in a day
max_half_life = 252 #20000 #~number of points in a year: 78*252

pairs_unsupervised, unique_tickers = series_analyser.get_candidate_pairs(clustered_series=clustered_series,
                                                            pricing_df_train=train_collated_price_df,
                                                            pricing_df_test=test_collated_price_df,
                                                            min_half_life=min_half_life,
                                                            max_half_life=max_half_life,
                                                            min_zero_crosings=12,
                                                            p_value_threshold=0.05,
                                                            hurst_threshold=0.5,
                                                            subsample=subsample
                                                            )


all_pairs = []
t_statistic = []
p_value = []
coint_coef = []
zero_cross = []
half_life = []
hurst_exponent = []
spread = []

pairing_metric = [t_statistic, p_value, coint_coef, zero_cross, half_life,hurst_exponent, spread]
pairing_metric_string = ['t_statistic', 'p_value', 'coint_coef', 'zero_cross', 'half_life', 'hurst_exponent', 'spread']

for pair in pairs_unsupervised:
    all_pairs.append((pair[0],pair[1]))
    for idx, metric in enumerate(pairing_metric):
        metric.append(pair[2][pairing_metric_string[idx]])
    # print(pair[2])

print(all_pairs)
print(hurst_exponent)


pairing_metric_df = pd.DataFrame({
    'pairs': all_pairs,
    't_statistic': t_statistic, 
    'p_value': p_value, 
    'coint_coef': coint_coef, 
    'zero_cross': zero_cross, 
    'half_life': half_life, 
    'hurst_exponent': hurst_exponent,
})




# 4. SIGNAL GENERATION AND BACKTESTING

# loop through all_pairs

# use the beta information to compute portfolio weights

Total_capital = 1000000

for idx, pair in enumerate(all_pairs):
    stock1 = pair[0]
    stock2 = pair[1] 

    pair_capital = Total_capital / len(all_pairs) # equal distribution of capital among pairs
    filtered_closing_data = {'train_data' : train_collated_price_df, 'test_data' : test_collated_price_df}

    model = Backtesting(filtered_closing_data, stock1, stock2,allocation = pair_capital,signal_effectiveness_metric='risk_reward',use_percentile=True)
    model.generate_trading_signal()
    model.evaluate_signals()
    model.plot_signals()
    model.backtest()
    model.plot_portfolio_performance()
    model.obtain_performance()


    # collate total portfolio value
    if idx == 0: # first iteration
        total_portfolio = model.portfolio['total asset']
    else:
        total_portfolio += model.portfolio['total asset']


# compute overall porfolio performance and give metrics

final_portfolio = total_portfolio.iloc[-1]

total_returns = total_portfolio.pct_change()
total_assets_value = final_portfolio

mdd = max_drawdown(total_returns) # I will need to convert to percentage myself
shp_rat = sharpe_ratio(total_returns, annualization = 245) # 245 for this data
ann_vol = annual_volatility(total_returns)
sortino = sortino_ratio(total_returns)
cagr_estimate = cagr(total_returns)


print("======================  Overall Performance Metrics  ========================")
print('Final portfolio value: {0:.2f}'.format(total_assets_value))
print('Compound Annual Growth Rate: {0: .4%}'.format(cagr_estimate))
print('Maximum Drawdown: {0: .4%}'.format(mdd))
print('Annual Volatility: {0: .4%}'.format(ann_vol))
print('Sharpe Ratio: ', shp_rat)
print('Sortino Ratio: ', sortino)
print('\n')
print('\n')


'''
SUMMARY:
In this project, the market-neutral trading strategy is created using python. Several steps were taken to achieve this task. In summary, the steps taken are;
1. The data was adequately explored and cleaned.
2. Rules were put in place to guide the pair selection process. First, DBSCAN clustering algorithm was used to cluster the returns of the stocks. The average volume of the stocks in the training period was used to improve the quality of the clustering process. Then an improved pair selection algorithm which involves multiple criteria (Hurst exponent, half life, zero crossing, cointegration etc) was used to select the best trading pairs.
3. The allocations of the assets for each pair was done equally since information such as Beta is not available and I do not want to do very complex portfolio optimization for each of the pairs.
4. The signals for each pair was generated using the z-score with the training data. The z-score in this solution used the 25th and 75th percentiles to compute the lower and upper thresholds for each pair. The strategy framework produced also made available the use of the traditional mean and standard deviation for threshold computation.
5. Also for the signal generation, feature selection optimization was introduced to help improve the robustness of the signals generated.
6. On the test data, the optimal feature from the signal generation stage is used to predict the signal using a simple trained machine learning model. Backtesting was carried out and the results of the process are presented.

'''