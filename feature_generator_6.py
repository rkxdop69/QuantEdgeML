import pandas as pd
import numpy as np
from indicators import *
from scipy.stats import linregress

def feature_gen(data:pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data['close_shift'] = data['close'].shift(1)
    data['open_shift'] = data['open'].shift(1)
    data['high_shift'] = data['high'].shift(1)
    data['low_shift'] = data['low'].shift(1)
    data["ema7"] = data["close"].ewm(span=7, adjust=False, min_periods=1).mean()
    data["ema12"] = data["close"].ewm(span=12, adjust=False, min_periods=1).mean()
    data['std12'] = data['close'].rolling(window=12).std()

    slopes=[]
    for i in range(len(data)):
        if i < 7:
            # Not enough temp_data points for the rolling window
            slopes.append(np.nan)
        else:
            # Extract the temp_data for the rolling window
            y = data['close'].iloc[i - 7:i]
            x = np.arange(len(y))  # Independent variable: time steps
            slope, _, _, _, _ = linregress(x, y)  # Fit a linear regression
            slopes.append(slope)
    data['lin_reg']=slopes


    slopes=[]
    for i in range(len(data)):
        if i < 7:
            # Not enough temp_data points for the rolling window
            slopes.append(np.nan)
        else:
            # Extract the temp_data for the rolling window
            y = data['lin_reg'].iloc[i - 7:i]
            x = np.arange(len(y))  # Independent variable: time steps
            slope, _, _, _, _ = linregress(x, y)  # Fit a linear regression
            slopes.append(slope)
    data['acc']=slopes

    data['f1'] = data['close'] / data['close_shift'] - 1
    data['f2'] = data['open'] / data['open_shift'] - 1
    data['f3'] = data['high'] / data['high_shift'] - 1
    data['f4'] = data['low'] / data['low_shift'] - 1
    data['f5'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['f6'] = (data['high'] - data['close']) / (data['high'] - data['low'])
    data['f7'] = (data['high'] - data['open']) / (data['high'] - data['low'])
    data['f8'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['f9'] = (data['open'] - data['low']) / (data['high'] - data['low'])
    data['f10'] = data['f6'] / data['f7']
    data['f11'] = data['f8'] / data['f9']
    data['f12'] = data['f6'] / data['f9']
    data['f13'] = (data['close_shift'] - data['open_shift']) / (data['high_shift'] - data['low_shift'])
    data['f14'] = (data['high_shift'] - data['close_shift']) / (data['high_shift'] - data['low_shift'])
    data['f15'] = (data['high_shift'] - data['open_shift']) / (data['high_shift'] - data['low_shift'])
    data['f16'] = (data['close_shift'] - data['low_shift']) / (data['high_shift'] - data['low_shift'])
    data['f17'] = (data['open_shift'] - data['low_shift']) / (data['high_shift'] - data['low_shift'])
    data['f18'] = rsi(data['close'], lookback=14)
    data['f19'] = williams_r(data, period=14)
    data['f20'] = data['ema7'] / data['ema12'] - 1
    data['f21'] = data['lin_reg'] / data['close']
    data['f22'] = data['acc'] / data['close']
    data['f23'] = data['ema12'] / data['close'] - 1
    data['f24'] = data['std12'] / data['close']
    data['f25'] = (data['ema12'] + 2 * data['std12']) / data['close'] - 1
    data['f26'] = (data['ema12'] - 2 * data['std12']) / data['close'] - 1



    return data.loc[:, 'f1':'f26']