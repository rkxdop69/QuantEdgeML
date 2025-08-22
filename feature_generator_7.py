import numpy as np
import pandas as pd
from scipy.stats import linregress
from indicators import *
import pandas_ta as ta


def feature_gen(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    eps = 1e-6  # for safe division

    # Lagged OHLC
    data['close_shift'] = data['close'].shift(1)
    data['open_shift'] = data['open'].shift(1)
    data['high_shift'] = data['high'].shift(1)
    data['low_shift'] = data['low'].shift(1)
    adx1 = ta.adx(high=data["high"], low=data["low"], close=data["close"])


    rollmax1 = data["close"].rolling(window=3, min_periods=1).max()
    rollmin1 = data["close"].rolling(window=3, min_periods=1).min()
    rollmax2 = data["close"].rolling(window=9, min_periods=1).max()
    rollmin2 = data["close"].rolling(window=9, min_periods=1).min()
    rollmax3 = data["close"].rolling(window=20, min_periods=1).max()
    rollmin3 = data["close"].rolling(window=20, min_periods=1).min()
    
    # EMA & Std Dev
    data['ema7'] = data['close'].ewm(span=7, adjust=False).mean()
    data['ema12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['std12'] = data['close'].rolling(window=12).std()
    data['zlema12'] = zlema(data['close'], period=12)
    data['zlema7'] = zlema(data['close'], period=7)

    macd = data['ema12'] - data['ema7']
    macd_signal = macd.ewm(span=10, adjust=False).mean()
    zlema_macd = data['zlema12'] - data['zlema7']
    zlema_macd_signal = zlema_macd.ewm(span=10, adjust=False).mean()

    # Linear regression slope (trend) and acceleration
    data['lin_reg'] = data['close'].rolling(7).apply(lambda y: linregress(np.arange(len(y)), y)[0], raw=False)
    data['acc'] = data['lin_reg'].rolling(7).apply(lambda y: linregress(np.arange(len(y)), y)[0], raw=False)

    # Basic returns
    data['f1'] = data['close'].pct_change()
    data['f1_shift'] = data['f1'].shift()
    data['f1_shift2'] = data['f1_shift'].shift()
    data['f2'] = data['open'] / (data['open_shift'] + eps) - 1
    data['f3'] = data['high'] / (data['high_shift'] + eps) - 1
    data['f4'] = data['low'] / (data['low_shift'] + eps) - 1

    # Candle ratios (using safe division)
    hl_range = (data['high'] - data['low']).replace(0, eps)
    hl_range_shift = (data['high_shift'] - data['low_shift']).replace(0, eps)

    data['f5'] = (data['close'] - data['open']) / hl_range
    data['f6'] = (data['high'] - data['close']) / hl_range
    data['f7'] = (data['high'] - data['open']) / hl_range
    data['f8'] = (data['close'] - data['low']) / hl_range
    data['f9'] = (data['open'] - data['low']) / hl_range
    data['f10'] = data['f6'] / (data['f7'] + eps)
    data['f11'] = data['f8'] / (data['f9'] + eps)
    data['f12'] = data['f6'] / (data['f9'] + eps)

    data['f13'] = (data['close_shift'] - data['open_shift']) / hl_range_shift
    data['f14'] = (data['high_shift'] - data['close_shift']) / hl_range_shift
    data['f15'] = (data['high_shift'] - data['open_shift']) / hl_range_shift
    data['f16'] = (data['close_shift'] - data['low_shift']) / hl_range_shift
    data['f17'] = (data['open_shift'] - data['low_shift']) / hl_range_shift

    # Technical indicators
    data['f18'] = rsi(data['close'], lookback=14)
    data['f19'] = williams_r(data, period=14)
    data['f20'] = data['ema7'] / (data['ema12'] + eps) - 1
    data['f21'] = data['lin_reg'] / (data['close'] + eps)
    data['f22'] = data['acc'] / (data['close'] + eps)
    data['f23'] = data['ema12'] / (data['close'] + eps) - 1
    data['f24'] = data['std12'] / (data['close'] + eps)
    data['f25'] = (data['ema12'] + 2 * data['std12']) / (data['close'] + eps) - 1
    data['f26'] = (data['ema12'] - 2 * data['std12']) / (data['close'] + eps) - 1

    # New: Z-Score of Close Price
    data['f27'] = (data['close'] - data['close'].rolling(20).mean()) / (data['close'].rolling(20).std() + eps)

    # New: Lagged versions of selected indicators
    for i in [1, 2, 5]:
        data[f'f20_lag{i}'] = data['f20'].shift(i)
        data[f'f21_lag{i}'] = data['f21'].shift(i)
        data[f'f18_lag{i}'] = data['f18'].shift(i)

    # New: Simple candlestick flags
    data['f28'] = macd / macd_signal - 1
    data['f29'] = zlema_macd / zlema_macd_signal - 1

    for i in [1, 2, 5]:
        data[f'f28_lag{i}'] = data['f28'].shift(i)
        data[f'f29_lag{i}'] = data['f29'].shift(i)


    data["f30"] = (rollmax1 - data["close"]) / (rollmax1 - rollmin1)
    data["f31"] = (rollmax2 - data["close"]) / (rollmax2 - rollmin2)
    data["f32"] = (rollmax3 - data["close"]) / (rollmax3 - rollmin3)

    data["f33"] = adx1["ADX_14"]
    data['f34'] = data['f33'].shift()

    # Drop rows with NaNs due to rolling/shift
    return data.loc[:, 'f1':'f34']
