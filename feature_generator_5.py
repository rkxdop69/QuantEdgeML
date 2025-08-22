import pandas as pd
import numpy as np
from indicators import *



def feature_gen(data:pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data['ema12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['ema26'] = data['close'].ewm(span=26, adjust=False).mean()
    data['ema_macd'] = data['ema12'] - data['ema26']
    data['ema_macd_signal'] = data['ema_macd'].ewm(span=9, adjust=False).mean()

    data['zlema12'] = zlema(data['close'], 12)
    data['zlema26'] = zlema(data['close'], 26)
    data['zlema_macd'] = data['zlema12'] - data['zlema26']
    data['zlema_macd_signal'] = zlema(data['zlema_macd'], 9)

    data['williams_r'] = williams_r(data, period=14)

    data['swing'] = no_lag_swing_filter(data['close'], 2.0, -1.0)

    data['rsi'] = rsi(data['close'], lookback=14)

    data['ema20'] = data['close'].ewm(span=20, adjust=False).mean()
    data['std20'] = data['close'].rolling(window=20).std()
    data['ema_bb_upper'] = data['ema20'] + (data['std20'] * 2)
    data['ema_bb_lower'] = data['ema20'] - (data['std20'] * 2)

    data['zlema20'] = zlema(data['close'], 20)
    data['zlema_bb_upper'] = data['zlema20'] + (data['std20'] * 2)
    data['zlema_bb_lower'] = data['zlema20'] - (data['std20'] * 2)

    #Features start here
    data['f1'] = data['close'] / data['close'].shift(1) - 1
    data['f2'] = data['close'] / data['close'].shift(2) - 1
    data['f3'] = data['close'] / data['close'].shift(3) - 1
    data['f4'] = data['close'] / data['close'].shift(4) - 1
    data['f5'] = data['close'] / data['close'].shift(5) - 1
    data['f6'] = data['close'] / data['close'].shift(6) - 1
    data['f7'] = data['close'] / data['close'].shift(7) - 1
    data['f8'] = data['close'] / data['close'].shift(8) - 1
    data['f9'] = data['close'] / data['close'].shift(9) - 1
    data['f10'] = data['close'] / data['close'].shift(10) - 1

    data['f11'] = data['ema_macd'] / data['ema_macd_signal'] - 1
    data['f12'] = data['ema_macd'] / data['ema_macd'].shift(1) - 1
    data['f13'] = data['ema_macd'] / data['ema_macd'].shift(2) - 1

    data['f14'] = data['zlema_macd'] / data['zlema_macd_signal'] - 1
    data['f15'] = data['zlema_macd'] / data['zlema_macd'].shift(1) - 1
    data['f16'] = data['zlema_macd'] / data['zlema_macd'].shift(2) - 1

    data['f17'] = data['williams_r']
    data['f18'] = data['williams_r'] / data['williams_r'].shift(1) - 1
    data['f19'] = data['williams_r'] / data['williams_r'].shift(2) - 1

    data['f20'] = data['swing'] / data['swing'].shift(1) - 1
    
    data['f21'] = data['rsi']
    data['f22'] = data['rsi'] / data['rsi'].shift(1) - 1
    data['f23'] = data['rsi'] / data['rsi'].shift(2) - 1

    data['f24'] = data['std20'] / data['std20'].shift(1) - 1

    data['f25'] = data['ema_bb_upper'] / data['close'] - 1
    data['f26'] = data['ema_bb_lower'] / data['close'] - 1

    data['f27'] = data['zlema_bb_upper'] / data['close'] - 1
    data['f28'] = data['zlema_bb_lower'] / data['close'] - 1

    return data.loc[:, 'f1':'f28']
