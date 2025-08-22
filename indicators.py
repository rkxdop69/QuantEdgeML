import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from pykalman import KalmanFilter

def recursive_median(series, window):
    med = series.copy()
    for i in range(window, len(series)):
        med.iloc[i] = np.median(series.iloc[i-window:i+1])
    return med

def zlema(series , period : int = 9):
    series = pd.Series(series)
    lag = (period - 1) // 2
    adjusted_series = series + (series - series.shift(lag))
    return pd.Series(adjusted_series.ewm(span=period, adjust=False).mean())

def kalman_filter(series):
    kf = KalmanFilter(initial_state_mean=series.iloc[0], n_dim_obs=1)
    state_means, _ = kf.filter(series.values.reshape(-1, 1))
    return pd.Series(state_means.flatten(), index=series.index)

def super_smoother(series, period):
    a1 = np.exp(-1.414 * np.pi / period)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    filt = np.zeros_like(series)
    filt[0] = series.iloc[0]
    filt[1] = series.iloc[1]
    for i in range(2, len(series)):
        filt[i] = c1 * (series.iloc[i] + series.iloc[i-1]) / 2 + c2 * filt[i-1] + c3 * filt[i-2]
    return pd.Series(filt, index=series.index)

def no_lag_swing_filter(close_series, threshold_up_percent=1.0, threshold_down_percent=-1.0):
    threshold_up = threshold_up_percent / 100
    threshold_down = threshold_down_percent / 100
    filtered = [close_series.iloc[0]]

    for i in range(1, len(close_series)):
        ref = filtered[-1]
        current = close_series.iloc[i]
        change = (current - ref) / ref

        if((change >= threshold_up) or (change <= threshold_down)):
            filtered.append(current)
        else:
            filtered.append(ref)
    
    return pd.Series(filtered, index=close_series.index)

def williams_r(df, period=14):
    """
    Calculate the Williams %R indicator.

    Parameters:
        df (pd.DataFrame): A DataFrame with columns 'high', 'low', 'close'.
        period (int): Lookback period for calculation (default is 14).

    Returns:
        pd.Series: A pandas Series containing the Williams %R values.
    """
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()
    williams_r = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
    return williams_r

def compute_atr(df, period=14): 
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def rsi(close, lookback=14):
    # Calculate price changes
    delta = close.diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate exponential moving averages of gains and losses
    avg_gain = gain.ewm(alpha=1/lookback, min_periods=lookback, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/lookback, min_periods=lookback, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def generate_signals_smoothed(df, price_col='close', smooth_window=11, smooth_polyorder=3, extrema_window=10):
    """
    Smooth price data and detect local minima/maxima for signal generation.
    """
    # Step 1: Smooth prices
    prices = df[price_col].values
    smoothed = savgol_filter(prices, window_length=smooth_window, polyorder=smooth_polyorder)
    
    # Step 2: Find local minima/maxima
    signals = np.zeros(len(prices))
    for i in range(extrema_window, len(prices) - extrema_window):
        window_slice = smoothed[i - extrema_window:i + extrema_window + 1]
        center = smoothed[i]

        if center == np.min(window_slice):
            signals[i] = 1  # Buy signal
        elif center == np.max(window_slice):
            signals[i] = -1  # Sell signal
        else:
            signals[i] = 0  # Hold

    df['smoothed'] = smoothed
    df['signal'] = signals
    return df

def expand_signals_by_adaptive_price(df, signal_col='signal', price_col='smoothed', atr_col='atr', atr_multiplier=1.0, max_window=10):
    """
    Expands signal labels to nearby points within a price range determined by recent volatility (ATR).
    Keeps labels as discrete values: -1, 0, or 1
    """
    signal = df[signal_col].values
    price = df[price_col].values
    atr = df[atr_col].values
    expanded = np.zeros_like(signal)

    extrema_idxs = np.where(np.abs(signal) == 1)[0]
    for idx in extrema_idxs:
        ref_price = price[idx]
        ref_atr = atr[idx] if not np.isnan(atr[idx]) else 0
        threshold = atr_multiplier * ref_atr

        for i in range(max(0, idx - max_window), min(len(df), idx + max_window + 1)):
            if abs(price[i] - ref_price) <= threshold:
                expanded[i] = signal[idx]  # Preserve -1 or 1 direction

    df[f'{signal_col}_expanded'] = expanded
    return df