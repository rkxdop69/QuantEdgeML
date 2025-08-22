import pandas as pd
import numpy as np
from indicators import zlema, williams_r
from scipy.stats import linregress


def feature_gen(data:pd.DataFrame) -> pd.DataFrame:
    try:
        df=data.copy()
        
        df["zlema_10"] = zlema(df["close"], 10)
        df["zlema_18"] = zlema(df["close"], 18)
        df["macd"] = df["zlema_18"] - df["zlema_10"]
        df["macd_signal"] = zlema(df["macd"], 9)
        df["williams_r"] = williams_r(df, 16)   
        df["williams_r_8"] = williams_r(df, 8)
        df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
        df["volatility"] = (df["high"] - df["low"]).rolling(window=10).mean()
        df["rollmax"] = df["close"].rolling(window=10).max()
        df["rollmin"] = df["close"].rolling(window=10).min()

        slopes=[]
        for i in range(len(df)):
            if i < 7:
                # Not enough temp_data points for the rolling window
                slopes.append(np.nan)
            else:
                # Extract the temp_data for the rolling window
                y = df['close'].iloc[i - 7:i]
                x = np.arange(len(y))  # Independent variable: time steps
                slope, _, _, _, _ = linregress(x, y)  # Fit a linear regression
                slopes.append(slope)
        df['lin_reg']=slopes

        df["f1"] = df["zlema_10"] / df["zlema_18"]
        df["f2"] = df["macd"] / df["macd_signal"]
        df["f3"] = df["williams_r"] / 100
        df["f4"] = df["williams_r_8"] / 100
        df["f5"] = df["ema_10"] / df["zlema_10"]
        df["f6"] = df["volatility"] / df["close"]
        df["f7"] = (df["close"] - df["rollmin"]) / (df["rollmax"] - df["rollmin"])
        df["f8"] = df["lin_reg"] / df["close"]

        return df.loc[:, "f1":"f8"]
    
    except Exception as e:
        print(e)
        return None