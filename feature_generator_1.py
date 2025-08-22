import pandas as pd
import numpy as np
from pathlib import Path
import ta
from statsmodels.tsa.stattools import acf, pacf, adfuller
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline 
from scipy.stats import linregress
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def feature_gen(data, column_names):

    df=ta.add_volatility_ta(df=data, high="high", low="low", close="ltp")
    df=ta.add_trend_ta(df=data, high="high", low="low", close="ltp")
    df=ta.add_momentum_ta(df=data, high="high", low="low", close="ltp", volume="ltp")
    df=ta.add_others_ta(df=data, close="ltp")
    # df.drop(columns=["momentum_pvo", "momentum_pvo_signal", "momentum_pvo_hist", "trend_psar_up", "trend_psar_down", "trend_psar_up_indicator", "trend_psar_down_indicator"], inplace=True)

    windows = [7, 16]
    for window in windows:
        df[f'close_mean_{window}'] = df['ltp'].rolling(window=window).mean()
        df[f'close_std_{window}'] = df['ltp'].rolling(window=window).std()
        df[f'close_skew_{window}'] = df['ltp'].rolling(window=window).apply(skew, raw=True)
        df[f'close_kurtosis_{window}'] = df['ltp'].rolling(window=window).apply(kurtosis, raw=True)

    # for window in windows:
    #     df[f'pacf_1_{window}'] = np.nan
    #     df[f'acf_1_{window}'] = np.nan
    #     df[f'adf_stat_{window}'] = np.nan
    new_columns = {}
    for window in windows:
        new_columns[f'pacf_1_{window}'] = df['ltp'].rolling(window=window).apply(lambda x: pacf(x, nlags=1)[1], raw=True)
        new_columns[f'adf_stat_{window}'] = df['ltp'].rolling(window=window).apply(lambda x: adfuller(x)[0], raw=True)
        new_columns[f'acf_1_{window}'] = df['ltp'].rolling(window=window).apply(lambda x: acf(x, nlags=1)[1], raw=True)

    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    
    for window in windows:
        df[f'adf_stat_{window}'] = df['ltp'].rolling(window=window).apply(lambda x: adfuller(x)[0], raw=True)
        df[f'acf_1_{window}'] = df['ltp'].rolling(window=window).apply(lambda x: acf(x, nlags=1)[1], raw=True)
        df[f'pacf_1_{window}'] = df['ltp'].rolling(window=window).apply(lambda x: pacf(x, nlags=1)[1], raw=True)

    df.drop(columns=column_names, inplace=True)
    return df

def feature_gen_w_signal(data, column_names):

    df=ta.add_volatility_ta(df=data, high="high", low="low", close="close")
    df=ta.add_trend_ta(df=data, high="high", low="low", close="close")
    df=ta.add_momentum_ta(df=data, high="high", low="low", close="close", volume="close")
    df=ta.add_others_ta(df=data, close="close")
    # df.drop(columns=["momentum_pvo", "momentum_pvo_signal", "momentum_pvo_hist", "trend_psar_up", "trend_psar_down", "trend_psar_up_indicator", "trend_psar_down_indicator"], inplace=True)

    windows = [7, 16]
    for window in windows:
        df[f'close_mean_{window}'] = df['close'].rolling(window=window).mean()
        df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
        df[f'close_skew_{window}'] = df['close'].rolling(window=window).apply(skew, raw=True)
        df[f'close_kurtosis_{window}'] = df['close'].rolling(window=window).apply(kurtosis, raw=True)

    # for window in windows:
    #     df[f'pacf_1_{window}'] = np.nan
    #     df[f'acf_1_{window}'] = np.nan
    #     df[f'adf_stat_{window}'] = np.nan
    new_columns = {}
    for window in windows:
        new_columns[f'pacf_1_{window}'] = df['close'].rolling(window=window).apply(lambda x: pacf(x, nlags=1)[1], raw=True)
        new_columns[f'adf_stat_{window}'] = df['close'].rolling(window=window).apply(lambda x: adfuller(x)[0], raw=True)
        new_columns[f'acf_1_{window}'] = df['close'].rolling(window=window).apply(lambda x: acf(x, nlags=1)[1], raw=True)

    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    
    for window in windows:
        df[f'adf_stat_{window}'] = df['close'].rolling(window=window).apply(lambda x: adfuller(x)[0], raw=True)
        df[f'acf_1_{window}'] = df['close'].rolling(window=window).apply(lambda x: acf(x, nlags=1)[1], raw=True)
        df[f'pacf_1_{window}'] = df['close'].rolling(window=window).apply(lambda x: pacf(x, nlags=1)[1], raw=True)

    # Signal Generation Starts

    max_prof=df["close"].max()-df["close"].min()
    df.loc[:, "prob"]=-1.0
    for i in range(5, 100):
        x=np.arange(len(data))
        spline=UnivariateSpline(x, df["close"], s=i*(df["close"].max()))
        fitted_curve=spline(x)
        neg_fitted_curve=-1*fitted_curve
        peaks, _= find_peaks(fitted_curve)
        troughs, _= find_peaks(neg_fitted_curve)
        if(len(peaks)==10 or len(peaks)==11):
            # print(i)
            break
    df["prob"][peaks]=1
    df["prob"][troughs]=0

    buy_price=0
    current_position=0
    buy_index=0

    # for i in range(len(df)):
    #     if(df.loc[i, "close"]==1 and current_position==0):
    #         current_position=1
    #         buy_price=df.loc[i, "close"]
    #         buy_index=i
    #     elif(df.loc[i, "close"]==1 and current_position==1):
    #         current_position=0
    #         sell_price=df.loc[i, "close"]
    #         if((sell_price-buy_price)<=0.01* buy_price):
    #             df.loc[i, "prob"]==0.5
    #             df.loc[buy_index, "prob"]=0.5


    start=df["close"].max()
    end=df["close"].min()
    diff=abs(start-end)
    minima=min(start, end)
    idx=0
    cp=0

    for i in range(len(df)):
        if(df.loc[i, "prob"]==0 or df.loc[i, "prob"]==1):
            start=df.loc[i, "close"]
            cp=df.loc[i, "prob"]
            min_start=i-7 if i-7>-1 else 0
            min_end=i+8 if i+8<len(df) else len(df)-1
            for j in range(min_start, min_end):
                if(cp==1):
                    start=max(start, df.loc[j, "close"])
                elif(cp==0):
                    start=min(start, df.loc[j, "close"])
            for j in range(i+1, len(df)):
                if(df.loc[j, "prob"]==0 or df.loc[j, "prob"]==1):
                    end=df.loc[j, "close"]
                    diff=abs(start-end)
                    minima=min(start, end)
                    # print(start, end, "idx", i, j)
                    idx=j
                    cp=df.loc[j, "prob"]
                    break
                cp=1 if cp==0 else 0
                end=df.loc[j, "close"]
                diff=abs(start-end)
                minima=min(start, end)
            max_end=idx+8 if idx+8<len(df) else len(df)-1
            max_start=idx-7 if idx-7>-1 else 0
            for j in range(max_start, max_end):
                if(cp==1):
                    end=max(end, df.loc[j, "close"])
                elif(cp==0):
                    end=min(end, df.loc[j, "close"])
                # if(end!=end2):
                    # print(start, end, "idx2", i, j)
        elif(df.loc[i, "prob"]==-1):
            df.loc[i, "prob"]=(df.loc[i, "close"]-minima)/diff



    for i in range(len(df)):
        if(df.loc[i, "prob"]>1):
            df.loc[i, "prob"]=1
        elif(df.loc[i, "prob"]<0):
            df.loc[i, "prob"]=0
    for i in range(len(df)):
        if(df.loc[i, "prob"]>0.85):
            df.loc[i, "prob"]=1
        elif(df.loc[i, "prob"]<0.25):
            df.loc[i, "prob"]=-1
        else:
            df.loc[i, "prob"]=0


    df.drop(columns=column_names, inplace=True)
    df["close"]=data["close"]
    return df


# data_path=Path("dfbig/ts1.csv")
# data=pd.read_csv(data_path)
# processed_data=feature_gen_w_signal(data=data, column_names=data.columns)

# plt.figure(1)
# plt.plot(processed_data["prob"], "red")
# plt.figure(2)
# plt.plot(processed_data["close"], "blue")
# # plt.plot(spline, "red")
# plt.show()