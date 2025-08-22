import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import acf, pacf, adfuller
from scipy.stats import linregress
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline 
import warnings
import matplotlib.pyplot as plt
import random
warnings.filterwarnings("ignore")

def feature_gen(data:pd.DataFrame) -> pd.DataFrame:
    try:
        df=data.copy()

        df["ema5"]=df["close"].ewm(span=7, adjust=False, min_periods=1).mean()
        df["ema10"]=df["close"].ewm(span=12, adjust=False, min_periods=1).mean()
        df["macd"]=df["ema10"]-df["ema5"]
        df["signal_macd"]=df["macd"].ewm(span=6, adjust=False, min_periods=1).mean()
        df['price_change']=df['close'].diff()
        df['gain']=df['price_change'].apply(lambda x: x if x > 0 else 0)
        df['loss'] = df['price_change'].apply(lambda x: abs(x) if x < 0 else 0)
        df['avg_gain'] = df['gain'].rolling(window=10, min_periods=1).mean()
        df['avg_loss'] = df['loss'].rolling(window=10, min_periods=1).mean()
        df['RS'] = df['avg_gain']/df['avg_loss'] 
        df['RSI'] = 100-(100/(1+df['RS']))
        df['sma']=df['close'].rolling(window=10, min_periods=1).mean()
        df['std']=df['close'].rolling(window=10, min_periods=1).std()
        df['ubb']=df['sma']+2*df['std']
        df['lbb']=df['sma']-2*df['std']
        df['lowestlow']=df['low'].rolling(window=10, min_periods=1).min()
        df['highesthigh']=df['high'].rolling(window=10, min_periods=1).max()
        df['%k']=(df['close']-df['lowestlow'])/(df['highesthigh']-df['lowestlow'])
        df['perd']=df['%k'].rolling(window=7, min_periods=1).mean()
        rollmax2=df["close"].rolling(window=9, min_periods=1).max()
        rollmin2=df["close"].rolling(window=9, min_periods=1).min()
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
        slopes=[]
        for i in range(len(df)):
            if i < 7:
                # Not enough temp_data points for the rolling window
                slopes.append(np.nan)
            else:
                # Extract the temp_data for the rolling window
                y = df['lin_reg'].iloc[i - 7:i]
                x = np.arange(len(y))  # Independent variable: time steps
                slope, _, _, _, _ = linregress(x, y)  # Fit a linear regression
                slopes.append(slope)
        df['acc']=slopes

        df["f1"]=df["ema10"]/df["ema5"]
        df["f2"]=df["RSI"]/100
        df["f3"]=df['macd']/df["signal_macd"]
        df["f4"]=df["%k"]-df["perd"]
        df["f5"]=df["lin_reg"]/df["close"].mean()
        df["f6"]=(df["ubb"]-df["lbb"])/(df["high"]-df["low"])
        df["f7"]=(df["ubb"]-df["close"])/(df["close"]-df["lbb"])
        df["f8"]=df["%k"]
        df["f9"]=df["acc"]/df["close"].mean()
        df["f10"]=(rollmax2-df["close"])/(rollmax2-rollmin2)

        return df.loc[:, "f1":"f10"]
    
    except Exception as e:
        print(e)

def feature_gen_w_signal(data:pd.DataFrame, upper_barrier:float = 10.0, lower_barrier:float = -10.0 , vertical_barrier:int = 5) -> pd.DataFrame:
    try:
        df=data.copy()

        df["ema5"]=df["close"].ewm(span=7, adjust=False, min_periods=1).mean()
        df["ema10"]=df["close"].ewm(span=12, adjust=False, min_periods=1).mean()
        df["macd"]=df["ema10"]-df["ema5"]
        df["signal_macd"]=df["macd"].ewm(span=6, adjust=False, min_periods=1).mean()
        df['price_change']=df['close'].diff()
        df['gain']=df['price_change'].apply(lambda x: x if x > 0 else 0)
        df['loss'] = df['price_change'].apply(lambda x: abs(x) if x < 0 else 0)
        df['avg_gain'] = df['gain'].rolling(window=10, min_periods=1).mean()
        df['avg_loss'] = df['loss'].rolling(window=10, min_periods=1).mean()
        df['RS'] = df['avg_gain']/df['avg_loss'] 
        df['RSI'] = 100-(100/(1+df['RS']))
        df['sma']=df['close'].rolling(window=10, min_periods=1).mean()
        df['std']=df['close'].rolling(window=10, min_periods=1).std()
        df['ubb']=df['sma']+2*df['std']
        df['lbb']=df['sma']-2*df['std']
        df['lowestlow']=df['low'].rolling(window=10, min_periods=1).min()
        df['highesthigh']=df['high'].rolling(window=10, min_periods=1).max()
        df['%k']=(df['close']-df['lowestlow'])/(df['highesthigh']-df['lowestlow'])
        df['perd']=df['%k'].rolling(window=7, min_periods=1).mean()
        rollmax2=df["close"].rolling(window=9, min_periods=1).max()
        rollmin2=df["close"].rolling(window=9, min_periods=1).min()
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
        slopes=[]
        for i in range(len(df)):
            if i < 7:
                # Not enough temp_data points for the rolling window
                slopes.append(np.nan)
            else:
                # Extract the temp_data for the rolling window
                y = df['lin_reg'].iloc[i - 7:i]
                x = np.arange(len(y))  # Independent variable: time steps
                slope, _, _, _, _ = linregress(x, y)  # Fit a linear regression
                slopes.append(slope)
        df['acc']=slopes

        df["f1"]=df["ema10"]/df["ema5"]
        df["f2"]=df["RSI"]/100
        df["f3"]=df['macd']/df["signal_macd"]
        df["f4"]=df["%k"]-df["perd"]
        df["f5"]=df["lin_reg"]/df["close"].mean()
        df["f6"]=(df["ubb"]-df["lbb"])/(df["high"]-df["low"])
        df["f7"]=(df["ubb"]-df["close"])/(df["close"]-df["lbb"])
        df["f8"]=df["%k"]
        df["f9"]=df["acc"]/df["close"].mean()
        df["f10"]=(rollmax2-df["close"])/(rollmax2-rollmin2)
    

        df["price"]=df["close"]

        #Signal Generation Start
        #Triple Barrier method  

        # df.loc[:, "signal"]=0
        # for i in range(len(df)-vertical_barrier):
        #     entry=df.loc[i, "price"]
        #     barrier_hit=True
        #     for j in range(vertical_barrier):
        #         if((df.loc[i+j, "high"]-entry)/entry>=upper_barrier/100):
        #             df.loc[i, "signal"]=1
        #             barrier_hit=False
        #             break
        #         elif((df.loc[i+j, "low"]-entry)/entry<=lower_barrier/100):
        #             df.loc[i, "signal"]=-1
        #             barrier_hit=False
        #             break
        #     if(barrier_hit):
        #         if((df.loc[i+j, "high"]-entry)/entry>=upper_barrier/200):
        #             df.loc[i, "signal"]=1

        # return df.loc[:, "f1":"signal"]

        #Own method
        df.loc[:, "signal"]=-1.0
        for i in range(5, 100):
            x=np.arange(len(data))
            spline=UnivariateSpline(x, df["close"], s=i*(df["close"].max()))
            fitted_curve=spline(x)
            neg_fitted_curve=-1*fitted_curve
            peaks, _= find_peaks(fitted_curve)
            troughs, _= find_peaks(neg_fitted_curve)
            if(len(peaks)==6 or len(peaks)==7 or len(peaks)==8 or len(peaks)==9):
                # print(i)
                break
        if(not(len(peaks)==6 or len(peaks)==7 or len(peaks)==8 or len(peaks)==9)):
            return
        df["signal"][peaks]=1
        df["signal"][troughs]=0

        start=df["close"].max()
        end=df["close"].min()
        diff=abs(start-end)
        minima=min(start, end)
        idx=0
        cp=0

        for i in range(len(df)):
            if(df.loc[i, "signal"]==0 or df.loc[i, "signal"]==1):
                start=df.loc[i, "close"]
                cp=df.loc[i, "signal"]
                min_start=i-7 if i-7>-1 else 0
                min_end=i+8 if i+8<len(df) else len(df)-1
                for j in range(min_start, min_end):
                    if(cp==1):
                        start=max(start, df.loc[j, "close"])
                    elif(cp==0):
                        start=min(start, df.loc[j, "close"])
                for j in range(i+1, len(df)):
                    if(df.loc[j, "signal"]==0 or df.loc[j, "signal"]==1):
                        end=df.loc[j, "close"]
                        diff=abs(start-end)
                        minima=min(start, end)
                        # print(start, end, "idx", i, j)
                        idx=j
                        cp=df.loc[j, "signal"]
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
            elif(df.loc[i, "signal"]==-1):
                df.loc[i, "signal"]=(df.loc[i, "close"]-minima)/diff

        for i in range(len(df)):
            if(df.loc[i, "signal"]>0.80):
                df.loc[i, "signal"]=1
            elif(df.loc[i, "signal"]<0.15):
                df.loc[i, "signal"]=-1
            else:
                df.loc[i, "signal"]=0
        for i in range(2, len(df)):
            if(df.loc[i-2, "signal"]==df.loc[i, "signal"]):
                df.loc[i-1, "signal"]=df.loc[i, "signal"]        

        return df.loc[:, "f1":"signal"]
    
    except Exception as e:
        print(e)

# all_profit=[]
# for x in range(5):
#     try:
#         num=random.randint(0, 4000)
#         datapath=Path(f"testingdata/ts{num}.csv")
#         test=pd.read_csv(datapath)
#         features=feature_gen_w_signal(test)
#         print(features)

#         positon=0
#         entry=0
#         profit=[]
#         for i in range(len(features)):
#             if(positon==0 and features.loc[i, "signal"]==-1):
#                 positon=1
#                 entry=features.loc[i, "price"]
#             elif(positon==1 and features.loc[i, "signal"]==1):
#                 positon=0
#                 profit.append(((features.loc[i, "price"]-entry)*100)/entry)

#         print(profit)
#         print(sum(profit))
#         all_profit.append(sum(profit))
#         print(features['signal'].value_counts())

#     except Exception as e:
#         print(e)

#     plt.figure(1)
#     plt.plot(features["signal"])
#     # plt.figure(2)
#     # plt.plot(features["price"])
#     plt.show()
# print(all_profit)
# print(sum(all_profit)/len(all_profit))
