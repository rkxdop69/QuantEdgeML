import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import pacf, adfuller
from scipy.stats import linregress
from scipy.signal import find_peaks, argrelextrema
from scipy.interpolate import UnivariateSpline 
import warnings
import pandas_ta as ta
import matplotlib.pyplot as plt
import random
warnings.filterwarnings("ignore")

def feature_gen(data:pd.DataFrame) -> pd.DataFrame:
    try:
        df=data.copy()

        macd1=ta.macd(close=df["close"], fast=4, slow=8, signal=5)
        macd2=ta.macd(close=df["close"], fast=10, slow=21, signal=9)
        bb1=ta.bbands(close=df["close"], length=12)
        bb2=ta.bbands(close=df["close"], lenght=4)
        adx1=ta.adx(high=df["high"], low=df["low"], close=df["close"])
        adx2=ta.adx(high=df["high"], low=df["low"], close=df["close"], length=4)
        stoch=ta.stoch(high=df["high"], low=df["low"], close=df["close"])
        rollmax1=df["close"].rolling(window=3, min_periods=1).max()
        rollmin1=df["close"].rolling(window=3, min_periods=1).min()
        rollmax2=df["close"].rolling(window=9, min_periods=1).max()
        rollmin2=df["close"].rolling(window=9, min_periods=1).min()
        rollmax3=df["close"].rolling(window=20, min_periods=1).max()
        rollmin3=df["close"].rolling(window=20, min_periods=1).min()

        slopes=[]
        for i in range(len(df)):
            if i < 5:
                # Not enough temp_data points for the rolling window
                slopes.append(np.nan)
            else:
                # Extract the temp_data for the rolling window
                y = df['close'].iloc[i - 5:i]
                x = np.arange(len(y))  # Independent variable: time steps
                slope, _, _, _, _ = linregress(x, y)  # Fit a linear regression
                slopes.append(slope)
        df['lin_reg']=slopes


        df["mf1"]=macd1["MACD_4_8_5"]/macd1["MACDs_4_8_5"]
        df["mf2"]=macd2["MACD_10_21_9"]/macd2["MACDs_10_21_9"]
        df["mf3"]=ta.rsi(close=df["close"], length=3)
        df["mf4"]=ta.rsi(close=df["close"], length=15)
        df["mf5"]=bb1["BBB_12_2.0"]
        df["mf6"]=bb1["BBP_12_2.0"]
        df["mf7"]=bb2["BBB_5_2.0"]
        df["mf8"]=bb2["BBP_5_2.0"]
        df["mf9"]=adx1["ADX_14"]
        df["mf10"]=adx2["ADX_4"]
        df["mf11"]=adx1["DMP_14"]/adx1["DMN_14"]
        df["mf12"]=adx2["DMP_4"]/adx2["DMN_4"]
        df["mf13"]=df["lin_reg"]/df["close"]
        df["mf14"]=df["close"].rolling(window=14).apply(lambda x: adfuller(x)[0], raw=True)
        df["mf15"]=df["close"].rolling(window=14).apply(lambda x: pacf(x, nlags=3)[3], raw=True)
        df["mf16"]=df["close"].rolling(window=14).apply(lambda x: pacf(x, nlags=4)[4], raw=True)
        df["mf17"]=stoch["STOCHk_14_3_3"]/stoch["STOCHd_14_3_3"]
        df["mf18"]=df["close"].ewm(span=4, adjust=False, min_periods=1).mean()/df["close"]
        df["mf19"]=df["close"].ewm(span=15, adjust=False, min_periods=1).mean()/df["close"]
        df["mf20"]=(rollmax1-df["close"])/(rollmax1-rollmin1)
        df["mf21"]=(rollmax2-df["close"])/(rollmax2-rollmin2)
        df["mf23"]=(rollmax3-df["close"])/(rollmax3-rollmin3)
        df["mf24"]=(df["high"]-df["close"])/(df["open"]-df["low"])
        df["mf25"]=(df["high"]-df["open"])/(df["close"]-df["low"])
        df["mf26"]=(df["high"]-df["low"])/(df["open"]-df["close"])
        df["mf27"]=df["close"].ewm(span=10, adjust=True, min_periods=1).mean()/df["close"].ewm(span=5, adjust=True, min_periods=1).mean()
        df["mf28"]=df["close"].pct_change()
        df["mf29"]=df["open"].pct_change()
        df["mf30"]=df["high"].pct_change()
        df["mf31"]=df["low"].pct_change()
        df["mf32"]=(df["high"] - df["low"])/df["low"]
        df["mf33"]=df["mf32"].pct_change()
        df["mf34"]=df["mf32"].rolling(window=5).mean()
        df["mf35"]=df["mf34"].pct_change()

        return df.loc[:, "mf1":"mf35"]

    except Exception as e:
        print(e)

def feature_gen_w_signal(data:pd.DataFrame, upper_barrier:float = 10.0, lower_barrier:float = -10.0 , vertical_barrier:int = 5) -> pd.DataFrame:
    try:
        df=data.copy()

        macd1=ta.macd(close=df["close"], fast=4, slow=8, signal=5)
        macd2=ta.macd(close=df["close"], fast=10, slow=21, signal=9)
        bb1=ta.bbands(close=df["close"], length=12)
        bb2=ta.bbands(close=df["close"], lenght=4)
        adx1=ta.adx(high=df["high"], low=df["low"], close=df["close"])
        adx2=ta.adx(high=df["high"], low=df["low"], close=df["close"], length=4)
        stoch=ta.stoch(high=df["high"], low=df["low"], close=df["close"])
        rollmax1=df["close"].rolling(window=3, min_periods=1).max()
        rollmin1=df["close"].rolling(window=3, min_periods=1).min()
        rollmax2=df["close"].rolling(window=9, min_periods=1).max()
        rollmin2=df["close"].rolling(window=9, min_periods=1).min()
        rollmax3=df["close"].rolling(window=20, min_periods=1).max()
        rollmin3=df["close"].rolling(window=20, min_periods=1).min()

        slopes=[]
        for i in range(len(df)):
            if i < 5:
                # Not enough temp_data points for the rolling window
                slopes.append(np.nan)
            else:
                # Extract the temp_data for the rolling window
                y = df['close'].iloc[i - 5:i]
                x = np.arange(len(y))  # Independent variable: time steps
                slope, _, _, _, _ = linregress(x, y)  # Fit a linear regression
                slopes.append(slope)
        df['lin_reg']=slopes


        df["f1"]=macd1["MACD_4_8_5"]/macd1["MACDs_4_8_5"]
        df["f2"]=macd2["MACD_10_21_9"]/macd2["MACDs_10_21_9"]
        df["f3"]=ta.rsi(close=df["close"], length=3)
        df["f4"]=ta.rsi(close=df["close"], length=15)
        df["f5"]=bb1["BBB_12_2.0"]
        df["f6"]=bb1["BBP_12_2.0"]
        df["f7"]=bb2["BBB_5_2.0"]
        df["f8"]=bb2["BBP_5_2.0"]
        df["f9"]=adx1["ADX_14"]
        df["f10"]=adx2["ADX_4"]
        df["f11"]=adx1["DMP_14"]/adx1["DMN_14"]
        df["f12"]=adx2["DMP_4"]/adx2["DMN_4"]
        df["f13"]=df["lin_reg"]/df["close"]
        df["f14"]=df["close"].rolling(window=14).apply(lambda x: adfuller(x)[0], raw=True)
        df["f15"]=df["close"].rolling(window=14).apply(lambda x: pacf(x, nlags=3)[3], raw=True)
        df["f16"]=df["close"].rolling(window=14).apply(lambda x: pacf(x, nlags=4)[4], raw=True)
        df["f17"]=stoch["STOCHk_14_3_3"]/stoch["STOCHd_14_3_3"]
        df["f18"]=df["close"].ewm(span=4, adjust=False, min_periods=1).mean()/df["close"]
        df["f19"]=df["close"].ewm(span=15, adjust=False, min_periods=1).mean()/df["close"]
        df["f20"]=(rollmax1-df["close"])/(rollmax1-rollmin1)
        df["f21"]=(rollmax2-df["close"])/(rollmax2-rollmin2)
        df["f23"]=(rollmax3-df["close"])/(rollmax3-rollmin3)
        df["f24"]=(df["high"]-df["close"])/(df["open"]-df["low"])
        df["f25"]=(df["high"]-df["open"])/(df["close"]-df["low"])
        df["f26"]=(df["high"]-df["low"])/(df["open"]-df["close"])
        df["f27"]=df["close"].ewm(span=10, adjust=True, min_periods=1).mean()/df["close"].ewm(span=5, adjust=True, min_periods=1).mean()
        df["price"]=df["close"]


        #Signal Generation Start
        #Triple Barrier method  (basic: the bands are set at a fixed percentage)

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
            if(len(peaks)==10 or len(peaks)==11):
                # print(i)
                break
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
#         print(features['signal'].value_counts(normalize=True))

#     except Exception as e:
#         print(e)

#     # plt.figure(1)
#     # plt.plot(features["signal"])
#     # # plt.figure(2)
#     # # plt.plot(features["price"])
#     # plt.show()
# print(all_profit)
# print(sum(all_profit)/len(all_profit))
