from StrategyTester import Strategy
import numpy as np
import pandas as pd


class Strategy_1(Strategy):
    def __init__(self, name = "Strategy_0_updated", train_data_directory = "", test_data_directory = "", is_ml = False, sl = -100, day_sl = -500, tp = 10000000, framework = "sklearn"):
        super().__init__(name, train_data_directory, test_data_directory, is_ml, sl, day_sl, tp, framework)


    def generate_signals(self, data):
        from indicators import rsi, zlema

        # The returned signal should be 1 for buy, -1 for sell, and 0 for hold/do nothing
        data.loc[:, "signal"] = 0

        # data["ema_long"] = data["close"].ewm(span=26, adjust=False).mean()
        # data["ema_short"] = data["close"].ewm(span=12, adjust=False).mean()


        #Using ZLEMA instead of EMA to reduce lag. ZLEMA is Zero Lag Exponential Moving Average mentioned by John Ehlers in his book "Cybernetic Analysis for Stocks and Futures"
        #ZLEMA improved the sharpe ratio by almost 1.5 in backtesting compared to EMA
        data["ema_long"] = zlema(data["close"], period=26)
        data["ema_short"] = zlema(data["close"], period=12)

        data["macd"] = data["ema_short"] - data["ema_long"]
    
        #data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
        data["macd_signal"] = zlema(data["macd"], period=9)

        data["rsi"] = rsi(data["close"], lookback=14)

        for i in range(1, len(data)):
            if(data.loc[i, "macd"] >= data.loc[i, "macd_signal"] and data.loc[i, "rsi"] < 70):
                data.loc[i, "signal"] = 1
            elif(data.loc[i, "macd"] <= data.loc[i, "macd_signal"] and data.loc[i, "rsi"] > 30):
                data.loc[i, "signal"] = -1
            else:
                data.loc[i, "signal"] = 0

        return data["signal"]


if __name__ == "__main__":
    strategy_2 = Strategy_1(train_data_directory="train_data", test_data_directory="test_data", sl=-5, day_sl=-20, framework="nil", is_ml=False)
    strategy_2.test()
    strategy_2.walkforward_permutation_test(n_iter=100)
