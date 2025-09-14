import pandas as pd
from tqdm.contrib.concurrent import process_map, thread_map
import os
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import joblib
from keras.models import save_model, load_model
import tensorflow as tf

class Strategy():

    def __init__(self, name : str = "Strategy_1", train_data_directory : str = "", test_data_directory : str = "", is_ml : bool = True, sl : float = -100.0, day_sl : float = -500, tp : float = 1e7, framework : str = "sklearn"):
        self.name = name
        self.train_data_directory : str = train_data_directory
        self.test_data_directory : str = test_data_directory
        self.is_ml : bool = is_ml
        self.sl : float = sl
        self.day_sl : float = day_sl
        self.tp : float = tp
        self.framework : str = framework
        self.model = None
        self.test_result = None

    def prepare_data_train(self, workers : int = 8):
        file_names = os.listdir(self.train_data_directory)
        dataframes = process_map(self.train_data_process, file_names, max_workers=workers, desc="Processing training files", chunksize=10)
        
        train_data = pd.concat(dataframes, axis=0, ignore_index=True)
        train_data.replace([np.inf, -np.inf], 1e6, inplace=True)
        train_data.dropna(axis=0, inplace=True)
        train_data.to_csv(f"compiled_data/{self.name}_train_data.csv", index=False)


    def train(self):
        data = pd.read_csv(f"compiled_data/{self.name}_train_data.csv")
        y_train = data['target']
        x_train = data.drop('target', axis=1)

        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.05, shuffle=False)

        model = self.define_and_fit_model(X_train, y_train)

        y_pred = model.predict(X_val)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        print("Y_pred shape:", y_pred.shape)
        print("Y_Pred ndim:", y_pred.ndim)

        if(self.framework == "tf"):
            if(y_pred.shape[1] >= 2):
                y_pred = np.argmax(y_pred, axis=1)
            else:
                y_pred = np.where(y_pred>0.5, 1, 0)

        print("Accuracy:", accuracy_score(y_val, y_pred))

        print("Precision:", precision_score(y_val, y_pred, average='macro'))
        print("Recall:", recall_score(y_val, y_pred, average='macro'))
        print("F1 Score:", f1_score(y_val, y_pred, average='macro'))

        from sklearn.metrics import ConfusionMatrixDisplay
        ConfusionMatrixDisplay.from_predictions(y_val, y_pred)

        if(self.framework == "sklearn"):
            plt.figure(figsize=(10, 5))
            feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
            sns.barplot(x=feat_imp.values, y=feat_imp.index)
            plt.title("Feature Importance")
            plt.show()

            joblib.dump(model, f'models/{self.name}_model.pkl')

        elif( self.framework == "tf"):
            save_model(model, f'models/{self.name}_model.h5')


    def test(self, workers : int = 8, chunksize : int = 10):
        file_names = os.listdir(self.test_data_directory)
        if(self.framework == "sklearn"):
            self.model = joblib.load(f'models/{self.name}_model.pkl')
            results = process_map(self.backtest, file_names, max_workers=workers, desc="Processing", chunksize=chunksize)
        elif(self.framework == "tf"):
            results = process_map(self.backtest, file_names, max_workers=workers, desc="Processing", chunksize=chunksize)
        elif(self.framework == "nil" and not self.is_ml):
            results = process_map(self.backtest, file_names, max_workers=workers, desc="Processing", chunksize=chunksize)

        # Unpack results from processes
        day_profits, day_trades, day_num_profit, day_num_loss = zip(*results)
        
        day_profits = list(day_profits)
        day_trades = list(day_trades)
        day_num_profit = list(day_num_profit)
        day_num_loss = list(day_num_loss)

        print(day_profits)
        print("Max profit ", max(day_profits))
        print("Min profit ", min(day_profits))

        print("per day prof ", sum(day_profits)/len(day_profits))
        print("avg trade per day ", sum(day_trades)/len(day_trades))
        print("avg profit trades ", sum(day_num_profit)/len(day_num_profit))
        print("avg loss trades ", sum(day_num_loss)/len(day_num_loss))

        starting_capital = 10000
        portfolio_values = [starting_capital]
        for ret in day_profits:
            if(portfolio_values[-1]<1000000):
                new_value = portfolio_values[-1] * (1 + ret / 100)
            else:
                new_value = (1000000 * (ret / 100)) + portfolio_values[-1] 
            portfolio_values.append(new_value)
        daily_returns = np.array(day_profits) / 100

        # Parameters
        risk_free_rate = 0.0698 / 252  # Assuming 6.98% risk-free rate
        trading_days = 252    # Annualization factor

        # Sharpe Ratio Calculation
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns, ddof=1)  # Sample standard deviation
        sharpe_ratio = (mean_return - risk_free_rate) / std_return * np.sqrt(trading_days)

        print("std ", std_return)
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(portfolio_values)
        plt.title("Portfolio Value Over Time (Percentage Returns)")
        plt.xlabel("Day")
        plt.ylabel("Portfolio Value")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/test/{self.name}_portfolio_value.png")
        plt.show()

        self.test_result = sum(day_profits) / len(day_profits)


        

    def train_data_process(self, name) -> pd.DataFrame:
        data = pd.read_csv(f"{self.train_data_directory}/{name}")
        data.columns = data.columns.str.lower()
        features = self.generate_features(data)
        target = self.generate_target_var(data)
        if(target is None):
            return pd.DataFrame()
        elif(features is None):
            return pd.DataFrame()
        data = pd.concat([features, target], axis=1)

        return data
    
    
    def backtest(self, datapath):
        try:
            data = pd.read_csv(f"{self.test_data_directory}/{datapath}")
            data.columns = data.columns.str.lower()
            if self.is_ml:
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                if(self.framework == "tf"):
                    gpus = tf.config.list_physical_devices('GPU')
                    if gpus:
                        try:
                            tf.config.experimental.set_memory_growth(gpus[0], True)
                        except RuntimeError as e:
                            print(e)
                    model = load_model(f'models/{self.name}_model.h5')
                
                if(self.framework == "sklearn"):
                    model = self.model

                features = self.generate_features(data)
                feature_columns = features.columns.to_list()
                data = pd.concat([data, features], axis=1)

                data.replace([np.inf, -np.inf], 1e6, inplace=True)
                data.dropna(inplace=True, axis=0)
                data.reset_index(inplace=True, drop=True)

                if (self.framework == "sklearn"):
                    y = model.predict(data.loc[:, feature_columns])
                    data['target'] = y
                elif (self.framework == "tf"):
                    y = model.predict(data.loc[:, feature_columns], verbose=0)
                    if(y.shape[0] >= 2):
                        data['target'] = np.argmax(y, axis=1)
                    else:
                        data['target'] = y

            data["signal"] = self.generate_signals(data)

            buy = 0
            position = 0
            profits = []

            for j in range(len(data)):
                if(sum(profits) < self.day_sl):
                    break
                if(position == 0 and data.loc[j, 'signal'] == 1):
                    buy = data.loc[j, 'close']
                    position = 1
                elif(position == 1 and data.loc[j, 'signal'] == -1):
                    position = 0
                    profit = (data.loc[j, "close"]/buy - 1)*100
                    profits.append(profit)

                elif(position == 1):
                    profit = (data.loc[j, "close"]/buy - 1)*100
                    if(profit < self.sl):
                        profits.append(profit)
                        position = 0
                    elif(profit > self.tp):
                        profits.append(profit)
                        position = 0

            if(len(profits) != 0):
                total_day = sum(profits)
                num_profit = sum(1 for x in profits if x > 0)
                num_loss = sum(1 for x in profits if x < 0)
                num_trades = len(profits)

                return (total_day, num_trades, num_profit, num_loss)
            return (0, 0, 0, 0)
        except Exception as e:
            print(f"Error processing {datapath}: {e}")
            return (0, 0, 0, 0)
        
    def _permutation_test_single(self, datapath, start_index=0):
        try:
            if self.is_ml:
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                if(self.framework == "tf"):
                    gpus = tf.config.list_physical_devices('GPU')
                    if gpus:
                        try:
                            tf.config.experimental.set_memory_growth(gpus[0], True)
                        except RuntimeError as e:
                            print(e)
                    model = load_model(f'models/{self.name}_model.h5')
                
                if(self.framework == "sklearn"):
                    model = self.model

            data = pd.read_csv(f"{self.test_data_directory}/{datapath}")
            data.columns = data.columns.str.lower()

            # Assume required columns are present: open, high, low, close
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in data.columns for col in required_cols):
                raise ValueError("Data must include 'open', 'high', 'low', 'close' columns.")

            n_bars = len(data)
            if n_bars < start_index + 2:
                return (0, 0, 0, 0)  # Not enough data for meaningful permutation

            # Prepare for permutation (single market)
            time_index = data.index
            log_bars = np.log(data[required_cols])

            # Get start bar
            start_bar = log_bars.iloc[start_index].to_numpy()

            perm_index = start_index + 1
            perm_n = n_bars - perm_index

            # Compute relatives
            r_o = (log_bars['open'] - log_bars['close'].shift()).to_numpy()[perm_index:]
            r_h = (log_bars['high'] - log_bars['open']).to_numpy()[perm_index:]
            r_l = (log_bars['low'] - log_bars['open']).to_numpy()[perm_index:]
            r_c = (log_bars['close'] - log_bars['open']).to_numpy()[perm_index:]

            idx = np.arange(perm_n)

            # Shuffle intra-bar relative values (high/low/close)
            perm1 = np.random.permutation(idx)
            r_h = r_h[perm1]
            r_l = r_l[perm1]
            r_c = r_c[perm1]

            # Shuffle last close to open (gaps) separately
            perm2 = np.random.permutation(idx)
            r_o = r_o[perm2]

            # Create permutation from relative prices
            perm_bars = np.zeros((n_bars, 4))

            # Copy over real data before start index
            perm_bars[:start_index] = log_bars.to_numpy()[:start_index]

            # Copy start bar
            perm_bars[start_index] = start_bar

            for i in range(perm_index, n_bars):
                k = i - perm_index
                perm_bars[i, 0] = perm_bars[i - 1, 3] + r_o[k]
                perm_bars[i, 1] = perm_bars[i, 0] + r_h[k]
                perm_bars[i, 2] = perm_bars[i, 0] + r_l[k]
                perm_bars[i, 3] = perm_bars[i, 0] + r_c[k]

            perm_bars = np.exp(perm_bars)
            perm_ohlc = pd.DataFrame(perm_bars, index=time_index, columns=['open', 'high', 'low', 'close'])

            # Merge back any non-OHLC columns from original data
            non_ohlc_cols = [col for col in data.columns if col not in required_cols]
            if non_ohlc_cols:
                perm_ohlc = pd.concat([perm_ohlc, data[non_ohlc_cols]], axis=1)

            # Now proceed with the rest of the backtest on the permuted data
            data = perm_ohlc

            if self.is_ml:
                features = self.generate_features(data)
                feature_columns = features.columns.to_list()
                data = pd.concat([data, features], axis=1)

                data.replace([np.inf, -np.inf], 1e6, inplace=True)
                data.dropna(inplace=True, axis=0)
                data.reset_index(inplace=True, drop=True)

                if (self.framework == "sklearn"):
                    y = model.predict(data.loc[:, feature_columns])
                    data['target'] = y
                elif (self.framework == "tf"):
                    y = model.predict(data.loc[:, feature_columns], verbose=0)
                    if(y.shape[0] >= 2):
                        data['target'] = np.argmax(y, axis=1)
                    else:
                        data['target'] = y

            data["signal"] = self.generate_signals(data)

            buy = 0
            position = 0
            profits = []

            for j in range(len(data)):
                if(sum(profits) < self.day_sl):
                    break
                if(position == 0 and data.loc[j, 'signal'] == 1):
                    buy = data.loc[j, 'close']
                    position = 1
                elif(position == 1 and data.loc[j, 'signal'] == -1):
                    position = 0
                    profit = (data.loc[j, "close"]/buy - 1)*100
                    profits.append(profit)

                elif(position == 1):
                    profit = (data.loc[j, "close"]/buy - 1)*100
                    if(profit < self.sl):
                        profits.append(profit)
                        position = 0
                    elif(profit > self.tp):
                        profits.append(profit)
                        position = 0

            if(len(profits) != 0):
                total_day = sum(profits)
                num_profit = sum(1 for x in profits if x > 0)
                num_loss = sum(1 for x in profits if x < 0)
                num_trades = len(profits)

                return (total_day, num_trades, num_profit, num_loss)
            return (0, 0, 0, 0)
        except Exception as e:
            print(f"Error processing {datapath}: {e}")
            return (0, 0, 0, 0)

    def walkforward_permutation_test(self, n_iter=100, workers: int=8, chunksize: int=10):
        file_names = os.listdir(self.test_data_directory)
        net_profits = []

        if self.framework == "sklearn":
            for x_num in range(n_iter):
                self.model = joblib.load(f'models/{self.name}_model.pkl')
                results = process_map(self._permutation_test_single, file_names, max_workers=workers, desc="Processing", chunksize=chunksize)
                # Unpack results from processes
                day_profits, day_trades, day_num_profit, day_num_loss = zip(*results)
                
                # Convert to lists for consistency with original code
                day_profits = list(day_profits)
                day_trades = list(day_trades)
                day_num_profit = list(day_num_profit)
                day_num_loss = list(day_num_loss)
                avg_profit = sum(day_profits) / len(day_profits)
                net_profits.append(avg_profit)
                print(f"Iteration {x_num+1}/{n_iter}: Avg Profit = {avg_profit}")
        elif self.framework == "tf":
            from multiprocessing import Pool
            with Pool(processes=workers) as pool:
                for x_num in range(n_iter):
                    # Use pool.map for parallel processing without recreating the pool each time
                    # Note: This doesn't include built-in tqdm progress; use manual progress if needed
                    results = pool.map(self._permutation_test_single, file_names)
                    # Unpack results from processes
                    day_profits, day_trades, day_num_profit, day_num_loss = zip(*results)
                    
                    # Convert to lists for consistency with original code
                    day_profits = list(day_profits)
                    day_trades = list(day_trades)
                    day_num_profit = list(day_num_profit)
                    day_num_loss = list(day_num_loss)
                    avg_profit = sum(day_profits) / len(day_profits)
                    net_profits.append(avg_profit)
                    print(f"Iteration {x_num+1}/{n_iter}: Avg Profit = {avg_profit}")

        elif not self.is_ml:
            for x_num in range(n_iter):
                results = process_map(self._permutation_test_single, file_names, max_workers=workers, desc="Processing", chunksize=chunksize)
                # Unpack results from processes
                day_profits, day_trades, day_num_profit, day_num_loss = zip(*results)
                
                # Convert to lists for consistency with original code
                day_profits = list(day_profits)
                day_trades = list(day_trades)
                day_num_profit = list(day_num_profit)
                day_num_loss = list(day_num_loss)
                avg_profit = sum(day_profits) / len(day_profits)
                net_profits.append(avg_profit)
                print(f"Iteration {x_num+1}/{n_iter}: Avg Profit = {avg_profit}")

        if len(net_profits) < 2:
            print("Not enough data points for a meaningful plot.")
            return
            
        data = np.array(net_profits)
                
        plt.figure(figsize=(8, 5))
        plt.hist(data, bins=25, color='blue', alpha=0.6, edgecolor='black')
        # Add vertical line at self.test_result
        plt.axvline(self.test_result, color='red', linestyle='dashed', linewidth=2, label=f'Test Result: {self.test_result}')
        plt.title("Histogram of Permuted Net Profits")
        plt.xlabel("Avg Profit Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"results/permutation_test/permutation_test_histogram_{self.name}.png")
        plt.show()
        count = sum(x <= self.test_result for x in net_profits)
        print("Test result is better then", count *100 / len(net_profits), "% of the permutations")
        print("test result:", self.test_result)
        print("mean permutation results:", np.mean(data))
        print("std of permutation results:", np.std(data))



        

    def generate_features(self, data) -> pd.DataFrame:
        pass


    def generate_target_var(self, data) -> pd.DataFrame:
        pass


    def define_and_fit_model(self, x_train, y_train):
        pass


    def generate_signals(self, data) -> pd.DataFrame:
        pass

    def permutation_test(self):
        pass
