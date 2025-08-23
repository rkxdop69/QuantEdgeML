from StrategyTester import Strategy
from feature_generator_4 import feature_gen
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import savgol_filter

class Strategy_1(Strategy):
    def __init__(self, name = "Strategy_0_updated", train_data_directory = "", test_data_directory = "", is_ml = True, sl = -100, day_sl = -500, tp = 10000000, framework = "sklearn"):
        super().__init__(name, train_data_directory, test_data_directory, is_ml, sl, day_sl, tp, framework)


    def generate_features(self, data):
        features = feature_gen(data)
        return features

    def generate_target_var(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['target'] = 0
        df["savgol_close"] = savgol_filter(df['close'], 11, 3)  # window size 11, polynomial order 3
        df['target'] = np.where(df['close'] > df['close'].shift(), 1, 0)
        df["target"] = df["target"].astype(int)
        return df[["target"]]


    def define_and_fit_model(self, x_train, y_train):
        """
        Defines, compiles, and fits the model for multi-class classification on tabular data.

        Args:
            x_train (np.array): The training input data (2D array).
            y_train (np.array): The training target data, with integer labels (e.g., 0, 1, 2).

        Returns:
            The trained model.
        """
        if x_train.ndim != 2:
            raise ValueError("Input x_train must be a 2D array for tabular data.")
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=10, min_samples_leaf = 60)

        model.fit(x_train, y_train)

         #Print training completion message

        print("\n--- Training finished ---")

        # 6. Return the fully trained model
        return model

    def generate_signals(self, data):
        # The returned signal should be 1 for buy, -1 for sell, and 0 for hold
        signal = np.where(data['target'] > 0.5, 1, -1)
        return signal


if __name__ == "__main__":
    # Prepare the training data
    strategy_2 = Strategy_1(train_data_directory="train_data", test_data_directory="test_data", sl=-5, day_sl=-20, framework="sklearn")
    # strategy_2.prepare_data_train()
    # strategy_2.train()
    strategy_2.test()
    strategy_2.walkforward_permutation_test(n_iter=1000)
