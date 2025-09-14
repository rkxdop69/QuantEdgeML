from StrategyTester import Strategy
from feature_generator_4 import feature_gen
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from keras import layers, models 
from keras.callbacks import EarlyStopping

class Strategy_1(Strategy):
    def __init__(self, name = "Strategy_0_dl", train_data_directory = "", test_data_directory = "", is_ml = True, sl = -100, day_sl = -500, tp = 10000000, framework = "sklearn"):
        super().__init__(name, train_data_directory, test_data_directory, is_ml, sl, day_sl, tp, framework)


    def generate_features(self, data):
        features = feature_gen(data)
        return features

    def generate_target_var(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        min_profit = 3.0

        df['target'] = 0
        df['savgol'] = savgol_filter(df['close'], 11, 3)
        df['savgol_shift'] = df['savgol'].shift()
        df['target'] = np.where(df['savgol'] > df['savgol_shift'], 2, 0)

        buy = 0
        i=0
        while(i<len(df)):
            if(df.loc[i, 'target'] == 2):
                buy = df.loc[i, 'close']
                for j in range(i+1, len(df)):
                    if(df.loc[j, 'target'] == 0):
                        profit = (df.loc[j, "close"]/buy - 1)*100
                        if(profit < min_profit):
                            for k in range(j+1, len(df)):
                                if(df.loc[k, 'target'] == 2 or k == len(df)-1):
                                    df.loc[i:k, 'target'] = 1
                                    i = k - 1
                                    break
                            break
                        else:
                            i = j
                            break
                    if(j == len(df)-1):
                        i = j
            i += 1
                

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
        
        input_shape = (x_train.shape[1],)
        num_classes = len(np.unique(y_train))
        print(f"Input shape: {input_shape}")
        print(f"Detected {num_classes} classes.")

        # 2. Build the model using the helper method
        model = models.Sequential([
            layers.Input(shape=(x_train.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])

        # 3. Compile the model
        # 'sparse_categorical_crossentropy' is used for multi-class integer labels.
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        # Print a summary of the model architecture
        model.summary()

        # 4. Set up the EarlyStopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=2,
            verbose=1,
            mode='min',
            restore_best_weights=True
        )

        # 5. Train the model
        print("\n--- Starting model training ---")
        model.fit(
            x_train,
            y_train,
            epochs=10,
            batch_size=256,
            validation_split=0.1,
            callbacks=[early_stopping],
        )
        print("\n--- Training finished ---")

        # 6. Return the fully trained model
        return model

    def generate_signals(self, data):
        # The returned signal should be 1 for buy, -1 for sell, and 0 for hold/do nothing
        data["signal"] = data["target"] - 1
        return data["signal"]


if __name__ == "__main__":
    strategy_2 = Strategy_1(train_data_directory="train_data", test_data_directory="test_data", sl=-5, day_sl=-20, framework="tf")
    strategy_2.prepare_data_train(workers=10)
    strategy_2.train()
    strategy_2.test(workers=5)
    strategy_2.walkforward_permutation_test(n_iter=200, workers=5)
