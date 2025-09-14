# QuantEdgeML

QuantEdgeML is a demonstration project showing how machine learning can be used to develop trading strategies for financial markets. It is not a full trading framework, but provides examples and code for feature engineering, model training, backtesting, and evaluation. Use this as a reference or starting point for your own research.

## Features
- Example strategies using machine learning and deep learning
- Feature generation for financial time series
- Backtesting and walkforward permutation testing (the method was talked about by Timothy Masters in his book)
- Model training and evaluation (sklearn, TensorFlow/Keras)
- Signal generation and filtering

## Strategies Explained
- **strategy_0.py**: The most common ML strategy. The target variable is 1 if the price moves up in the next candle, 0 otherwise. Features are generated in `feature_generator_4.py`. The ML model used is Random Forest Classifier, as suggested in "Advances in Financial Machine Learning" by Marcos Lopez de Prado.
- **strategy_1.py**: The target variable is smoothed using the Savitzky-Golay filter (centered window) on close prices to reduce noise. Trades with less than a threshold return are converted to hold/do nothing signals. The ML model and features are the same as strategy_0.
- **strategy_2.py**: Same as strategy_1, but uses a Deep Neural Network (DNN) instead of Random Forest Classifier.
- **strategy_3.py**: A standard, unoptimized MACD+RSI strategy with default parameters. Included for reference purposes only.

## Datasets
### train_data
- Contains `.csv` files
- Each file: OHLCV data for 1 day, 1-minute candlesticks
- Data: NSE stocks/index options for the year 2022 (I was only able to get 2022 data)

### test_data
- Contains `.csv` files
- Each file: OHLCV data for 1 day, 1-minute candlesticks
- Data: NIFTY index options for a few days in January and February 2025 (it was manually recorded from a broker API)

## File Structure
- `StrategyTester.py` — Main backtesting and evaluation engine
- `feature_generator_*.py` — Feature engineering modules
- `strategy_*.py` — Example strategies (ML, DL, etc.)
- `models/` — Saved model files
- `compiled_data/` — Preprocessed datasets
- `train_data/`, `test_data/` — Raw data folders
- `.gitignore` — Ignore large files and models

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Disclaimer
This software is for research and educational purposes only. Use at your own risk. No financial advice is provided.

## Contributing
Pull requests and issues are welcome! Please open an issue to discuss major changes first.

## Contact
For questions or collaboration, open an issue or contact the repo owner via GitHub. You can also email me at rkxdop69@gmail.com
