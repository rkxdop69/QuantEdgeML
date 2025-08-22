import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def plot_candlestick(df, title="Candlestick Chart", date_col='date', open_col='open', high_col='high', low_col='low', close_col='close'):
    """
    Generates a candlestick chart using Matplotlib and returns the figure and axes.
    This approach is well-suited for handling intraday data and allows for further customization.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing the financial data.
        title (str): The title for the chart.
        date_col (str): The name of the column containing the date information.
        open_col (str): The name of the column containing the open price.
        high_col (str): The name of the column containing the high price.
        low_col (str): The name of the column containing the low price.
        close_col (str): The name of the column containing the close price.

    Returns:
        fig (matplotlib.figure.Figure): The figure object.
        ax (matplotlib.axes.Axes): The axes object on which the chart is plotted.
    """
    # --- Input Validation and Data Preparation ---
    required_columns = {date_col, open_col, high_col, low_col, close_col}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    plot_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(plot_df[date_col]):
        try:
            plot_df[date_col] = pd.to_datetime(plot_df[date_col])
        except Exception as e:
            raise TypeError(f"Could not convert '{date_col}' column to datetime. Error: {e}")

    # Use a numerical index for plotting to avoid time gaps
    plot_df.reset_index(inplace=True)
    x = plot_df.index

    # --- Create Figure and Axes ---
    fig, ax = plt.subplots(figsize=(15, 7))

    # --- Define Colors ---
    up_color = 'green'
    down_color = 'red'

    # --- Plotting Wicks (High-Low lines) ---
    # Plot the vertical lines for the wicks
    ax.vlines(x=x, ymin=plot_df[low_col], ymax=plot_df[high_col], color='black', linewidth=1)

    # --- Plotting Candle Bodies (Open-Close rectangles) ---
    # Loop through each data point to draw the candle body
    for i in range(len(plot_df)):
        current_open = plot_df[open_col].iloc[i]
        current_close = plot_df[close_col].iloc[i]
        
        if current_close >= current_open:
            # Bullish (green) candle
            # A rectangle is defined by its bottom-left corner, width, and height
            ax.add_patch(plt.Rectangle((x[i] - 0.4, current_open), 0.8, current_close - current_open, facecolor=up_color))
        else:
            # Bearish (red) candle
            ax.add_patch(plt.Rectangle((x[i] - 0.4, current_close), 0.8, current_open - current_close, facecolor=down_color))

    # --- Formatting the Chart ---
    ax.set_title(title, fontsize=16)
    ax.set_ylabel('Price (USD)')
    ax.set_xlabel('Date')

    # Format x-axis to show dates instead of numerical indices
    # We select a subset of ticks to prevent them from overlapping
    num_ticks = 6
    tick_indices = plot_df.index[::len(plot_df)//num_ticks if len(plot_df) > num_ticks else 1]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(plot_df[date_col].iloc[tick_indices].dt.strftime('%Y-%m-%d %H:%M'), rotation=45, ha='right')

    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout() # Adjust layout to make room for rotated tick labels

    return fig, ax


# --- Example Usage with 1-Minute Intraday Data ---
if __name__ == '__main__':
    # Create a sample DataFrame with 1-minute OHLC data
    intraday_data = {
        'Timestamp': [
            datetime(2023, 8, 21, 15, 30), datetime(2023, 8, 21, 15, 31), datetime(2023, 8, 21, 15, 32),
            datetime(2023, 8, 22, 9, 30), datetime(2023, 8, 22, 9, 31), datetime(2023, 8, 22, 9, 32),
            datetime(2023, 8, 22, 9, 33), datetime(2023, 8, 22, 9, 34), datetime(2023, 8, 22, 9, 35)
        ],
        'O':  [150.0, 150.2, 150.1, 151.0, 151.2, 151.1, 151.3, 151.5, 151.4],
        'H':  [150.3, 150.5, 150.3, 151.4, 151.5, 151.3, 151.6, 151.7, 151.5],
        'L':  [149.9, 150.1, 150.0, 150.8, 151.0, 150.9, 151.1, 151.3, 151.2],
        'C':  [150.2, 150.3, 150.0, 151.3, 151.1, 151.2, 151.5, 151.4, 151.3]
    }
    sample_df_intraday = pd.DataFrame(intraday_data)

    print("Displaying intraday chart using Matplotlib.")
    
    # Call the function to get the figure and axes objects
    fig, ax = plot_candlestick(sample_df_intraday,
                               title="1-Minute Intraday Stock Price (Matplotlib)",
                               date_col='Timestamp',
                               open_col='O',
                               high_col='H',
                               low_col='L',
                               close_col='C')

    # --- Example of plotting an additional line (e.g., a moving average) ---
    # Calculate a 3-period simple moving average on the close price
    sma = sample_df_intraday['C'].rolling(window=3).mean()
    
    # Plot the SMA on the same axes returned by the function
    # We use the same numerical index for the x-axis to align it correctly
    ax.plot(sample_df_intraday.index, sma, color='blue', linestyle='--', label='3-Period SMA')
    ax.legend()

    # Show the final plot with both the candlesticks and the SMA line
    plt.show()
