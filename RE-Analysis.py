import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from matplotlib.ticker import MaxNLocator

# Example real estate asset data (cap rates)
data = {
    'Office': {'symbol': 'OFFICE', 'cap_rate': [0.06, 0.061, 0.059, 0.058, 0.057, 0.056]},
    'Retail': {'symbol': 'RETAIL', 'cap_rate': [0.07, 0.068, 0.069, 0.07, 0.071, 0.072]},
    'Industrial': {'symbol': 'IND', 'cap_rate': [0.05, 0.051, 0.052, 0.053, 0.054, 0.055]}
}

# Convert the example data to DataFrames
def create_dataframe(asset_data, dates):
    df = pd.DataFrame(asset_data)
    df['Date'] = pd.to_datetime(dates)
    df.set_index('Date', inplace=True)
    return df

dates = pd.date_range(start='2023-01-01', periods=len(data['Office']['cap_rate']), freq='ME')
dfs = {asset: create_dataframe(data[asset], dates) for asset in data}

# Prepare and fit models for each asset type
def process_asset(asset, df):
    df.rename(columns={'cap_rate': 'y'}, inplace=True)

    # Fit the Prophet model
    model = Prophet()
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(df.reset_index().rename(columns={'Date': 'ds'}))

    # Create future dataframe for projections
    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)
    forecast_data = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Projected_Cap_Rate'})
    forecast_data['SMA_12'] = forecast_data['Projected_Cap_Rate'].rolling(window=12).mean()

    # Calculate 12-month SMA for historical data
    df['SMA_12'] = df['y'].rolling(window=12).mean()

    # Define buy/sell signals
    generate_signals(df)
    generate_signals(forecast_data, historical=False)

    # Calculate Trend Wave Oscillator (TWO)
    calculate_two(df, column_name='y')
    forecast_data.set_index('Date', inplace=True)
    calculate_two(forecast_data, column_name='Projected_Cap_Rate')

    return df, forecast_data

# Define buy/sell signals
def generate_signals(df, historical=True):
    df['Signal'] = 0
    if historical:
        df.loc[df['y'] > df['SMA_12'], 'Signal'] = 1
        df.loc[df['y'] < df['SMA_12'], 'Signal'] = -1
    else:
        df['SMA_12'] = df['Projected_Cap_Rate'].rolling(window=12).mean()
        df.loc[df['Projected_Cap_Rate'] > df['SMA_12'], 'Signal'] = 1
        df.loc[df['Projected_Cap_Rate'] < df['SMA_12'], 'Signal'] = -1

# Calculate Trend Wave Oscillator (TWO)
def calculate_two(df, short_window=12, long_window=24, column_name='y'):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    df['TWO'] = df[column_name].rolling(window=short_window).mean() - df[column_name].rolling(window=long_window).mean()
    df['TWO_Buy_Marker'] = np.where(df['TWO'] > 0, df['TWO'], np.nan)
    df['TWO_Sell_Marker'] = np.where(df['TWO'] < 0, df['TWO'], np.nan)

# Plot buy/sell signals
def plot_signals(ax, df, historical=True):
    color_map = {'buy': ('g', 'lime'), 'sell': ('r', 'darkred')}
    suffix = '(Historical)' if historical else '(Projected)'
    price_col = 'y' if historical else 'Projected_Cap_Rate'
    ax.scatter(df[df['Signal'] == 1].index, df[df['Signal'] == 1][price_col],
               marker='^', color=color_map['buy'][0 if historical else 1], label=f'Buy Signal {suffix}')
    ax.scatter(df[df['Signal'] == -1].index, df[df['Signal'] == -1][price_col],
               marker='v', color=color_map['sell'][0 if historical else 1], label=f'Sell Signal {suffix}')

# Plot data
def plot_data(df, forecast_data, asset):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 20), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.5})

    ax1.plot(df.index, df['y'], label=f'Historical {asset} Cap Rates', color='blue')
    ax1.plot(forecast_data.index, forecast_data['Projected_Cap_Rate'], label=f'Projected {asset} Cap Rates', color='red')
    ax1.plot(df.index, df['SMA_12'], label='12-Month SMA (Historical)', color='green', linestyle='--')
    ax1.plot(forecast_data.index, forecast_data['SMA_12'], label='12-Month SMA (Projected)', color='orange', linestyle='--')

    # Plot signals
    plot_signals(ax1, df, historical=True)
    plot_signals(ax1, forecast_data, historical=False)

    # Projection start marker
    ax1.axvline(x=df.index.max(), color='red', linestyle='--', label='Projection Start')
    ax1.set_title(f'{asset} Cap Rate Projection Until Next Year with Buy/Sell Signals')
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'{asset} Cap Rate')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Plot TWO
    ax2.plot(df.index, df['TWO'], label=f'Trend Wave Oscillator (Historical)', color='blue')
    ax2.plot(forecast_data.index, forecast_data['TWO'], label=f'Trend Wave Oscillator (Projected)', color='red')

    # Plot TWO markers
    ax2.scatter(df.index, df['TWO_Buy_Marker'], color='green', marker='^', label='Buy Marker (Historical)')
    ax2.scatter(df.index, df['TWO_Sell_Marker'], color='red', marker='v', label='Sell Marker (Historical)')
    ax2.scatter(forecast_data.index, forecast_data['TWO_Buy_Marker'], color='lime', marker='^', label='Buy Marker (Projected)')
    ax2.scatter(forecast_data.index, forecast_data['TWO_Sell_Marker'], color='darkred', marker='v', label='Sell Marker (Projected)')

    ax2.axhline(0, color='black', linestyle='--', label='Zero Line')
    ax2.set_title('Trend Wave Oscillator with Markers')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('TWO')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=6))

    plt.tight_layout(pad=2.0)
    plt.show()

# Main execution
for asset, df in dfs.items():
    df, forecast_data = process_asset(asset, df)
    plot_data(df, forecast_data, asset)
