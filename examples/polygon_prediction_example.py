"""
Polygon.io Data Integration Example for Kronos Prediction

This script demonstrates how to fetch real-time S&P 500 data from Polygon.io
and use it with the Kronos prediction model.
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor
from utils.polygon_fetcher import PolygonDataFetcher


def plot_prediction(kline_df, pred_df, symbol):
    """Plot prediction results vs ground truth"""
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax1.set_ylabel('Close Price ($)', fontsize=14)
    ax1.set_title(f'{symbol} Stock Price Prediction', fontsize=16)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.plot(volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax2.plot(volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.set_xlabel('Time', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """Main function to demonstrate Polygon.io + Kronos integration"""
    
    # Configuration
    API_KEY = "o3GPaqqnnAucZzpseeH1_i3qpG57ZWtj"
    SYMBOL = "AMZN"  # Apple Inc. - you can change this to any S&P 500 symbol
    LOOKBACK_DAYS = 7  # Number of days to fetch historical data
    LOOKBACK_POINTS = 400  # Number of data points for model input
    PRED_LEN = 120  # Number of points to predict
    
    print("="*60)
    print("Polygon.io + Kronos Financial Prediction Demo")
    print("="*60)
    
    # 1. Initialize Polygon.io data fetcher
    print(f"\n1. Initializing Polygon.io data fetcher...")
    fetcher = PolygonDataFetcher(API_KEY)
    
    # Display available symbols
    print(f"Available S&P 500 symbols: {len(fetcher.get_sp500_symbols())}")
    print(f"Selected symbol: {SYMBOL}")
    
    # 2. Fetch real-time data
    print(f"\n2. Fetching {SYMBOL} data from Polygon.io...")
    print(f"   - Lookback period: {LOOKBACK_DAYS} days")
    print(f"   - Data interval: 5-minute candles")
    
    # Set current timestamp for demo (using a weekday to ensure market data)
    # Use last Friday if today is weekend
    current_dt = datetime.now()
    if current_dt.weekday() >= 5:  # Saturday (5) or Sunday (6)
        days_to_subtract = current_dt.weekday() - 4  # Go back to Friday
        current_dt = current_dt - timedelta(days=days_to_subtract)
    current_timestamp = current_dt.strftime('%Y-%m-%d %H:%M:%S')
    
    df = fetcher.get_stock_data(
        symbol=SYMBOL,
        current_timestamp=current_timestamp,
        lookback_days=LOOKBACK_DAYS,
        multiplier=5,  # 5-minute intervals
        timespan="minute"
    )
    
    if df is None or len(df) == 0:
        print("Failed to fetch data from Polygon.io")
        print("   Please check your API key and internet connection")
        print("   Note: Markets may be closed (weekends/holidays)")
        return
    
    print(f"Successfully fetched {len(df)} data points")
    print(f"   Date range: {df['timestamps'].iloc[0]} to {df['timestamps'].iloc[-1]}")
    
    # 3. Save fetched data
    saved_path = fetcher.save_data(df, f"{SYMBOL}_polygon", "data")
    print(f"   Data saved to: {saved_path}")
    
    # 4. Prepare data for Kronos model
    print(f"\n3. Preparing data for Kronos model...")
    
    # Ensure we have enough data
    if len(df) < LOOKBACK_POINTS + PRED_LEN:
        print(f"Warning: Not enough data points. Need {LOOKBACK_POINTS + PRED_LEN}, got {len(df)}")
        print("   Adjusting parameters...")
        available_points = len(df)
        LOOKBACK_POINTS = min(400, available_points - PRED_LEN) if available_points > PRED_LEN else available_points // 2
        PRED_LEN = min(120, available_points - LOOKBACK_POINTS)
        print(f"   Adjusted: lookback={LOOKBACK_POINTS}, prediction={PRED_LEN}")
    
    # Convert timestamps to datetime
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    
    # Prepare model inputs
    x_df = df.iloc[:LOOKBACK_POINTS][['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.iloc[:LOOKBACK_POINTS]['timestamps']
    y_timestamp = df.iloc[LOOKBACK_POINTS:LOOKBACK_POINTS+PRED_LEN]['timestamps']
    
    print(f"   Model input: {len(x_df)} historical points")
    print(f"   Prediction target: {len(y_timestamp)} future points")
    
    # 5. Load Kronos model
    print(f"\n4. Loading Kronos model...")
    try:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return
    
    # 6. Initialize predictor
    print(f"\n5. Initializing predictor...")
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
    print("Predictor initialized")
    
    # 7. Make prediction
    print(f"\n6. Making prediction...")
    print(f"   This may take a few minutes on CPU...")
    
    try:
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=PRED_LEN,
            T=1.2,          # Slightly higher temperature for diversity
            top_p=0.95,     # Nucleus sampling
            sample_count=1,  # Single prediction for speed
            verbose=True
        )
        print("Prediction completed successfully")
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        return
    
    # 8. Display results
    print(f"\n7. Results Summary")
    print("="*40)
    print(f"Symbol: {SYMBOL}")
    print(f"Prediction period: {y_timestamp.iloc[0]} to {y_timestamp.iloc[-1]}")
    print(f"Historical close price: ${x_df['close'].iloc[-1]:.2f}")
    print(f"Predicted close price: ${pred_df['close'].iloc[-1]:.2f}")
    
    price_change = pred_df['close'].iloc[-1] - x_df['close'].iloc[-1]
    price_change_pct = (price_change / x_df['close'].iloc[-1]) * 100
    
    print(f"Predicted price change: ${price_change:.2f} ({price_change_pct:+.2f}%)")
    
    print(f"\nForecasted Data Sample:")
    print(pred_df.head())
    
    # 9. Visualize results
    print(f"\n8. Generating visualization...")
    
    # Combine historical and predicted data for visualization
    kline_df = df.iloc[:LOOKBACK_POINTS+PRED_LEN].copy()
    
    # Plot results
    plot_prediction(kline_df, pred_df, SYMBOL)
    
    print("Demo completed successfully!")
    print(f"\nNext steps:")
    print(f"- Try different symbols: {', '.join(fetcher.get_sp500_symbols()[:10])}...")
    print(f"- Adjust prediction parameters (T, top_p, sample_count)")
    print(f"- Use longer lookback periods for better predictions")


if __name__ == "__main__":
    main()