"""
Feature Engineering Module
Creates technical indicators and labels for stock trend prediction.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the stock data.

    Computes:
    - Daily returns
    - Moving averages (5, 10, 20 day)
    - Volatility (5 and 10 day rolling std)
    - Volume features
    - Price position relative to moving averages
    - Lag features

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added technical indicator columns
    """
    df = df.copy()

    # Daily returns
    df['daily_return'] = df['Close'].pct_change()

    # Moving averages
    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_10'] = df['Close'].rolling(window=10).mean()
    df['ma_20'] = df['Close'].rolling(window=20).mean()

    # Volatility (rolling standard deviation of returns)
    df['volatility_5'] = df['daily_return'].rolling(window=5).std()
    df['volatility_10'] = df['daily_return'].rolling(window=10).std()

    # Volume features
    df['volume_change'] = df['Volume'].pct_change()
    df['volume_ma_5'] = df['Volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma_5']

    # Price position relative to moving averages
    df['close_to_ma5'] = (df['Close'] - df['ma_5']) / df['ma_5']
    df['close_to_ma10'] = (df['Close'] - df['ma_10']) / df['ma_10']
    df['close_to_ma20'] = (df['Close'] - df['ma_20']) / df['ma_20']

    # High-Low range
    df['high_low_range'] = (df['High'] - df['Low']) / df['Close']

    # Lag features - previous day values
    df['prev_close'] = df['Close'].shift(1)
    df['prev_return'] = df['daily_return'].shift(1)
    df['prev_volume'] = df['Volume'].shift(1)

    # Rolling return features (5-day cumulative return)
    df['return_5d'] = df['Close'].pct_change(periods=5)
    df['return_10d'] = df['Close'].pct_change(periods=10)

    # Momentum indicators
    df['rsi_14'] = compute_rsi(df['Close'], window=14)

    # Drop rows with NaN values created by rolling windows
    # The first ~20 rows will have NaNs due to 20-day MA
    df = df.dropna()

    return df


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).

    Args:
        prices: Series of closing prices
        window: RSI window period (default 14)

    Returns:
        Series of RSI values (0-100)
    """
    delta = prices.diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def create_labels(
    df: pd.DataFrame,
    up_threshold: float = 0.005,
    down_threshold: float = -0.005
) -> pd.DataFrame:
    """
    Create trend labels based on next-day price movement.

    Labels:
    - 1 (UP): Next day close >= today close * (1 + up_threshold)
    - -1 (DOWN): Next day close <= today close * (1 + down_threshold)
    - 0 (FLAT): Otherwise

    Args:
        df: DataFrame with Close price
        up_threshold: Threshold for UP trend (default 0.005 = 0.5%)
        down_threshold: Threshold for DOWN trend (default -0.005 = -0.5%)

    Returns:
        DataFrame with 'label' column added
    """
    df = df.copy()

    # Calculate next day's close
    df['next_close'] = df['Close'].shift(-1)

    # Calculate percent change to next day
    df['next_return'] = (df['next_close'] - df['Close']) / df['Close']

    # Create labels
    df['label'] = 0  # Default to FLAT

    # UP trend
    df.loc[df['next_return'] >= up_threshold, 'label'] = 1

    # DOWN trend
    df.loc[df['next_return'] <= down_threshold, 'label'] = -1

    # Drop the last row (no next_close available)
    df = df[:-1]

    # Drop temporary columns
    df = df.drop(columns=['next_close', 'next_return'])

    return df


def prepare_dataset(
    df: pd.DataFrame,
    up_threshold: float = 0.005,
    down_threshold: float = -0.005
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Complete feature engineering pipeline.

    Args:
        df: Raw OHLCV DataFrame
        up_threshold: Threshold for UP trend
        down_threshold: Threshold for DOWN trend

    Returns:
        Tuple of (X, y) where:
        - X: Feature DataFrame
        - y: Label Series
    """
    # Add technical indicators
    df = add_technical_indicators(df)

    # Create labels
    df = create_labels(df, up_threshold, down_threshold)

    # Define feature columns (exclude raw OHLCV, Date, and label)
    exclude_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'label']

    feature_columns = [col for col in df.columns if col not in exclude_columns]

    # Separate features and labels
    X = df[feature_columns].copy()
    y = df['label'].copy()

    # Store feature names for later use
    X.feature_names = feature_columns

    return X, y


def get_latest_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from the most recent data point for prediction.

    Args:
        df: DataFrame with technical indicators already computed

    Returns:
        Single-row DataFrame with latest features
    """
    # Add indicators if not present
    if 'ma_5' not in df.columns:
        df = add_technical_indicators(df)

    # Get feature columns (same as in prepare_dataset)
    exclude_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'label']
    feature_columns = [col for col in df.columns if col not in exclude_columns]

    # Get last row
    latest = df[feature_columns].iloc[[-1]].copy()

    return latest


if __name__ == "__main__":
    # Quick test
    from data_loader import load_stock_data

    print("Testing feature engineering...")
    try:
        # Load sample data
        df = load_stock_data("AAPL", "2023-01-01", "2024-01-01")
        print(f"Loaded {len(df)} rows")

        # Prepare dataset
        X, y = prepare_dataset(df)
        print(f"\nFeatures shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"\nFeature columns ({len(X.columns)}):")
        print(X.columns.tolist())
        print(f"\nLabel distribution:")
        print(y.value_counts())
        print(f"\nSample features:")
        print(X.head())

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
