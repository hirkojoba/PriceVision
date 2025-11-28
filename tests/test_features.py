"""
Unit tests for features module
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import (
    add_technical_indicators,
    create_labels,
    prepare_dataset,
    compute_rsi,
    get_latest_features
)


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n = len(dates)

    # Generate synthetic price data
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(n) * 2)

    df = pd.DataFrame({
        'Date': dates,
        'Open': close_prices + np.random.randn(n) * 0.5,
        'High': close_prices + np.abs(np.random.randn(n) * 1),
        'Low': close_prices - np.abs(np.random.randn(n) * 1),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, n)
    })

    return df


def test_add_technical_indicators(sample_stock_data):
    """Test technical indicator calculation."""
    df = add_technical_indicators(sample_stock_data)

    # Check that indicators were added
    expected_columns = [
        'daily_return', 'ma_5', 'ma_10', 'ma_20',
        'volatility_5', 'volatility_10',
        'volume_change', 'volume_ma_5', 'volume_ratio',
        'close_to_ma5', 'close_to_ma10', 'close_to_ma20',
        'high_low_range', 'prev_close', 'prev_return',
        'return_5d', 'return_10d', 'rsi_14'
    ]

    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

    # Check that NaN rows were dropped
    assert df.isnull().sum().sum() == 0, "DataFrame contains NaN values"

    # Check that data was reduced (some rows dropped due to rolling windows)
    assert len(df) < len(sample_stock_data), "No rows were dropped"


def test_create_labels(sample_stock_data):
    """Test label creation."""
    df = create_labels(sample_stock_data)

    # Check that label column exists
    assert 'label' in df.columns

    # Check label values are valid (-1, 0, 1)
    unique_labels = df['label'].unique()
    assert all(label in [-1, 0, 1] for label in unique_labels)

    # Check that last row was dropped
    assert len(df) == len(sample_stock_data) - 1


def test_prepare_dataset(sample_stock_data):
    """Test complete dataset preparation."""
    X, y = prepare_dataset(sample_stock_data)

    # Check shapes
    assert len(X) == len(y), "Feature and label lengths don't match"
    assert len(X) > 0, "Dataset is empty"

    # Check that features don't contain raw OHLCV
    raw_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']
    for col in raw_columns:
        assert col not in X.columns, f"Raw column {col} should not be in features"

    # Check label values
    assert all(label in [-1, 0, 1] for label in y.unique())

    # Check for NaN values
    assert X.isnull().sum().sum() == 0, "Features contain NaN values"
    assert y.isnull().sum() == 0, "Labels contain NaN values"


def test_compute_rsi():
    """Test RSI calculation."""
    # Create simple price series
    prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                        111, 110, 112, 114, 113, 115, 117, 116, 118, 120])

    rsi = compute_rsi(prices, window=14)

    # Check that RSI is in valid range (0-100)
    valid_rsi = rsi.dropna()
    assert all((valid_rsi >= 0) & (valid_rsi <= 100)), "RSI values out of range"


def test_get_latest_features(sample_stock_data):
    """Test extracting latest features."""
    df_with_indicators = add_technical_indicators(sample_stock_data)
    latest = get_latest_features(df_with_indicators)

    # Check it's a single row
    assert len(latest) == 1, "Should return single row"

    # Check it doesn't contain raw OHLCV
    raw_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']
    for col in raw_columns:
        assert col not in latest.columns, f"Raw column {col} should not be in features"


def test_label_thresholds():
    """Test label creation with custom thresholds."""
    # Create simple test data
    df = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=10),
        'Open': [100] * 10,
        'High': [101] * 10,
        'Low': [99] * 10,
        'Close': [100, 101, 100, 102, 100, 99, 100, 101, 100, 100],
        'Volume': [1000000] * 10
    })

    # Test with tight thresholds
    df_labeled = create_labels(df, up_threshold=0.01, down_threshold=-0.01)

    # Verify labels were created
    assert 'label' in df_labeled.columns
    assert len(df_labeled) == 9  # One less than original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
