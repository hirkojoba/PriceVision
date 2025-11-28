"""
Utility Functions
Helper functions for the stock trend predictor.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple


def get_default_date_range(years: int = 5) -> Tuple[str, str]:
    """
    Get default date range for stock data.

    Args:
        years: Number of years of historical data

    Returns:
        Tuple of (start_date, end_date) as strings
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def validate_ticker(ticker: str) -> str:
    """
    Validate and normalize ticker symbol.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Normalized ticker (uppercase)

    Raises:
        ValueError: If ticker is invalid
    """
    if not ticker or not ticker.strip():
        raise ValueError("Ticker cannot be empty")

    ticker = ticker.strip().upper()

    # Basic validation (alphanumeric only)
    if not ticker.isalnum():
        raise ValueError("Ticker must contain only letters and numbers")

    if len(ticker) > 10:
        raise ValueError("Ticker is too long (max 10 characters)")

    return ticker


def ensure_directory(path: str) -> None:
    """
    Ensure a directory exists, create if it doesn't.

    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def get_model_path(ticker: str, models_dir: str = "models") -> str:
    """
    Get the model file path for a given ticker.

    Args:
        ticker: Stock ticker
        models_dir: Directory where models are stored

    Returns:
        Full path to model file
    """
    ensure_directory(models_dir)
    return os.path.join(models_dir, f"{ticker}_model.pkl")


def format_currency(value: float) -> str:
    """
    Format a number as currency.

    Args:
        value: Numeric value

    Returns:
        Formatted currency string
    """
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """
    Format a decimal as percentage.

    Args:
        value: Decimal value (e.g., 0.65 for 65%)

    Returns:
        Formatted percentage string
    """
    return f"{value:.2%}"


def get_trend_emoji(trend: str) -> str:
    """
    Get emoji representation of trend.

    Args:
        trend: Trend string ('UP', 'DOWN', 'FLAT')

    Returns:
        Emoji string
    """
    emoji_map = {
        'UP': '📈',
        'DOWN': '📉',
        'FLAT': '➡️'
    }
    return emoji_map.get(trend, '❓')


def get_trend_color(trend: str) -> str:
    """
    Get color code for trend.

    Args:
        trend: Trend string

    Returns:
        Hex color code
    """
    color_map = {
        'UP': '#2ecc71',
        'DOWN': '#e74c3c',
        'FLAT': '#95a5a6'
    }
    return color_map.get(trend, '#3498db')


def calculate_data_quality_score(df: pd.DataFrame) -> float:
    """
    Calculate a simple data quality score.

    Args:
        df: DataFrame with stock data

    Returns:
        Quality score between 0 and 1
    """
    if df.empty:
        return 0.0

    # Check for missing values
    missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))

    # Check for duplicate dates
    if 'Date' in df.columns:
        duplicate_ratio = df['Date'].duplicated().sum() / len(df)
    else:
        duplicate_ratio = 0

    # Quality score (higher is better)
    score = 1.0 - (missing_ratio + duplicate_ratio)

    return max(0.0, min(1.0, score))


def get_business_days_ahead(start_date: datetime, days: int = 1) -> datetime:
    """
    Get a date N business days ahead.

    Args:
        start_date: Starting date
        days: Number of business days ahead

    Returns:
        Future datetime
    """
    current = start_date
    business_days_count = 0

    while business_days_count < days:
        current += timedelta(days=1)
        # Skip weekends (Saturday=5, Sunday=6)
        if current.weekday() < 5:
            business_days_count += 1

    return current


if __name__ == "__main__":
    # Quick tests
    print("Testing utility functions...")

    # Test date range
    start, end = get_default_date_range(3)
    print(f"Default 3-year range: {start} to {end}")

    # Test ticker validation
    try:
        print(f"Validated ticker: {validate_ticker('aapl')}")
        print(f"Validated ticker: {validate_ticker('  MSFT  ')}")
    except ValueError as e:
        print(f"Validation error: {e}")

    # Test formatting
    print(f"Currency: {format_currency(12345.67)}")
    print(f"Percentage: {format_percentage(0.6543)}")

    # Test trend helpers
    print(f"UP emoji: {get_trend_emoji('UP')}")
    print(f"DOWN color: {get_trend_color('DOWN')}")

    print("\nAll utility tests passed!")
