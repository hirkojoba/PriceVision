"""
Data Loader Module
Handles downloading historical stock data using yfinance.
"""

import pandas as pd
import yfinance as yf
from typing import Optional
from datetime import datetime
import time


def load_stock_data(ticker: str, start: str, end: str, retry_count: int = 3) -> pd.DataFrame:
    """
    Load historical OHLCV stock data for a given ticker.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format
        retry_count: Number of retry attempts (default 3)

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume

    Raises:
        ValueError: If ticker is invalid or no data is available
        Exception: If data download fails
    """
    last_error = None

    for attempt in range(retry_count):
        try:
            # Use Ticker object for more reliable data fetching
            ticker_obj = yf.Ticker(ticker)

            # Download historical data using the Ticker object
            data = ticker_obj.history(start=start, end=end, auto_adjust=False)

            # Check if data is empty
            if data.empty:
                raise ValueError(f"No data available for ticker '{ticker}' in the specified date range.")

            # Reset index to make Date a column
            data = data.reset_index()

            # Ensure we have the required columns
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

            # yfinance returns 'Adj Close' as well, but we only need these columns
            # Filter to keep only required columns
            available_columns = [col for col in required_columns if col in data.columns]

            if len(available_columns) != len(required_columns):
                missing = set(required_columns) - set(available_columns)
                raise ValueError(f"Missing required columns: {missing}")

            data = data[required_columns].copy()

            # Convert Date to datetime if not already
            data['Date'] = pd.to_datetime(data['Date'])

            # Sort by date ascending
            data = data.sort_values('Date').reset_index(drop=True)

            # Remove any rows with NaN values
            data = data.dropna()

            # If we got here, data was successfully loaded
            if len(data) > 0:
                return data
            else:
                raise ValueError(f"No valid data returned for ticker '{ticker}'")

        except Exception as e:
            last_error = e
            if attempt < retry_count - 1:
                # Wait before retrying (exponential backoff)
                wait_time = (attempt + 1) * 2
                print(f"Attempt {attempt + 1} failed. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                # Final attempt failed
                if "No data found" in str(e) or "No timezone found" in str(e) or "symbol may be delisted" in str(e):
                    raise ValueError(f"Invalid ticker symbol '{ticker}' or no data available. Please check the ticker symbol and try again.")
                raise Exception(f"Error downloading data for {ticker} after {retry_count} attempts: {str(last_error)}")

    raise Exception(f"Failed to download data for {ticker}: {str(last_error)}")


def validate_date_range(start: str, end: str) -> bool:
    """
    Validate that start and end dates are in correct format and order.

    Args:
        start: Start date string
        end: End date string

    Returns:
        True if valid, raises ValueError otherwise
    """
    try:
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)

        if start_date >= end_date:
            raise ValueError("Start date must be before end date.")

        if end_date > pd.Timestamp.now():
            raise ValueError("End date cannot be in the future.")

        return True

    except Exception as e:
        raise ValueError(f"Invalid date format. Use 'YYYY-MM-DD'. Error: {str(e)}")


if __name__ == "__main__":
    # Quick test
    print("Testing data loader...")
    try:
        df = load_stock_data("AAPL", "2023-01-01", "2024-01-01")
        print(f"Successfully loaded {len(df)} rows of data for AAPL")
        print(df.head())
        print(f"\nColumns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error: {e}")
