"""
Unit tests for predict module
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import (
    prepare_latest_features,
    predict_next_day,
    predict_with_context,
    batch_predict,
    get_prediction_confidence_level
)
from src.features import prepare_dataset, add_technical_indicators
from src.model import train_model, time_series_train_test_split


@pytest.fixture
def sample_stock_data():
    """Create sample stock data."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n = len(dates)

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


@pytest.fixture
def trained_model(sample_stock_data):
    """Create a trained model for testing."""
    X, y = prepare_dataset(sample_stock_data)
    X_train, X_test, y_train, y_test = time_series_train_test_split(X, y, test_size=0.2)

    model = train_model(X_train, y_train, n_estimators=50)
    feature_names = X.columns.tolist()

    return model, feature_names


def test_prepare_latest_features(sample_stock_data):
    """Test preparing latest features."""
    df_with_features = add_technical_indicators(sample_stock_data)
    latest = prepare_latest_features(df_with_features)

    # Check it's a single row
    assert len(latest) == 1

    # Check it doesn't contain raw OHLCV or Date
    exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'label']
    for col in exclude_cols:
        assert col not in latest.columns


def test_predict_next_day(trained_model, sample_stock_data):
    """Test making a prediction."""
    model, feature_names = trained_model

    # Prepare features
    df_with_features = add_technical_indicators(sample_stock_data)
    latest = prepare_latest_features(df_with_features)

    # Make prediction
    result = predict_next_day(model, latest, feature_names)

    # Check result structure
    assert 'trend' in result
    assert 'confidence' in result
    assert 'predicted_class' in result
    assert 'probabilities' in result

    # Check trend is valid
    assert result['trend'] in ['UP', 'DOWN', 'FLAT']

    # Check confidence structure
    confidence = result['confidence']
    assert 'UP' in confidence
    assert 'DOWN' in confidence
    assert 'FLAT' in confidence

    # Check probabilities sum to ~1
    prob_sum = sum(confidence.values())
    assert np.isclose(prob_sum, 1.0, atol=0.01)

    # Check all probabilities are in valid range
    for prob in confidence.values():
        assert 0 <= prob <= 1


def test_predict_with_context(trained_model, sample_stock_data):
    """Test prediction with context."""
    model, feature_names = trained_model

    result = predict_with_context(model, sample_stock_data, feature_names)

    # Check all expected fields are present
    assert 'trend' in result
    assert 'confidence' in result
    assert 'latest_close' in result
    assert 'latest_date' in result
    assert 'next_day_prediction_date' in result
    assert 'latest_volume' in result

    # Check data types
    assert isinstance(result['trend'], str)
    assert isinstance(result['latest_close'], float)
    assert isinstance(result['latest_volume'], float)


def test_batch_predict(trained_model, sample_stock_data):
    """Test batch predictions."""
    model, feature_names = trained_model

    # Prepare dataset
    X, y = prepare_dataset(sample_stock_data)

    # Make batch predictions
    results = batch_predict(model, X.head(10), feature_names)

    # Check result structure
    assert len(results) == 10
    assert 'prediction' in results.columns
    assert 'trend' in results.columns
    assert 'prob_UP' in results.columns
    assert 'prob_DOWN' in results.columns
    assert 'prob_FLAT' in results.columns

    # Check all trends are valid
    assert all(trend in ['UP', 'DOWN', 'FLAT'] for trend in results['trend'])

    # Check probabilities are in valid range
    for col in ['prob_UP', 'prob_DOWN', 'prob_FLAT']:
        assert all((results[col] >= 0) & (results[col] <= 1))


def test_get_prediction_confidence_level():
    """Test confidence level categorization."""
    # High confidence
    high_conf = {'UP': 0.7, 'FLAT': 0.2, 'DOWN': 0.1}
    assert get_prediction_confidence_level(high_conf) == 'HIGH'

    # Medium confidence
    medium_conf = {'UP': 0.5, 'FLAT': 0.3, 'DOWN': 0.2}
    assert get_prediction_confidence_level(medium_conf) == 'MEDIUM'

    # Low confidence
    low_conf = {'UP': 0.35, 'FLAT': 0.35, 'DOWN': 0.30}
    assert get_prediction_confidence_level(low_conf) == 'LOW'


def test_prediction_consistency(trained_model, sample_stock_data):
    """Test that predictions are consistent."""
    model, feature_names = trained_model

    df_with_features = add_technical_indicators(sample_stock_data)
    latest = prepare_latest_features(df_with_features)

    # Make multiple predictions
    result1 = predict_next_day(model, latest, feature_names)
    result2 = predict_next_day(model, latest, feature_names)

    # Check consistency
    assert result1['trend'] == result2['trend']
    assert result1['predicted_class'] == result2['predicted_class']

    # Check confidence values are the same
    for key in result1['confidence']:
        assert np.isclose(
            result1['confidence'][key],
            result2['confidence'][key]
        )


def test_prediction_without_feature_names(trained_model, sample_stock_data):
    """Test prediction without providing feature names."""
    model, _ = trained_model

    df_with_features = add_technical_indicators(sample_stock_data)
    latest = prepare_latest_features(df_with_features)

    # Should work without feature_names
    result = predict_next_day(model, latest, feature_names=None)

    assert 'trend' in result
    assert result['trend'] in ['UP', 'DOWN', 'FLAT']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
