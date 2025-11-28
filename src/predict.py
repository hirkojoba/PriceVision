"""
Prediction Module
Handles next-day trend predictions using trained models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta


def prepare_latest_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the most recent feature set for prediction.

    Args:
        df: DataFrame with features already computed

    Returns:
        Single-row DataFrame with latest features
    """
    from src.features import add_technical_indicators

    # Add indicators if not present
    if 'ma_5' not in df.columns:
        df = add_technical_indicators(df)

    # Define feature columns (same as in prepare_dataset)
    exclude_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'label', 'next_close', 'next_return']
    feature_columns = [col for col in df.columns if col not in exclude_columns]

    # Get last row
    latest = df[feature_columns].iloc[[-1]].copy()

    return latest


def predict_next_day(
    model: Any,
    latest_features: pd.DataFrame,
    feature_names: Optional[list] = None
) -> Dict[str, Any]:
    """
    Predict next-day trend using trained model.

    Args:
        model: Trained classification model
        latest_features: Single-row DataFrame with feature values
        feature_names: Optional list of expected feature names for validation

    Returns:
        Dictionary with:
        - trend: Predicted trend ('UP', 'DOWN', 'FLAT')
        - confidence: Probability distribution
        - probabilities: Raw class probabilities
    """
    # Validate features if feature_names provided
    if feature_names is not None:
        missing_features = set(feature_names) - set(latest_features.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Reorder columns to match training order
        latest_features = latest_features[feature_names]

    # Get prediction and probabilities
    prediction = model.predict(latest_features)[0]
    probabilities = model.predict_proba(latest_features)[0]

    # Map class indices to trend names
    # Model classes: [-1, 0, 1] -> DOWN, FLAT, UP
    class_names = ['DOWN', 'FLAT', 'UP']

    # Get class order from model
    classes = model.classes_

    # Create probability mapping
    prob_dict = {}
    for cls, prob in zip(classes, probabilities):
        if cls == -1:
            prob_dict['DOWN'] = float(prob)
        elif cls == 0:
            prob_dict['FLAT'] = float(prob)
        elif cls == 1:
            prob_dict['UP'] = float(prob)

    # Map prediction to trend name
    if prediction == 1:
        trend = 'UP'
    elif prediction == -1:
        trend = 'DOWN'
    else:
        trend = 'FLAT'

    result = {
        'trend': trend,
        'confidence': prob_dict,
        'predicted_class': int(prediction),
        'probabilities': probabilities.tolist()
    }

    return result


def predict_with_context(
    model: Any,
    df: pd.DataFrame,
    feature_names: Optional[list] = None
) -> Dict[str, Any]:
    """
    Make prediction with additional context information.

    Args:
        model: Trained model
        df: Historical DataFrame with data
        feature_names: Optional feature names for validation

    Returns:
        Dictionary with prediction and context (latest price, date, etc.)
    """
    from src.features import add_technical_indicators

    # Ensure features are computed
    if 'ma_5' not in df.columns:
        df = add_technical_indicators(df)

    # Get latest features
    latest_features = prepare_latest_features(df)

    # Make prediction
    prediction_result = predict_next_day(model, latest_features, feature_names)

    # Add context
    latest_row = df.iloc[-1]

    # Estimate next prediction date (next business day)
    if 'Date' in df.columns:
        last_date = pd.to_datetime(latest_row['Date'])
        # Add 1 day (simple approach, doesn't account for weekends/holidays)
        next_date = last_date + timedelta(days=1)
    else:
        next_date = None

    result = {
        **prediction_result,
        'latest_close': float(latest_row['Close']) if 'Close' in latest_row else None,
        'latest_date': str(last_date.date()) if 'Date' in df.columns else None,
        'next_day_prediction_date': str(next_date.date()) if next_date else None,
        'latest_volume': float(latest_row['Volume']) if 'Volume' in latest_row else None
    }

    return result


def batch_predict(
    model: Any,
    X: pd.DataFrame,
    feature_names: Optional[list] = None
) -> pd.DataFrame:
    """
    Make predictions for multiple samples.

    Args:
        model: Trained model
        X: Feature DataFrame
        feature_names: Optional feature names for validation

    Returns:
        DataFrame with predictions and probabilities
    """
    if feature_names is not None:
        X = X[feature_names]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Map predictions to trend names
    trend_map = {-1: 'DOWN', 0: 'FLAT', 1: 'UP'}
    trends = [trend_map[p] for p in predictions]

    # Create results DataFrame
    results = pd.DataFrame({
        'prediction': predictions,
        'trend': trends,
        'prob_DOWN': probabilities[:, 0] if -1 in model.classes_ else 0,
        'prob_FLAT': probabilities[:, 1] if 0 in model.classes_ else 0,
        'prob_UP': probabilities[:, 2] if 1 in model.classes_ else 0
    })

    return results


def get_prediction_confidence_level(confidence: Dict[str, float]) -> str:
    """
    Categorize prediction confidence.

    Args:
        confidence: Dictionary with trend probabilities

    Returns:
        Confidence level: 'HIGH', 'MEDIUM', or 'LOW'
    """
    max_prob = max(confidence.values())

    if max_prob >= 0.6:
        return 'HIGH'
    elif max_prob >= 0.4:
        return 'MEDIUM'
    else:
        return 'LOW'


if __name__ == "__main__":
    # Quick test
    from data_loader import load_stock_data
    from features import prepare_dataset
    from model import train_model, time_series_train_test_split

    print("Testing prediction module...")
    try:
        # Load and prepare data
        df = load_stock_data("AAPL", "2023-01-01", "2024-01-01")
        X, y = prepare_dataset(df)

        # Quick train
        X_train, X_test, y_train, y_test = time_series_train_test_split(X, y, test_size=0.2)
        model = train_model(X_train, y_train)

        # Prepare features from original df
        from features import add_technical_indicators
        df_with_features = add_technical_indicators(df)

        # Make prediction
        result = predict_with_context(model, df_with_features, X.columns.tolist())

        print("\n=== Prediction Result ===")
        print(f"Trend: {result['trend']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Latest Close: ${result['latest_close']:.2f}")
        print(f"Latest Date: {result['latest_date']}")
        print(f"Prediction for: {result['next_day_prediction_date']}")
        print(f"Confidence Level: {get_prediction_confidence_level(result['confidence'])}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
