"""
Model Training and Evaluation Module
Handles ML model training, evaluation, and persistence.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Tuple, Dict, Any, Optional
import os


def time_series_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data chronologically for time series (no shuffling).

    Args:
        X: Feature DataFrame
        y: Label Series
        test_size: Proportion of data for testing (default 0.2)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    split_idx = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "random_forest",
    **kwargs
) -> Any:
    """
    Train a classification model.

    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model ('random_forest' or 'gradient_boosting')
        **kwargs: Additional parameters for the model

    Returns:
        Trained model
    """
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', None),
            min_samples_split=kwargs.get('min_samples_split', 5),
            min_samples_leaf=kwargs.get('min_samples_leaf', 2),
            random_state=kwargs.get('random_state', 42),
            n_jobs=kwargs.get('n_jobs', -1)
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=kwargs.get('n_estimators', 200),
            learning_rate=kwargs.get('learning_rate', 0.1),
            max_depth=kwargs.get('max_depth', 5),
            random_state=kwargs.get('random_state', 42)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train the model
    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluate model performance.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary containing:
        - accuracy: Overall accuracy
        - classification_report: Detailed metrics per class
        - confusion_matrix: Confusion matrix array
        - precision, recall, f1: Per-class metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(
        y_test,
        y_pred,
        target_names=['DOWN', 'FLAT', 'UP'],
        output_dict=False
    )

    # Get precision, recall, f1 per class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=[-1, 0, 1],
        average=None,
        zero_division=0
    )

    metrics = {
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'precision': {
            'DOWN': precision[0],
            'FLAT': precision[1],
            'UP': precision[2]
        },
        'recall': {
            'DOWN': recall[0],
            'FLAT': recall[1],
            'UP': recall[2]
        },
        'f1_score': {
            'DOWN': f1[0],
            'FLAT': f1[1],
            'UP': f1[2]
        },
        'support': {
            'DOWN': support[0],
            'FLAT': support[1],
            'UP': support[2]
        }
    }

    return metrics


def get_feature_importance(
    model: Any,
    feature_names: list
) -> pd.DataFrame:
    """
    Extract feature importance from trained model.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names

    Returns:
        DataFrame with features and their importance scores, sorted by importance
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not support feature importance extraction")

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })

    # Sort by importance descending
    importance_df = importance_df.sort_values('importance', ascending=False)

    return importance_df


def save_model(model: Any, path: str, feature_names: Optional[list] = None) -> None:
    """
    Save trained model to disk.

    Args:
        model: Trained model
        path: File path to save the model
        feature_names: Optional list of feature names to save with model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save model and metadata
    model_data = {
        'model': model,
        'feature_names': feature_names
    }

    joblib.dump(model_data, path)


def load_model(path: str) -> Tuple[Any, Optional[list]]:
    """
    Load trained model from disk.

    Args:
        path: File path to the saved model

    Returns:
        Tuple of (model, feature_names)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    model_data = joblib.load(path)

    # Handle both old format (just model) and new format (dict)
    if isinstance(model_data, dict):
        model = model_data['model']
        feature_names = model_data.get('feature_names', None)
    else:
        model = model_data
        feature_names = None

    return model, feature_names


def train_and_evaluate_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    model_type: str = "random_forest",
    save_path: Optional[str] = None,
    **model_kwargs
) -> Tuple[Any, Dict[str, Any]]:
    """
    Complete pipeline: split, train, and evaluate.

    Args:
        X: Features
        y: Labels
        test_size: Test split proportion
        model_type: Type of model to train
        save_path: Optional path to save the model
        **model_kwargs: Additional model parameters

    Returns:
        Tuple of (trained_model, evaluation_metrics)
    """
    # Split data
    X_train, X_test, y_train, y_test = time_series_train_test_split(
        X, y, test_size=test_size
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Train label distribution:\n{y_train.value_counts()}")
    print(f"Test label distribution:\n{y_test.value_counts()}")

    # Train model
    print(f"\nTraining {model_type} model...")
    model = train_model(X_train, y_train, model_type=model_type, **model_kwargs)

    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])

    # Save if path provided
    if save_path:
        feature_names = X.columns.tolist()
        save_model(model, save_path, feature_names)
        print(f"\nModel saved to: {save_path}")

    return model, metrics


if __name__ == "__main__":
    # Quick test
    from data_loader import load_stock_data
    from features import prepare_dataset

    print("Testing model training...")
    try:
        # Load and prepare data
        df = load_stock_data("AAPL", "2022-01-01", "2024-01-01")
        X, y = prepare_dataset(df)

        # Train and evaluate
        model, metrics = train_and_evaluate_pipeline(
            X, y,
            test_size=0.2,
            model_type="random_forest"
        )

        # Show feature importance
        feature_names = X.columns.tolist()
        importance = get_feature_importance(model, feature_names)
        print("\nTop 10 Most Important Features:")
        print(importance.head(10))

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
